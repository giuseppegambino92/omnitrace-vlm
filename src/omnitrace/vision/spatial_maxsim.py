"""Vectorised late-interaction (MaxSim) retrieval, 2-D activation heatmaps and
bounding-box grounding, backed by a single-file SQLite index.

Scoring
-------
* **Page score** (ColBERT / ColPali MaxSim):  ``sum_i max_j <q_i, p_j>`` over query
  token vectors ``q_i`` and page patch vectors ``p_j``.  All pages share the same
  grid so the whole corpus is scored with one matmul + reshape + max.
* **Two-stage search.**  1-bit sign packs give an asymmetric MaxSim estimate
  (``<q, sign(p)> * ||p|| * sqrt(pi / 2D)``) to shortlist pages; the shortlist is
  re-scored at the requested precision (float32 / int8 / 2-bit).
* **Activation heatmap** (grounding).  Per token similarities are normalised by the
  token's best patch on the page, tokens are weighted by how informative they are,
  and patches are combined with a weighted *geometric* mean - a soft AND, so the
  winning patch is the one that satisfies every query constraint at once (the cell
  at the intersection of the queried row and column, not the column header).
* **Bounding box.**  With a structure layer: the leaf region with the highest mean
  heat.  Without one (raw screenshots): the connected component of the
  thresholded heatmap that contains the peak.
"""

from __future__ import annotations

import io
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from collections import Counter

from omnitrace.config import VisionConfig
from omnitrace.textfeat import tokenize
from omnitrace.vision.patch_encoder import (
    DESCRIPTOR_ROLES, NON_LEAF_ROLES, Page, PageEncoding, PatchEncoder, Region, binarize, dequantize_int2,
    dequantize_int8, quantize_int2, quantize_int8, unpack_signs,
)

__all__ = ["GroundedHit", "VisualIndex", "PRECISIONS", "maxsim_scores", "activation_heatmap"]

PRECISIONS = ("float32", "int8", "int2", "binary")


@dataclass
class GroundedHit:
    page_id: str
    doc_id: str
    rank: int
    score: float
    bbox: tuple[float, float, float, float]
    grounding: str                        # "region" | "heatmap"
    heatmap: np.ndarray = field(repr=False)
    region_id: str | None = None
    region_text: str = ""
    region_role: str = ""
    token_peaks: dict[str, tuple[int, int, float]] = field(default_factory=dict)
    latency_ms: float = 0.0

    def to_dict(self, include_heatmap: bool = False) -> dict:
        d = {
            "page_id": self.page_id, "doc_id": self.doc_id, "rank": self.rank,
            "score": round(float(self.score), 5), "bbox": [round(float(v), 1) for v in self.bbox],
            "grounding": self.grounding, "region_id": self.region_id,
            "region_text": self.region_text, "region_role": self.region_role,
            "token_peaks": {k: [int(r), int(c), round(float(s), 4)] for k, (r, c, s) in self.token_peaks.items()},
            "latency_ms": round(self.latency_ms, 3),
        }
        if include_heatmap:
            d["heatmap"] = np.round(self.heatmap, 4).tolist()
        return d


def token_maxima(Q: np.ndarray, P: np.ndarray, n_pages: int, n_patches: int) -> np.ndarray:
    """Q: (nq, D); P: (n_pages * n_patches, D) -> (nq, n_pages) per-token best-patch similarity."""
    if len(Q) == 0 or n_pages == 0:
        return np.zeros((len(Q), n_pages), dtype=np.float32)
    return (Q @ P.T).reshape(len(Q), n_pages, n_patches).max(axis=2)


def maxsim_scores(Q: np.ndarray, P: np.ndarray, n_pages: int, n_patches: int) -> np.ndarray:
    """Classic ColBERT / ColPali late interaction: ``sum_i max_j <q_i, p_j>`` per page."""
    return token_maxima(Q, P, n_pages, n_patches).sum(axis=0)


def calibrated_scores(M: np.ndarray, idf: np.ndarray, noise: float, floor: float) -> np.ndarray:
    """Aggregate per-token maxima ``M`` (nq, n_pages) with per-token calibration.

    Each token contributes ``idf * clip((m - noise) / (best - noise), 0, 1)`` where ``best`` is
    the token's best page: a rare identifier matched inside a long label counts as much as a
    common word matched by a one-word header.  Tokens with no real match anywhere
    (``best < floor``) contribute nothing."""
    best = M.max(axis=1, keepdims=True)
    rel = np.clip((M - noise) / np.maximum(best - noise, 1e-9), 0.0, 1.0)
    rel[best[:, 0] < floor] = 0.0
    return (idf[:, None] * rel).sum(axis=0)


def activation_heatmap(sims: np.ndarray, grid: tuple[int, int], eps: float, floor: float,
                       power: float = -1.0, feasible: float = 0.0,
                       scope: np.ndarray | None = None, noise: float = 0.0,
                       prior: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Soft-AND heatmap from per-token patch similarities ``sims`` (nq, n_patches).

    * similarities below ``noise`` (the hashing noise level, ~2.5 / sqrt(text_dim)) carry
      no evidence and are zeroed; the rest are normalised by the token's best patch;
    * tokens are weighted by informativeness (how far their peak clears ``floor``);
    * ``scope`` marks page-scope tokens (best matched by the page title) - they are
      satisfied by every element of the page and are excluded from grounding;
    * patches combine tokens with a weighted power mean (``power`` < 0 = soft minimum:
      a patch must satisfy *every* constraint);
    * if no patch reaches ``feasible``, the token blocking the conjunction is dropped
      (greedily) - e.g. a widget-type word like "card" that is lexically matched
      somewhere unrelated.  ``prior`` (per patch / region multiplier) takes part in that
      choice, so the relaxation never favours an element the prior rules against.

    Returns (heatmap in [0, 1] of shape grid, final per-token weights)."""
    s = np.clip(sims, 0.0, None)
    peak = s.max(axis=1)
    w = np.clip((peak - floor) / max(1e-6, 0.3 - floor), 0.0, 1.0)
    if scope is not None and (w * ~scope).sum() > 0:
        w = np.where(scope, 0.0, w)
    if w.sum() <= 0:
        heat = s.sum(axis=0) * (1.0 if prior is None else prior)
        w = np.ones_like(peak)
    else:
        norm = eps + np.clip(s - noise, 0.0, None) / np.maximum(peak[:, None] - noise, 1e-9)

        pr = np.ones(s.shape[1], dtype=np.float64) if prior is None else prior

        def conj(wv: np.ndarray) -> np.ndarray:
            if power == 0:
                h = np.exp((wv[:, None] * np.log(norm)).sum(axis=0) / wv.sum()) - eps
            else:
                h = ((wv[:, None] * norm ** power).sum(axis=0) / wv.sum()) ** (1.0 / power) - eps
            return h * pr

        heat = conj(w)
        while heat.max() < feasible and (w > 0).sum() > 1:
            best = None
            for i in np.nonzero(w > 0)[0]:
                w2 = w.copy()
                w2[i] = 0.0
                h2 = conj(w2)
                if best is None or h2.max() > best[0]:
                    best = (h2.max(), i, h2)
            w[best[1]] = 0.0
            heat = best[2]
    heat = np.clip(heat, 0.0, None)
    m = heat.max()
    if m > 0:
        heat = heat / m
    return heat.reshape(grid).astype(np.float32), w


class VisualIndex:
    """In-memory multi-vector page index with SQLite persistence."""

    def __init__(self, cfg: VisionConfig | None = None):
        self.cfg = cfg or VisionConfig()
        self.encoder = PatchEncoder(self.cfg)
        self.pages: list[PageEncoding] = []
        self.images: dict[str, bytes] = {}
        self._f32: np.ndarray | None = None
        self._i8: tuple[np.ndarray, np.ndarray] | None = None
        self._i2: tuple[np.ndarray, np.ndarray] | None = None
        self._bin: tuple[np.ndarray, np.ndarray] | None = None
        self._matrix_cache: dict[str, np.ndarray] = {}
        self._df: Counter = Counter()
        self._dirty = True

    # ------------------------------------------------------------------ building
    @property
    def n_patches(self) -> int:
        return self.cfg.grid_rows * self.cfg.grid_cols

    def __len__(self) -> int:
        return len(self.pages)

    def add_page(self, page: Page, store_image: bool = True) -> PageEncoding:
        if any(p.page_id == page.page_id for p in self.pages):
            raise ValueError(f"duplicate page_id {page.page_id!r}")
        enc = self.encoder.encode_page(page)
        self.pages.append(enc)
        self._count_df(enc.regions)
        if store_image:
            buf = io.BytesIO()
            Image.fromarray(np.asarray(page.image, dtype=np.uint8)).save(buf, format="PNG", optimize=False)
            self.images[page.page_id] = buf.getvalue()
        self._dirty = True
        return enc

    def _count_df(self, regions: list[Region]) -> None:
        self._df.update({t for r in regions for t in tokenize(r.text, bind="doc")})

    def idf(self, tokens: list[str]) -> np.ndarray:
        """BM25-style inverse page frequency, normalised to mean 1 over the query."""
        n = max(1, len(self.pages))
        v = np.array([np.log(1.0 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5)) for t in tokens],
                     dtype=np.float32)
        return v / max(float(v.mean()), 1e-9) if len(v) else v

    def add_pages(self, pages, store_image: bool = True) -> None:
        for p in pages:
            self.add_page(p, store_image=store_image)

    def _build(self) -> None:
        if not self._dirty:
            return
        if not self.pages:
            d = self.cfg.dim
            self._f32 = np.zeros((0, d), dtype=np.float32)
        else:
            self._f32 = np.concatenate([p.vectors for p in self.pages]).astype(np.float32)
        self._i8 = quantize_int8(self._f32)
        self._i2 = quantize_int2(self._f32)
        self._bin = binarize(self._f32)
        self._matrix_cache = {}
        self._dirty = False

    def matrix(self, precision: str) -> np.ndarray:
        """Patch matrix (dequantised to float32 for BLAS) at a given storage precision."""
        self._build()
        if precision == "float32":
            return self._f32
        if precision == "int8":
            return dequantize_int8(*self._i8)
        if precision == "int2":
            return dequantize_int2(*self._i2, self.cfg.dim)
        if precision == "binary":
            bits, norms = self._bin
            return unpack_signs(bits, self.cfg.dim) * (norms[:, None] * np.sqrt(np.pi / (2 * self.cfg.dim)))
        raise ValueError(f"unknown precision {precision!r}; expected one of {PRECISIONS}")

    def bytes_per_patch(self) -> dict[str, int]:
        d = self.cfg.dim
        return {"float32": 4 * d, "int8": d + 4, "int2": (d + 3) // 4 + 4, "binary": (d + 7) // 8 + 4}

    # ------------------------------------------------------------------ search
    def _page_slice(self, i: int) -> slice:
        return slice(i * self.n_patches, (i + 1) * self.n_patches)

    def search(self, query: str, k: int = 3, precision: str = "int8", two_stage: bool = True,
               ground: bool = True) -> list[GroundedHit]:
        t0 = time.perf_counter()
        self._build()
        n = len(self.pages)
        if precision not in PRECISIONS:
            raise ValueError(f"unknown precision {precision!r}; expected one of {PRECISIONS}")
        toks, Q = self.encoder.encode_query(query)
        if n == 0 or len(toks) == 0:
            return []
        idf = self.idf(toks)
        noise = 2.5 / np.sqrt(self.cfg.text_dim)

        def page_scores(Pm: np.ndarray, n_pages: int) -> np.ndarray:
            M = token_maxima(Q, Pm, n_pages, self.n_patches)
            if self.cfg.calibrated:
                return calibrated_scores(M, idf, noise, self.cfg.match_floor)
            return M.sum(axis=0)  # raw ColBERT / ColPali MaxSim

        P = self._cached_matrix(precision)
        if two_stage and precision != "binary" and n > self.cfg.candidate_pages:
            coarse = page_scores(self._cached_matrix("binary"), n)
            cand = np.argsort(-coarse)[: max(self.cfg.candidate_pages, k)]
        else:
            cand = np.arange(n)
        Pc = np.concatenate([P[self._page_slice(int(i))] for i in cand])
        scores = page_scores(Pc, len(cand))
        order = np.argsort(-scores)[:k]
        hits = []
        for rank, o in enumerate(order):
            pi = int(cand[o])
            enc = self.pages[pi]
            hits.append(self._ground(enc, toks, Q, P[self._page_slice(pi)], float(scores[o]), rank) if ground else
                        GroundedHit(enc.page_id, enc.doc_id, rank, float(scores[o]),
                                    (0.0, 0.0, float(enc.width), float(enc.height)), "page",
                                    np.zeros(enc.grid, dtype=np.float32)))
        dt = (time.perf_counter() - t0) * 1000
        for h in hits:
            h.latency_ms = dt
        return hits

    def _cached_matrix(self, precision: str) -> np.ndarray:
        self._build()
        if precision not in self._matrix_cache:
            self._matrix_cache[precision] = self.matrix(precision)
        return self._matrix_cache[precision]

    def _ground(self, enc: PageEncoding, toks: list[str], Q: np.ndarray, P: np.ndarray, score: float,
                rank: int) -> GroundedHit:
        """Heatmap + bbox for one page.  ``Q``/``P`` are rotated query / page patch matrices."""
        cfg = self.cfg
        sims = Q @ P.T
        gr, gc = enc.grid
        noise = 2.5 / np.sqrt(cfg.text_dim)
        if enc.leaf_text is None:
            enc.leaf_ids, enc.leaf_text = self.encoder.leaf_vectors(enc.regions)
        by_id = {r.id: r for r in enc.regions}
        leaves = [by_id[i] for i in enc.leaf_ids if by_id[i].area > 0]
        keep = [k for k, i in enumerate(enc.leaf_ids) if by_id[i].area > 0]
        scope = None
        if leaves:
            pb = self.encoder.patch_boxes(enc.width, enc.height)
            rb = np.array([r.bbox for r in leaves], dtype=np.float64)
            inter = PatchEncoder.overlap_matrix(rb, pb)          # (regions, patches)
            # region embeddings: exact contextual text vector + pooled visual / positional patch features
            cover = inter / np.maximum(inter.sum(axis=1, keepdims=True), 1e-9)
            R = self.encoder.R
            pooled = (cover @ P) @ R                              # un-rotated pooled patch vectors
            pooled[:, : cfg.text_dim] = enc.leaf_text[keep]
            region_sims = (Q @ R) @ pooled.T                      # (nq, regions)
            is_title = np.array([r.role == "title" for r in leaves])
            if is_title.any() and not is_title.all():
                pos = np.clip(region_sims, 0, None)
                t_max, o_max = pos[:, is_title].max(axis=1), pos[:, ~is_title].max(axis=1)
                scope = (t_max >= 0.6 * o_max) & (t_max > noise)
        heat, w = activation_heatmap(sims, enc.grid, cfg.heat_eps, cfg.token_floor,
                                     cfg.heat_power, cfg.and_feasible, scope, noise=noise)
        peaks = {}
        for t, row in zip(toks, sims):
            j = int(np.argmax(row))
            peaks[t] = (j // gc, j % gc, float(row[j]))
        if leaves:
            # answer-bearing prior: a descriptor (header / label / caption...) whose words are
            # all restated by the query is a *pointer* to the answer, not the answer itself
            qset = set(toks)
            prior = np.array([0.5 if (r.role in DESCRIPTOR_ROLES and tokenize(r.text, bind="query")
                                      and set(tokenize(r.text, bind="query")) <= qset) else 1.0
                              for r in leaves])
            region_heat, _ = activation_heatmap(
                region_sims, (1, len(leaves)), cfg.heat_eps, cfg.token_floor, cfg.heat_power,
                cfg.and_feasible, scope=scope, noise=noise, prior=prior)
            patch_mean = (inter @ heat.ravel()) / np.maximum(inter.sum(axis=1), 1e-9)
            score_r = 0.75 * region_heat.ravel() + 0.25 * patch_mean * prior
            best = int(np.argmax(score_r))
            r = leaves[best]
            return GroundedHit(enc.page_id, enc.doc_id, rank, score, tuple(map(float, r.bbox)), "region",
                               heat, r.id, r.text, r.role, peaks)
        bbox = heatmap_bbox(heat, enc.width, enc.height)
        return GroundedHit(enc.page_id, enc.doc_id, rank, score, bbox, "heatmap", heat, None, "", "", peaks)

    # ------------------------------------------------------------------ crops
    def page_image(self, page_id: str) -> np.ndarray:
        if page_id not in self.images:
            raise KeyError(f"no stored image for page {page_id!r}")
        return np.asarray(Image.open(io.BytesIO(self.images[page_id])).convert("RGB"))

    def crop(self, hit: GroundedHit, pad: int = 4) -> np.ndarray:
        """High-resolution crop of the grounded bounding box (the visual citation)."""
        img = self.page_image(hit.page_id)
        x1, y1, x2, y2 = hit.bbox
        h, w = img.shape[:2]
        x1, y1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
        x2, y2 = min(w, int(np.ceil(x2)) + pad), min(h, int(np.ceil(y2)) + pad)
        return img[y1:y2, x1:x2].copy()

    # ------------------------------------------------------------------ persistence
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
    CREATE TABLE IF NOT EXISTS pages(
        ord INTEGER PRIMARY KEY, page_id TEXT UNIQUE, doc_id TEXT, width INTEGER, height INTEGER,
        meta TEXT, image BLOB);
    CREATE TABLE IF NOT EXISTS patches(
        page_id TEXT PRIMARY KEY, f32 BLOB, i8 BLOB, i8_scale BLOB, i2 BLOB, i2_sigma BLOB,
        bits BLOB, norms BLOB);
    CREATE TABLE IF NOT EXISTS regions(
        page_id TEXT, ord INTEGER, region_id TEXT, x1 REAL, y1 REAL, x2 REAL, y2 REAL,
        role TEXT, text TEXT, parent TEXT, links TEXT, PRIMARY KEY(page_id, ord));
    """

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        self._build()
        if path.exists():
            path.unlink()
        con = sqlite3.connect(path)
        try:
            con.executescript(self._SCHEMA)
            cfg = {k: getattr(self.cfg, k) for k in self.cfg.__dataclass_fields__}
            con.execute("INSERT INTO meta VALUES('vision_config', ?)", (json.dumps(cfg),))
            i8, i8s = self._i8
            i2, i2s = self._i2
            bits, norms = self._bin
            for i, p in enumerate(self.pages):
                sl = self._page_slice(i)
                con.execute("INSERT INTO pages VALUES(?,?,?,?,?,?,?)",
                            (i, p.page_id, p.doc_id, p.width, p.height, json.dumps(p.meta),
                             self.images.get(p.page_id)))
                con.execute("INSERT INTO patches VALUES(?,?,?,?,?,?,?,?)",
                            (p.page_id, self._f32[sl].tobytes(), i8[sl].tobytes(), i8s[sl].tobytes(),
                             i2[sl].tobytes(), i2s[sl].tobytes(), bits[sl].tobytes(), norms[sl].tobytes()))
                con.executemany(
                    "INSERT INTO regions VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    [(p.page_id, j, r.id, *r.bbox, r.role, r.text, r.parent, json.dumps(r.links))
                     for j, r in enumerate(p.regions)])
            con.commit()
        finally:
            con.close()
        return path

    @classmethod
    def load(cls, path: str | Path) -> "VisualIndex":
        con = sqlite3.connect(Path(path))
        try:
            cfg_json = json.loads(con.execute("SELECT value FROM meta WHERE key='vision_config'").fetchone()[0])
            idx = cls(VisionConfig(**cfg_json))
            d = idx.cfg.dim
            np_ = idx.n_patches
            for page_id, doc_id, width, height, meta, image in con.execute(
                    "SELECT page_id, doc_id, width, height, meta, image FROM pages ORDER BY ord"):
                (f32,) = con.execute("SELECT f32 FROM patches WHERE page_id=?", (page_id,)).fetchone()
                vec = np.frombuffer(f32, dtype=np.float32).reshape(np_, d).copy()
                regions = [Region(rid, (x1, y1, x2, y2), text, role, parent, json.loads(links))
                           for rid, x1, y1, x2, y2, role, text, parent, links in con.execute(
                               "SELECT region_id,x1,y1,x2,y2,role,text,parent,links FROM regions "
                               "WHERE page_id=? ORDER BY ord", (page_id,))]
                idx.pages.append(PageEncoding(page_id, doc_id, width, height, (idx.cfg.grid_rows, idx.cfg.grid_cols),
                                              vec, regions, json.loads(meta)))
                idx._count_df(regions)
                if image is not None:
                    idx.images[page_id] = image
        finally:
            con.close()
        idx._dirty = True
        return idx


def heatmap_bbox(heat: np.ndarray, width: int, height: int, rel_threshold: float = 0.5
                 ) -> tuple[float, float, float, float]:
    """Pixel bbox of the above-threshold connected component containing the heat peak."""
    gr, gc = heat.shape
    if heat.max() <= 0:
        return (0.0, 0.0, float(width), float(height))
    mask = heat >= rel_threshold * heat.max()
    lab, _ = ndimage.label(mask)
    pr, pc = np.unravel_index(int(np.argmax(heat)), heat.shape)
    rows, cols = np.nonzero(lab == lab[pr, pc])
    return (cols.min() * width / gc, rows.min() * height / gr,
            (cols.max() + 1) * width / gc, (rows.max() + 1) * height / gr)
