"""2-D spatial-grid patch encoder (ColPali-style multi-vector pages, built from first principles).

A page is an RGB raster plus an optional *structure layer*: the text / DOM / SVG
regions that PDF text layers, browser DOM snapshots and SVG exports carry
natively.  No OCR is ever performed - pixels are described by colour and
gradient statistics, text comes from the structure layer.

Pipeline per page
-----------------
1. **Layout graph.**  Leaf regions become nodes; edges encode the 2-D reading
   relations that flat text linearisation destroys: row headers / labels to the
   left, column headers above, the nearest section heading above, the caption
   of the enclosing container, explicit connectors (schematic wires) and the
   immediate left / above neighbours.
2. **Contextual region vectors.**  ``normalize(own + lambda * normalize(sum w_e * neighbour))``
   - a table cell therefore "knows" its row and column headers.
3. **Rasterisation.**  Region vectors are splatted onto a ``grid_rows x grid_cols``
   patch grid weighted by overlap (``|r & p| / min(|r|, |p|)``).
4. **Visual descriptors.**  Per patch: 12-bin hue histogram of saturated pixels,
   achromatic fractions, 8-bin magnitude-weighted gradient-orientation histogram,
   edge density, luminance stats and colour moments (32 dims).
5. **Spatial coordinates.**  Linear + RBF encodings of the patch centre (16 dims).
6. **Rotation + quantisation.**  The 512-d patch vectors are multiplied by a
   seeded random orthogonal matrix (dot products are preserved, energy is spread
   evenly across dimensions) and stored as float32, int8 (per-vector absmax),
   2-bit (Lloyd-Max Gaussian levels) and 1-bit sign packs (SimHash).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from PIL import Image

from omnitrace.config import VisionConfig
from omnitrace.textfeat import text_vector, token_vector, tokenize

__all__ = [
    "Region", "Page", "PageEncoding", "PatchEncoder", "build_layout_graph",
    "quantize_int8", "dequantize_int8", "quantize_int2", "dequantize_int2",
    "binarize", "unpack_signs", "COLOR_WORDS", "POSITION_WORDS",
]

HEADER_ROLES = {"header", "colheader"}
DESCRIPTOR_ROLES = {"header", "colheader", "rowheader", "label", "heading", "caption", "title"}
ROW_ANCHOR_ROLES = {"rowheader", "label", "header"}
NON_LEAF_ROLES = {"container", "edge"}


@dataclass
class Region:
    """A structure-layer element (PDF text block, DOM node, SVG group)."""

    id: str
    bbox: tuple[float, float, float, float]  # pixel x1, y1, x2, y2
    text: str = ""
    role: str = "text"
    parent: str | None = None
    links: list[str] = field(default_factory=list)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def to_dict(self) -> dict:
        return {"id": self.id, "bbox": list(self.bbox), "text": self.text, "role": self.role,
                "parent": self.parent, "links": list(self.links)}

    @classmethod
    def from_dict(cls, d: dict) -> "Region":
        return cls(d["id"], tuple(float(v) for v in d["bbox"]), d.get("text", ""),
                   d.get("role", "text"), d.get("parent"), list(d.get("links", [])))


@dataclass
class Page:
    page_id: str
    image: np.ndarray  # H x W x 3 uint8
    regions: list[Region] = field(default_factory=list)
    doc_id: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


@dataclass
class PageEncoding:
    page_id: str
    doc_id: str
    width: int
    height: int
    grid: tuple[int, int]
    vectors: np.ndarray            # (n_patches, D) float32, rotated space
    regions: list[Region]
    meta: dict = field(default_factory=dict)
    leaf_ids: list[str] = field(default_factory=list)          # leaf regions (proposals)
    leaf_text: np.ndarray | None = None                         # (n_leaves, text_dim) contextual unit vectors


# ----------------------------------------------------------------------------------------
# layout graph
# ----------------------------------------------------------------------------------------

def _overlap_1d(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


def _near_w(a: Region, b: Region) -> float:
    """Weight of the immediately adjacent (left / above) neighbour ``b`` as context for ``a``."""
    if a.role == b.role and a.role in DESCRIPTOR_ROLES:
        return 0.05  # sibling headers / labels do not describe each other
    return 0.3 if b.role in DESCRIPTOR_ROLES or a.role == "cell" else 0.05


def build_layout_graph(regions: list[Region]) -> dict[str, dict[str, float]]:
    """Return ``{region_id: {neighbour_id: weight}}`` of *incoming* context edges."""
    by_id = {r.id: r for r in regions}
    leaves = [r for r in regions if r.role not in NON_LEAF_ROLES]
    groups: dict[str | None, list[Region]] = defaultdict(list)
    for r in leaves:
        groups[r.parent].append(r)

    graph: dict[str, dict[str, float]] = {r.id: {} for r in leaves}

    def put(a: str, b: str, w: float) -> None:
        if a != b and w > graph[a].get(b, 0.0):
            graph[a][b] = w

    for parent, members in groups.items():
        for a in members:
            ax1, ay1, ax2, ay2 = a.bbox
            ah, aw = max(1e-6, ay2 - ay1), max(1e-6, ax2 - ax1)
            row_before, col_above = [], []
            for b in members:
                if b is a:
                    continue
                bx1, by1, bx2, by2 = b.bbox
                bh, bw = max(1e-6, by2 - by1), max(1e-6, bx2 - bx1)
                ov_y = _overlap_1d(ay1, ay2, by1, by2) / min(ah, bh)
                ov_x = _overlap_1d(ax1, ax2, bx1, bx2) / min(aw, bw)
                if ov_y >= 0.5 and bx2 <= ax1 + 2:
                    row_before.append(b)
                elif ov_x >= 0.5 and by2 <= ay1 + 2:
                    col_above.append(b)
                if b.role == "heading" and by2 <= ay1 + 2 and 0 < ov_x < 0.5:
                    col_above.append(b)  # heading spanning several columns
            if row_before:
                leftmost = min(row_before, key=lambda r: r.bbox[0])
                nearest = max(row_before, key=lambda r: r.bbox[2])
                for b in row_before:
                    put(a.id, b.id, 0.05)
                put(a.id, nearest.id, _near_w(a, nearest))
                if a.role not in HEADER_ROLES and (
                        leftmost.role in ROW_ANCHOR_ROLES or (a.role == "cell" and leftmost.role != "cell")):
                    put(a.id, leftmost.id, 1.0)
                for b in row_before:
                    if b.role in ROW_ANCHOR_ROLES and not (a.role in HEADER_ROLES and b.role in HEADER_ROLES):
                        put(a.id, b.id, 1.0)
            if col_above:
                nearest = max(col_above, key=lambda r: r.bbox[3])
                if a.bbox[1] - nearest.bbox[3] < 2.5 * ah:
                    put(a.id, nearest.id, _near_w(a, nearest))
                for b in col_above:
                    put(a.id, b.id, 1.0 if b.role in HEADER_ROLES else 0.05)
                headings = [b for b in col_above if b.role == "heading"]
                if headings:
                    put(a.id, max(headings, key=lambda r: r.bbox[3]).id, 1.0)
            for b in members:
                if b.role in ("title", "caption") and b is not a:
                    put(a.id, b.id, 0.3)
        has_caption = any(m.role in ("title", "caption") for m in members)
        if parent is not None and parent in by_id and by_id[parent].text and not has_caption:
            for a in members:
                graph[a.id][f"__parent__{parent}"] = 0.3
    for a in leaves:
        for l in a.links:
            if l in graph:
                put(a.id, l, 0.6)
                put(l, a.id, 0.6)
    return graph


# ----------------------------------------------------------------------------------------
# quantisation
# ----------------------------------------------------------------------------------------

_LM_THRESH = 0.9816   # Lloyd-Max 4-level Gaussian quantiser (units of sigma)
_LM_LEVELS = np.array([-1.510, -0.4528, 0.4528, 1.510], dtype=np.float32)


def quantize_int8(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.abs(x).max(axis=1) / 127.0
    scale[scale == 0] = 1.0
    q = np.clip(np.rint(x / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


def dequantize_int8(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * scale[:, None]


def quantize_int2(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sigma = x.std(axis=1)
    sigma[sigma == 0] = 1.0
    z = x / sigma[:, None]
    codes = (z > -_LM_THRESH).astype(np.uint8) + (z > 0).astype(np.uint8) + (z > _LM_THRESH).astype(np.uint8)
    n, d = codes.shape
    pad = (-d) % 4
    if pad:
        codes = np.pad(codes, ((0, 0), (0, pad)))
    c = codes.reshape(n, (d + pad) // 4, 4)
    packed = (c[:, :, 0] | (c[:, :, 1] << 2) | (c[:, :, 2] << 4) | (c[:, :, 3] << 6)).astype(np.uint8)
    return packed, sigma.astype(np.float32)


def dequantize_int2(packed: np.ndarray, sigma: np.ndarray, dim: int) -> np.ndarray:
    codes = np.stack([(packed >> s) & 3 for s in (0, 2, 4, 6)], axis=-1).reshape(packed.shape[0], -1)[:, :dim]
    return _LM_LEVELS[codes] * sigma[:, None]


def binarize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """1-bit SimHash pack (sign of the rotated vector) + L2 norm for asymmetric scoring."""
    return np.packbits(x > 0, axis=1), np.linalg.norm(x, axis=1).astype(np.float32)


def unpack_signs(bits: np.ndarray, dim: int) -> np.ndarray:
    return np.unpackbits(bits, axis=1)[:, :dim].astype(np.float32) * 2.0 - 1.0


# ----------------------------------------------------------------------------------------
# query-side visual / positional vocabulary
# ----------------------------------------------------------------------------------------

# hue bins: 12 bins of 30 degrees centred on 0 (red), 30 (orange), 60 (yellow), ...
COLOR_WORDS: dict[str, list[int]] = {
    "red": [0], "crimson": [0, 11], "orange": [1], "amber": [1, 2], "yellow": [2],
    "green": [3, 4], "teal": [5, 6], "cyan": [6], "blue": [7, 8], "navy": [8],
    "purple": [9], "violet": [9], "magenta": [10], "pink": [10, 11],
}
ACHROMATIC_WORDS = {"black": 12, "dark": 12, "gray": 13, "grey": 13, "white": 14, "blank": 14}
POSITION_WORDS: dict[str, list[int]] = {
    "top": [0], "upper": [0], "bottom": [1], "lower": [1],
    "left": [2], "right": [3], "center": [4, 5], "centre": [4, 5], "middle": [4, 5],
}


@lru_cache(maxsize=8)
def _rotation(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
    q *= np.sign(np.diag(r))
    return q.astype(np.float32)


class PatchEncoder:
    def __init__(self, cfg: VisionConfig | None = None, patch_px: int = 16):
        self.cfg = cfg or VisionConfig()
        self.patch_px = patch_px
        self.R = _rotation(self.cfg.dim, self.cfg.rotation_seed)
        c = self.cfg
        self._sl_text = slice(0, c.text_dim)
        self._sl_vis = slice(c.text_dim, c.text_dim + c.visual_dim)
        self._sl_pos = slice(c.text_dim + c.visual_dim, c.dim)

    # ---------------------------------------------------------------- page side
    def region_vectors(self, regions: list[Region]) -> tuple[list[Region], np.ndarray, np.ndarray]:
        """Leaf regions, their own text vectors and their (unit) layout-graph context vectors."""
        dim = self.cfg.text_dim
        graph = build_layout_graph(regions)
        by_id = {r.id: r for r in regions}
        leaves = [r for r in regions if r.role not in NON_LEAF_ROLES and r.id in graph]
        own = {r.id: text_vector(r.text, dim, bind="doc") for r in regions}
        own_m = np.zeros((len(leaves), dim), dtype=np.float32)
        ctx_m = np.zeros((len(leaves), dim), dtype=np.float32)
        for i, r in enumerate(leaves):
            ctx = np.zeros(dim, dtype=np.float32)
            for nid, w in graph[r.id].items():
                if nid.startswith("__parent__"):
                    ctx += w * own[nid[len("__parent__"):]]
                elif nid in by_id:
                    ctx += w * own[nid]
            # strong context (two anchoring headers) saturates at unit norm; weak context
            # (a lone caption at weight 0.3) stays weak instead of being inflated
            cn = float(np.linalg.norm(ctx))
            own_m[i] = own[r.id]
            ctx_m[i] = ctx / max(cn, 1.0)
        return leaves, own_m, ctx_m

    def _lambda(self, n_tokens: int) -> float:
        # adaptive context: a bare number lives entirely off its headers, a rich label is
        # mostly self-describing
        return self.cfg.context_weight * float(np.clip(4.0 / max(n_tokens, 1), 0.35, 1.0))

    def leaf_vectors(self, regions: list[Region]) -> tuple[list[str], np.ndarray]:
        """Contextual unit vector of every leaf region (used to score bbox proposals)."""
        leaves, own, ctx = self.region_vectors(regions)
        out = np.zeros_like(own)
        for i, r in enumerate(leaves):
            v = own[i] + self._lambda(len(tokenize(r.text, bind="doc"))) * ctx[i]
            out[i] = v / max(float(np.linalg.norm(v)), 1e-9)
        return [r.id for r in leaves], out

    def contextual_units(self, regions: list[Region], patch_h: float) -> tuple[np.ndarray, np.ndarray]:
        """Split long multi-line regions into horizontal bands (word-flow approximation of a
        text layer) and return (unit boxes, unit vectors).  A band keeps its local words at
        full strength plus the whole region's gist and layout context."""
        leaves, own, ctx = self.region_vectors(regions)
        boxes, vecs = [], []
        for r, o, c in zip(leaves, own, ctx):
            x1, y1, x2, y2 = r.bbox
            toks = tokenize(r.text, bind="doc")
            lam = self._lambda(len(toks))
            n_bands = int(min(round((y2 - y1) / max(patch_h, 1e-6)), np.ceil(len(toks) / 4)))
            if len(toks) > 8 and n_bands >= 2:
                chunks = np.array_split(np.array(toks, dtype=object), n_bands)
                for k, ch in enumerate(chunks):
                    local = np.zeros_like(o)
                    for t in ch:
                        local += token_vector(str(t), self.cfg.text_dim)
                    ln = float(np.linalg.norm(local))
                    v = (local / ln if ln > 0 else local) + 0.5 * o + lam * c
                    boxes.append((x1, y1 + k * (y2 - y1) / n_bands, x2, y1 + (k + 1) * (y2 - y1) / n_bands))
                    vecs.append(v / max(float(np.linalg.norm(v)), 1e-9))
            else:
                v = o + lam * c
                boxes.append(r.bbox)
                vecs.append(v / max(float(np.linalg.norm(v)), 1e-9))
        if not boxes:
            return np.zeros((0, 4)), np.zeros((0, self.cfg.text_dim), dtype=np.float32)
        return np.array(boxes, dtype=np.float64), np.array(vecs, dtype=np.float32)

    def patch_boxes(self, width: int, height: int) -> np.ndarray:
        gr, gc = self.cfg.grid_rows, self.cfg.grid_cols
        xs = np.linspace(0, width, gc + 1)
        ys = np.linspace(0, height, gr + 1)
        x1, y1 = np.meshgrid(xs[:-1], ys[:-1])
        x2, y2 = np.meshgrid(xs[1:], ys[1:])
        return np.stack([x1.ravel(), y1.ravel(), x2.ravel(), y2.ravel()], axis=1)

    @staticmethod
    def overlap_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """Intersection areas, shape (len(a), len(b))."""
        ix = np.clip(np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
                     - np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0]), 0, None)
        iy = np.clip(np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
                     - np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1]), 0, None)
        return ix * iy

    def text_raster(self, regions: list[Region], width: int, height: int) -> np.ndarray:
        n_p = self.cfg.grid_rows * self.cfg.grid_cols
        if not regions:
            return np.zeros((n_p, self.cfg.text_dim), dtype=np.float32)
        rb, rv = self.contextual_units(regions, height / self.cfg.grid_rows)
        if len(rb) == 0:
            return np.zeros((n_p, self.cfg.text_dim), dtype=np.float32)
        pb = self.patch_boxes(width, height)
        inter = self.overlap_matrix(pb, rb)
        ra = np.maximum((rb[:, 2] - rb[:, 0]) * (rb[:, 3] - rb[:, 1]), 1e-6)
        pa = (pb[:, 2] - pb[:, 0]) * (pb[:, 3] - pb[:, 1])
        w = inter / np.minimum(pa[:, None], ra[None, :])
        return (w.astype(np.float32) @ rv).astype(np.float32)

    def visual_descriptors(self, image: np.ndarray) -> np.ndarray:
        gr, gc, ps = self.cfg.grid_rows, self.cfg.grid_cols, self.patch_px
        im = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
        im = im.resize((gc * ps, gr * ps), Image.Resampling.BOX)
        rgb = np.asarray(im, dtype=np.float32) / 255.0
        hsv = np.asarray(im.convert("HSV"), dtype=np.float32) / 255.0
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

        def pool(x: np.ndarray) -> np.ndarray:  # (H, W[, C]) -> (gr*gc[, C]) means
            if x.ndim == 2:
                return x.reshape(gr, ps, gc, ps).mean(axis=(1, 3)).reshape(-1)
            return x.reshape(gr, ps, gc, ps, -1).mean(axis=(1, 3)).reshape(gr * gc, -1)

        sat = (s > 0.35) & (v > 0.25)
        hue_bin = (((h * 360.0 + 15.0) % 360.0) // 30.0).astype(np.int32)  # bin 0 centred on red
        hue_hist = np.zeros(h.shape + (12,), dtype=np.float32)
        np.put_along_axis(hue_hist, hue_bin[..., None], sat[..., None].astype(np.float32), axis=-1)
        dark = (~sat) & (v < 0.25)
        white = (~sat) & (v > 0.85)
        gray = (~sat) & ~dark & ~white
        achrom = np.stack([dark, gray, white], axis=-1).astype(np.float32)

        gy, gx = np.gradient(lum)
        mag = np.hypot(gx, gy)
        ang = np.mod(np.arctan2(gy, gx), np.pi)
        ob = np.minimum((ang / np.pi * 8).astype(np.int32), 7)
        ori = np.zeros(lum.shape + (8,), dtype=np.float32)
        np.put_along_axis(ori, ob[..., None], mag[..., None], axis=-1)
        edges = (mag > 0.08).astype(np.float32)

        rg = rgb[..., 0] - rgb[..., 1]
        yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
        feats = np.concatenate([
            pool(hue_hist),                                   # 12
            pool(achrom),                                     # 3
            pool(ori) * 4.0,                                  # 8
            pool(edges)[:, None],                             # 1
            pool(lum)[:, None], (pool(lum ** 2) - pool(lum) ** 2).clip(0)[:, None] ** 0.5,  # 2
            pool(rgb),                                        # 3
            np.hypot(pool(rg), pool(yb))[:, None],            # 1
            pool(s)[:, None], pool(v)[:, None],               # 2
        ], axis=1).astype(np.float32)
        out = np.zeros((gr * gc, self.cfg.visual_dim), dtype=np.float32)
        out[:, : feats.shape[1]] = feats[:, : self.cfg.visual_dim]
        n = np.maximum(1.0, np.linalg.norm(out, axis=1, keepdims=True))
        return out / n

    def position_features(self) -> np.ndarray:
        gr, gc = self.cfg.grid_rows, self.cfg.grid_cols
        y, x = np.meshgrid((np.arange(gr) + 0.5) / gr, (np.arange(gc) + 0.5) / gc, indexing="ij")
        y, x = y.ravel(), x.ravel()
        cols = [1 - y, y, 1 - x, x, 1 - np.abs(2 * y - 1), 1 - np.abs(2 * x - 1)]
        for cy in (1 / 6, 0.5, 5 / 6):
            for cx in (1 / 6, 0.5, 5 / 6):
                cols.append(np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / 0.05))
        p = np.stack(cols, axis=1).astype(np.float32)
        out = np.zeros((gr * gc, self.cfg.pos_dim), dtype=np.float32)
        out[:, : p.shape[1]] = p[:, : self.cfg.pos_dim]
        return out

    def encode_page(self, page: Page) -> PageEncoding:
        c = self.cfg
        n_p = c.grid_rows * c.grid_cols
        vec = np.zeros((n_p, c.dim), dtype=np.float32)
        vec[:, self._sl_text] = self.text_raster(page.regions, page.width, page.height)
        vec[:, self._sl_vis] = c.visual_weight * self.visual_descriptors(page.image)
        vec[:, self._sl_pos] = c.pos_weight * self.position_features() / 2.0
        leaf_ids, leaf_text = self.leaf_vectors(page.regions)
        return PageEncoding(page.page_id, page.doc_id, page.width, page.height,
                            (c.grid_rows, c.grid_cols), vec @ self.R.T, list(page.regions), dict(page.meta),
                            leaf_ids, leaf_text)

    # ---------------------------------------------------------------- query side
    def encode_query(self, query: str) -> tuple[list[str], np.ndarray]:
        """Multi-vector query: one rotated vector per informative token."""
        c = self.cfg
        toks = list(dict.fromkeys(tokenize(query, bind="query")))
        if not toks:
            return [], np.zeros((0, c.dim), dtype=np.float32)
        Q = np.zeros((len(toks), c.dim), dtype=np.float32)
        for i, t in enumerate(toks):
            tv = token_vector(t, c.text_dim)
            if t in COLOR_WORDS or t in ACHROMATIC_WORDS:
                vis = np.zeros(c.visual_dim, dtype=np.float32)
                if t in COLOR_WORDS:
                    vis[COLOR_WORDS[t]] = 1.0
                else:
                    vis[ACHROMATIC_WORDS[t]] = 1.0
                Q[i, self._sl_text] = 0.3 * tv
                Q[i, self._sl_vis] = vis / np.linalg.norm(vis)
            elif t in POSITION_WORDS:
                pos = np.zeros(c.pos_dim, dtype=np.float32)
                pos[POSITION_WORDS[t]] = 1.0
                Q[i, self._sl_text] = 0.5 * tv
                Q[i, self._sl_pos] = pos / np.linalg.norm(pos)
            else:
                Q[i, self._sl_text] = tv
        return toks, Q @ self.R.T
