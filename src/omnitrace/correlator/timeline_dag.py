"""Cross-modal temporal alignment: UI frames <-> DOM mutations <-> network / backend spans <-> logs.

Data model
----------
``TraceBundle`` carries one recorded execution: OpenTelemetry-style spans, structured
logs, DOM mutation records (with ``getBoundingClientRect`` boxes) and, optionally, the
screen recording itself.

Anomaly detection (first principles, no models)
-----------------------------------------------
Visual (on keyframes from :mod:`omnitrace.vision.phash_dedup`):

* ``error_banner``     - a newly appeared saturated-red region (HSV), persisting;
* ``warning_toast``    - a newly appeared saturated orange/amber region;
* ``modal_overlay``    - the page periphery darkens while a bright box stays in the centre;
* ``blank_region``     - previously textured tiles become flat and stay flat;
* ``stalled_spinner``  - a confined animation (cycle-aware dedup) running past the stall threshold.

DOM: error classes / ``role=alert``, poisoned text (``NaN``, ``undefined``, ``null``,
``[object Object]``), busy indicators never cleared, emptied containers.

Causal DAG
----------
Nodes are spans, DOM mutations and anomalies.  Edges always point cause -> effect and
respect time (with a clock-skew tolerance):

* child span -> parent span            (failure / latency propagates upward)
* client span end -> DOM mutation       (response rendered), weighted by lexical affinity
                                        between the request and the DOM node
* client span start -> busy indicator   (request put a spinner on screen)
* DOM mutation -> anomaly               (bbox overlap within the visual window)
* in-flight client span -> stall        (spinner still waiting on that request)
* client span -> anomaly                (weak, purely temporal fallback)
"""

from __future__ import annotations

import base64
import io
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Iterable

import numpy as np
from PIL import Image
from scipy import ndimage

from omnitrace.config import CorrelatorConfig, DedupConfig
from omnitrace.textfeat import tokenize
from omnitrace.vision.phash_dedup import KeyframeExtractor, KeyframeResult

BBox = tuple[float, float, float, float]

__all__ = [
    "Span", "LogRecord", "DomMutation", "Anomaly", "TraceBundle", "CausalDAG", "Edge",
    "detect_visual_anomalies", "detect_dom_anomalies", "build_causal_dag", "lexical_affinity",
]


# --------------------------------------------------------------------------- data model
@dataclass
class Span:
    span_id: str
    trace_id: str
    service: str
    name: str
    start: float
    end: float
    parent_id: str | None = None
    kind: str = "internal"          # client | server | internal | consumer
    status_code: int | None = None
    error: bool = False
    attributes: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class LogRecord:
    t: float
    level: str
    service: str
    message: str
    span_id: str | None = None
    trace_id: str | None = None


@dataclass
class DomMutation:
    t: float
    op: str                         # insert | remove | attr | text
    selector: str
    bbox: BBox | None = None
    attrs: dict = field(default_factory=dict)
    text: str = ""


@dataclass
class Anomaly:
    kind: str
    t_start: float                  # when the symptom first became visible
    t_detect: float                 # when it qualified as an anomaly (e.g. stall threshold)
    bbox: BBox | None
    severity: float
    source: str                     # visual | dom
    detail: str = ""
    frame_index: int | None = None

    def key(self, i: int) -> str:
        return f"anom:{i}"


@dataclass
class TraceBundle:
    spans: list[Span]
    logs: list[LogRecord] = field(default_factory=list)
    dom: list[DomMutation] = field(default_factory=list)
    frames: list[np.ndarray] | None = None
    frame_times: list[float] | None = None
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ JSON IO
    def to_dict(self, include_frames: bool = False) -> dict:
        d = {
            "spans": [asdict(s) for s in self.spans],
            "logs": [asdict(l) for l in self.logs],
            "dom": [asdict(m) for m in self.dom],
            "meta": self.meta,
        }
        if include_frames and self.frames is not None:
            enc = []
            for f in self.frames:
                buf = io.BytesIO()
                Image.fromarray(f).save(buf, format="PNG")
                enc.append(base64.b64encode(buf.getvalue()).decode("ascii"))
            d["frames_png_b64"] = enc
            d["frame_times"] = list(self.frame_times or [])
        return d

    def horizon(self) -> float:
        """End of the observation window: recording end if known, else the last observed event
        (span *starts* are used because a hung request may end long after the session)."""
        if "recording_end" in self.meta:
            return float(self.meta["recording_end"])
        if self.frame_times:
            return float(self.frame_times[-1])
        ts = [m.t for m in self.dom] + [l.t for l in self.logs] + [s.start for s in self.spans]
        return max(ts, default=0.0)

    @classmethod
    def from_dict(cls, d: dict) -> "TraceBundle":
        spans = [Span(**s) for s in d.get("spans", [])]
        logs = [LogRecord(**l) for l in d.get("logs", [])]
        dom = []
        for m in d.get("dom", []):
            m = dict(m)
            if m.get("bbox") is not None:
                m["bbox"] = tuple(float(v) for v in m["bbox"])
            dom.append(DomMutation(**m))
        frames = None
        if "frames_png_b64" in d:
            frames = [np.asarray(Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB"))
                      for b in d["frames_png_b64"]]
        return cls(spans, logs, dom, frames, d.get("frame_times"), d.get("meta", {}))


# --------------------------------------------------------------------------- helpers
def _overlap_frac(a: BBox | None, b: BBox | None) -> float:
    """|a & b| / min(|a|, |b|)  (containment-friendly overlap)."""
    if a is None or b is None:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    aa = max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))
    ab = max(1e-9, (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / min(aa, ab)


_URL_SPLIT = re.compile(r"[^a-z0-9]+")
_GENERIC = frozenset("""api http https get post put patch delete www com v1 v2 service services server client
    error errors failed fail alert danger class role children container true false text insert select
    from into update where the and for with please retry try again could not static img data badge
    toast warning modal dialog grid spinner busy aria""".split())


def _toks(*parts: str) -> set[str]:
    out = set()
    for p in parts:
        for t in _URL_SPLIT.split(str(p).lower()):
            if len(t) > 2 and not t.isdigit() and t not in _GENERIC:
                out.add(t[:-1] if t.endswith("s") and len(t) > 4 else t)
    return out


def _req_tokens(span: Span) -> set[str]:
    return _toks(span.name, span.service, span.attributes.get("http.url", ""), span.attributes.get("http.route", ""))


def _dom_tokens(m: DomMutation) -> set[str]:
    return _toks(m.selector, m.text, *(f"{k} {v}" for k, v in m.attrs.items() if k != "data-endpoint"))


_FAILURE_DOM = re.compile(r"error|alert|danger|fail|nan|undefined|unavailable|expired|denied", re.I)


def _dom_is_failure(m: DomMutation) -> bool:
    blob = " ".join([m.selector, m.text, " ".join(map(str, m.attrs.values()))])
    return bool(_FAILURE_DOM.search(blob)) or (m.attrs.get("children") == 0 and bool(m.attrs.get("container")))


def _span_is_failure(s: Span) -> bool:
    return s.error or (s.status_code is not None and (s.status_code >= 400 or s.status_code == 0)) \
        or bool(s.attributes.get("http.response.null_fields"))


def lexical_affinity(span: Span, m: DomMutation, trace_tokens: set[str] | None = None) -> float:
    """How plausibly DOM node ``m`` renders the response of request ``span`` (0.25..1).

    * explicit ``data-endpoint`` binding -> 1.0;
    * shared content words between the DOM node (selector, attributes, text) and the
      request's *whole downstream trace* (routes, services, operations) - a banner that
      says "Payment failed" binds to the trace that touched ``payment-service``;
    * a failure-looking DOM change and a failed response are mutually consistent (>= 0.5).
    """
    ep = m.attrs.get("data-endpoint")
    route = str(span.attributes.get("http.route", span.attributes.get("http.url", "")))
    if ep and route and (ep in route or route.split("?")[0].endswith(ep)):
        return 1.0
    a = trace_tokens if trace_tokens is not None else _req_tokens(span)
    shared = len(a & _dom_tokens(m))
    aff = 0.25 + 0.75 * (1.0 - math.exp(-1.5 * shared))
    if _dom_is_failure(m) and _span_is_failure(span):
        aff = max(aff, 0.5)
    return aff


# --------------------------------------------------------------------------- visual anomalies
def _hsv(frame: np.ndarray, reduce: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    im = Image.fromarray(frame)
    if reduce > 1:
        im = im.reduce(reduce)
    hsv = np.asarray(im.convert("HSV"), dtype=np.float32)
    gray = np.asarray(im.convert("L"), dtype=np.float32)
    return hsv[..., 0] * (360.0 / 255.0), hsv[..., 1] / 255.0, hsv[..., 2] / 255.0, gray


def _color_mask(h, s, v, lo: float, hi: float) -> np.ndarray:
    hue_ok = ((h >= lo) & (h <= hi)) if lo <= hi else ((h >= lo) | (h <= hi))
    return hue_ok & (s > 0.55) & (v > 0.45)


def _largest_bbox(mask: np.ndarray, scale: float) -> tuple[BBox | None, int]:
    lab, n = ndimage.label(mask)
    if n == 0:
        return None, 0
    sizes = ndimage.sum(mask, lab, index=range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    ys, xs = np.nonzero(lab == k)
    return (xs.min() * scale, ys.min() * scale, (xs.max() + 1) * scale, (ys.max() + 1) * scale), int(sizes[k - 1])


def _tile_std(gray: np.ndarray, tr: int = 9, tc: int = 16) -> np.ndarray:
    h, w = gray.shape
    g = gray[: h - h % tr, : w - w % tc]
    blocks = g.reshape(tr, g.shape[0] // tr, tc, g.shape[1] // tc)
    return blocks.std(axis=(1, 3))


def detect_visual_anomalies(frames: list[np.ndarray], times: list[float], cfg: CorrelatorConfig | None = None,
                            kf: KeyframeResult | None = None, dedup: DedupConfig | None = None
                            ) -> tuple[list[Anomaly], KeyframeResult]:
    cfg = cfg or CorrelatorConfig()
    if not frames:
        return [], KeyframeResult([], {}, [], 0, 0.0)
    kf = kf or KeyframeExtractor(dedup).extract(frames, times)
    H, W = frames[0].shape[:2]
    red_lo, red_hi = 345.0, 12.0
    org_lo, org_hi = 20.0, 45.0
    anomalies: list[Anomaly] = []
    feats = {}

    def F(i: int):
        if i not in feats:
            feats[i] = _hsv(frames[i])
        return feats[i]

    def at_time(t: float) -> int:
        return int(min(len(times) - 1, np.searchsorted(times, t)))

    min_area = 0.004 * (H / 2) * (W / 2)
    kfs = kf.keyframes
    for a, b in zip(kfs, kfs[1:]):
        h0, s0, v0, g0 = F(a)
        h1, s1, v1, g1 = F(b)
        t = times[b]
        for kind, lo, hi, sev in (("error_banner", red_lo, red_hi, 1.0), ("warning_toast", org_lo, org_hi, 0.6)):
            new = _color_mask(h1, s1, v1, lo, hi) & ~_color_mask(h0, s0, v0, lo, hi)
            bbox, area = _largest_bbox(ndimage.binary_opening(new, iterations=1), 2.0)
            if bbox is None or area < min_area:
                continue
            j = at_time(t + 0.5)
            hj, sj, vj, _ = F(j)
            x1, y1, x2, y2 = (int(c / 2) for c in bbox)
            if _color_mask(hj, sj, vj, lo, hi)[y1:y2, x1:x2].mean() < 0.3:
                continue  # flashed and vanished: not a persistent symptom
            anomalies.append(Anomaly(kind, t, t, bbox, sev, "visual", f"new {kind.split('_')[0]} region {area * 4} px", b))
        # modal overlay: periphery darkens, a bright centre box remains
        ring = np.ones_like(g1, dtype=bool)
        hh, ww = g1.shape
        ring[hh // 6: -hh // 6, ww // 6: -ww // 6] = False
        dimmed = False
        if g0[ring].mean() > 1e-6 and g1[ring].mean() / g0[ring].mean() < 0.65:
            bright = (g1 > 0.85 * g0.max()) & ~ring
            bbox, area = _largest_bbox(bright, 2.0)
            if bbox is not None and area > 0.05 * hh * ww:
                anomalies.append(Anomaly("modal_overlay", t, t, bbox, 0.7, "visual", "page dimmed behind dialog", b))
                dimmed = True
        if dimmed:
            continue  # texture loss behind a dimming overlay is not a blank region
        # blank region: textured tiles become flat and stay flat
        sd0, sd1 = _tile_std(g0), _tile_std(g1)
        lost = (sd0 > 6.0) & (sd1 < 2.5)
        if lost.sum() >= 0.06 * lost.size:
            lab, n = ndimage.label(lost)
            sizes = ndimage.sum(lost, lab, index=range(1, n + 1))
            k = int(np.argmax(sizes)) + 1
            if sizes[k - 1] >= 0.04 * lost.size:
                ys, xs = np.nonzero(lab == k)
                tr, tc = lost.shape
                bbox = (xs.min() * W / tc, ys.min() * H / tr, (xs.max() + 1) * W / tc, (ys.max() + 1) * H / tr)
                j = at_time(t + cfg.blank_persist_s)
                if j > b and (_tile_std(F(j)[3])[lab == k] < 2.5).mean() > 0.8:
                    anomalies.append(Anomaly("blank_region", t, times[j], bbox, 0.9, "visual",
                                             f"{int(sizes[k - 1])} tiles lost all content", b))
    for run in kf.animations:
        if run.duration >= cfg.stall_threshold_s:
            anomalies.append(Anomaly("stalled_spinner", run.t_start, run.t_start + cfg.stall_threshold_s,
                                     run.bbox(W, H), 0.9, "visual",
                                     f"animation running {run.duration:.1f}s without progress", run.start))
    anomalies.sort(key=lambda a: a.t_start)
    return anomalies, kf


# --------------------------------------------------------------------------- DOM anomalies
_BAD_TEXT = re.compile(r"\b(NaN|undefined|null|\[object Object\])\b|\bError\b|failed|unavailable", re.I)
_ERR_CLASS = re.compile(r"error|alert|danger|fail", re.I)


def detect_dom_anomalies(dom: list[DomMutation], cfg: CorrelatorConfig | None = None,
                         horizon: float | None = None) -> list[Anomaly]:
    """``horizon`` = the end of the observation window (defaults to the last DOM record)."""
    cfg = cfg or CorrelatorConfig()
    out: list[Anomaly] = []
    busy_open: dict[str, DomMutation] = {}
    for m in sorted(dom, key=lambda m: m.t):
        cls = " ".join(str(m.attrs.get(k, "")) for k in ("class", "role"))
        if m.op in ("insert", "attr") and _ERR_CLASS.search(cls):
            out.append(Anomaly("dom_error", m.t, m.t, m.bbox, 0.8, "dom", f"{m.selector} [{cls.strip()}]"))
        elif m.op in ("insert", "text") and m.text and _BAD_TEXT.search(m.text):
            out.append(Anomaly("dom_bad_text", m.t, m.t, m.bbox, 0.8, "dom", f"{m.selector}: {m.text[:60]!r}"))
        elif m.op in ("remove", "text") and m.attrs.get("children") == 0 and m.attrs.get("container"):
            out.append(Anomaly("dom_emptied", m.t, m.t, m.bbox, 0.6, "dom", f"{m.selector} emptied"))
        if str(m.attrs.get("aria-busy", "")).lower() == "true":
            busy_open[m.selector] = m
        elif m.op == "remove" or str(m.attrs.get("aria-busy", "")).lower() == "false":
            busy_open.pop(m.selector, None)
    if horizon is None:
        horizon = max((m.t for m in dom), default=0.0)
    for sel, m in busy_open.items():
        if horizon - m.t >= cfg.stall_threshold_s:
            out.append(Anomaly("dom_stall", m.t, m.t + cfg.stall_threshold_s, m.bbox, 0.7, "dom",
                               f"{sel} busy for {horizon - m.t:.1f}s"))
    return out


# --------------------------------------------------------------------------- causal DAG
@dataclass
class Edge:
    src: str
    dst: str
    weight: float
    kind: str


@dataclass
class CausalDAG:
    nodes: dict[str, object]
    incoming: dict[str, list[Edge]]

    @property
    def n_edges(self) -> int:
        return sum(len(v) for v in self.incoming.values())

    def add(self, src: str, dst: str, w: float, kind: str) -> None:
        if src == dst or w <= 0:
            return
        lst = self.incoming.setdefault(dst, [])
        for e in lst:
            if e.src == src:
                if w > e.weight:
                    e.weight, e.kind = w, kind
                return
        lst.append(Edge(src, dst, w, kind))

    def reach(self, target: str) -> dict[str, tuple[float, list[str]]]:
        """Max-product path weight from every ancestor to ``target`` (with the best path)."""
        best: dict[str, tuple[float, list[str]]] = {target: (1.0, [target])}
        frontier = [target]
        # Bellman-Ford style relaxation on the reversed graph; weights <= 1 so it converges
        for _ in range(len(self.nodes) + 1):
            nxt = []
            for v in frontier:
                wv, pv = best[v]
                for e in self.incoming.get(v, ()):
                    cand = wv * e.weight
                    if cand > best.get(e.src, (0.0, []))[0] + 1e-12 and e.src not in pv:
                        best[e.src] = (cand, [e.src] + pv)
                        nxt.append(e.src)
            if not nxt:
                break
            frontier = nxt
        return best


_BUSY = re.compile(r"spinner|loading|busy|skeleton", re.I)


def build_causal_dag(bundle: TraceBundle, anomalies: list[Anomaly], cfg: CorrelatorConfig | None = None) -> CausalDAG:
    cfg = cfg or CorrelatorConfig()
    skew = cfg.max_clock_skew_s
    dskew = cfg.dom_clock_skew_s
    nodes: dict[str, object] = {}
    dag = CausalDAG(nodes, {})
    spans = {s.span_id: s for s in bundle.spans}
    for s in bundle.spans:
        nodes[f"span:{s.span_id}"] = s
    for i, m in enumerate(bundle.dom):
        nodes[f"dom:{i}"] = m
    for i, a in enumerate(anomalies):
        nodes[a.key(i)] = a

    # 1. propagation: child -> parent
    for s in bundle.spans:
        if s.parent_id and s.parent_id in spans:
            dag.add(f"span:{s.span_id}", f"span:{s.parent_id}", cfg.propagation_weight, "propagates-to")

    client = [s for s in bundle.spans if s.kind == "client"]
    trace_toks: dict[str, set[str]] = {}
    for s in bundle.spans:
        trace_toks.setdefault(s.trace_id, set()).update(_req_tokens(s))
    # 2. client span -> DOM mutation
    for i, m in enumerate(bundle.dom):
        busy = bool(_BUSY.search(m.selector + " " + " ".join(map(str, m.attrs.values())))) and m.op == "insert"
        for s in client:
            aff = lexical_affinity(s, m, trace_toks.get(s.trace_id))
            if s.end - dskew <= m.t <= s.end + cfg.dom_window_s:
                decay = math.exp(-max(0.0, m.t - s.end) / cfg.dom_window_s)
                dag.add(f"span:{s.span_id}", f"dom:{i}", aff * (0.4 + 0.6 * decay), "rendered-as")
            if busy and s.start - dskew <= m.t <= s.start + 1.0:
                dag.add(f"span:{s.span_id}", f"dom:{i}", 0.8 * aff, "started-busy-indicator")

    # 3. DOM mutation -> anomaly ; 4. in-flight / temporal span -> anomaly
    for j, a in enumerate(anomalies):
        ak = a.key(j)
        for i, m in enumerate(bundle.dom):
            if not (m.t - skew <= a.t_start <= m.t + cfg.visual_window_s):
                continue
            ov = _overlap_frac(m.bbox, a.bbox)
            if a.source == "dom" and a.bbox == m.bbox and abs(a.t_start - m.t) < 1e-6:
                ov = 1.0
            if ov > 0.05:
                dag.add(f"dom:{i}", ak, 0.95 * (0.5 + 0.5 * min(1.0, ov)), "visible-as")
        for s in client:
            if a.kind in ("stalled_spinner", "dom_stall"):
                if s.start <= a.t_start + skew + 1.0 and s.end >= a.t_detect - skew:
                    dag.add(f"span:{s.span_id}", ak, 0.6, "in-flight-during")
            elif a.t_start - cfg.visual_window_s - cfg.dom_window_s <= s.end <= a.t_start + skew:
                decay = math.exp(-max(0.0, a.t_start - s.end) / cfg.temporal_tau_s)
                dag.add(f"span:{s.span_id}", ak, 0.25 * decay, "temporal-only")
    # 5. visual <-> DOM anomaly identity (a DOM symptom that is the same visible symptom)
    for j, a in enumerate(anomalies):
        for k, b in enumerate(anomalies):
            if a.source == "dom" and b.source == "visual" and _overlap_frac(a.bbox, b.bbox) > 0.3 \
                    and a.t_start - skew <= b.t_start <= a.t_start + cfg.visual_window_s:
                dag.add(a.key(j), b.key(k), 0.95, "same-symptom")
    return dag


def iter_service_logs(logs: Iterable[LogRecord], span: Span) -> Iterable[LogRecord]:
    for l in logs:
        if l.span_id == span.span_id or (l.span_id is None and l.service == span.service
                                          and span.start - 0.05 <= l.t <= span.end + 0.05):
            yield l
