"""Automated cross-modal root-cause analysis.

For every span the engine computes

* **evidence** ``e(s)`` in [0, 1] - noisy-OR of independent failure signals: HTTP status
  class, error flag, attached ERROR/FATAL logs (exception text weighs more), robust latency
  outliers (median / MAD against same-operation spans), response-schema drift against the
  operation's majority schema, and null-valued response fields;
* **origin factor** ``o(s)`` - spans whose anomaly is *explained by a child* with the same
  class of evidence are propagating, not originating, the failure (``origin_penalty``);
  failures recovered by a successful retry are discounted further;
* **causal reach** ``r(s, a)`` - the max-product path weight from the span to each
  user-visible anomaly ``a`` in the cross-modal DAG.

``score(s) = max_a severity(a) * r(s, a) * e(s) * o(s)`` and the ranked list carries
the explaining path (span -> ... -> DOM node -> visible anomaly) as its citation.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from omnitrace.config import CorrelatorConfig, DedupConfig
from omnitrace.correlator.timeline_dag import (
    Anomaly, CausalDAG, Span, TraceBundle, build_causal_dag, detect_dom_anomalies,
    detect_visual_anomalies, iter_service_logs,
)

__all__ = ["Evidence", "Candidate", "RCAReport", "span_evidence", "origin_factors", "analyze"]

_EXC_WORDS = ("exception", "traceback", "timeout", "deadlock", "refused", "panic", "fatal", "oom")


@dataclass
class Evidence:
    score: float
    reasons: list[str] = field(default_factory=list)
    classes: set[str] = field(default_factory=set)


@dataclass
class Candidate:
    span: Span
    score: float
    evidence: Evidence
    origin: float
    reach: float
    anomaly_index: int
    path: list[str]

    def to_dict(self, anomalies: list[Anomaly]) -> dict:
        a = anomalies[self.anomaly_index]
        return {
            "span_id": self.span.span_id, "service": self.span.service, "name": self.span.name,
            "score": round(self.score, 5), "evidence": round(self.evidence.score, 3),
            "reasons": self.evidence.reasons, "origin_factor": self.origin, "reach": round(self.reach, 4),
            "anomaly": {"kind": a.kind, "t_start": round(a.t_start, 3), "bbox": a.bbox, "source": a.source},
            "path": self.path,
        }


@dataclass
class RCAReport:
    anomalies: list[Anomaly]
    candidates: list[Candidate]
    dag: CausalDAG
    keyframes: list[int] = field(default_factory=list)
    n_frames: int = 0
    elapsed_ms: float = 0.0

    @property
    def root(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    def to_dict(self, top: int = 5) -> dict:
        return {
            "root_cause": self.root.to_dict(self.anomalies) if self.root else None,
            "candidates": [c.to_dict(self.anomalies) for c in self.candidates[:top]],
            "anomalies": [{"kind": a.kind, "t_start": round(a.t_start, 3), "t_detect": round(a.t_detect, 3),
                           "bbox": [round(v, 1) for v in a.bbox] if a.bbox else None, "severity": a.severity,
                           "source": a.source, "detail": a.detail} for a in self.anomalies],
            "dag": {"nodes": len(self.dag.nodes), "edges": self.dag.n_edges},
            "frames": {"total": self.n_frames, "keyframes": len(self.keyframes)},
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


def _noisy_or(xs: list[float]) -> float:
    p = 1.0
    for x in xs:
        p *= 1.0 - x
    return 1.0 - p


def span_evidence(bundle: TraceBundle) -> dict[str, Evidence]:
    by_op: dict[tuple[str, str], list[Span]] = defaultdict(list)
    for s in bundle.spans:
        by_op[(s.service, s.name)].append(s)
    out: dict[str, Evidence] = {}
    for s in bundle.spans:
        sig: list[float] = []
        ev = Evidence(0.0)
        code = s.status_code
        if code is not None:
            if code == 0:
                sig.append(0.85); ev.reasons.append("network failure (status 0)"); ev.classes.add("error")
            elif code in (408, 504):
                sig.append(0.8); ev.reasons.append(f"HTTP {code} timeout"); ev.classes.add("error")
            elif code >= 500:
                sig.append(0.8); ev.reasons.append(f"HTTP {code}"); ev.classes.add("error")
            elif code in (401, 403):
                sig.append(0.6); ev.reasons.append(f"HTTP {code} auth"); ev.classes.add("error")
            elif code == 429:
                sig.append(0.65); ev.reasons.append("HTTP 429 rate limited"); ev.classes.add("error")
            elif code == 404:
                sig.append(0.3); ev.reasons.append("HTTP 404"); ev.classes.add("error")
            elif code >= 400:
                sig.append(0.4); ev.reasons.append(f"HTTP {code}"); ev.classes.add("error")
        if s.error and not any(c in ev.classes for c in ("error",)):
            sig.append(0.7); ev.reasons.append("span status ERROR"); ev.classes.add("error")
        for log in iter_service_logs(bundle.logs, s):
            lvl = log.level.upper()
            if lvl in ("ERROR", "FATAL", "CRITICAL"):
                exc = any(w in log.message.lower() for w in _EXC_WORDS)
                sig.append(0.8 if exc else 0.7)
                ev.reasons.append(f"{lvl} log: {log.message[:70]}")
                ev.classes.add("error")
            elif lvl in ("WARN", "WARNING"):
                sig.append(0.2)
                ev.reasons.append(f"WARN log: {log.message[:70]}")
        peers = [p.duration for p in by_op[(s.service, s.name)] if p.span_id != s.span_id]
        if len(peers) >= 2:
            med = float(np.median(peers))
            mad = float(np.median(np.abs(np.array(peers) - med))) * 1.4826
            z = (s.duration - med) / (mad + 0.05 * med + 1e-3)
            if z > 6 and s.duration > 3 * med and s.duration > 0.25:
                sig.append(min(0.85, 0.45 + 0.02 * z))
                ev.reasons.append(f"latency outlier {s.duration:.2f}s vs median {med:.3f}s (z={z:.0f})")
                ev.classes.add("latency")
        if s.duration >= 10.0 and "latency" not in ev.classes:
            sig.append(0.7); ev.reasons.append(f"latency {s.duration:.1f}s"); ev.classes.add("latency")
        schema = s.attributes.get("http.response.schema")
        if schema is not None:
            others = [p.attributes.get("http.response.schema") for p in by_op[(s.service, s.name)]
                      if p.span_id != s.span_id and p.attributes.get("http.response.schema") is not None]
            if others:
                major, cnt = Counter(others).most_common(1)[0]
                if schema != major and cnt >= 1:
                    missing = sorted(set(str(major).split(",")) - set(str(schema).split(",")))
                    sig.append(0.75)
                    ev.reasons.append(f"response schema drift (missing: {', '.join(missing) or '-'})")
                    ev.classes.add("drift")
        nulls = s.attributes.get("http.response.null_fields")
        if nulls:
            sig.append(0.6); ev.reasons.append(f"null fields in response: {nulls}"); ev.classes.add("nulls")
        ev.score = max(0.02, _noisy_or(sig))
        out[s.span_id] = ev
    return out


def origin_factors(bundle: TraceBundle, ev: dict[str, Evidence], cfg: CorrelatorConfig) -> dict[str, tuple[float, str]]:
    children: dict[str, list[Span]] = defaultdict(list)
    for s in bundle.spans:
        if s.parent_id:
            children[s.parent_id].append(s)
    spans = {s.span_id: s for s in bundle.spans}
    out: dict[str, tuple[float, str]] = {}
    # recovered by retry: an error followed by a successful sibling of the same operation
    recovered = set()
    for pid, kids in children.items():
        for c in kids:
            if "error" in ev[c.span_id].classes:
                if any(k.name == c.name and k.service == c.service and k.start >= c.end - 1e-6
                       and "error" not in ev[k.span_id].classes for k in kids):
                    recovered.add(c.span_id)
    for s in bundle.spans:
        e = ev[s.span_id]
        if s.span_id in recovered:
            out[s.span_id] = (0.15, "recovered by retry")
            continue
        explained = []
        for c in children.get(s.span_id, []):
            if c.span_id in recovered:
                continue
            ce = ev[c.span_id]
            if "error" in e.classes and "error" in ce.classes:
                explained.append("error")
            if "latency" in e.classes and "latency" in ce.classes and c.duration >= 0.6 * s.duration:
                explained.append("latency")
            if "drift" in e.classes and "drift" in ce.classes:
                explained.append("drift")
            if "nulls" in e.classes and "nulls" in ce.classes:
                explained.append("nulls")
        own = e.classes - set(explained)
        if explained and not own:
            out[s.span_id] = (cfg.origin_penalty, f"propagated from child ({', '.join(sorted(set(explained)))})")
        else:
            out[s.span_id] = (1.0, "originating")
    _ = spans
    return out


def analyze(bundle: TraceBundle, cfg: CorrelatorConfig | None = None, dedup: DedupConfig | None = None,
            extra_anomalies: list[Anomaly] | None = None) -> RCAReport:
    """Full pipeline: keyframes -> anomalies -> causal DAG -> ranked root causes."""
    cfg = cfg or CorrelatorConfig()
    t0 = time.perf_counter()
    anomalies: list[Anomaly] = []
    keyframes: list[int] = []
    n_frames = 0
    if bundle.frames:
        times = bundle.frame_times or [i * 0.2 for i in range(len(bundle.frames))]
        vis, kf = detect_visual_anomalies(bundle.frames, times, cfg, dedup=dedup)
        anomalies.extend(vis)
        keyframes, n_frames = kf.keyframes, kf.n_frames
    anomalies.extend(detect_dom_anomalies(bundle.dom, cfg, horizon=bundle.horizon()))
    if extra_anomalies:
        anomalies.extend(extra_anomalies)
    anomalies.sort(key=lambda a: a.t_start)
    dag = build_causal_dag(bundle, anomalies, cfg)
    ev = span_evidence(bundle)
    origin = origin_factors(bundle, ev, cfg)
    spans = {s.span_id: s for s in bundle.spans}
    best: dict[str, Candidate] = {}
    for j, a in enumerate(anomalies):
        for node, (w, path) in dag.reach(a.key(j)).items():
            if not node.startswith("span:"):
                continue
            sid = node[5:]
            o, _why = origin[sid]
            score = a.severity * w * ev[sid].score * o
            if sid not in best or score > best[sid].score:
                best[sid] = Candidate(spans[sid], score, ev[sid], o, w, j, path)
    ranked = sorted(best.values(), key=lambda c: -c.score)
    return RCAReport(anomalies, ranked, dag, keyframes, n_frames, (time.perf_counter() - t0) * 1000)
