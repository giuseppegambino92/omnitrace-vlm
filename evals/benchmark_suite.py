"""End-to-end benchmark suite for omnitrace.

    python -m evals.benchmark_suite            # writes evals/latest_report.json + README metrics

Four benchmarks, each with baselines / ablations, a *dev* split (the data the heuristics
were developed against) and *held-out* splits (unseen generator seeds or a separately
written corpus), so tuning effects are visible rather than hidden:

1. 2-D bounding-box grounding (Region Recall@1 at IoU >= 0.5, page Recall@1, mean IoU)
2. UI-recording keyframe deduplication (pruning, ms/frame, transition recall)
3. Cross-modal root-cause identification (Top-1 root span)
4. Indirect prompt-injection + agent-loop defence (recall, FPR, latency)
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from evals.baselines import FlatChunkBM25, RegionBM25
from evals.datasets.injection_corpus import build_heldout_corpus, build_security_corpus, export_json
from evals.datasets.layouts import build_layout_corpus
from evals.datasets.traces import build_trace_cases
from evals.datasets.ui_sim import dedup_session
from omnitrace import __version__
from omnitrace.config import DedupConfig, VisionConfig
from omnitrace.correlator.rca_engine import analyze
from omnitrace.security.injection_shield import InjectionShield
from omnitrace.security.loop_breaker import LoopBreaker
from omnitrace.vision.phash_dedup import KeyframeExtractor, frame_features, hamming
from omnitrace.vision.spatial_maxsim import PRECISIONS, VisualIndex

ROOT = Path(__file__).resolve().parent.parent
DEV_LAYOUT_SEED, HELDOUT_LAYOUT_SEEDS = 7, (11, 23, 42)
DEV_TRACE_SEED, HELDOUT_TRACE_SEEDS = 11, (5, 23)


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def pct(xs, q):
    return float(np.percentile(np.asarray(xs, dtype=float), q)) if xs else 0.0


def _log(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------------------ 1. grounding
def _eval_grounding(searcher, pages, queries) -> dict:
    regions = {r.id: r for p in pages for r in p.regions}
    hits = page_hits = 0
    ious, fam = [], {}
    for q in queries:
        h = searcher(q.text)
        gt = regions[q.answer_region]
        i = iou(h[1], gt.bbox) if h[0] == q.page_id else 0.0
        ok = i >= 0.5
        hits += ok
        page_hits += h[0] == q.page_id
        ious.append(i)
        f = fam.setdefault(q.family, [0, 0])
        f[0] += ok
        f[1] += 1
    n = len(queries)
    return {"region_recall@1": hits / n, "page_recall@1": page_hits / n, "mean_iou": float(np.mean(ious)),
            "n_queries": n, "per_family_region_recall@1": {k: v[0] / v[1] for k, v in fam.items()}}


def bench_grounding() -> dict:
    _log("[1/4] 2-D bounding-box grounding")
    out: dict = {"dataset": "15 synthetic technical pages x 4 queries (tables, schematics, multi-column specs, "
                            "UI viewports, charts); hit = correct page and IoU >= 0.5 with the answer region"}
    pages, queries = build_layout_corpus(DEV_LAYOUT_SEED)
    t0 = time.perf_counter()
    idx = VisualIndex()
    idx.add_pages(pages)
    index_s = time.perf_counter() - t0
    out["index"] = {"pages": len(pages), "patches_per_page": idx.n_patches, "dim": idx.cfg.dim,
                    "encode_ms_per_page": 1000 * index_s / len(pages), "bytes_per_patch": idx.bytes_per_patch()}
    dev = {}
    for prec in PRECISIONS:
        idx.search("warm up", k=1, precision=prec)
        lat = []

        def s(text, prec=prec):
            t = time.perf_counter()
            h = idx.search(text, k=1, precision=prec)[0]
            lat.append(1000 * (time.perf_counter() - t))
            return h.page_id, h.bbox
        r = _eval_grounding(s, pages, queries)
        r["query_latency_ms_p50"], r["query_latency_ms_p95"] = pct(lat, 50), pct(lat, 95)
        dev[f"omnitrace_{prec}"] = r
    flat = FlatChunkBM25(pages)
    reg = RegionBM25(pages)
    dev["baseline_flat_text_chunk_bm25"] = _eval_grounding(lambda t: (lambda h: (h.page_id, h.bbox))(flat.search(t)), pages, queries)
    dev["baseline_region_bm25_no_layout_context"] = _eval_grounding(lambda t: (lambda h: (h.page_id, h.bbox))(reg.search(t)), pages, queries)
    raw = VisualIndex(replace(VisionConfig(), calibrated=False))
    raw.add_pages(pages, store_image=False)
    dev["ablation_raw_colbert_maxsim_page_ranking"] = _eval_grounding(
        lambda t: (lambda h: (h.page_id, h.bbox))(raw.search(t, k=1)[0]), pages, queries)
    # bbox from the heatmap alone (no structure-layer proposals): raw-screenshot mode
    from omnitrace.vision.spatial_maxsim import heatmap_bbox

    def heat_only(t):
        h = idx.search(t, k=1)[0]
        enc = next(p for p in idx.pages if p.page_id == h.page_id)
        return h.page_id, heatmap_bbox(h.heatmap, enc.width, enc.height)
    dev["ablation_heatmap_only_bbox_no_structure_layer"] = _eval_grounding(heat_only, pages, queries)
    out["dev_seed_%d" % DEV_LAYOUT_SEED] = dev
    held = {}
    for seed in HELDOUT_LAYOUT_SEEDS:
        p2, q2 = build_layout_corpus(seed)
        i2 = VisualIndex()
        i2.add_pages(p2, store_image=False)
        f2 = FlatChunkBM25(p2)
        held[str(seed)] = {
            "omnitrace_int8": _eval_grounding(lambda t: (lambda h: (h.page_id, h.bbox))(i2.search(t, k=1)[0]), p2, q2),
            "baseline_flat_text_chunk_bm25": _eval_grounding(lambda t: (lambda h: (h.page_id, h.bbox))(f2.search(t)), p2, q2),
        }
    out["heldout_seeds"] = held
    out["heldout_mean_region_recall@1_int8"] = float(np.mean([v["omnitrace_int8"]["region_recall@1"] for v in held.values()]))
    _log(f"      dev int8 region R@1 = {dev['omnitrace_int8']['region_recall@1']:.3f}  "
         f"flat-BM25 = {dev['baseline_flat_text_chunk_bm25']['region_recall@1']:.3f}  "
         f"held-out = {out['heldout_mean_region_recall@1_int8']:.3f}")
    return out


# ------------------------------------------------------------------------------ 2. dedup
def bench_dedup() -> dict:
    _log("[2/4] keyframe deduplication")
    rec = dedup_session()
    ex = KeyframeExtractor()
    ex.extract(rec.frames[:10], rec.times[:10])  # warm-up
    runs = [ex.extract(rec.frames, rec.times) for _ in range(7)]
    res = runs[-1]
    gt = set(rec.transitions)
    recalled = gt & set(res.keyframes)
    # baseline: global dHash only (keyframe when Hamming distance to previous frame > 4)
    cfg = DedupConfig()
    feats = [frame_features(f, cfg) for f in rec.frames]
    dh_kf = [0] + [i for i in range(1, len(feats)) if hamming(feats[i].dhash, feats[i - 1].dhash) > 4]
    out = {
        "dataset": f"{len(rec.frames)} frames (480x270 @10 fps, Gaussian sensor noise sigma=1.5) with "
                   f"{len(gt)} ground-truth transitions incl. a 1-digit badge change, a 25-frame spinner, "
                   "modal open/close, toast in/out and a 4-frame fade",
        "omnitrace": {
            "frames": res.n_frames, "keyframes": len(res.keyframes),
            "redundant_frames_eliminated": res.pruned_fraction,
            "ms_per_frame_median": statistics.median(r.ms_per_frame for r in runs),
            "ms_per_frame_max": max(r.ms_per_frame for r in runs),
            "transition_recall": len(recalled) / len(gt), "missed_transitions": sorted(gt - recalled),
            "spurious_keyframes": sorted(set(res.keyframes) - gt - {0, 165, 166, 167}),
            "animation_runs": [(a.start, a.end) for a in res.animations],
            "noise_adaptive_threshold": ex.threshold_used,
        },
        "baseline_global_dhash_hamming_gt4": {
            "keyframes": len(dh_kf), "redundant_frames_eliminated": 1 - len(dh_kf) / len(rec.frames),
            "transition_recall": len(gt & set(dh_kf)) / len(gt),
        },
    }
    o = out["omnitrace"]
    _log(f"      pruned {o['redundant_frames_eliminated']:.1%} at {o['ms_per_frame_median']:.3f} ms/frame, "
         f"recall {o['transition_recall']:.0%}")
    return out


# ------------------------------------------------------------------------------ 3. RCA
def _baselines(case):
    b = case.bundle
    anomalies_t = min([m.t for m in b.dom] + [99.0])
    errs = [s for s in b.spans if s.error or (s.status_code or 0) >= 400 or s.status_code == 0]
    first = min(errs, key=lambda s: s.end, default=None)
    before = [s for s in errs if s.end <= case.bundle.horizon()]
    sym_t = max((m.t for m in b.dom if any(k in (m.attrs.get("class", "") + m.selector + m.text).lower()
                                             for k in ("error", "alert", "toast", "nan", "modal", "spinner", "grid"))),
                default=anomalies_t)
    last_before = max([s for s in before if s.end <= sym_t], key=lambda s: s.end, default=None)
    longest = max(b.spans, key=lambda s: s.duration)
    log_counts = {}
    for l in b.logs:
        if l.level.upper() in ("ERROR", "FATAL") and l.span_id:
            log_counts[l.span_id] = log_counts.get(l.span_id, 0) + 1
    most_logs = max(log_counts, key=log_counts.get) if log_counts else None
    return {
        "first_error_span": first.span_id if first else None,
        "last_error_span_before_symptom": last_before.span_id if last_before else None,
        "longest_span": longest.span_id,
        "most_error_logs_span": most_logs,
    }


def _rca_split(seed: int, with_frames: bool = True) -> dict:
    cases = build_trace_cases(seed)
    rows, lat, base = [], [], {}
    for c in cases:
        bundle = c.bundle if with_frames else replace(c.bundle, frames=None, frame_times=None)
        rep = analyze(bundle)
        lat.append(rep.elapsed_ms)
        ok = rep.root is not None and rep.root.span.span_id in c.root_span_ids
        rank = next((i + 1 for i, cand in enumerate(rep.candidates) if cand.span.span_id in c.root_span_ids), None)
        rows.append({"scenario": c.name, "top1": ok, "rank_of_true_root": rank, "spans": len(c.bundle.spans),
                     "frames": rep.n_frames, "keyframes": len(rep.keyframes),
                     "anomalies": [a.kind for a in rep.anomalies],
                     "predicted": f"{rep.root.span.service}:{rep.root.span.name}" if rep.root else None,
                     "truth": c.root_desc})
        for k, sid in _baselines(c).items():
            base.setdefault(k, []).append(sid in c.root_span_ids)
    return {"top1_accuracy": float(np.mean([r["top1"] for r in rows])),
            "mrr": float(np.mean([1.0 / r["rank_of_true_root"] if r["rank_of_true_root"] else 0.0 for r in rows])),
            "latency_ms_mean": float(np.mean(lat)), "latency_ms_max": float(np.max(lat)),
            "mean_spans_per_trace": float(np.mean([r["spans"] for r in rows])),
            "baselines_top1": {k: float(np.mean(v)) for k, v in base.items()},
            "cases": rows}


def bench_rca() -> dict:
    _log("[3/4] cross-modal root-cause analysis")
    dev = _rca_split(DEV_TRACE_SEED)
    out = {"dataset": "10 failing 12 s storefront sessions (60 frames @5 fps, DOM log, ~66 spans across ~14 services, "
                      "logs) with injected root causes and distractors",
           "dev_seed_%d" % DEV_TRACE_SEED: dev,
           "ablation_no_video_frames": {k: v for k, v in _rca_split(DEV_TRACE_SEED, with_frames=False).items()
                                        if k in ("top1_accuracy", "mrr")},
           "heldout_seeds": {str(s): {k: v for k, v in _rca_split(s).items() if k != "cases"} for s in HELDOUT_TRACE_SEEDS}}
    _log(f"      top-1 {dev['top1_accuracy']:.0%} (baselines: " +
         ", ".join(f"{k} {v:.0%}" for k, v in dev["baselines_top1"].items()) + ")")
    return out


# ------------------------------------------------------------------------------ 4. security
def bench_security() -> dict:
    _log("[4/4] injection shield + loop breaker")
    docs, trajs = build_security_corpus()
    sh = InjectionShield()
    for d in docs:
        sh.scan(d.text, d.source)  # warm-up (regex caches, lazy imports)

    def run_docs(ds):
        tp = fp = fn = tn = 0
        lat, per_cat, misses, fps = [], {}, [], []
        for d in ds:
            ts = []
            for _ in range(5):
                v = sh.scan(d.text, d.source)
                ts.append(v.latency_ms)
            lat.append(statistics.median(ts))
            mal = d.label == "malicious"
            if mal:
                c = per_cat.setdefault(d.category, [0, 0])
                c[1] += 1
                if v.blocked:
                    tp += 1
                    c[0] += 1
                else:
                    fn += 1
                    misses.append(d.id)
            else:
                if v.blocked:
                    fp += 1
                    fps.append(d.id)
                else:
                    tn += 1
        return {"benign": fp + tn, "malicious": tp + fn, "recall": tp / max(1, tp + fn),
                "false_positive_rate": fp / max(1, fp + tn), "precision": tp / max(1, tp + fp),
                "latency_ms_p50": pct(lat, 50), "latency_ms_p95": pct(lat, 95), "latency_ms_max": max(lat),
                "per_category_recall": {k: v[0] / v[1] for k, v in per_cat.items()},
                "missed": misses, "false_positives": fps}

    dev = run_docs(docs)
    held = run_docs(build_heldout_corpus())
    loops, lat = [], []
    for t in trajs:
        lb = LoopBreaker(token_budget=t.budget_steps * 1500)
        t0 = time.perf_counter()
        v = lb.run(t.steps)
        lat.append(1000 * (time.perf_counter() - t0) / max(1, v.step))
        loops.append({"id": t.id, "label": t.label, "halted": v.halt, "step": v.step, "reason": v.reason,
                      "tokens_saved": v.tokens_saved})
    lp = [l for l in loops if l["label"] == "loop"]
    lb_ = [l for l in loops if l["label"] == "benign"]
    loop_res = {"loops": len(lp), "benign_trajectories": len(lb_),
                "loop_detection_recall": float(np.mean([l["halted"] for l in lp])),
                "false_halt_rate": float(np.mean([l["halted"] for l in lb_])),
                "mean_steps_to_halt": float(np.mean([l["step"] for l in lp if l["halted"]])),
                "mean_tokens_saved_of_60k_budget": float(np.mean([l["tokens_saved"] or 0 for l in lp if l["halted"]])),
                "ms_per_step_mean": float(np.mean(lat)), "trajectories": loops}
    n_adv = dev["malicious"] + len(lp)
    caught = dev["recall"] * dev["malicious"] + loop_res["loop_detection_recall"] * len(lp)
    n_ben = dev["benign"] + len(lb_)
    false_alarms = dev["false_positive_rate"] * dev["benign"] + loop_res["false_halt_rate"] * len(lb_)
    out = {"dataset": "dev: 30 benign technical docs (hard negatives) + 25 injection payloads + 5 cyclic agent "
                      "trajectories + 5 progressing trajectories; held-out: 12 benign + 12 payloads written after "
                      "the detectors were frozen",
           "combined_dev": {"adversarial_items": n_adv, "recall": caught / n_adv, "benign_items": n_ben,
                            "false_positive_rate": false_alarms / n_ben},
           "documents_dev": dev, "documents_heldout": held, "loop_breaker": loop_res}
    _log(f"      dev recall {out['combined_dev']['recall']:.0%} FPR {out['combined_dev']['false_positive_rate']:.0%}; "
         f"held-out recall {held['recall']:.0%} FPR {held['false_positive_rate']:.0%}; p95 {dev['latency_ms_p95']:.2f} ms")
    return out


# ------------------------------------------------------------------------------ export + README
def export_datasets() -> None:
    d = ROOT / "evals" / "datasets"
    export_json(d / "injection_corpus.json")
    pages, queries = build_layout_corpus(DEV_LAYOUT_SEED)
    rd = d / "rendered"
    rd.mkdir(exist_ok=True)
    for p in pages:
        Image.fromarray(p.image).save(rd / f"{p.page_id}.png", optimize=True)
        (rd / f"{p.page_id}.json").write_text(json.dumps({"meta": p.meta, "regions": [r.to_dict() for r in p.regions]}))
    (d / "layout_queries.json").write_text(json.dumps([q.__dict__ for q in queries], indent=1))
    traces = []
    for c in build_trace_cases(DEV_TRACE_SEED):
        traces.append({"name": c.name, "description": c.description, "root_span_ids": sorted(c.root_span_ids),
                       "root": c.root_desc, "bundle": c.bundle.to_dict()})
    (d / "traces_dev.json").write_text(json.dumps(traces))


READMES = {"en": ROOT / "README.md", "it": ROOT / "README.it.md"}


def _fmt(x: float, lang: str = "en") -> str:
    out = f"{100 * x:.1f}%"
    return out.replace(".", ",") if lang == "it" else out


def _num(x: float, nd: int, lang: str = "en") -> str:
    out = f"{x:.{nd}f}"
    return out.replace(".", ",") if lang == "it" else out


_BASELINE_NAMES = {
    "en": {"first_error_span": "first error span", "last_error_span_before_symptom": "last error span before symptom",
           "longest_span": "longest span", "most_error_logs_span": "span with most error logs"},
    "it": {"first_error_span": "primo span in errore", "last_error_span_before_symptom": "ultimo span in errore prima del sintomo",
           "longest_span": "span più lungo", "most_error_logs_span": "span con più log di errore"},
}

_T = {
    "en": {
        "measured": "_Measured {ts} on {plat} (Python {py}), omnitrace {ver}, total suite time {secs} s._",
        "header": "| Benchmark | Metric | Target | **omnitrace** | Best baseline |",
        "g_dev": "| 1. 2-D grounding (dev, 60 queries) | Region Recall@1 (IoU>=0.5), int8 | >= 90% | **{v}** | flat-chunk BM25 {b1} / region BM25 {b2} |",
        "g_ho": "| 1. 2-D grounding (held-out seeds {seeds}) | Region Recall@1, int8 | >= 90% | **{v}** | flat-chunk BM25 {b1} |",
        "d_prune": "| 2. Frame dedup (200 frames) | redundant frames eliminated | > 90% | **{v}** | global dHash {b1} |",
        "d_lat": "| 2. Frame dedup | latency per frame (median of 7 runs) | < 0.5 ms | **{v} ms** | - |",
        "d_rec": "| 2. Frame dedup | keyframe transition recall | 100% | **{v}** | global dHash {b1} |",
        "r_dev": "| 3. Cross-modal RCA (dev, 10 traces) | Top-1 root-cause span | >= 90% | **{v}** | {bn} {b1} |",
        "r_ho": "| 3. Cross-modal RCA (held-out seeds) | Top-1 | >= 90% | **{v}** | - |",
        "s_rec": "| 4. Injection + loops (dev, 30 adversarial / 35 benign) | recall | >= 95% | **{v}** | - |",
        "s_fpr": "| 4. Injection + loops (dev) | false-positive rate | 0% | **{v}** | - |",
        "s_ho": "| 4. Injection (held-out, 12 / 12) | recall / FPR | >= 95% / 0% | **{v1} / {v2}** | - |",
        "s_lat": "| 4. Shield latency (dev docs) | p95 / max per document | < 2 ms | **{v1} / {v2} ms** | - |",
        "q_title": "**Quantisation (dev grounding, page Recall@1 / Region Recall@1, bytes per patch vector):**",
        "q_header": "| precision | page R@1 | region R@1 | mean IoU | bytes/patch | query p50 |",
        "a_title": "**Ablations / extra measurements:**",
        "a_raw": "- raw (uncalibrated) ColBERT MaxSim page ranking: page R@1 {v1}, region R@1 {v2}",
        "a_heat": "- bbox from the activation heatmap alone (no structure-layer proposals, raw-screenshot mode): region R@1 {v1}, mean IoU {v2}",
        "a_flat": "- flat-chunk BM25 finds the right *page* {v1} of the time but its citation box has mean IoU {v2}",
        "a_novid": "- RCA without the video frames (DOM + spans + logs only): Top-1 {v}",
        "a_rbase": "- RCA baselines (dev): {v}",
        "a_rlat": "- RCA mean latency {v1} ms per trace (incl. keyframe extraction of 60 frames), {v2} spans per trace",
        "a_loop": "- loop breaker: {v1} of loops halted after {v2} steps on average, {v3} false halts, ~{v4} of 60k budget tokens saved per loop",
        "a_index": "- index encoding {v1} ms/page ({v2} patches x {v3} dims)",
    },
    "it": {
        "measured": "_Misurato il {ts} su {plat} (Python {py}), omnitrace {ver}, durata totale della suite {secs} s._",
        "header": "| Benchmark | Metrica | Obiettivo | **omnitrace** | Miglior baseline |",
        "g_dev": "| 1. Grounding 2-D (dev, 60 query) | Region Recall@1 (IoU>=0,5), int8 | >= 90% | **{v}** | BM25 a chunk piatti {b1} / BM25 per regione {b2} |",
        "g_ho": "| 1. Grounding 2-D (seed held-out {seeds}) | Region Recall@1, int8 | >= 90% | **{v}** | BM25 a chunk piatti {b1} |",
        "d_prune": "| 2. Deduplicazione frame (200 frame) | frame ridondanti eliminati | > 90% | **{v}** | dHash globale {b1} |",
        "d_lat": "| 2. Deduplicazione frame | latenza per frame (mediana di 7 esecuzioni) | < 0,5 ms | **{v} ms** | - |",
        "d_rec": "| 2. Deduplicazione frame | recall delle transizioni (keyframe) | 100% | **{v}** | dHash globale {b1} |",
        "r_dev": "| 3. RCA cross-modale (dev, 10 tracce) | Top-1 span causa radice | >= 90% | **{v}** | {bn} {b1} |",
        "r_ho": "| 3. RCA cross-modale (seed held-out) | Top-1 | >= 90% | **{v}** | - |",
        "s_rec": "| 4. Iniezione + loop (dev, 30 avversari / 35 benigni) | recall | >= 95% | **{v}** | - |",
        "s_fpr": "| 4. Iniezione + loop (dev) | tasso di falsi positivi | 0% | **{v}** | - |",
        "s_ho": "| 4. Iniezione (held-out, 12 / 12) | recall / FPR | >= 95% / 0% | **{v1} / {v2}** | - |",
        "s_lat": "| 4. Latenza dello shield (documenti dev) | p95 / max per documento | < 2 ms | **{v1} / {v2} ms** | - |",
        "q_title": "**Quantizzazione (grounding dev, Recall@1 di pagina / Recall@1 di regione, byte per vettore di patch):**",
        "q_header": "| precisione | R@1 pagina | R@1 regione | IoU medio | byte/patch | query p50 |",
        "a_title": "**Ablazioni / misure aggiuntive:**",
        "a_raw": "- ranking di pagina con MaxSim ColBERT grezzo (non calibrato): R@1 pagina {v1}, R@1 regione {v2}",
        "a_heat": "- bbox dalla sola heatmap di attivazione (senza proposte dallo structure layer, modalità screenshot grezzo): R@1 regione {v1}, IoU medio {v2}",
        "a_flat": "- il BM25 a chunk piatti trova la *pagina* giusta nel {v1} dei casi, ma il suo box di citazione ha IoU medio {v2}",
        "a_novid": "- RCA senza i frame video (solo DOM + span + log): Top-1 {v}",
        "a_rbase": "- baseline RCA (dev): {v}",
        "a_rlat": "- latenza media della RCA {v1} ms per traccia (inclusa l'estrazione dei keyframe da 60 frame), {v2} span per traccia",
        "a_loop": "- loop breaker: {v1} dei loop fermati in media dopo {v2} passi, {v3} di arresti errati, ~{v4} token risparmiati per loop su un budget di 60k",
        "a_index": "- codifica dell'indice {v1} ms/pagina ({v2} patch x {v3} dimensioni)",
    },
}


def metrics_markdown(r: dict, lang: str = "en") -> str:
    T, B = _T[lang], _BASELINE_NAMES[lang]
    f = lambda x: _fmt(x, lang)  # noqa: E731
    n = lambda x, d: _num(x, d, lang)  # noqa: E731
    g, dd, rc, s = r["grounding"], r["dedup"], r["rca"], r["security"]
    gd = g[f"dev_seed_{DEV_LAYOUT_SEED}"]
    rd = rc[f"dev_seed_{DEV_TRACE_SEED}"]
    o = dd["omnitrace"]
    dh = dd["baseline_global_dhash_hamming_gt4"]
    best_rca = max(rd["baselines_top1"], key=rd["baselines_top1"].get)
    held_flat = float(np.mean([v["baseline_flat_text_chunk_bm25"]["region_recall@1"] for v in g["heldout_seeds"].values()]))
    m = r["meta"]
    lines = [
        T["measured"].format(ts=m["timestamp"], plat=m["platform"], py=m["python"], ver=m["version"],
                             secs=f"{m['suite_seconds']:.0f}"),
        "", T["header"], "|---|---|---|---|---|",
        T["g_dev"].format(v=f(gd["omnitrace_int8"]["region_recall@1"]),
                          b1=f(gd["baseline_flat_text_chunk_bm25"]["region_recall@1"]),
                          b2=f(gd["baseline_region_bm25_no_layout_context"]["region_recall@1"])),
        T["g_ho"].format(seeds=", ".join(map(str, HELDOUT_LAYOUT_SEEDS)),
                         v=f(g["heldout_mean_region_recall@1_int8"]), b1=f(held_flat)),
        T["d_prune"].format(v=f(o["redundant_frames_eliminated"]), b1=f(dh["redundant_frames_eliminated"])),
        T["d_lat"].format(v=n(o["ms_per_frame_median"], 3)),
        T["d_rec"].format(v=f(o["transition_recall"]), b1=f(dh["transition_recall"])),
        T["r_dev"].format(v=f(rd["top1_accuracy"]), bn=B[best_rca], b1=f(rd["baselines_top1"][best_rca])),
        T["r_ho"].format(v=" / ".join(f(v["top1_accuracy"]) for v in rc["heldout_seeds"].values())),
        T["s_rec"].format(v=f(s["combined_dev"]["recall"])),
        T["s_fpr"].format(v=f(s["combined_dev"]["false_positive_rate"])),
        T["s_ho"].format(v1=f(s["documents_heldout"]["recall"]), v2=f(s["documents_heldout"]["false_positive_rate"])),
        T["s_lat"].format(v1=n(s["documents_dev"]["latency_ms_p95"], 2), v2=n(s["documents_dev"]["latency_ms_max"], 2)),
        "", T["q_title"], "", T["q_header"], "|---|---|---|---|---|---|",
    ]
    bpp = g["index"]["bytes_per_patch"]
    for p in PRECISIONS:
        e = gd[f"omnitrace_{p}"]
        lines.append(f"| {p} | {f(e['page_recall@1'])} | {f(e['region_recall@1'])} | {n(e['mean_iou'], 3)} | {bpp[p]} | "
                     f"{n(e['query_latency_ms_p50'], 1)} ms |")
    raw = gd["ablation_raw_colbert_maxsim_page_ranking"]
    heat = gd["ablation_heatmap_only_bbox_no_structure_layer"]
    flat = gd["baseline_flat_text_chunk_bm25"]
    lb = s["loop_breaker"]
    lines += [
        "", T["a_title"], "",
        T["a_raw"].format(v1=f(raw["page_recall@1"]), v2=f(raw["region_recall@1"])),
        T["a_heat"].format(v1=f(heat["region_recall@1"]), v2=n(heat["mean_iou"], 3)),
        T["a_flat"].format(v1=f(flat["page_recall@1"]), v2=n(flat["mean_iou"], 3)),
        T["a_novid"].format(v=f(rc["ablation_no_video_frames"]["top1_accuracy"])),
        T["a_rbase"].format(v=", ".join(f"{B[k]} {f(v)}" for k, v in rd["baselines_top1"].items())),
        T["a_rlat"].format(v1=n(rd["latency_ms_mean"], 1), v2=f"{rd['mean_spans_per_trace']:.0f}"),
        T["a_loop"].format(v1=f(lb["loop_detection_recall"]), v2=n(lb["mean_steps_to_halt"], 1),
                           v3=f(lb["false_halt_rate"]), v4=f"{lb['mean_tokens_saved_of_60k_budget']:.0f}"),
        T["a_index"].format(v1=f"{g['index']['encode_ms_per_page']:.0f}", v2=g["index"]["patches_per_page"],
                            v3=g["index"]["dim"]),
    ]
    return "\n".join(lines)


def replace_block(path: Path, tag: str, content: str) -> bool:
    """Replace ``<!-- TAG:START --> ... <!-- TAG:END -->`` in ``path``; False if the markers are absent."""
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    pat = re.compile(rf"<!-- {tag}:START -->.*?<!-- {tag}:END -->", re.S)
    if not pat.search(text):
        return False
    block = f"<!-- {tag}:START -->\n{content}\n<!-- {tag}:END -->"
    path.write_text(pat.sub(lambda _: block, text), encoding="utf-8")
    return True


def write_readme_metrics(r: dict) -> None:
    for lang, path in READMES.items():
        replace_block(path, "METRICS", metrics_markdown(r, lang))


def main(output: Path | None = None, update_readme: bool = True, export: bool = True) -> dict:
    t0 = time.perf_counter()
    report = {"grounding": bench_grounding(), "dedup": bench_dedup(), "rca": bench_rca(), "security": bench_security()}
    report["meta"] = {"version": __version__, "python": platform.python_version(),
                      "platform": f"{platform.system()} {platform.machine()}",
                      "timestamp": time.strftime("%Y-%m-%d %H:%M"), "suite_seconds": time.perf_counter() - t0}
    output = output or ROOT / "evals" / "latest_report.json"
    output.write_text(json.dumps(report, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o)))
    if export:
        export_datasets()
    if update_readme:
        write_readme_metrics(report)
    _log(f"report written to {output} ({report['meta']['suite_seconds']:.0f} s)")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--no-readme", action="store_true")
    ap.add_argument("--no-export", action="store_true")
    a = ap.parse_args()
    main(a.output, update_readme=not a.no_readme, export=not a.no_export)
    sys.exit(0)
