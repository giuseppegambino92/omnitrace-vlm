"""Repeatability check: run the benchmark suite N times (default 3) on a quiet machine.

    python -m evals.repeat_runs                 # 3 runs, wait for load <= 0.35 per core
    python -m evals.repeat_runs --runs 5 --max-load 0.25 --wait-timeout 1800

* **Quiet-machine gate.** Before every run the script waits until the 1-minute load
  average divided by the logical CPU count is at or below ``--max-load``, then lets the
  machine settle for ``--settle`` seconds. If the gate never opens within
  ``--wait-timeout`` it aborts (exit 3); ``--force`` runs anyway and records the load.
* **Isolation.** Each run is a fresh ``python -m evals.benchmark_suite`` subprocess (no
  shared caches, no warm imports).
* **Verdict.** Every non-timing metric in the report (accuracy, recall, FPR, IoU, counts)
  must be *bit-identical* across runs - the pipeline is deterministic, so any difference
  is a bug. Timing metrics are reported as ``[min, max]`` intervals with their coefficient
  of variation, and their targets must hold in *every* run.

Writes ``evals/repeatability_report.json`` and refreshes the ``REPEAT`` block of
``README.md`` and ``README.it.md``. With ``--update-metrics`` the last run also becomes
``evals/latest_report.json`` and refreshes the ``METRICS`` blocks. Exit code 0 = repeatable
and all targets met, 1 = not repeatable or a target missed, 3 = machine never quiet.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "evals" / "repeatability_report.json"

# timing metrics: (dotted path, target as (op, value) or None, label en, label it)
TIMINGS: list[tuple[str, tuple[str, float] | None, str, str]] = [
    ("dedup.omnitrace.ms_per_frame_median", ("<", 0.5), "Frame dedup, ms/frame (median of 7)",
     "Deduplicazione frame, ms/frame (mediana di 7)"),
    ("security.documents_dev.latency_ms_p95", ("<", 2.0), "Shield latency p95, ms/doc",
     "Latenza shield p95, ms/doc"),
    ("security.documents_dev.latency_ms_max", ("<", 2.0), "Shield latency max, ms/doc",
     "Latenza shield max, ms/doc"),
    ("grounding.dev_seed_7.omnitrace_int8.query_latency_ms_p50", None, "Visual query p50 (int8), ms",
     "Query visuale p50 (int8), ms"),
    ("rca.dev_seed_11.latency_ms_mean", None, "RCA mean latency per trace, ms", "Latenza media RCA per traccia, ms"),
    ("security.loop_breaker.ms_per_step_mean", None, "Loop breaker, ms/step", "Loop breaker, ms/passo"),
]
# headline accuracy metrics shown in the README (all non-timing metrics are compared anyway)
HEADLINES: list[tuple[str, str, str]] = [
    ("grounding.dev_seed_7.omnitrace_int8.region_recall@1", "Grounding Region R@1 (dev, int8)", "Grounding R@1 regione (dev, int8)"),
    ("grounding.heldout_mean_region_recall@1_int8", "Grounding Region R@1 (held-out)", "Grounding R@1 regione (held-out)"),
    ("dedup.omnitrace.redundant_frames_eliminated", "Frames eliminated", "Frame eliminati"),
    ("dedup.omnitrace.transition_recall", "Transition recall", "Recall transizioni"),
    ("rca.dev_seed_11.top1_accuracy", "RCA Top-1 (dev)", "RCA Top-1 (dev)"),
    ("security.combined_dev.recall", "Injection + loop recall (dev)", "Recall iniezioni + loop (dev)"),
    ("security.combined_dev.false_positive_rate", "Injection + loop FPR (dev)", "FPR iniezioni + loop (dev)"),
    ("security.documents_heldout.recall", "Injection recall (held-out)", "Recall iniezioni (held-out)"),
]
_TIMING_KEY = re.compile(r"latency|_ms|ms_|seconds|timestamp|threshold_used")


def load_per_core() -> float:
    return os.getloadavg()[0] / (os.cpu_count() or 1)


def wait_until_quiet(max_load: float, timeout: float, settle: float, poll: float = 10.0) -> tuple[bool, float]:
    t0 = time.monotonic()
    while True:
        lpc = load_per_core()
        if lpc <= max_load:
            if settle > 0:
                time.sleep(settle)
            return True, load_per_core()
        if time.monotonic() - t0 >= timeout:
            return False, lpc
        print(f"  waiting for a quiet machine: load/core {lpc:.2f} > {max_load:.2f}", flush=True)
        time.sleep(poll)


def flatten(d, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(flatten(v, f"{prefix}{i}."))
    elif isinstance(d, (bool, int, float)) and not isinstance(d, str):
        out[prefix.rstrip(".")] = float(d)
    return out


def get(d: dict, path: str) -> float:
    for part in path.split("."):
        d = d[part]
    return float(d)


def run_suite(out: Path) -> float:
    t0 = time.perf_counter()
    subprocess.run([sys.executable, "-m", "evals.benchmark_suite", "--output", str(out), "--no-readme", "--no-export"],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    return time.perf_counter() - t0


def analyse(reports: list[dict]) -> dict:
    flats = [{k: v for k, v in flatten(r).items() if not k.startswith("meta.")} for r in reports]
    keys = sorted(set().union(*flats))
    exact = [k for k in keys if not _TIMING_KEY.search(k)]
    diffs = {k: [f.get(k) for f in flats] for k in exact if len({f.get(k) for f in flats}) > 1}
    timings = []
    for path, target, en, it in TIMINGS:
        vals = [get(r, path) for r in reports]
        mean = statistics.fmean(vals)
        cv = statistics.pstdev(vals) / mean if mean else 0.0
        ok = True
        if target:
            op, lim = target
            ok = all(v < lim for v in vals) if op == "<" else all(v > lim for v in vals)
        timings.append({"metric": path, "label_en": en, "label_it": it, "target": target, "values": vals,
                        "min": min(vals), "max": max(vals), "cv": cv, "target_met_every_run": ok})
    headlines = [{"metric": p, "label_en": en, "label_it": it, "values": [get(r, p) for r in reports]}
                 for p, en, it in HEADLINES]
    return {"compared_exact_metrics": len(exact), "nonrepeatable_metrics": diffs, "timings": timings,
            "headlines": headlines, "repeatable": not diffs,
            "all_targets_met": all(t["target_met_every_run"] for t in timings)}


def _pct(x: float, lang: str) -> str:
    s = f"{100 * x:.1f}%"
    return s.replace(".", ",") if lang == "it" else s


def _ms(x: float, lang: str) -> str:
    s = f"{x:.3f}" if x < 1 else f"{x:.2f}"
    return s.replace(".", ",") if lang == "it" else s


def markdown(rep: dict, lang: str) -> str:
    n = len(rep["runs"])
    it = lang == "it"
    loads = ", ".join(_ms(r["load_per_core_at_start"], lang) for r in rep["runs"])
    head = (f"_{n} esecuzioni consecutive il {rep['finished']} su {rep['platform']} ({rep['cpus']} CPU logiche), "
            f"ciascuna in un processo separato; carico a 1 minuto per core all'avvio di ogni esecuzione: {loads} "
            f"(soglia {_ms(rep['max_load_per_core'], lang)})._"
            if it else
            f"_{n} consecutive runs on {rep['finished']} on {rep['platform']} ({rep['cpus']} logical CPUs), each in a "
            f"fresh process; 1-minute load per core at the start of each run: {loads} "
            f"(gate {_ms(rep['max_load_per_core'], lang)})._")
    a = rep["analysis"]
    verdict_ok = a["repeatable"]
    if it:
        verdict = (f"**{a['compared_exact_metrics']} metriche non temporali confrontate: "
                   + ("tutte identiche bit per bit in ogni esecuzione.**" if verdict_ok else
                      f"{len(a['nonrepeatable_metrics'])} differiscono (vedi `evals/repeatability_report.json`).**"))
        th = "| Metrica di tempo | Obiettivo | " + " | ".join(f"Esec. {i + 1}" for i in range(n)) + " | Intervallo [min, max] | CV | Obiettivo in ogni esecuzione |"
        hh = "| Metrica di accuratezza | " + " | ".join(f"Esec. {i + 1}" for i in range(n)) + " |"
        yes, no = "sì", "**no**"
    else:
        verdict = (f"**{a['compared_exact_metrics']} non-timing metrics compared: "
                   + ("all bit-identical in every run.**" if verdict_ok else
                      f"{len(a['nonrepeatable_metrics'])} differ (see `evals/repeatability_report.json`).**"))
        th = "| Timing metric | Target | " + " | ".join(f"Run {i + 1}" for i in range(n)) + " | Interval [min, max] | CV | Target met every run |"
        hh = "| Accuracy metric | " + " | ".join(f"Run {i + 1}" for i in range(n)) + " |"
        yes, no = "yes", "**no**"
    lines = [head, "", verdict, "", th, "|---|---|" + "---|" * n + "---|---|---|"]
    for t in a["timings"]:
        tgt = f"{t['target'][0]} {_ms(t['target'][1], lang)}" if t["target"] else "-"
        vals = " | ".join(_ms(v, lang) for v in t["values"])
        lines.append(f"| {t['label_' + lang]} | {tgt} | {vals} | [{_ms(t['min'], lang)}, {_ms(t['max'], lang)}] | "
                     f"{_pct(t['cv'], lang)} | {yes if t['target_met_every_run'] else no} |")
    lines += ["", hh, "|---|" + "---|" * n]
    for h in a["headlines"]:
        lines.append(f"| {h['label_' + lang]} | " + " | ".join(_pct(v, lang) for v in h["values"]) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-load", type=float, default=0.35, help="max 1-minute load average per logical CPU")
    ap.add_argument("--wait-timeout", type=float, default=900.0, help="seconds to wait for a quiet machine per run")
    ap.add_argument("--settle", type=float, default=15.0, help="seconds to idle once the gate opens")
    ap.add_argument("--force", action="store_true", help="run even if the machine never becomes quiet")
    ap.add_argument("--no-readme", action="store_true")
    ap.add_argument("--update-metrics", action="store_true",
                    help="promote the last run to evals/latest_report.json and refresh the METRICS blocks")
    a = ap.parse_args(argv)
    import platform

    from evals.benchmark_suite import READMES, replace_block, write_readme_metrics

    runs, reports = [], []
    with tempfile.TemporaryDirectory(prefix="omnitrace-repeat-") as tmp:
        for i in range(a.runs):
            print(f"[run {i + 1}/{a.runs}]", flush=True)
            quiet, lpc = wait_until_quiet(a.max_load, a.wait_timeout, a.settle)
            if not quiet and not a.force:
                print(f"machine never became quiet (load/core {lpc:.2f} > {a.max_load}); use --force to run anyway",
                      file=sys.stderr)
                return 3
            out = Path(tmp) / f"run_{i + 1}.json"
            secs = run_suite(out)
            reports.append(json.loads(out.read_text()))
            runs.append({"run": i + 1, "load_per_core_at_start": lpc, "quiet_gate_passed": quiet, "wall_seconds": secs})
            print(f"  done in {secs:.0f} s (load/core at start {lpc:.2f})", flush=True)
        if a.update_metrics:
            shutil.copy(Path(tmp) / f"run_{a.runs}.json", ROOT / "evals" / "latest_report.json")
    rep = {"finished": time.strftime("%Y-%m-%d %H:%M"), "platform": f"{platform.system()} {platform.machine()}",
           "cpus": os.cpu_count(), "max_load_per_core": a.max_load, "runs": runs, "analysis": analyse(reports)}
    REPORT.write_text(json.dumps(rep, indent=1))
    if not a.no_readme:
        for lang, path in READMES.items():
            replace_block(path, "REPEAT", markdown(rep, lang))
        if a.update_metrics:
            write_readme_metrics(reports[-1])
    an = rep["analysis"]
    print(f"repeatable: {an['repeatable']} ({an['compared_exact_metrics']} exact metrics), "
          f"all timing targets met in every run: {an['all_targets_met']}")
    for t in an["timings"]:
        print(f"  {t['metric']}: [{t['min']:.3f}, {t['max']:.3f}] cv={t['cv']:.1%}")
    return 0 if an["repeatable"] and an["all_targets_met"] else 1


if __name__ == "__main__":
    sys.exit(main())
