"""Rich terminal UI: ASCII 2-D activation heatmaps with the grounded bbox, the cross-modal
causal timeline / DAG path, and security alerts."""

from __future__ import annotations

import numpy as np
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from omnitrace.correlator.rca_engine import RCAReport
from omnitrace.correlator.timeline_dag import Anomaly, DomMutation, Span, TraceBundle
from omnitrace.security.injection_shield import ShieldVerdict
from omnitrace.security.loop_breaker import LoopVerdict
from omnitrace.vision.spatial_maxsim import GroundedHit

_SHADES = " .:-=+*#%@"
_HEAT_STYLES = ["grey23", "grey37", "grey50", "dark_orange3", "orange3", "dark_orange", "orange1", "red3", "red1", "bright_red"]


def heatmap_text(heat: np.ndarray, bbox_cells: tuple[int, int, int, int] | None = None) -> Text:
    """Render a heatmap (values in [0,1]) as shaded characters; bbox cells are outlined."""
    t = Text()
    gr, gc = heat.shape
    for r in range(gr):
        for c in range(gc):
            v = float(np.clip(heat[r, c], 0, 1))
            k = min(len(_SHADES) - 1, int(v * (len(_SHADES) - 1) + 0.5))
            ch = _SHADES[k]
            style = _HEAT_STYLES[k]
            if bbox_cells:
                r1, c1, r2, c2 = bbox_cells
                on_edge = (r1 <= r <= r2 and c1 <= c <= c2) and (r in (r1, r2) or c in (c1, c2))
                if on_edge:
                    ch, style = ("█" if k > 4 else "▒"), "bold bright_cyan"
            t.append(ch * 2, style=style)
        t.append("\n")
    return t


def bbox_to_cells(bbox, width: int, height: int, grid: tuple[int, int]) -> tuple[int, int, int, int]:
    gr, gc = grid
    x1, y1, x2, y2 = bbox
    return (int(y1 / height * gr), int(x1 / width * gc),
            min(gr - 1, int(np.ceil(y2 / height * gr)) - 1), min(gc - 1, int(np.ceil(x2 / width * gc)) - 1))


def _pool(heat: np.ndarray, f: int) -> np.ndarray:
    gr, gc = heat.shape
    gr2, gc2 = gr // f, gc // f
    return heat[: gr2 * f, : gc2 * f].reshape(gr2, f, gc2, f).max(axis=(1, 3))


def render_hit(hit: GroundedHit, width: int, height: int, max_rows: int = 16) -> Panel:
    f = max(1, int(np.ceil(hit.heatmap.shape[0] / max_rows)))
    heat = _pool(hit.heatmap, f) if f > 1 else hit.heatmap
    cells = bbox_to_cells(hit.bbox, width, height, heat.shape)
    meta = Table.grid(padding=(0, 1))
    meta.add_row("[bold]page[/]", hit.page_id, "[bold]score[/]", f"{hit.score:.3f}")
    meta.add_row("[bold]bbox[/]", "(" + ", ".join(f"{v:.0f}" for v in hit.bbox) + ")", "[bold]grounding[/]", hit.grounding)
    meta.add_row("[bold]region[/]", f"{hit.region_role}: {hit.region_text[:60]!r}", "[bold]latency[/]", f"{hit.latency_ms:.1f} ms")
    peaks = ", ".join(f"{t}@({r},{c}) {s:.2f}" for t, (r, c, s) in hit.token_peaks.items())
    return Panel(Group(meta, Text(f"token peaks: {peaks}", style="dim"), heatmap_text(heat, cells)),
                 title=f"#{hit.rank + 1} visual citation", border_style="cyan")


def _node_label(node: object) -> str:
    if isinstance(node, Span):
        st = f" [{node.status_code}]" if node.status_code is not None else ""
        err = " [red]ERR[/]" if node.error else ""
        return f"[yellow]span[/] {node.service} · {node.name}{st}{err}  [dim]{node.start:.2f}-{node.end:.2f}s[/]"
    if isinstance(node, DomMutation):
        return f"[magenta]dom[/] {node.op} {node.selector} {node.text[:40]!r}  [dim]@{node.t:.2f}s[/]"
    if isinstance(node, Anomaly):
        return f"[bold red]symptom[/] {node.kind} ({node.source}) bbox={tuple(round(v) for v in node.bbox) if node.bbox else None}  [dim]@{node.t_start:.2f}s[/]"
    return str(node)


def render_rca(report: RCAReport, bundle: TraceBundle | None = None, top: int = 5) -> Group:
    parts = []
    an = Table(title="visible anomalies", show_lines=False, expand=False)
    for col in ("t", "kind", "source", "severity", "detail"):
        an.add_column(col)
    for a in report.anomalies:
        an.add_row(f"{a.t_start:.2f}", a.kind, a.source, f"{a.severity:.1f}", a.detail[:60])
    parts.append(an)
    ct = Table(title="root-cause ranking", expand=False)
    for col in ("#", "score", "service", "operation", "evidence", "origin"):
        ct.add_column(col)
    for i, c in enumerate(report.candidates[:top]):
        ct.add_row(str(i + 1), f"{c.score:.4f}", c.span.service, c.span.name, "; ".join(c.evidence.reasons)[:70],
                   f"{c.origin:.2f}")
    parts.append(ct)
    if report.root:
        tree = Tree("[bold]causal path (root → symptom)[/]")
        node = tree
        for key in report.root.path:
            node = node.add(_node_label(report.dag.nodes.get(key, key)))
        parts.append(tree)
    if bundle is not None:
        parts.append(timeline_strip(report, bundle))
    parts.append(Text(f"frames {report.n_frames} → keyframes {len(report.keyframes)} · DAG {len(report.dag.nodes)} nodes / "
                      f"{report.dag.n_edges} edges · {report.elapsed_ms:.1f} ms", style="dim"))
    return Group(*parts)


def timeline_strip(report: RCAReport, bundle: TraceBundle, width: int = 56) -> Text:
    t_end = max([bundle.horizon()] + [a.t_start for a in report.anomalies]) or 1.0

    def col(t: float) -> int:
        return max(0, min(width - 1, int(t / t_end * (width - 1))))

    rows = {"frames ": [" "] * width, "dom    ": [" "] * width, "spans  ": [" "] * width, "symptom": [" "] * width}
    for i in report.keyframes:
        if bundle.frame_times and i < len(bundle.frame_times):
            rows["frames "][col(bundle.frame_times[i])] = "▮"
    for m in bundle.dom:
        rows["dom    "][col(m.t)] = "◆"
    for s in bundle.spans:
        if s.error or (s.status_code or 0) >= 400:
            rows["spans  "][col(min(s.end, t_end))] = "✖"
    for a in report.anomalies:
        rows["symptom"][col(a.t_start)] = "▲"
    out = Text("timeline (0 → %.1fs)\n" % t_end, style="bold")
    styles = {"frames ": "cyan", "dom    ": "magenta", "spans  ": "red", "symptom": "bold red"}
    for k, r in rows.items():
        out.append(f"{k} │", style="dim")
        out.append("".join(r), style=styles[k])
        out.append("│\n", style="dim")
    return out


def render_security(items: list[tuple[str, ShieldVerdict]]) -> Table:
    tb = Table(title="injection shield", expand=False)
    for col in ("item", "verdict", "risk", "top findings", "ms"):
        tb.add_column(col)
    for name, v in items:
        verdict = "[bold red]BLOCK[/]" if v.blocked else ("[yellow]warn[/]" if v.findings else "[green]pass[/]")
        tops = ", ".join(f"{f.detector}({f.severity:.2f})" for f in sorted(v.findings, key=lambda f: -f.severity)[:3])
        tb.add_row(name, verdict, f"{v.risk:.2f}", tops or "-", f"{v.latency_ms:.2f}")
    return tb


def render_loops(items: list[tuple[str, LoopVerdict]]) -> Table:
    tb = Table(title="loop breaker", expand=False)
    for col in ("trajectory", "halt", "step", "entropy", "novelty", "reason"):
        tb.add_column(col)
    for name, v in items:
        tb.add_row(name, "[bold red]HALT[/]" if v.halt else "[green]ok[/]", str(v.step), f"{v.entropy:.2f}",
                   f"{v.novelty:.2f}", v.reason)
    return tb


def dashboard(console: Console | None = None) -> None:
    """Self-contained demo over the synthetic corpora (no network, no models)."""
    from evals.datasets.injection_corpus import build_security_corpus
    from evals.datasets.layouts import build_layout_corpus
    from evals.datasets.traces import build_trace_cases
    from omnitrace.correlator.rca_engine import analyze
    from omnitrace.security.injection_shield import InjectionShield
    from omnitrace.security.loop_breaker import LoopBreaker
    from omnitrace.vision.spatial_maxsim import VisualIndex

    console = console or Console()
    pages, queries = build_layout_corpus()
    idx = VisualIndex()
    idx.add_pages(pages)
    q = next(x for x in queries if x.family == "table")
    hit = idx.search(q.text, k=1)[0]
    page = next(p for p in pages if p.page_id == hit.page_id)
    console.rule(f"[bold cyan]Subsystem A · visual query: {q.text!r}")
    console.print(render_hit(hit, page.width, page.height))
    case = build_trace_cases()[0]
    rep = analyze(case.bundle)
    console.rule(f"[bold yellow]Subsystem B · RCA: {case.name}")
    console.print(render_rca(rep, case.bundle))
    docs, trajs = build_security_corpus()
    sh = InjectionShield()
    sample = [d for d in docs if d.label == "benign"][:3] + [d for d in docs if d.label == "malicious"][::5]
    console.rule("[bold red]Subsystem C · shield & loop breaker")
    console.print(render_security([(d.id, sh.scan(d.text, d.source)) for d in sample]))
    console.print(render_loops([(t.id, LoopBreaker().run(t.steps)) for t in trajs]))
