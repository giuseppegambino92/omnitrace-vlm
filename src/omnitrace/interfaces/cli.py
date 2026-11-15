"""``omnitrace`` command line (Typer)."""

from __future__ import annotations

import json
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Optional

import numpy as np
import typer
from PIL import Image
from rich.console import Console
from typer.core import TyperGroup

from omnitrace import i18n
from omnitrace.config import OmnitraceConfig
from omnitrace.i18n import t


class Lang(str, Enum):
    en = "en"
    it = "it"


def _scan_lang(args: list[str], commands: Any) -> str | None:
    """The ``--lang`` value given before the sub-command, read ahead of Click's own parsing so that
    help texts (including ``--help``, which Click handles eagerly) can be localised too."""
    for i, tok in enumerate(args):
        if tok in commands:
            break
        if tok == "--lang" and i + 1 < len(args):
            return args[i + 1]
        if tok.startswith("--lang="):
            return tok.split("=", 1)[1]
    return None


def _localize_help(group: Any) -> None:
    """Rewrite the help texts of the command tree in the current language (keys ``help.*``)."""
    group.help = t("help.app")
    for p in group.params:
        if i18n.has_key(f"help.{p.name}"):
            p.help = t(f"help.{p.name}")
    for name, cmd in group.commands.items():
        cmd.help = t(f"help.cmd.{name}")
        for p in cmd.params:
            if i18n.has_key(key := f"help.opt.{name}.{p.name}"):
                p.help = t(key)


class LocalizedGroup(TyperGroup):
    def make_context(self, info_name: str | None, args: list[str], parent: Any = None, **extra: Any) -> Any:
        i18n.set_lang(_scan_lang(args, self.commands))     # None -> $OMNITRACE_LANG -> en
        _localize_help(self)
        return super().make_context(info_name, args, parent, **extra)


app = typer.Typer(cls=LocalizedGroup, add_completion=False, no_args_is_help=True, help=t("help.app"))
console = Console()


@app.callback()
def _root(lang: Annotated[Lang | None, typer.Option("--lang", case_sensitive=False)] = None) -> None:
    # help texts of the whole command tree, this option included, come from the catalog: see _localize_help
    i18n.set_lang(lang.value if lang else None)


def _cfg(db: Optional[Path]) -> OmnitraceConfig:
    cfg = OmnitraceConfig.from_env()
    if db is not None:
        from dataclasses import replace
        cfg = replace(cfg, db_path=db)
    return cfg


def _load_pages(path: Path):
    """A directory of ``<name>.png`` (+ optional ``<name>.json`` structure layer: a list of regions)."""
    from omnitrace.vision.patch_encoder import Page, Region

    pages = []
    for img in sorted(list(path.glob("*.png")) + list(path.glob("*.jpg")) + list(path.glob("*.jpeg"))):
        side = img.with_suffix(".json")
        regions = []
        meta = {}
        if side.exists():
            data = json.loads(side.read_text())
            items = data.get("regions", data) if isinstance(data, dict) else data
            meta = data.get("meta", {}) if isinstance(data, dict) else {}
            regions = [Region.from_dict(r) for r in items]
        arr = np.asarray(Image.open(img).convert("RGB"))
        pages.append(Page(img.stem, arr, regions, doc_id=path.name, meta=meta))
    return pages


@app.command()
def index(path: Optional[Path] = typer.Argument(None), db: Optional[Path] = typer.Option(None),
          demo: bool = typer.Option(False)):
    """Encode pages into a multi-vector visual index."""
    from omnitrace.vision.spatial_maxsim import VisualIndex

    cfg = _cfg(db)
    if demo:
        from evals.datasets.layouts import build_layout_corpus
        pages, _ = build_layout_corpus()
    elif path is not None:
        pages = _load_pages(path)
    else:
        raise typer.BadParameter(t("cli.err.index_source"))
    if not pages:
        raise typer.BadParameter(t("cli.err.no_images", path=path))
    idx = VisualIndex(cfg.vision)
    with console.status(t("cli.status.encoding", n=len(pages))):
        idx.add_pages(pages)
    out = idx.save(cfg.db_path)
    bpp = idx.bytes_per_patch()
    console.print(t("cli.indexed", n=len(pages), patches=idx.n_patches, out=out, f32=bpp["float32"],
                    int8=bpp["int8"], int2=bpp["int2"], binary=bpp["binary"]))


@app.command()
def query(text: str, db: Optional[Path] = typer.Option(None), k: int = 3,
          precision: str = typer.Option("int8"),
          crop: Optional[Path] = typer.Option(None),
          vlm: Optional[str] = typer.Option(None),
          as_json: bool = typer.Option(False, "--json")):
    """Retrieve pages and ground the answer to a bounding box."""
    from omnitrace.interfaces.tui import render_hit
    from omnitrace.vision.spatial_maxsim import VisualIndex

    cfg = _cfg(db)
    idx = VisualIndex.load(cfg.db_path)
    hits = idx.search(text, k=k, precision=precision)
    if vlm:
        from omnitrace.vision.vlm_adapters import answer_with_citation, get_adapter
        res = answer_with_citation(idx, text, get_adapter(vlm), precision)
        res.pop("crop_png_b64", None)
        console.print_json(json.dumps(res))
    if as_json:
        console.print_json(json.dumps([h.to_dict() for h in hits]))
    else:
        for h in hits:
            enc = next(p for p in idx.pages if p.page_id == h.page_id)
            console.print(render_hit(h, enc.width, enc.height))
    if crop and hits:
        Image.fromarray(idx.crop(hits[0])).save(crop)
        console.print(t("cli.crop_written", path=crop))


@app.command()
def rca(trace: Optional[Path] = typer.Argument(None),
        demo: Optional[str] = typer.Option(None),
        as_json: bool = typer.Option(False, "--json")):
    """Cross-modal root-cause analysis of one failing execution."""
    from omnitrace.correlator.rca_engine import analyze
    from omnitrace.correlator.timeline_dag import TraceBundle
    from omnitrace.interfaces.tui import render_rca

    if demo:
        from evals.datasets.traces import SCENARIOS, build_trace_cases
        if demo == "list":
            for n, d, _ in SCENARIOS:
                console.print(f"[bold]{n}[/]: {d}")
            return
        case = next((c for c in build_trace_cases() if c.name == demo), None)
        if case is None:
            raise typer.BadParameter(t("cli.err.unknown_scenario", name=demo))
        bundle = case.bundle
    elif trace:
        bundle = TraceBundle.from_dict(json.loads(trace.read_text()))
    else:
        raise typer.BadParameter(t("cli.err.rca_source"))
    rep = analyze(bundle)
    if as_json:
        console.print_json(json.dumps(rep.to_dict()))
    else:
        console.print(render_rca(rep, bundle))


@app.command()
def shield(file: str = typer.Argument("-"),
           source: str = typer.Option("document"),
           spotlight: str = typer.Option("none"),
           as_json: bool = typer.Option(False, "--json")):
    """Scan untrusted content for indirect prompt injection; exit code 2 when blocked."""
    from omnitrace.interfaces.tui import render_security
    from omnitrace.security.injection_shield import InjectionShield

    text = sys.stdin.read() if file == "-" else Path(file).read_text(encoding="utf-8")
    sh = InjectionShield()
    v = sh.scan(text, source)
    if as_json:
        out = v.to_dict()
        if spotlight != "none" and not v.blocked:
            out["spotlight"] = sh.spotlight(text, spotlight)
        console.print_json(json.dumps(out))
    else:
        console.print(render_security([(file, v)]))
        for f in v.findings:
            console.print(f"  [red]•[/] {f.detector} ({f.severity:.2f}): {f.evidence}")
        if spotlight != "none" and not v.blocked:
            sp = sh.spotlight(text, spotlight)
            console.print(f"[dim]{t('cli.system_note')}[/] {sp['system_note']}\n{sp['content']}")
    raise typer.Exit(2 if v.blocked else 0)


@app.command("eval")
def eval_(output: Path = typer.Option(Path("evals/latest_report.json")),
          readme: bool = typer.Option(False)):
    """Run the full benchmark suite."""
    from evals.benchmark_suite import main as run_suite
    run_suite(output=output, update_readme=readme)


@app.command()
def dash():
    """Demo dashboard: heatmap citation, causal timeline and security alerts."""
    from omnitrace.interfaces.tui import dashboard
    dashboard(console)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8765, stdio: bool = typer.Option(False),
          db: Optional[Path] = typer.Option(None)):
    """Run the MCP server (streamable HTTP on /mcp, or stdio)."""
    from omnitrace.interfaces.mcp_server import MCPServer, create_app, serve_stdio

    srv = MCPServer(_cfg(db))
    if stdio:
        serve_stdio(srv)
        return
    import uvicorn
    uvicorn.run(create_app(srv), host=host, port=port, log_level="info")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
