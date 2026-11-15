"""Regenerate the terminal screenshots in docs/images/{en,it}/ from real ``omnitrace`` runs.

Every screenshot is the recorded output of the actual CLI commands, run in-process on the
bundled synthetic data (``evals/datasets``): nothing is typed by hand. Each scenario is run
once in English (the default) and once with ``--lang it``. No network and no VLM are used.

    python scripts/make_screenshots.py            # writes docs/images/en/*.svg and docs/images/it/*.svg

Latency figures in the output are measured on the machine that runs the script, so they
change from run to run.
"""

from __future__ import annotations

import io
import json
import os
import shlex
import tempfile
from pathlib import Path

import typer
from rich.console import Console

from omnitrace import i18n
from omnitrace.interfaces import cli

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images"
DATASETS = ROOT / "evals" / "datasets"
WIDTH = 112


def _query_text() -> str:
    """First table query of the exported benchmark queries (same corpus as ``index --demo``)."""
    queries = json.loads((DATASETS / "layout_queries.json").read_text(encoding="utf-8"))
    return next(q["text"] for q in queries if q["family"] == "table")


def _shield_doc() -> tuple[str, str]:
    """A malicious document of the exported injection corpus: (file name, text)."""
    corpus = json.loads((DATASETS / "injection_corpus.json").read_text(encoding="utf-8"))
    doc = next(d for d in corpus["documents"] if d["id"] == "attack/calendar_invite")
    return "calendar_invite.txt", doc["text"]


def _run(console: Console, args: list[str], lang: str) -> int:
    """Echo the command line, then run it through the real Click/Typer entry point."""
    full = (["--lang", lang] if lang != "en" else []) + args
    console.print(f"[bold green]$[/] omnitrace {shlex.join(full)}", highlight=False)
    cli.console = console
    try:
        rv = typer.main.get_command(cli.app).main(full, prog_name="omnitrace", standalone_mode=False)
    except typer.Exit as e:                      # shield exits with 2 when the content is blocked
        rv = e.exit_code
    finally:
        i18n.set_lang(None)
    console.print()
    return int(rv or 0)


def _scenarios() -> dict[str, list]:
    name, text = _shield_doc()
    return {
        "query": [["index", "--demo", "--db", "pages.sqlite"],
                  ["query", _query_text(), "--db", "pages.sqlite", "--k", "1"]],
        "rca": [["rca", "--demo", "payment_db_timeout"]],
        "shield": [("cat", name, text), ["shield", name, "--source", "tool_output"]],
    }


def main() -> None:
    os.environ.pop(i18n.ENV_VAR, None)
    for lang in i18n.LANGS:
        (OUT / lang).mkdir(parents=True, exist_ok=True)
        for shot, steps in _scenarios().items():
            console = Console(record=True, width=WIDTH, file=io.StringIO(), force_terminal=False,
                              color_system="truecolor", legacy_windows=False)
            with tempfile.TemporaryDirectory() as tmp:
                cwd = Path.cwd()
                os.chdir(tmp)
                try:
                    for step in steps:
                        if isinstance(step, tuple):          # ("cat", file, text): write the input file, show it
                            _, fname, body = step
                            Path(fname).write_text(body, encoding="utf-8")
                            console.print(f"[bold green]$[/] cat {fname}", highlight=False)
                            console.print(body, markup=False, highlight=False)
                            console.print()
                            continue
                        _run(console, step, lang)
                finally:
                    os.chdir(cwd)
            path = OUT / lang / f"{shot}.svg"
            title = "omnitrace " + " · ".join(s[0] for s in steps if isinstance(s, list))
            path.write_text(console.export_svg(title=title), encoding="utf-8")
            print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
