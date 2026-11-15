"""Message catalog for human-facing CLI / TUI output (English and Italian).

Only text meant for a person goes through :func:`t`: console messages, Rich table titles,
headers and labels, status lines, user-facing errors and the Typer help texts. Machine-readable
output (``--json``, MCP / API responses, report files), identifiers, engine-generated evidence
strings and prompts sent to a VLM are never translated.

The language is chosen, in order of precedence, by :func:`set_lang` (the CLI's ``--lang``
option), the ``OMNITRACE_LANG`` environment variable (``it``, ``it_IT.UTF-8``, ...) and finally
the English default.
"""

from __future__ import annotations

import os

LANGS: tuple[str, ...] = ("en", "it")
DEFAULT_LANG = "en"
ENV_VAR = "OMNITRACE_LANG"

_override: str | None = None


def normalize(code: str | None) -> str | None:
    """``"it"``, ``"IT"``, ``"it_IT.UTF-8"`` or ``"it-IT"`` -> ``"it"``; unknown or empty -> ``None``."""
    if not code:
        return None
    base = code.strip().lower().replace("-", "_").split(".")[0].split("_")[0]
    return base if base in LANGS else None


def set_lang(code: str | None) -> None:
    """Force a language for this process; ``None`` (or an unknown code) falls back to the environment."""
    global _override
    _override = normalize(code)


def get_lang() -> str:
    return _override or normalize(os.environ.get(ENV_VAR)) or DEFAULT_LANG


def t(key: str, **kwargs: object) -> str:
    """The message ``key`` in the current language, formatted with ``kwargs`` (``str.format``)."""
    msg = CATALOG[get_lang()].get(key)
    if msg is None:
        msg = CATALOG[DEFAULT_LANG][key]
    return msg.format(**kwargs) if kwargs else msg


def has_key(key: str) -> bool:
    return key in CATALOG[DEFAULT_LANG]


CATALOG: dict[str, dict[str, str]] = {
    "en": {
        # -------------------------------------------------------------- Typer help texts
        "help.app": "omnitrace: visual late-interaction RAG, cross-modal RCA and AI security engine.",
        "help.lang": "Language of the human-readable output (default: $OMNITRACE_LANG or en). JSON output is never translated.",
        "help.cmd.index": "Encode pages into a multi-vector visual index.",
        "help.opt.index.path": "directory of page images (+ .json structure layers)",
        "help.opt.index.db": "SQLite index path (default: $OMNITRACE_DB or ./omnitrace_index.sqlite)",
        "help.opt.index.demo": "index the synthetic 15-page benchmark corpus",
        "help.cmd.query": "Retrieve pages and ground the answer to a bounding box.",
        "help.opt.query.text": "natural-language query",
        "help.opt.query.db": "SQLite index path (default: $OMNITRACE_DB or ./omnitrace_index.sqlite)",
        "help.opt.query.k": "number of pages to return",
        "help.opt.query.precision": "float32 | int8 | int2 | binary",
        "help.opt.query.crop": "write the top hit's bbox crop to this PNG",
        "help.opt.query.vlm": "read the crop with a VLM: ollama:<model> | openai:<model>@<url>",
        "help.opt.query.as_json": "print the hits as JSON",
        "help.cmd.rca": "Cross-modal root-cause analysis of one failing execution.",
        "help.opt.rca.trace": "TraceBundle JSON",
        "help.opt.rca.demo": "run a synthetic scenario by name (or 'list')",
        "help.opt.rca.as_json": "print the report as JSON",
        "help.cmd.shield": "Scan untrusted content for indirect prompt injection; exit code 2 when blocked.",
        "help.opt.shield.file": "file to scan, or - for stdin",
        "help.opt.shield.source": "where the content comes from (document, dom_text, tool_output, ...)",
        "help.opt.shield.spotlight": "none | datamark | delimit | base64",
        "help.opt.shield.as_json": "print the verdict as JSON",
        "help.cmd.eval": "Run the full benchmark suite.",
        "help.opt.eval.output": "where to write the JSON report",
        "help.opt.eval.readme": "also refresh the metrics block in README.md",
        "help.cmd.dash": "Demo dashboard: heatmap citation, causal timeline and security alerts.",
        "help.cmd.serve": "Run the MCP server (streamable HTTP on /mcp, or stdio).",
        "help.opt.serve.host": "address to bind the HTTP transport to",
        "help.opt.serve.port": "port of the HTTP transport",
        "help.opt.serve.stdio": "use the stdio transport",
        "help.opt.serve.db": "SQLite index path (default: $OMNITRACE_DB or ./omnitrace_index.sqlite)",
        # -------------------------------------------------------------- CLI messages
        "cli.err.index_source": "give a directory or --demo",
        "cli.err.no_images": "no images found in {path}",
        "cli.status.encoding": "encoding {n} pages",
        "cli.indexed": "[green]indexed[/] {n} pages × {patches} patches → {out} "
                       "(bytes/patch: f32 {f32}, int8 {int8}, 2-bit {int2}, 1-bit {binary})",
        "cli.crop_written": "[green]crop written[/] {path}",
        "cli.err.unknown_scenario": "unknown scenario {name!r}",
        "cli.err.rca_source": "give a trace file or --demo",
        "cli.system_note": "system note:",
        # -------------------------------------------------------------- TUI: visual citation
        "tui.hit.page": "page",
        "tui.hit.score": "score",
        "tui.hit.bbox": "bbox",
        "tui.hit.grounding": "grounding",
        "tui.hit.region": "region",
        "tui.hit.latency": "latency",
        "tui.hit.token_peaks": "token peaks: {peaks}",
        "tui.hit.title": "#{rank} visual citation",
        # -------------------------------------------------------------- TUI: RCA
        "tui.node.span": "span",
        "tui.node.dom": "dom",
        "tui.node.symptom": "symptom",
        "tui.rca.anomalies": "visible anomalies",
        "tui.rca.col.t": "t",
        "tui.rca.col.kind": "kind",
        "tui.rca.col.source": "source",
        "tui.rca.col.severity": "severity",
        "tui.rca.col.detail": "detail",
        "tui.rca.ranking": "root-cause ranking",
        "tui.rca.col.rank": "#",
        "tui.rca.col.score": "score",
        "tui.rca.col.service": "service",
        "tui.rca.col.operation": "operation",
        "tui.rca.col.evidence": "evidence",
        "tui.rca.col.origin": "origin",
        "tui.rca.path": "causal path (root → symptom)",
        "tui.rca.footer": "frames {frames} → keyframes {keyframes} · DAG {nodes} nodes / {edges} edges · {ms:.1f} ms",
        "tui.timeline.title": "timeline (0 → {t_end:.1f}s)",
        "tui.timeline.frames": "frames",
        "tui.timeline.dom": "dom",
        "tui.timeline.spans": "spans",
        "tui.timeline.symptom": "symptom",
        # -------------------------------------------------------------- TUI: security
        "tui.shield.title": "injection shield",
        "tui.shield.col.item": "item",
        "tui.shield.col.verdict": "verdict",
        "tui.shield.col.risk": "risk",
        "tui.shield.col.findings": "top findings",
        "tui.shield.col.ms": "ms",
        "tui.shield.block": "BLOCK",
        "tui.shield.warn": "warn",
        "tui.shield.pass": "pass",
        "tui.loops.title": "loop breaker",
        "tui.loops.col.trajectory": "trajectory",
        "tui.loops.col.halt": "halt",
        "tui.loops.col.step": "step",
        "tui.loops.col.entropy": "entropy",
        "tui.loops.col.novelty": "novelty",
        "tui.loops.col.reason": "reason",
        "tui.loops.halt": "HALT",
        "tui.loops.ok": "ok",
        # -------------------------------------------------------------- TUI: dashboard
        "tui.dash.visual": "Subsystem A · visual query: {query!r}",
        "tui.dash.rca": "Subsystem B · RCA: {name}",
        "tui.dash.security": "Subsystem C · shield & loop breaker",
        # -------------------------------------------------------------- benchmark suite progress
        "eval.step.grounding": "[1/4] 2-D bounding-box grounding",
        "eval.res.grounding": "      dev int8 region R@1 = {dev:.3f}  flat-BM25 = {bm25:.3f}  held-out = {held:.3f}",
        "eval.step.dedup": "[2/4] keyframe deduplication",
        "eval.res.dedup": "      pruned {pruned:.1%} at {ms:.3f} ms/frame, recall {recall:.0%}",
        "eval.step.rca": "[3/4] cross-modal root-cause analysis",
        "eval.res.rca": "      top-1 {top1:.0%} (baselines: {baselines})",
        "eval.step.security": "[4/4] injection shield + loop breaker",
        "eval.res.security": "      dev recall {recall:.0%} FPR {fpr:.0%}; held-out recall {held_recall:.0%} "
                             "FPR {held_fpr:.0%}; p95 {p95:.2f} ms",
        "eval.done": "report written to {path} ({seconds:.0f} s)",
    },
    "it": {
        # -------------------------------------------------------------- Typer help texts
        "help.app": "omnitrace: RAG visuale late-interaction, RCA cross-modale e motore di sicurezza per agenti AI.",
        "help.lang": "Lingua dell'output leggibile (default: $OMNITRACE_LANG o en). L'output JSON non viene mai tradotto.",
        "help.cmd.index": "Codifica le pagine in un indice visuale multi-vettore.",
        "help.opt.index.path": "cartella di immagini di pagina (+ structure layer .json)",
        "help.opt.index.db": "percorso dell'indice SQLite (default: $OMNITRACE_DB o ./omnitrace_index.sqlite)",
        "help.opt.index.demo": "indicizza il corpus sintetico di benchmark da 15 pagine",
        "help.cmd.query": "Recupera le pagine e ancora la risposta a un bounding box.",
        "help.opt.query.text": "query in linguaggio naturale",
        "help.opt.query.db": "percorso dell'indice SQLite (default: $OMNITRACE_DB o ./omnitrace_index.sqlite)",
        "help.opt.query.k": "numero di pagine da restituire",
        "help.opt.query.precision": "float32 | int8 | int2 | binary",
        "help.opt.query.crop": "salva in questo PNG il ritaglio del bbox del primo risultato",
        "help.opt.query.vlm": "fai leggere il ritaglio a un VLM: ollama:<modello> | openai:<modello>@<url>",
        "help.opt.query.as_json": "stampa i risultati in JSON",
        "help.cmd.rca": "Analisi cross-modale della causa radice di un'esecuzione fallita.",
        "help.opt.rca.trace": "JSON TraceBundle",
        "help.opt.rca.demo": "esegui uno scenario sintetico per nome (oppure 'list')",
        "help.opt.rca.as_json": "stampa il report in JSON",
        "help.cmd.shield": "Analizza contenuto non fidato alla ricerca di prompt injection indirette; exit code 2 se bloccato.",
        "help.opt.shield.file": "file da analizzare, oppure - per lo stdin",
        "help.opt.shield.source": "provenienza del contenuto (document, dom_text, tool_output, ...)",
        "help.opt.shield.spotlight": "none | datamark | delimit | base64",
        "help.opt.shield.as_json": "stampa il verdetto in JSON",
        "help.cmd.eval": "Esegue l'intera suite di benchmark.",
        "help.opt.eval.output": "dove scrivere il report JSON",
        "help.opt.eval.readme": "aggiorna anche il blocco metriche in README.md",
        "help.cmd.dash": "Dashboard demo: citazione con heatmap, timeline causale e allarmi di sicurezza.",
        "help.cmd.serve": "Avvia il server MCP (HTTP streamable su /mcp, oppure stdio).",
        "help.opt.serve.host": "indirizzo su cui esporre il transport HTTP",
        "help.opt.serve.port": "porta del transport HTTP",
        "help.opt.serve.stdio": "usa il transport stdio",
        "help.opt.serve.db": "percorso dell'indice SQLite (default: $OMNITRACE_DB o ./omnitrace_index.sqlite)",
        # -------------------------------------------------------------- CLI messages
        "cli.err.index_source": "indica una cartella oppure --demo",
        "cli.err.no_images": "nessuna immagine trovata in {path}",
        "cli.status.encoding": "codifica di {n} pagine in corso",
        "cli.indexed": "[green]indicizzate[/] {n} pagine × {patches} patch → {out} "
                       "(byte/patch: f32 {f32}, int8 {int8}, 2-bit {int2}, 1-bit {binary})",
        "cli.crop_written": "[green]ritaglio salvato[/] {path}",
        "cli.err.unknown_scenario": "scenario sconosciuto {name!r}",
        "cli.err.rca_source": "indica un file di trace oppure --demo",
        "cli.system_note": "nota di sistema:",
        # -------------------------------------------------------------- TUI: visual citation
        "tui.hit.page": "pagina",
        "tui.hit.score": "score",
        "tui.hit.bbox": "bbox",
        "tui.hit.grounding": "grounding",
        "tui.hit.region": "regione",
        "tui.hit.latency": "latenza",
        "tui.hit.token_peaks": "picchi per token: {peaks}",
        "tui.hit.title": "#{rank} citazione visuale",
        # -------------------------------------------------------------- TUI: RCA
        "tui.node.span": "span",
        "tui.node.dom": "dom",
        "tui.node.symptom": "sintomo",
        "tui.rca.anomalies": "anomalie visibili",
        "tui.rca.col.t": "t",
        "tui.rca.col.kind": "tipo",
        "tui.rca.col.source": "sorgente",
        "tui.rca.col.severity": "severità",
        "tui.rca.col.detail": "dettaglio",
        "tui.rca.ranking": "classifica delle cause radice",
        "tui.rca.col.rank": "#",
        "tui.rca.col.score": "score",
        "tui.rca.col.service": "servizio",
        "tui.rca.col.operation": "operazione",
        "tui.rca.col.evidence": "evidenze",
        "tui.rca.col.origin": "origine",
        "tui.rca.path": "percorso causale (causa radice → sintomo)",
        "tui.rca.footer": "frame {frames} → keyframe {keyframes} · DAG {nodes} nodi / {edges} archi · {ms:.1f} ms",
        "tui.timeline.title": "timeline (0 → {t_end:.1f}s)",
        "tui.timeline.frames": "frame",
        "tui.timeline.dom": "dom",
        "tui.timeline.spans": "span",
        "tui.timeline.symptom": "sintomo",
        # -------------------------------------------------------------- TUI: security
        "tui.shield.title": "injection shield",
        "tui.shield.col.item": "elemento",
        "tui.shield.col.verdict": "verdetto",
        "tui.shield.col.risk": "rischio",
        "tui.shield.col.findings": "rilevamenti principali",
        "tui.shield.col.ms": "ms",
        "tui.shield.block": "BLOCCATO",
        "tui.shield.warn": "avviso",
        "tui.shield.pass": "ok",
        "tui.loops.title": "loop breaker",
        "tui.loops.col.trajectory": "traiettoria",
        "tui.loops.col.halt": "arresto",
        "tui.loops.col.step": "step",
        "tui.loops.col.entropy": "entropia",
        "tui.loops.col.novelty": "novità",
        "tui.loops.col.reason": "motivo",
        "tui.loops.halt": "STOP",
        "tui.loops.ok": "ok",
        # -------------------------------------------------------------- TUI: dashboard
        "tui.dash.visual": "Sottosistema A · query visuale: {query!r}",
        "tui.dash.rca": "Sottosistema B · RCA: {name}",
        "tui.dash.security": "Sottosistema C · shield e loop breaker",
        # -------------------------------------------------------------- benchmark suite progress
        "eval.step.grounding": "[1/4] grounding 2-D con bounding box",
        "eval.res.grounding": "      dev int8 R@1 regione = {dev:.3f}  flat-BM25 = {bm25:.3f}  held-out = {held:.3f}",
        "eval.step.dedup": "[2/4] deduplicazione dei keyframe",
        "eval.res.dedup": "      eliminati {pruned:.1%} a {ms:.3f} ms/frame, recall {recall:.0%}",
        "eval.step.rca": "[3/4] analisi cross-modale della causa radice",
        "eval.res.rca": "      top-1 {top1:.0%} (baseline: {baselines})",
        "eval.step.security": "[4/4] injection shield + loop breaker",
        "eval.res.security": "      dev recall {recall:.0%} FPR {fpr:.0%}; held-out recall {held_recall:.0%} "
                             "FPR {held_fpr:.0%}; p95 {p95:.2f} ms",
        "eval.done": "report scritto in {path} ({seconds:.0f} s)",
    },
}
