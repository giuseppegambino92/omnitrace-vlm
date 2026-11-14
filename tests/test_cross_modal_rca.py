import json
from dataclasses import replace

from omnitrace.config import CorrelatorConfig
from omnitrace.correlator.rca_engine import analyze, origin_factors, span_evidence
from omnitrace.correlator.timeline_dag import (
    Anomaly, DomMutation, LogRecord, Span, TraceBundle, build_causal_dag, detect_dom_anomalies, lexical_affinity,
)


def _chain():
    """frontend -> gateway -> svc -> db(fails); banner rendered after the client span."""
    c = Span("c", "t1", "web-frontend", "POST /api/checkout", 1.0, 1.5, None, "client", 500, True, {"http.route": "/api/checkout"})
    g = Span("g", "t1", "api-gateway", "POST /api/checkout", 1.01, 1.49, "c", "server", 500, True)
    s = Span("s", "t1", "payment-service", "Charge", 1.02, 1.48, "g", "server", 500, True)
    d = Span("d", "t1", "postgres", "INSERT payments", 1.03, 1.47, "s", "internal", None, True)
    logs = [LogRecord(1.47, "ERROR", "postgres", "deadlock detected", "d"),
            LogRecord(1.48, "ERROR", "payment-service", "charge failed", "s")]
    dom = [DomMutation(1.6, "insert", "#error-banner", (0, 0, 100, 20), {"class": "alert alert-danger"}, "Payment failed")]
    return TraceBundle([c, g, s, d], logs, dom)


def test_evidence_signals():
    b = _chain()
    ev = span_evidence(b)
    assert ev["d"].score > 0.7 and "error" in ev["d"].classes
    peers = [Span(f"p{i}", "t", "es", "query", i, i + 0.05) for i in range(5)]
    slow = Span("slow", "t", "es", "query", 10, 40)
    ev2 = span_evidence(TraceBundle(peers + [slow]))
    assert "latency" in ev2["slow"].classes and ev2["p0"].score < 0.1
    a = Span("a", "t", "catalog", "List", 0, 1, attributes={"http.response.schema": "items,page"})
    a2 = Span("a2", "t", "catalog", "List", 2, 3, attributes={"http.response.schema": "items,page"})
    bad = Span("bad", "t", "catalog", "List", 4, 5, attributes={"http.response.schema": "page"})
    ev3 = span_evidence(TraceBundle([a, a2, bad]))
    assert "drift" in ev3["bad"].classes and any("items" in r for r in ev3["bad"].reasons)


def test_origin_factor_blames_deepest_failure_and_discounts_retries():
    b = _chain()
    cfg = CorrelatorConfig()
    o = origin_factors(b, span_evidence(b), cfg)
    assert o["d"][0] == 1.0 and o["s"][0] == cfg.origin_penalty and o["c"][0] == cfg.origin_penalty
    p = Span("p", "t", "cart", "Check", 0, 1)
    f1 = Span("f1", "t", "inventory", "Get", 0.1, 0.2, "p", status_code=503, error=True)
    ok = Span("ok", "t", "inventory", "Get", 0.3, 0.4, "p", status_code=200)
    b2 = TraceBundle([p, f1, ok])
    assert origin_factors(b2, span_evidence(b2), cfg)["f1"][0] < 0.2


def test_dag_edges_respect_time_and_geometry():
    b = _chain()
    anomalies = [Anomaly("error_banner", 1.7, 1.7, (0, 0, 100, 20), 1.0, "visual")]
    dag = build_causal_dag(b, anomalies)
    assert any(e.src == "span:c" and e.kind == "rendered-as" for e in dag.incoming["dom:0"])
    assert any(e.src == "dom:0" for e in dag.incoming["anom:0"])
    # a DOM change *before* the response cannot be its rendering
    early = replace(b, dom=[DomMutation(0.5, "insert", "#error-banner", (0, 0, 100, 20), {"class": "alert"}, "x")])
    dag2 = build_causal_dag(early, [])
    assert not any(e.src == "span:c" for e in dag2.incoming.get("dom:0", []))
    # a symptom elsewhere on screen gets no DOM edge
    far = [Anomaly("error_banner", 1.7, 1.7, (300, 300, 400, 330), 1.0, "visual")]
    dag3 = build_causal_dag(b, far)
    assert not any(e.src.startswith("dom:") for e in dag3.incoming.get("anom:0", []))


def test_analyze_ranks_db_span_first():
    rep = analyze(_chain())
    assert rep.root.span.span_id == "d"
    assert rep.root.path[0] == "span:d" and rep.root.path[-1].startswith("anom:")


def test_affinity_and_dom_anomalies():
    s = Span("x", "t", "web-frontend", "GET /api/products", 0, 1, kind="client", attributes={"http.route": "/api/products"})
    assert lexical_affinity(s, DomMutation(1, "insert", "#grid", attrs={"data-endpoint": "/api/products"})) == 1.0
    assert lexical_affinity(s, DomMutation(1, "insert", "#product-grid")) > lexical_affinity(s, DomMutation(1, "insert", "#footer"))
    dom = [DomMutation(1, "text", "#total", text="Total: $NaN"),
           DomMutation(2, "insert", "#spin", attrs={"class": "spinner", "aria-busy": "true"}),
           DomMutation(3, "insert", "#b", attrs={"role": "alert"})]
    kinds = {a.kind for a in detect_dom_anomalies(dom, horizon=9.0)}
    assert kinds == {"dom_bad_text", "dom_stall", "dom_error"}


def test_trace_bundle_json_roundtrip_with_frames(trace_cases):
    b = trace_cases[0].bundle
    small = replace(b, frames=b.frames[:3], frame_times=b.frame_times[:3])
    d = json.loads(json.dumps(small.to_dict(include_frames=True)))
    b2 = TraceBundle.from_dict(d)
    assert len(b2.spans) == len(b.spans) and len(b2.frames) == 3
    assert (b2.frames[1] == small.frames[1]).all()


def test_benchmark_scenarios_top1(trace_cases):
    hits = [analyze(c.bundle).root.span.span_id in c.root_span_ids for c in trace_cases]
    assert sum(hits) / len(hits) >= 0.9


def test_visual_anomalies_detected_from_pixels(trace_cases):
    by = {c.name: c for c in trace_cases}
    kinds = lambda n: {a.kind for a in analyze(by[n].bundle).anomalies if a.source == "visual"}
    assert "stalled_spinner" in kinds("slow_search_frozen_spinner")
    assert "modal_overlay" in kinds("auth_expired_modal")
    assert "blank_region" in kinds("schema_drift_blank_grid")
    assert "warning_toast" in kinds("inventory_503_behind_502")
