"""Ten failing end-to-end executions with ground-truth root-cause spans (benchmark 3).

Every trace is a 12 s storefront session recorded at 5 fps (60 frames, rendered by
:mod:`evals.datasets.ui_sim`) plus ~50 spans across ~14 services, structured logs and a
DOM mutation log.  The failure is injected at one span deep in a call chain and
propagates upward (status codes, ERROR logs, latency, schema drift, null fields) exactly
the way real backends do.

Noise / distractors present in every trace:
* page-load traffic (flags, profile, products, recommendations, cart, CDN images);
* a 404 favicon, periodic analytics beacons, a cache miss;
* an inventory call that fails once with 503 and is retried successfully;
* an unrelated erroring consumer (email-service SMTP failure with ERROR logs);
* a slow but healthy cron job (reports-worker, 8 s) with WARN logs;
* in some traces a 500 *after* the symptom, a concurrent unrelated failure, and a
  benign red "SALE" badge that is visually indistinguishable from an error colour.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace

import numpy as np

from evals.datasets.ui_sim import DOM_GEOMETRY, UIState, render
from omnitrace.correlator.timeline_dag import DomMutation, LogRecord, Span, TraceBundle

FPS = 5.0
DURATION = 12.0
RECORDER_OFFSET = 0.08  # screen recorder clock vs browser clock


@dataclass
class TraceCase:
    name: str
    description: str
    bundle: TraceBundle
    root_span_ids: set[str]
    root_desc: str
    ui_events: list[tuple[float, dict]] = field(default_factory=list)


class _TB:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.spans: list[Span] = []
        self.logs: list[LogRecord] = []
        self.dom: list[DomMutation] = []
        self.ui: list[tuple[float, dict]] = []
        self._n = 0
        self._t = 0

    def _id(self) -> str:
        self._n += 1
        return f"{self._n:04x}{self.rng.randrange(16 ** 8):08x}"

    def _trace(self) -> str:
        self._t += 1
        return f"tr{self._t:03d}"

    def span(self, service, name, start, end, parent=None, kind="server", status=None, error=False,
             trace=None, **attrs) -> Span:
        s = Span(self._id(), trace or (parent.trace_id if parent else self._trace()), service, name,
                 round(start, 4), round(end, 4), parent.span_id if parent else None, kind, status, error, dict(attrs))
        self.spans.append(s)
        return s

    def log(self, t, level, service, msg, span: Span | None = None):
        self.logs.append(LogRecord(round(t, 4), level, service, msg, span.span_id if span else None,
                                   span.trace_id if span else None))

    def dom_event(self, t, op, selector, attrs=None, text="", ui=None):
        self.dom.append(DomMutation(round(t, 4), op, selector, DOM_GEOMETRY.get(selector), attrs or {}, text))
        if ui is not None:
            self.ui.append((t, ui))

    def request(self, t0: float, method: str, route: str, hops: list[tuple[str, str, float]],
                fail: dict | None = None, schema: str | None = None, status_ok: int = 200,
                client_only_status: int | None = None) -> list[Span]:
        """Build ``client -> api-gateway -> hops[0] -> hops[1] ...`` and return that chain.

        ``fail`` injects a failure at ``hops[fail["at"]]``:
        ``status`` / ``error`` / ``latency`` / ``schema`` / ``nulls`` on the failing hop,
        ``ancestor_status`` on every caller above it (default: the same status), and
        ``logs`` = [(level, hop_index, message)] (hop_index -1 = gateway, -2 = client).
        """
        trace = self._trace()
        rng = self.rng
        if client_only_status is not None:
            return [self.span("web-frontend", f"{method} {route}", t0, t0 + 0.03, kind="client",
                              status=client_only_status, error=True, trace=trace, **{"http.route": route})]
        fail = fail or {}
        fi = fail.get("at")
        own = [h[2] * rng.uniform(0.85, 1.15) for h in hops]
        if fi is not None and "latency" in fail:
            own[fi] = fail["latency"]
        total, acc = [0.0] * len(hops), 0.0
        for k in range(len(hops) - 1, -1, -1):
            acc = own[k] + (acc + 0.004 if k < len(hops) - 1 else 0.0)
            total[k] = acc
        gw_d = (total[0] if hops else 0.002) + 0.006
        client = self.span("web-frontend", f"{method} {route}", t0, t0 + gw_d + 0.016, kind="client",
                           status=status_ok, trace=trace, **{"http.route": route, "http.method": method})
        gw = self.span("api-gateway", f"{method} {route}", t0 + 0.004, t0 + 0.004 + gw_d, parent=client,
                       status=status_ok, **{"http.route": route})
        chain, parent, start = [client, gw], gw, gw.start + 0.003
        for k, (svc, op, _) in enumerate(hops):
            st = None if svc in _DB_LIKE else status_ok
            s = self.span(svc, op, start, start + total[k], parent=parent, status=st)
            chain.append(s)
            parent, start = s, start + 0.002
        if schema:
            for s in chain:
                if s.service not in _DB_LIKE:
                    s.attributes["http.response.schema"] = schema
        if fi is None:
            return chain
        node = chain[2 + fi]
        above = chain[: 2 + fi]
        if "status" in fail:
            node.status_code = fail["status"]
        node.error = bool(fail.get("error")) or (node.status_code is not None and node.status_code >= 500)
        anc = fail.get("ancestor_status", fail.get("status"))
        if anc is not None:
            for s in above:
                s.status_code = anc
                s.error = anc >= 500 or anc == 0
        for key, attr in (("schema", "http.response.schema"), ("nulls", "http.response.null_fields")):
            if key in fail:
                for s in above + [node]:
                    s.attributes[attr] = fail[key]
        for lvl, idx, msg in fail.get("logs", []):
            tgt = chain[2 + idx]
            self.log(tgt.end - 0.001, lvl, tgt.service, msg, tgt)
        return chain


_DB_LIKE = {"postgres", "redis", "elasticsearch", "smtp"}
PRODUCTS_SCHEMA = "items,page,total"
FLAGS_SCHEMA = "checkout_v2,dark_mode,new_checkout"


def _baseline(tb: _TB, ui0: dict, sale_badge: bool = False, post_error_at: float | None = None) -> None:
    """Shared, healthy-but-noisy page-load + background traffic (~44 spans)."""
    r = tb.rng
    tb.request(0.10, "GET", "/api/flags", [("flags-service", "GetFlags", 0.01), ("redis", "GET flags", 0.002)],
               schema=FLAGS_SCHEMA)
    tb.request(0.15, "GET", "/api/profile", [("auth-service", "GetProfile", 0.02), ("postgres", "SELECT users", 0.006)])
    tb.span("web-frontend", "GET /static/favicon.ico", 0.2, 0.23, kind="client", status=404, **{"http.route": "/static/favicon.ico"})
    prod = tb.request(0.30, "GET", "/api/products", [("catalog-service", "ListProducts", 0.03),
                                                     ("postgres", "SELECT products", 0.012)], schema=PRODUCTS_SCHEMA)
    tb.dom_event(prod[0].end + 0.07, "insert", "#product-grid", {"class": "grid", "children": 6}, "6 products",
                 ui={"cards": 6})
    for k in range(3):
        t = 0.5 + 0.1 * k
        tb.span("web-frontend", f"GET /static/img/p{k}.webp", t, t + r.uniform(0.05, 0.4), kind="client", status=200)
    recs = tb.request(0.9, "GET", "/api/recommendations",
                      [("recommendation-service", "Recommend", 0.05), ("redis", "GET recs:user", 0.002)])
    tb.log(recs[3].end, "INFO", "recommendation-service", "cache miss recs:user, computing", recs[3])
    for k, t in enumerate([1.0, 3.0, 5.0, 7.0, 9.0]):
        st = 500 if (post_error_at is not None and abs(t - post_error_at) < 1e-6) else 204
        tb.request(t, "POST", "/analytics/events", [("analytics-collector", "Ingest", 0.004)],
                       fail={"at": 0, "status": 500, "logs": [("ERROR", 0, "kafka producer queue full")]} if st == 500 else None,
                       status_ok=204)
    tb.request(1.5, "GET", "/api/search/suggest", [("search-service", "Suggest", 0.01), ("elasticsearch", "query suggest", 0.03)])
    tb.request(2.2, "GET", "/api/search/suggest", [("search-service", "Suggest", 0.01), ("elasticsearch", "query suggest", 0.03)])
    tb.request(2.6, "GET", "/api/search/suggest", [("search-service", "Suggest", 0.01), ("elasticsearch", "query suggest", 0.03)])
    cart = tb.request(2.0, "GET", "/api/cart", [("cart-service", "GetCart", 0.015), ("redis", "GET cart", 0.002)])
    tb.dom_event(cart[0].end + 0.05, "text", "#cart-badge", {"class": "badge"}, "cart 1", ui={"cart_count": 1})
    # inventory call fails once (503) and is retried successfully -> recovered distractor
    inv = tb.request(2.8, "GET", "/api/inventory", [("cart-service", "CheckStock", 0.01)])
    first = tb.span("inventory-service", "GetStock", inv[2].start + 0.001, inv[2].start + 0.02, parent=inv[2], status=503, error=True)
    tb.log(first.end, "ERROR", "inventory-service", "upstream timeout talking to warehouse-db", first)
    tb.span("inventory-service", "GetStock", first.end + 0.05, first.end + 0.07, parent=inv[2], status=200)
    if sale_badge:
        promo = tb.request(2.4, "GET", "/api/promotions", [("promo-service", "ActivePromos", 0.02)])
        tb.dom_event(promo[0].end + 0.06, "insert", "#promo-badge", {"class": "badge badge-sale"}, "SALE",
                     ui={"promo": True})
    # unrelated erroring consumer + slow healthy cron
    em = tb.span("email-service", "consume order-confirmation", 3.4, 3.9, kind="consumer", error=True)
    tb.span("smtp", "SEND", 3.45, 3.88, parent=em, error=True)
    tb.log(3.88, "ERROR", "email-service", "SMTP connection refused: smtp.internal:587", em)
    cron = tb.span("reports-worker", "build-daily-report", 0.5, 8.5, kind="internal")
    for k in range(3):
        tb.span("postgres", "SELECT orders aggregate", 1.0 + 2.2 * k, 2.9 + 2.2 * k, parent=cron)
    tb.log(6.0, "WARN", "reports-worker", "report build slower than SLO (8s)", cron)
    tb.ui.append((0.0, ui0))


# ------------------------------------------------------------------------------ scenarios
def _s1(tb: _TB):
    _baseline(tb, {"page": "checkout"}, post_error_at=7.0)
    ch = tb.request(4.0, "POST", "/api/checkout", [("payment-service", "Charge", 0.04), ("postgres", "INSERT payments", 1.8)],
                    fail={"at": 1, "error": True, "latency": 2.0, "ancestor_status": 500,
                          "logs": [("ERROR", 1, "canceling statement due to statement timeout"),
                                   ("ERROR", 0, "charge failed: database error")]})
    tb.request(5.9, "GET", "/api/recommendations", [("recommendation-service", "Recommend", 0.05)],
               fail={"at": 0, "status": 500, "logs": [("ERROR", 0, "model server returned 500")]})
    tb.dom_event(ch[0].end + 0.09, "insert", "#error-banner", {"class": "alert alert-danger", "role": "alert"},
                 "Payment failed. Please retry.", ui={"banner": "Payment failed. Please retry."})
    return ch[3], "postgres INSERT payments statement timeout"


def _s2(tb: _TB):
    _baseline(tb, {"page": "products"})
    pr = tb.request(4.2, "GET", "/api/products", [("catalog-service", "ListProducts", 0.03), ("postgres", "SELECT products", 0.012)],
                    schema=PRODUCTS_SCHEMA, fail={"at": 0, "schema": "page,total"})
    tb.dom_event(pr[0].end + 0.06, "text", "#product-grid", {"class": "grid", "children": 0, "container": True}, "",
                 ui={"content_blank": True})
    return pr[2], "catalog-service ListProducts response dropped 'items' (schema drift, HTTP 200)"


def _s3(tb: _TB):
    _baseline(tb, {"page": "products"})
    se = tb.request(4.0, "GET", "/api/search", [("search-service", "Search", 0.02), ("elasticsearch", "query products", 30.0)],
                    fail={"at": 1, "latency": 30.0})
    tb.dom_event(4.03, "insert", "#results-spinner", {"class": "spinner", "aria-busy": "true"}, "",
                 ui={"page": "search", "spinner_phase": 0})
    return se[3], "elasticsearch query stalled for 30 s"


def _s4(tb: _TB):
    _baseline(tb, {"page": "products"}, sale_badge=True)
    od = tb.request(4.5, "GET", "/api/orders", [("auth-service", "ValidateToken", 0.01), ("redis", "GET session", 0.002)],
                    fail={"at": 0, "status": 401, "logs": [("WARN", 0, "token expired for session")]})
    tb.dom_event(od[0].end + 0.05, "insert", "#modal", {"role": "dialog"}, "Session expired - sign in again",
                 ui={"modal": "Session expired"})
    return od[2], "auth-service ValidateToken rejected an expired token (401)"


def _s5(tb: _TB):
    _baseline(tb, {"page": "products"}, post_error_at=9.0)
    ca = tb.request(4.1, "POST", "/api/cart/items", [("cart-service", "AddItem", 0.02), ("inventory-service", "Reserve", 0.03)],
                    fail={"at": 1, "status": 503, "ancestor_status": 502,
                          "logs": [("ERROR", 1, "connection pool exhausted (max=20)"),
                                   ("ERROR", 0, "inventory unavailable, giving up")]})
    tb.dom_event(ca[0].end + 0.08, "insert", "#toast", {"class": "toast toast-warning"},
                 "Could not add item - inventory unavailable", ui={"toast": "Could not add item"})
    return ca[3], "inventory-service Reserve 503 (pool exhausted), gateway returned 502"


def _s6(tb: _TB):
    _baseline(tb, {"page": "products"})
    rv = tb.request(4.3, "GET", "/api/reviews", [], client_only_status=0)[0]
    tb.log(rv.end, "ERROR", "web-frontend", "CORS policy: No 'Access-Control-Allow-Origin' header on reviews.cdn", rv)
    tb.dom_event(rv.end + 0.05, "text", "#product-grid", {"class": "grid", "children": 0, "container": True}, "",
                 ui={"page": "search", "content_blank": True})
    return rv, "browser blocked /api/reviews (CORS, status 0)"


def _s7(tb: _TB):
    _baseline(tb, {"page": "products"}, sale_badge=True)
    se = tb.request(4.4, "GET", "/api/search", [])     # gateway short-circuits: no upstream hop
    gw = se[1]
    for s_ in se:
        s_.status_code = 429
    tb.log(gw.end, "WARN", "api-gateway", "rate limit exceeded for tenant acme (100 rps)", gw)
    tb.dom_event(se[0].end + 0.05, "insert", "#error-banner", {"class": "alert", "role": "alert"},
                 "Too many requests - slow down", ui={"banner": "Too many requests - slow down"})
    return gw, "api-gateway rate limiter (429)"


def _s8(tb: _TB):
    _baseline(tb, {"page": "cart", "total_text": "Total: $39.98"})
    ca = tb.request(4.0, "GET", "/api/cart", [("cart-service", "GetCart", 0.015), ("pricing-service", "PriceLines", 0.02)],
                    fail={"at": 1, "nulls": "unit_price"})
    for s_ in ca[:3]:
        s_.attributes["http.response.null_fields"] = "total"
    tb.dom_event(ca[0].end + 0.05, "text", "#price-total", {"class": "total"}, "Total: $NaN",
                 ui={"total_text": "Total: $NaN"})
    return ca[3], "pricing-service returned null unit_price"


def _s9(tb: _TB):
    _baseline(tb, {"page": "products"})
    fl = tb.request(4.6, "GET", "/api/flags", [("flags-service", "GetFlags", 0.01), ("redis", "GET flags", 0.002)],
                    schema=FLAGS_SCHEMA, fail={"at": 0, "schema": "checkout_v2,dark_mode"})
    tb.dom_event(fl[0].end + 0.07, "insert", "#modal", {"role": "dialog", "class": "modal onboarding"},
                 "Try the new checkout!", ui={"modal": "Try the new checkout!"})
    return fl[2], "flags-service served a config without 'new_checkout' (drift)"


def _s10(tb: _TB):
    _baseline(tb, {"page": "checkout"})
    od = tb.request(4.0, "POST", "/api/orders", [("order-service", "CreateOrder", 0.03)],
                    fail={"at": 0, "status": 500, "logs": [("ERROR", 0, "retries exhausted creating order")]})
    svc = od[2]
    roots = []
    t = svc.start + 0.005
    for k in range(3):
        db = tb.span("postgres", "INSERT orders", t, t + 0.12, parent=svc, error=True)
        tb.log(db.end, "ERROR", "postgres", "deadlock detected while inserting into orders", db)
        roots.append(db)
        t = db.end + 0.05
    svc.end = max(svc.end, t + 0.01)
    od[1].end = svc.end + 0.004
    od[0].end = od[1].end + 0.01
    tb.dom_event(od[0].end + 0.08, "insert", "#error-banner", {"class": "alert alert-danger", "role": "alert"},
                 "Order could not be placed", ui={"banner": "Order could not be placed"})
    return roots, "postgres deadlock on INSERT orders (retries exhausted)"


SCENARIOS = [
    ("payment_db_timeout", "checkout -> red error banner; root = DB statement timeout 3 hops down", _s1),
    ("schema_drift_blank_grid", "HTTP 200 with drifted payload -> blank product grid", _s2),
    ("slow_search_frozen_spinner", "30 s Elasticsearch query -> spinner never stops", _s3),
    ("auth_expired_modal", "401 from token validation -> session-expired modal (plus a benign red SALE badge)", _s4),
    ("inventory_503_behind_502", "inventory 503 behind a gateway 502 -> warning toast", _s5),
    ("cors_network_failure", "browser-blocked request (status 0) -> blank widget", _s6),
    ("gateway_rate_limit", "gateway 429 -> red banner (plus SALE badge)", _s7),
    ("null_price_nan_total", "null unit_price -> 'Total: $NaN' (DOM-only symptom)", _s8),
    ("feature_flag_drift_modal", "flags payload drift -> unexpected onboarding modal", _s9),
    ("deadlock_retries_exhausted", "3 deadlocked INSERT attempts -> red banner", _s10),
]


def _frames(tb: _TB, seed: int) -> tuple[list[np.ndarray], list[float]]:
    rng = np.random.default_rng(seed)
    events = sorted(tb.ui, key=lambda e: e[0])
    n = int(DURATION * FPS)
    frames, times = [], []
    state = UIState(cards=0, items_text="")
    spin_start = None
    for i in range(n):
        t_browser = i / FPS
        for t, ch in events:
            if t <= t_browser and not ch.get("_applied"):
                state = replace(state, **ch)
                if ch.get("spinner_phase") is not None:
                    spin_start = i
                ch["_applied"] = True
        st = state
        if state.spinner_phase is not None and spin_start is not None:
            st = replace(state, spinner_phase=(i - spin_start) % 8)
        frames.append(render(st, noise_sigma=1.5, rng=rng))
        times.append(round(t_browser + RECORDER_OFFSET, 4))
    for _, ch in events:
        ch.pop("_applied", None)
    return frames, times


def build_trace_cases(seed: int = 11) -> list[TraceCase]:
    cases = []
    for k, (name, desc, fn) in enumerate(SCENARIOS):
        tb = _TB(seed * 100 + k)
        root, root_desc = fn(tb)
        roots = root if isinstance(root, list) else [root]
        tb.spans.sort(key=lambda s: s.start)
        frames, times = _frames(tb, seed * 100 + k)
        bundle = TraceBundle(tb.spans, tb.logs, sorted(tb.dom, key=lambda m: m.t), frames, times,
                             {"scenario": name})
        cases.append(TraceCase(name, desc, bundle, {r.span_id for r in roots}, root_desc, tb.ui))
    return cases
