"""Deterministic synthetic corpus of complex multi-region technical pages.

15 pages (1024 x 1024 px), three of each family:

* ``table``      - two spec tables per page (header row, row headers, numeric cells)
* ``schematic``  - process diagrams: labelled nodes, connectors, a red FAULT node, legend
* ``multicol``   - 2-3 column engineering specs: numbered section headings + paragraphs
* ``ui``         - application viewports: nav, forms, a field error in red, toast, metric cards
* ``chart``      - bar charts (SVG-style bars carry their aria-label), legend, summary table

Every page carries its structure layer (what a PDF text layer / DOM / SVG export
provides) and 4 queries with the ground-truth answer region id.  Vocabulary is
deliberately shared across pages (every table has "Max Voltage", every schematic
has pumps and valves...) so retrieval has to rely on the specific entities and
their 2-D relations.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from omnitrace.vision.patch_encoder import Page, Region

W = H = 1024


@dataclass
class LayoutQuery:
    qid: str
    page_id: str
    text: str
    answer_region: str
    family: str


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


class _Builder:
    def __init__(self, page_id: str, doc_id: str, bg=(255, 255, 255)):
        self.page_id, self.doc_id = page_id, doc_id
        self.img = Image.new("RGB", (W, H), bg)
        self.d = ImageDraw.Draw(self.img)
        self.regions: list[Region] = []

    def region(self, rid: str, bbox, text: str = "", role: str = "text", parent=None, links=None,
               fill=None, outline=None, text_fill=(20, 20, 20), size: int = 14, draw_text: str | None = None,
               width: int = 1) -> Region:
        x1, y1, x2, y2 = bbox
        if fill is not None or outline is not None:
            self.d.rectangle(bbox, fill=fill, outline=outline, width=width)
        label = text if draw_text is None else draw_text
        if label and role not in ("container", "edge"):
            self.d.text((x1 + 5, y1 + max(2, (y2 - y1 - size) // 2)), label, font=_font(size), fill=text_fill)
        r = Region(f"{self.page_id}/{rid}", (float(x1), float(y1), float(x2), float(y2)), text, role,
                   f"{self.page_id}/{parent}" if parent else None,
                   [f"{self.page_id}/{l}" for l in (links or [])])
        self.regions.append(r)
        return r

    def page(self, family: str) -> Page:
        return Page(self.page_id, np.asarray(self.img, dtype=np.uint8).copy(), self.regions, self.doc_id,
                    {"family": family})


# --------------------------------------------------------------------------------- tables
_COMPONENTS = [
    ("P", "Feed Pump"), ("P", "Booster Pump"), ("C", "Compressor"), ("V", "Control Valve"),
    ("M", "Drive Motor"), ("HX", "Heat Exchanger"), ("F", "Cooling Fan"), ("T", "Transformer"),
    ("G", "Generator"), ("B", "Blower"), ("K", "Chiller"), ("S", "Separator"),
]
_SPEC_COLS = ["Max Voltage", "Max Current", "Rated Power", "Operating Temp", "Status"]
_MAINT_COLS = ["Interval", "Last Service", "Next Service", "Technician"]
_TECHS = ["A. Rossi", "B. Chen", "C. Okafor", "D. Novak", "E. Silva", "F. Haddad", "G. Larsen"]
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _table(b: _Builder, tid: str, x: int, y: int, col_w: list[int], header: list[str], rows: list[list[str]],
           caption: str, row_h: int = 34) -> dict[tuple[int, int], str]:
    total_w = sum(col_w)
    b.region(f"{tid}-cap", (x, y, x + total_w, y + 30), caption, "caption", tid, size=17)
    top = y + 34
    b.region(tid, (x, top, x + total_w, top + row_h * (len(rows) + 1)), caption, "container", outline=(90, 90, 90))
    ids: dict[tuple[int, int], str] = {}
    cx = x
    for c, (hdr, w) in enumerate(zip(header, col_w)):
        b.region(f"{tid}-h{c}", (cx, top, cx + w, top + row_h), hdr, "header", tid,
                 fill=(225, 230, 240), outline=(120, 120, 120), size=13)
        cx += w
    for r, row in enumerate(rows):
        cx = x
        ry = top + row_h * (r + 1)
        for c, (val, w) in enumerate(zip(row, col_w)):
            role = "rowheader" if c == 0 else "cell"
            fill = (255, 235, 235) if val in ("FAULT", "OFFLINE") else None
            tf = (190, 20, 20) if val in ("FAULT", "OFFLINE") else (20, 20, 20)
            rid = f"{tid}-r{r}c{c}"
            reg = b.region(rid, (cx, ry, cx + w, ry + row_h), val, role, tid, fill=fill, outline=(170, 170, 170),
                           text_fill=tf, size=13)
            ids[(r, c)] = reg.id
            cx += w
    return ids


def _table_page(i: int, rng: random.Random) -> tuple[Page, list[LayoutQuery]]:
    pid = f"table-{i}"
    b = _Builder(pid, f"plant-spec-{i}")
    b.region("title", (40, 24, 984, 64), f"Unit {i + 1} Equipment Datasheet - Rev {chr(65 + i)}", "title", size=22)
    comps = rng.sample(_COMPONENTS, 8)
    names = [f"{p}-{100 * (i + 2) + rng.randint(1, 98)} {n}" for p, n in comps]
    rows = []
    for n in names:
        status = rng.choice(["OK", "OK", "OK", "STANDBY", "FAULT"])
        rows.append([n, f"{rng.choice([230, 400, 480, 690])} V", f"{rng.uniform(2, 60):.1f} A",
                     f"{rng.randint(5, 250)} kW", f"{rng.randint(-20, 10)} to {rng.randint(40, 95)} C", status])
    spec = _table(b, "spec", 40, 90, [250, 130, 130, 130, 160, 120], ["Component"] + _SPEC_COLS, rows,
                  f"Table {i + 1}.1 Electrical ratings")
    m_rows = []
    for n in names[:6]:
        mo = rng.randrange(12)
        m_rows.append([n, f"{rng.choice([3, 6, 12])} months", f"{rng.randint(1, 28)} {_MONTHS[mo]} 2025",
                       f"{rng.randint(1, 28)} {_MONTHS[(mo + 6) % 12]} 2026", rng.choice(_TECHS)])
    maint = _table(b, "maint", 40, 520, [250, 150, 170, 170, 160], ["Component"] + _MAINT_COLS, m_rows,
                   f"Table {i + 1}.2 Maintenance schedule")
    b.region("note", (40, 820, 984, 900),
             "Notes: ratings at 40 C ambient. Derate current 2% per degree above ambient limit.", "text", size=13)
    q = []
    r1, r2, r3, r4 = rng.sample(range(8), 4)
    ident = lambda n: n.split()[0]
    q.append(LayoutQuery(f"{pid}-q0", pid, f"max current of {ident(names[r1])}", spec[(r1, 2)], "table"))
    q.append(LayoutQuery(f"{pid}-q1", pid, f"rated power {names[r2]}", spec[(r2, 3)], "table"))
    m = rng.randrange(6)
    q.append(LayoutQuery(f"{pid}-q2", pid, f"next service date for {ident(names[m])}", maint[(m, 3)], "table"))
    q.append(LayoutQuery(f"{pid}-q3", pid, f"operating temp range of {ident(names[r4])} {names[r4].split(' ', 1)[1]}",
                         spec[(r4, 4)], "table"))
    return b.page("table"), q


# ----------------------------------------------------------------------------- schematics
_STAGES = ["Intake Filter", "Feed Pump", "Pressure Regulator", "Heat Exchanger", "Flow Meter", "Separator",
           "Control Valve", "Storage Tank", "Temperature Sensor", "Bypass Valve", "Dosing Pump", "Chiller"]


def _schematic_page(i: int, rng: random.Random) -> tuple[Page, list[LayoutQuery]]:
    pid = f"schematic-{i}"
    b = _Builder(pid, f"pid-diagram-{i}")
    b.region("title", (40, 24, 984, 64), f"Process Line {i + 1} P&ID Overview", "title", size=22)
    stages = rng.sample(_STAGES, 10)
    codes = {"Pump": "P", "Valve": "V", "Regulator": "PR", "Exchanger": "HX", "Meter": "FM", "Sensor": "TS",
             "Separator": "S", "Tank": "TK", "Filter": "F", "Chiller": "K"}
    labels = []
    for s in stages:
        code = next(v for k, v in codes.items() if k in s)
        labels.append(f"{code}-{i + 1}{rng.randint(0, 9)} {s}")
    fault = rng.randrange(10)
    positions = []
    for k in range(10):
        row, col = divmod(k, 4)
        x = 60 + col * 240 + (40 if row % 2 else 0)
        y = 140 + row * 230
        positions.append((x, y, x + 190, y + 90))
    for k in range(9):  # connectors first so boxes paint over them
        a, c = positions[k], positions[k + 1]
        b.d.line(((a[0] + a[2]) // 2, (a[1] + a[3]) // 2, (c[0] + c[2]) // 2, (c[1] + c[3]) // 2),
                 fill=(60, 60, 160), width=3)
    ids = []
    for k, (lab, pos) in enumerate(zip(labels, positions)):
        status = "FAULT" if k == fault else rng.choice(["NOMINAL", "NOMINAL", "STANDBY"])
        fill = (235, 70, 60) if status == "FAULT" else (215, 235, 215) if status == "NOMINAL" else (235, 235, 200)
        links = [f"n{k - 1}"] if k else []
        txt = f"{lab} {status}"
        r = b.region(f"n{k}", pos, txt, "node", links=links, fill=fill, outline=(40, 40, 40), size=13,
                     draw_text=lab, width=2)
        b.d.text((pos[0] + 6, pos[1] + 58), status, font=_font(13), fill=(255, 255, 255) if status == "FAULT" else (30, 90, 30))
        ids.append(r.id.split("/")[1])
    b.region("legend", (60, 880, 600, 990), "Legend: green NOMINAL, yellow STANDBY, red FAULT; arrows show flow direction",
             "caption", fill=(245, 245, 245), outline=(150, 150, 150), size=13)
    b.region("rev", (640, 880, 984, 990), f"Drawing PID-{i + 1:03d} sheet 1 of 1, approved", "text", size=13)
    q = []
    k1, k2 = rng.sample([k for k in range(10) if k != fault], 2)
    q.append(LayoutQuery(f"{pid}-q0", pid, f"red FAULT node on process line {i + 1}", f"{pid}/{ids[fault]}", "schematic"))
    q.append(LayoutQuery(f"{pid}-q1", pid, f"{labels[k1].split()[0]} {labels[k1].split(' ', 1)[1]}",
                         f"{pid}/{ids[k1]}", "schematic"))
    q.append(LayoutQuery(f"{pid}-q2", pid, f"status of the {labels[k2].split(' ', 1)[1].lower()} {labels[k2].split()[0]}",
                         f"{pid}/{ids[k2]}", "schematic"))
    q.append(LayoutQuery(f"{pid}-q3", pid, f"drawing PID-{i + 1:03d} sheet approval", f"{pid}/rev", "schematic"))
    return b.page("schematic"), q


# ---------------------------------------------------------------------------- multi-column
_MODULES = ["XR-7", "XR-9", "TC-4", "LM-2", "QS-5", "RF-3", "DX-8", "PW-6", "HB-1"]
_TOPICS = [
    ("Thermal Management", "The {m} thermal shutdown threshold is {a} C; fan speed ramps linearly from {b} C."),
    ("Power Sequencing", "Rails of the {m} power up in order 3V3, 1V8, 0V9 with {a} ms spacing; brownout trips at {b} mV."),
    ("Watchdog Timer", "The {m} watchdog timeout is {a} ms; the window opens after {b} ms and a missed kick forces reset."),
    ("Bus Interface", "The {m} SPI bus clock is limited to {a} MHz; CRC errors above {b} per hour raise a fault."),
    ("Calibration", "Factory calibration of the {m} stores {a} coefficients; drift beyond {b} ppm requires recalibration."),
    ("Firmware Update", "{m} firmware images are signed; rollback is blocked below version {a}.{b}."),
]


def _multicol_page(i: int, rng: random.Random, pool: list[str]) -> tuple[Page, list[LayoutQuery]]:
    pid = f"multicol-{i}"
    b = _Builder(pid, f"hw-spec-{i}")
    b.region("title", (40, 24, 984, 64), f"Controller Family Hardware Specification - Chapter {i + 4}", "title", size=22)
    ncols = 3 if i == 1 else 2
    gutter = 30
    cw = (944 - gutter * (ncols - 1)) // ncols
    mods = [pool.pop() for _ in range(ncols)]
    sections = []
    sec_no = 1
    for c in range(ncols):
        x = 40 + c * (cw + gutter)
        b.region(f"col{c}", (x, 90, x + cw, 1000), "", "container")
        y = 90
        topics = rng.sample(_TOPICS, 4)
        for t_name, t_body in topics:
            m = mods[c]
            head = f"{i + 4}.{sec_no} {m} {t_name}"
            hb = b.region(f"s{sec_no}h", (x, y, x + cw, y + 28), head, "heading", f"col{c}", size=15,
                          text_fill=(10, 40, 120))
            body = t_body.format(m=m, a=rng.randint(10, 120), b=rng.randint(5, 99))
            lines = max(3, len(body) // (cw // 8) + 1)
            ph = 20 * lines + 12
            pr = b.region(f"s{sec_no}p", (x, y + 32, x + cw, y + 32 + ph), body, "text", f"col{c}", size=12,
                          draw_text="")
            # wrap the paragraph text visually
            words, line, ly = body.split(), "", y + 36
            for w_ in words:
                if len(line) + len(w_) + 1 > cw // 7:
                    b.d.text((x + 4, ly), line, font=_font(12), fill=(30, 30, 30))
                    line, ly = w_, ly + 20
                else:
                    line = f"{line} {w_}".strip()
            b.d.text((x + 4, ly), line, font=_font(12), fill=(30, 30, 30))
            sections.append((m, t_name, hb.id, pr.id))
            sec_no += 1
            y += 32 + ph + 26
    q = []
    picks = rng.sample(sections, 4)
    kw = {"Thermal Management": "thermal shutdown threshold", "Power Sequencing": "brownout trip level",
          "Watchdog Timer": "watchdog timeout", "Bus Interface": "SPI bus clock limit",
          "Calibration": "calibration drift limit", "Firmware Update": "firmware rollback version"}
    for n, (m, t_name, _h, p_id) in enumerate(picks):
        q.append(LayoutQuery(f"{pid}-q{n}", pid, f"{m} {kw[t_name]}", p_id, "multicol"))
    return b.page("multicol"), q


# ------------------------------------------------------------------------------------ UI
def _ui_page(i: int, rng: random.Random) -> tuple[Page, list[LayoutQuery]]:
    pid = f"ui-{i}"
    b = _Builder(pid, f"app-screens-{i}", bg=(246, 247, 250))
    app = ["Checkout", "Account Settings", "Deploy Console"][i]
    b.region("topbar", (0, 0, 1024, 60), f"{app} - Acme Cloud", "title", fill=(30, 40, 70), text_fill=(255, 255, 255), size=20)
    nav = ["Dashboard", "Orders", "Billing", "Deployments", "Settings", "Support"]
    for k, n in enumerate(nav):
        b.region(f"nav{k}", (0, 70 + k * 50, 200, 115 + k * 50), n, "widget", "navc", fill=(232, 235, 242), size=15)
    b.region("navc", (0, 60, 200, 1024), "", "container")
    fields = {
        0: ["Cardholder Name", "Card Number", "Expiry Date", "Billing ZIP"],
        1: ["Display Name", "Email Address", "Phone Number", "Time Zone"],
        2: ["Service Name", "Container Image", "Replica Count", "Region"],
    }[i]
    err_field = rng.randrange(len(fields))
    err_msg = {0: "{f} is invalid", 1: "{f} format not recognised", 2: "{f} must be set"}[i].format(f=fields[err_field])
    b.region("form", (240, 90, 700, 640), f"{app} form", "container", fill=(255, 255, 255), outline=(200, 200, 210))
    y = 120
    err_id = None
    input_ids = []
    for k, f in enumerate(fields):
        b.region(f"lab{k}", (260, y, 420, y + 36), f, "label", "form", size=14)
        bad = k == err_field
        b.region(f"in{k}", (430, y, 680, y + 36), f"{f} input", "widget", "form", fill=(255, 255, 255),
                 outline=(220, 40, 40) if bad else (160, 160, 170), width=3 if bad else 1, draw_text="")
        input_ids.append(f"{pid}/in{k}")
        y += 44
        if bad:
            r = b.region(f"err{k}", (430, y, 680, y + 26), f"Error: {err_msg}", "text", "form",
                         fill=(253, 228, 228), text_fill=(200, 20, 20), size=12)
            err_id = r.id
            y += 34
        y += 18
    submit = {0: "Place Order", 1: "Save Changes", 2: "Deploy Now"}[i]
    b.region("submit", (430, y + 10, 600, y + 52), submit, "widget", "form", fill=(40, 110, 220), text_fill=(255, 255, 255), size=15)
    b.region("cancel", (260, y + 10, 400, y + 52), "Cancel", "widget", "form", fill=(230, 230, 235), size=15)
    metrics = [
        [("Conversion Rate", f"{rng.uniform(1, 9):.1f}%"), ("Cart Abandonment", f"{rng.randint(20, 80)}%"),
         ("Payment Latency", f"{rng.randint(80, 900)} ms"), ("Fraud Score", f"{rng.uniform(0, 1):.2f}")],
        [("Active Sessions", f"{rng.randint(1, 40)}"), ("Login Failures", f"{rng.randint(0, 30)} today"),
         ("MFA Adoption", f"{rng.randint(20, 99)}%"), ("Storage Used", f"{rng.uniform(1, 16):.1f} GiB")],
        [("CPU Utilization", f"{rng.randint(20, 95)}%"), ("Memory Usage", f"{rng.uniform(1, 16):.1f} GiB"),
         ("Pod Restarts", f"{rng.randint(0, 12)}"), ("Error Budget", f"{rng.randint(5, 99)}% left")],
    ][i]
    rng.shuffle(metrics)
    for k, (m, v) in enumerate(metrics):
        y0 = 90 + k * 130
        b.region(f"card{k}", (740, y0, 1000, y0 + 110), f"{m} {v}", "widget", "cards", fill=(255, 255, 255),
                 outline=(200, 200, 210), draw_text=m, size=14)
        b.d.text((750, y0 + 50), v, font=_font(26), fill=(30, 30, 30))
    b.region("cards", (720, 80, 1024, 640), "", "container")
    toast_txt = {0: "Payment service unavailable, retrying", 1: "Profile sync delayed", 2: "Registry rate limit reached"}[i]
    toast = b.region("toast", (240, 900, 760, 960), toast_txt, "widget", fill=(255, 140, 0), text_fill=(20, 20, 20), size=15)
    q = [
        LayoutQuery(f"{pid}-q0", pid, f"red error message for {fields[err_field].lower()}", err_id, "ui"),
        LayoutQuery(f"{pid}-q1", pid, f"{submit} button", f"{pid}/submit", "ui"),
        LayoutQuery(f"{pid}-q2", pid, f"{metrics[2][0].lower()} card", f"{pid}/card2", "ui"),
        LayoutQuery(f"{pid}-q3", pid, f"orange toast {toast_txt.split()[0].lower()} notification at bottom", toast.id, "ui"),
    ]
    return b.page("ui"), q


# --------------------------------------------------------------------------------- charts
def _chart_page(i: int, rng: random.Random) -> tuple[Page, list[LayoutQuery]]:
    pid = f"chart-{i}"
    b = _Builder(pid, f"perf-report-{i}")
    metric = ["p99 latency", "error rate", "throughput"][i]
    unit = ["ms", "%", "rps"][i]
    b.region("title", (40, 24, 984, 64), f"Quarterly {metric} report - service tier {i + 1}", "title", size=22)
    b.region("plot", (60, 100, 980, 600), f"{metric} by quarter", "container", outline=(180, 180, 180))
    groups = ["Q1", "Q2", "Q3", "Q4"]
    series = [("API", (50, 110, 200)), ("Batch", (230, 130, 40))]
    bar_ids = {}
    gx = 110
    for g in groups:
        for s_name, col in series:
            v = rng.uniform(0.15, 0.95)
            top = int(560 - v * 420)
            val = {"ms": f"{int(v * 900)} ms", "%": f"{v * 5:.2f} %", "rps": f"{int(v * 9000)} rps"}[unit]
            r = b.region(f"bar-{g}-{s_name}", (gx, top, gx + 70, 560), f"{g} {s_name} {metric} {val}", "bar", "plot",
                         fill=col, draw_text="")
            b.d.text((gx + 4, top - 18), val, font=_font(11), fill=(40, 40, 40))
            bar_ids[(g, s_name)] = r.id
            gx += 80
        b.region(f"lbl-{g}", (gx - 160, 565, gx, 590), g, "label", "plot", size=13)
        gx += 60
    b.region("legend", (700, 110, 960, 170), "Legend: blue API tier, orange Batch tier", "caption", "plot",
             fill=(250, 250, 250), outline=(200, 200, 200), size=12)
    rows = [[g, *(f"{rng.uniform(0, 9):.2f}" for _ in range(3))] for g in groups]
    tbl = _table(b, "sum", 60, 640, [180, 220, 220, 220], ["Quarter", "SLO Target", "Observed", "Delta"], rows,
                 f"Summary vs SLO for {metric}")
    q = [
        LayoutQuery(f"{pid}-q0", pid, f"Q3 API {metric} bar", bar_ids[("Q3", "API")], "chart"),
        LayoutQuery(f"{pid}-q1", pid, f"orange batch {metric} bar Q1", bar_ids[("Q1", "Batch")], "chart"),
        LayoutQuery(f"{pid}-q2", pid, f"Q4 observed {metric} value", tbl[(3, 2)], "chart"),
        LayoutQuery(f"{pid}-q3", pid, f"Q2 SLO target for {metric}", tbl[(1, 1)], "chart"),
    ]
    return b.page("chart"), q


def build_layout_corpus(seed: int = 7) -> tuple[list[Page], list[LayoutQuery]]:
    rng = random.Random(seed)
    pages, queries = [], []
    pool = list(_MODULES)
    rng.shuffle(pool)
    for fam in (_table_page, _schematic_page, _multicol_page, _ui_page, _chart_page):
        for i in range(3):
            p, q = fam(i, rng, pool) if fam is _multicol_page else fam(i, rng)
            pages.append(p)
            queries.extend(q)
    return pages, queries
