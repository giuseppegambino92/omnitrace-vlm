"""Deterministic web-app UI renderer for synthetic session recordings.

``render(state)`` draws one 480 x 270 frame of a small storefront app.  ``DOM_GEOMETRY``
gives the bounding box of every DOM node the renderer can draw, so DOM mutation logs
and frames are geometrically consistent (as they are in a real browser recording).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FW, FH = 480, 270

DOM_GEOMETRY: dict[str, tuple[int, int, int, int]] = {
    "#header": (0, 0, 480, 34),
    "#cart-badge": (420, 6, 470, 28),
    "#nav": (0, 34, 96, 270),
    "#content": (104, 42, 472, 262),
    "#product-grid": (104, 70, 472, 262),
    "#page-title": (104, 42, 472, 66),
    "#results-spinner": (264, 140, 312, 188),
    "#error-banner": (104, 42, 472, 70),
    "#toast": (250, 222, 470, 258),
    "#modal": (150, 80, 430, 220),
    "#price-total": (330, 236, 470, 258),
    "#promo-badge": (110, 74, 170, 92),
    "#checkout-form": (104, 70, 472, 230),
}

_FONT = None


def _font(size=11):
    global _FONT
    if _FONT is None:
        _FONT = {}
    if size not in _FONT:
        _FONT[size] = ImageFont.load_default(size=size)
    return _FONT[size]


@dataclass(frozen=True)
class UIState:
    page: str = "products"           # products | cart | checkout | search
    cards: int = 6
    cart_count: int = 0
    spinner_phase: int | None = None # 0..7 when a spinner is shown
    content_blank: bool = False      # content area rendered but empty
    banner: str | None = None        # red error banner text
    toast: str | None = None         # orange warning toast
    modal: str | None = None         # modal dialog title (dims the page)
    total_text: str = "Total: $0.00"
    promo: bool = False              # benign red "SALE" badge
    fade: float = 1.0                # 0..1 opacity of content during transitions
    items_text: str = "0 items"


def render(state: UIState, noise_sigma: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
    img = Image.new("RGB", (FW, FH), (244, 245, 248))
    d = ImageDraw.Draw(img)
    d.rectangle(DOM_GEOMETRY["#header"], fill=(28, 36, 60))
    d.text((10, 10), "ACME STORE", font=_font(13), fill=(255, 255, 255))
    d.rectangle(DOM_GEOMETRY["#cart-badge"], fill=(70, 90, 150))
    d.text((428, 10), f"cart {state.cart_count}", font=_font(11), fill=(255, 255, 255))
    d.rectangle(DOM_GEOMETRY["#nav"], fill=(226, 229, 238))
    for k, n in enumerate(["Products", "Search", "Cart", "Checkout", "Account"]):
        sel = n.lower() == state.page
        if sel:
            d.rectangle((0, 44 + k * 28, 96, 68 + k * 28), fill=(200, 208, 230))
        d.text((10, 50 + k * 28), n, font=_font(11), fill=(30, 30, 50))
    x1, y1, x2, y2 = DOM_GEOMETRY["#content"]
    d.rectangle((x1, y1, x2, y2), fill=(255, 255, 255))
    title = {"products": "All products", "cart": "Your cart", "checkout": "Checkout", "search": "Search results"}[state.page]
    d.text((x1 + 8, y1 + 6), f"{title} - {state.items_text}", font=_font(12), fill=(20, 20, 30))

    content = Image.new("RGB", (x2 - x1, y2 - 70), (255, 255, 255))
    cd = ImageDraw.Draw(content)
    if not state.content_blank and state.spinner_phase is None:
        if state.page in ("products", "search"):
            for k in range(state.cards):
                cx, cy = 8 + (k % 3) * 120, 6 + (k // 3) * 92
                cd.rectangle((cx, cy, cx + 110, cy + 84), outline=(200, 200, 210), fill=(250, 250, 252))
                cd.rectangle((cx + 6, cy + 6, cx + 104, cy + 46), fill=(120 + 20 * k % 100, 150, 190 - 15 * k))
                cd.text((cx + 6, cy + 52), f"Item {k + 1}", font=_font(11), fill=(30, 30, 30))
                cd.text((cx + 6, cy + 66), f"${19 + 7 * k}.99", font=_font(11), fill=(30, 110, 30))
        elif state.page == "cart":
            for k in range(3):
                cd.rectangle((8, 6 + k * 40, 360, 40 + k * 40), outline=(210, 210, 215))
                cd.text((16, 16 + k * 40), f"Line item {k + 1}  x1", font=_font(11), fill=(30, 30, 30))
        elif state.page == "checkout":
            for k, lab in enumerate(["Name", "Card number", "Expiry", "ZIP"]):
                cd.text((10, 10 + k * 34), lab, font=_font(11), fill=(40, 40, 40))
                cd.rectangle((110, 6 + k * 34, 340, 30 + k * 34), outline=(160, 160, 170), fill=(255, 255, 255))
    if state.fade < 1.0:
        content = Image.blend(Image.new("RGB", content.size, (255, 255, 255)), content, state.fade)
    img.paste(content, (x1, 70))
    d = ImageDraw.Draw(img)
    if state.promo:
        d.rectangle(DOM_GEOMETRY["#promo-badge"], fill=(225, 30, 45))
        d.text((118, 76), "SALE", font=_font(11), fill=(255, 255, 255))
    if state.page in ("cart", "checkout"):
        d.text((DOM_GEOMETRY["#price-total"][0] + 4, DOM_GEOMETRY["#price-total"][1] + 4), state.total_text,
               font=_font(12), fill=(20, 20, 20))
    if state.spinner_phase is not None:
        sx1, sy1, sx2, sy2 = DOM_GEOMETRY["#results-spinner"]
        cx, cy, r = (sx1 + sx2) / 2, (sy1 + sy2) / 2, 16
        for k in range(8):
            a = 2 * math.pi * k / 8
            shade = 70 + 22 * ((k - state.spinner_phase) % 8)
            px, py = cx + r * math.cos(a), cy + r * math.sin(a)
            d.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(shade, shade, min(255, shade + 40)))
    if state.banner:
        d.rectangle(DOM_GEOMETRY["#error-banner"], fill=(214, 38, 38))
        d.text((112, 50), state.banner, font=_font(12), fill=(255, 255, 255))
    if state.modal:
        arr = np.asarray(img, dtype=np.float32) * 0.45
        img = Image.fromarray(arr.astype(np.uint8))
        d = ImageDraw.Draw(img)
        d.rectangle(DOM_GEOMETRY["#modal"], fill=(255, 255, 255), outline=(90, 90, 90))
        d.text((164, 96), state.modal, font=_font(13), fill=(20, 20, 20))
        d.rectangle((300, 180, 410, 206), fill=(40, 110, 220))
        d.text((318, 186), "Sign in", font=_font(11), fill=(255, 255, 255))
    if state.toast:
        d.rectangle(DOM_GEOMETRY["#toast"], fill=(250, 150, 20))
        d.text((258, 232), state.toast, font=_font(11), fill=(30, 20, 0))
    out = np.asarray(img, dtype=np.float32)
    if noise_sigma > 0:
        rng = rng or np.random.default_rng(0)
        out = out + rng.normal(0, noise_sigma, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


@dataclass
class SessionRecording:
    frames: list[np.ndarray]
    times: list[float]
    transitions: list[int] = field(default_factory=list)  # ground-truth first frame of each new state


def dedup_session(n_frames: int = 200, fps: float = 10.0, seed: int = 3) -> SessionRecording:
    """A 200-frame UI recording with sparse transitions (benchmark 2).

    Includes: a one-digit badge change, page navigation, a 25-frame loading spinner,
    toast in/out, modal open/close (returning to an earlier state), a red banner, a
    4-frame fade, a small text change, plus per-frame sensor noise everywhere."""
    rng = np.random.default_rng(seed)
    base = UIState()
    timeline: list[tuple[int, UIState]] = []
    s = base
    plan: list[tuple[int, dict]] = [
        (0, {}),
        (18, {"cart_count": 1}),
        (30, {"page": "cart", "items_text": "1 items", "total_text": "Total: $19.99"}),
        (45, {"page": "checkout", "spinner_phase": 0}),
        (70, {"spinner_phase": None}),
        (85, {"toast": "Saved card expired"}),
        (100, {"toast": None}),
        (112, {"modal": "Confirm your address"}),
        (125, {"modal": None}),
        (140, {"banner": "Payment declined - try another card"}),
        (155, {"cart_count": 2}),
        (165, {"page": "products", "banner": None, "fade": 0.25}),
        (166, {"fade": 0.5}),
        (167, {"fade": 0.75}),
        (168, {"fade": 1.0}),
        (180, {"items_text": "2 items"}),
    ]
    transitions = [f for f, _ in plan if f not in (0, 165, 166, 167)]  # fade: the settled frame counts
    plan_d = dict(plan)
    frames, times = [], []
    for i in range(n_frames):
        if i in plan_d:
            s = replace(s, **plan_d[i])
        st = s
        if s.spinner_phase is not None:
            st = replace(s, spinner_phase=(i - 45) % 8)
        frames.append(render(st, noise_sigma=1.5, rng=rng))
        times.append(i / fps)
    return SessionRecording(frames, times, transitions)
