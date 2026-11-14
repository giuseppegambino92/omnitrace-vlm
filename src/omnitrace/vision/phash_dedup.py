"""Perceptual hashing and cycle-aware keyframe extraction for UI recordings.

Hashes
------
* ``dhash``  - difference hash: box-downsample to (h, h+1) grey, compare horizontal neighbours.
* ``phash``  - DCT hash: 32x32 grey -> 2-D orthonormal DCT-II -> low-frequency 8x8 block
               thresholded at its median (DC excluded from the median).

Keyframe extraction
-------------------
Each frame is reduced once (integer box reduction in C via Pillow) and summarised by

* a 64-bit dHash (global structure), and
* a ``tile_rows x tile_cols`` grid of tile means (local structure; catches a single
  changed digit that a global hash cannot see).

Frames are then segmented into *change runs*.  A run whose changes stay confined
to the same small set of tiles and keeps changing is an **animation** (spinner,
progress bar, blinking caret): only its first frame is a keyframe.  The frame on
which any run *settles* (followed by ``settle`` static frames) is always a
keyframe, so the final state of fades / slides / spinner disappearance is never
lost.  Static frames and animation phases are discarded.

The extractor also reports the animation runs themselves - the correlator uses
them to detect frozen spinners.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy.fft import dctn

from omnitrace.config import DedupConfig

__all__ = [
    "dhash", "phash", "hamming", "FrameFeatures", "AnimationRun", "KeyframeResult",
    "KeyframeExtractor", "frame_features",
]


def _as_pil_gray(image: np.ndarray | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("L") if image.mode != "L" else image
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 3 and arr.flags.c_contiguous:
        # zero-copy wrap of the frame buffer (Image.fromarray copies)
        return Image.frombuffer("RGB", (arr.shape[1], arr.shape[0]), arr, "raw", "RGB", 0, 1).convert("L")
    im = Image.fromarray(arr)
    return im.convert("L") if im.mode != "L" else im


def _bits_to_int(bits: np.ndarray) -> int:
    out = 0
    for b in np.packbits(bits.ravel().astype(np.uint8)).tolist():
        out = (out << 8) | b
    return out


def dhash(image: np.ndarray | Image.Image, hash_size: int = 8) -> int:
    """64-bit (for hash_size=8) difference hash."""
    g = np.asarray(_as_pil_gray(image).resize((hash_size + 1, hash_size), Image.Resampling.BOX),
                   dtype=np.int16)
    return _bits_to_int(g[:, 1:] > g[:, :-1])


def phash(image: np.ndarray | Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """DCT perceptual hash (robust to small rescales / compression)."""
    n = hash_size * highfreq_factor
    g = np.asarray(_as_pil_gray(image).resize((n, n), Image.Resampling.LANCZOS), dtype=np.float64)
    low = dctn(g, type=2, norm="ortho")[:hash_size, :hash_size]
    med = np.median(low.ravel()[1:])
    return _bits_to_int(low > med)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


_EDGE_CACHE: dict[tuple[int, int, int, int], tuple[np.ndarray, np.ndarray]] = {}


def _tile_edges(h: int, w: int, tr: int, tc: int) -> tuple[np.ndarray, np.ndarray]:
    key = (h, w, tr, tc)
    if key not in _EDGE_CACHE:
        _EDGE_CACHE[key] = (np.linspace(0, h, tr + 1).astype(np.intp), np.linspace(0, w, tc + 1).astype(np.intp))
    return _EDGE_CACHE[key]


@dataclass(slots=True)
class FrameFeatures:
    dhash: int
    tiles: np.ndarray  # (tile_rows, tile_cols) float32 tile means


def frame_features(frame: np.ndarray | Image.Image, cfg: DedupConfig) -> FrameFeatures:
    im = _as_pil_gray(frame)
    if cfg.work_reduce > 1:
        im = im.reduce(cfg.work_reduce)
    g = np.asarray(im, dtype=np.float32)
    h, w = g.shape
    if h < cfg.tile_rows or w < cfg.tile_cols:
        raise ValueError(f"frame too small for a {cfg.tile_rows}x{cfg.tile_cols} tile grid")
    tr, tc = cfg.tile_rows, cfg.tile_cols
    if h % tr == 0 and w % tc == 0:   # fast path: exact block means
        tiles = g.reshape(tr, h // tr, tc, w // tc).mean(axis=(1, 3))
    else:                             # general sizes: uneven blocks via reduceat
        ri, ci = _tile_edges(h, w, tr, tc)
        tiles = np.add.reduceat(np.add.reduceat(g, ri[:-1], axis=0), ci[:-1], axis=1)
        tiles /= np.outer(np.diff(ri), np.diff(ci))
    hs = cfg.hash_size
    small = np.asarray(im.resize((hs + 1, hs), Image.Resampling.BOX), dtype=np.int16)
    return FrameFeatures(_bits_to_int(small[:, 1:] > small[:, :-1]), tiles)


@dataclass
class AnimationRun:
    start: int                 # first animated frame index
    end: int                   # last animated frame index (inclusive)
    tiles: np.ndarray          # bool mask (tile_rows, tile_cols) of animated tiles
    t_start: float = 0.0
    t_end: float = 0.0

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    def bbox(self, width: int, height: int) -> tuple[float, float, float, float]:
        rows, cols = np.nonzero(self.tiles)
        tr, tc = self.tiles.shape
        return (cols.min() * width / tc, rows.min() * height / tr,
                (cols.max() + 1) * width / tc, (rows.max() + 1) * height / tr)


@dataclass
class KeyframeResult:
    keyframes: list[int]
    reasons: dict[int, str]
    animations: list[AnimationRun]
    n_frames: int
    elapsed_s: float
    features: list[FrameFeatures] = field(repr=False, default_factory=list)

    @property
    def pruned_fraction(self) -> float:
        return 1.0 - len(self.keyframes) / max(1, self.n_frames)

    @property
    def ms_per_frame(self) -> float:
        return 1000.0 * self.elapsed_s / max(1, self.n_frames)


class KeyframeExtractor:
    """Offline (look-ahead) keyframe extractor; see module docstring."""

    def __init__(self, cfg: DedupConfig | None = None, settle: int = 3):
        self.cfg = cfg or DedupConfig()
        self.settle = settle
        self.threshold_used = self.cfg.tile_threshold

    def _change_masks(self, feats: list[FrameFeatures]) -> tuple[list[np.ndarray], float]:
        """Per-frame changed-tile masks, with a noise-adaptive threshold.

        Sensor / compression noise is estimated robustly from the median absolute tile
        delta of the whole recording (most tiles of most frames are static); the
        threshold is ``max(tile_threshold, 8 * sigma_hat)``."""
        cfg = self.cfg
        if len(feats) > 1:
            stack = np.stack([f.tiles for f in feats])
            dstack = np.abs(np.diff(stack, axis=0))
            deltas = list(dstack)
            sample = dstack.ravel()[::7]
            sigma = float(np.partition(sample, len(sample) // 2)[len(sample) // 2]) / 0.6745
        else:
            deltas, sigma = [], 0.0
        thr = max(cfg.tile_threshold, 8.0 * sigma)
        self.threshold_used = thr
        masks = [np.ones_like(feats[0].tiles, dtype=bool)]
        for prev, cur, d in zip(feats, feats[1:], deltas):
            m = d > thr
            if (not m.any() and hamming(cur.dhash, prev.dhash) > cfg.hamming_threshold
                    and d.mean() > 0.25 * thr):
                m = np.ones_like(m)  # global change too diffuse for any single tile
            masks.append(m)
        return masks, thr

    def extract(self, frames, times=None) -> KeyframeResult:
        t0 = time.perf_counter()
        cfg = self.cfg
        feats = [frame_features(f, cfg) for f in frames]
        n = len(feats)
        if n == 0:
            return KeyframeResult([], {}, [], 0, 0.0)
        times = list(times) if times is not None else [float(i) for i in range(n)]
        masks, thr = self._change_masks(feats)
        counts = [int(m.sum()) for m in masks]
        changed = [c > 0 for c in counts]
        max_tiles = max(1, int(cfg.animation_max_frac * cfg.tile_rows * cfg.tile_cols))
        confined = [0 < c <= max_tiles for c in counts]
        confined[0] = False

        # 1. animation segments: runs of confined changes over the same small tile set
        segments: list[tuple[list[int], np.ndarray]] = []
        k = 1
        while k < n:
            if not confined[k]:
                k += 1
                continue
            seg, union, j = [k], masks[k].copy(), k + 1
            while True:
                nxt = next((m for m in range(j, min(n, j + self.settle)) if changed[m]), None)
                if nxt is None or not confined[nxt]:
                    break
                u2 = union | masks[nxt]
                if u2.sum() > max_tiles:
                    break
                seg.append(nxt)
                union, j = u2, nxt + 1
            if len(seg) >= cfg.animation_min_frames:
                segments.append((seg, union))
            k = seg[-1] + 1

        in_segment = {f for seg, _ in segments for f in seg}
        keyframes: dict[int, str] = {0: "first"}
        for i in range(1, n):
            if changed[i] and i not in in_segment:
                keyframes[i] = "transition"

        animations: list[AnimationRun] = []
        for seg, union in segments:
            first, last = seg[0], seg[-1]
            if not (changed[first - 1] and (first - 1) not in in_segment) and first - 1 not in keyframes:
                keyframes[first] = "animation-start"
            # the last confined change may be the animation disappearing: keep it if the
            # settled state was never seen during the animation
            last_new = len(seg) > 1 and all(
                np.abs(feats[last].tiles - feats[f].tiles)[union].max() > thr
                for f in seg[:-1])
            anim_end = last
            if last_new:
                keyframes[last] = "animation-end"
                anim_end = seg[-2]
            animations.append(AnimationRun(first, anim_end, union, times[first], times[anim_end]))

        elapsed = time.perf_counter() - t0
        return KeyframeResult(sorted(keyframes), keyframes, animations, n, elapsed, feats)
