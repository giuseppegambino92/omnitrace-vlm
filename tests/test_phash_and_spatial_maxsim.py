from dataclasses import replace

import numpy as np
import pytest

from evals.datasets.ui_sim import UIState, dedup_session, render
from omnitrace.config import VisionConfig
from omnitrace.vision.patch_encoder import (
    Page, PatchEncoder, Region, binarize, build_layout_graph, dequantize_int2, dequantize_int8, quantize_int2,
    quantize_int8, unpack_signs,
)
from omnitrace.vision.phash_dedup import KeyframeExtractor, dhash, hamming, phash
from omnitrace.vision.spatial_maxsim import VisualIndex, activation_heatmap, heatmap_bbox, maxsim_scores


# ----------------------------------------------------------------------------- perceptual hashes
def test_hashes_identical_noisy_and_different():
    rng = np.random.default_rng(0)
    a = render(UIState())
    noisy = np.clip(a + rng.normal(0, 2, a.shape), 0, 255).astype(np.uint8)
    b = render(UIState(page="checkout", modal="Hello"))
    for h in (dhash, phash):
        assert hamming(h(a), h(a)) == 0
        assert hamming(h(a), h(noisy)) <= 4
        assert hamming(h(a), h(b)) >= 10
    assert 0 <= dhash(a) < 2 ** 64


def test_static_recording_has_single_keyframe():
    rng = np.random.default_rng(1)
    base = render(UIState())
    frames = [np.clip(base + rng.normal(0, 1.5, base.shape), 0, 255).astype(np.uint8) for _ in range(30)]
    res = KeyframeExtractor().extract(frames)
    assert res.keyframes == [0]
    assert res.pruned_fraction > 0.96


def test_single_digit_change_is_a_keyframe():
    frames = [render(UIState(cart_count=1))] * 5 + [render(UIState(cart_count=2))] * 5
    assert KeyframeExtractor().extract(frames).keyframes == [0, 5]


def test_spinner_animation_is_absorbed_and_reported():
    states = [UIState()] * 4 + [UIState(page="search", spinner_phase=k % 8) for k in range(30)] + [UIState(page="search")] * 6
    frames = [render(s) for s in states]
    res = KeyframeExtractor().extract(frames, [i * 0.1 for i in range(len(frames))])
    assert 4 in res.keyframes and 34 in res.keyframes          # spinner appears / disappears
    assert len(res.keyframes) <= 4
    assert len(res.animations) == 1 and res.animations[0].duration > 2.5
    x1, y1, x2, y2 = res.animations[0].bbox(480, 270)
    assert 230 < (x1 + x2) / 2 < 340 and 120 < (y1 + y2) / 2 < 210   # localised on the spinner


def test_benchmark_session_targets():
    rec = dedup_session()
    res = KeyframeExtractor().extract(rec.frames, rec.times)
    assert set(rec.transitions) <= set(res.keyframes)
    assert res.pruned_fraction > 0.90


# ----------------------------------------------------------------------------- quantisation
def test_quantizers_preserve_dot_products():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((64, 1024)).astype(np.float32)
    q = rng.standard_normal(1024).astype(np.float32)
    exact = x @ q
    i8 = dequantize_int8(*quantize_int8(x)) @ q
    i2 = dequantize_int2(*quantize_int2(x), 1024) @ q
    bits, norms = binarize(x)
    b = (unpack_signs(bits, 1024) * norms[:, None] * np.sqrt(np.pi / 2048)) @ q
    corr = lambda a: np.corrcoef(exact, a)[0, 1]
    assert corr(i8) > 0.999 and corr(i2) > 0.9 and corr(b) > 0.75
    assert quantize_int2(x)[0].shape == (64, 256) and bits.shape == (64, 128)


def test_rotation_is_orthogonal():
    enc = PatchEncoder()
    R = enc.R
    assert np.allclose(R @ R.T, np.eye(R.shape[0]), atol=1e-4)


# ----------------------------------------------------------------------------- layout graph
def _mini_table():
    regs = [Region("t", (0, 0, 400, 120), "Ratings", "container")]
    hdr = ["Component", "Max Voltage", "Max Current"]
    for c, h in enumerate(hdr):
        regs.append(Region(f"h{c}", (c * 130, 0, c * 130 + 130, 30), h, "header", "t"))
    for r, name in enumerate(["P-101 Pump", "V-202 Valve"]):
        regs.append(Region(f"r{r}c0", (0, 30 + r * 30, 130, 60 + r * 30), name, "rowheader", "t"))
        for c in (1, 2):
            regs.append(Region(f"r{r}c{c}", (c * 130, 30 + r * 30, c * 130 + 130, 60 + r * 30), f"{r}{c}", "cell", "t"))
    return regs


def test_layout_graph_links_cells_to_row_and_column_headers():
    g = build_layout_graph(_mini_table())
    assert g["r1c2"]["r1c0"] == 1.0          # row header
    assert g["r1c2"]["h2"] == 1.0            # column header
    assert g["r1c2"].get("r0c0", 0) < 0.3    # other row's header is not context
    assert g["h2"].get("h1", 0) <= 0.05      # sibling headers do not describe each other


def test_grounds_table_cell_at_row_column_intersection():
    regs = _mini_table()
    img = np.full((240, 400, 3), 255, np.uint8)
    idx = VisualIndex(replace(VisionConfig(), grid_rows=16, grid_cols=16))
    idx.add_page(Page("p", img, regs))
    hit = idx.search("max current of V-202", k=1)[0]
    assert hit.region_id == "r1c2"
    assert hit.grounding == "region"


# ----------------------------------------------------------------------------- retrieval
def test_maxsim_matches_bruteforce():
    rng = np.random.default_rng(3)
    Q, P = rng.standard_normal((4, 32)), rng.standard_normal((3 * 10, 32))
    s = maxsim_scores(Q, P, 3, 10)
    brute = [sum(max(q @ p for p in P[i * 10:(i + 1) * 10]) for q in Q) for i in range(3)]
    assert np.allclose(s, brute)


def test_soft_and_heatmap_prefers_patch_satisfying_all_tokens():
    sims = np.array([[0.9, 0.0, 0.4], [0.9, 0.0, 0.4], [0.0, 0.8, 0.35]])  # 3 tokens x 3 patches
    heat, w = activation_heatmap(sims, (1, 3), eps=0.02, floor=0.1, power=-1.0)
    assert int(np.argmax(heat)) == 2


def test_heatmap_bbox_component():
    heat = np.zeros((8, 8))
    heat[2:4, 5:7] = 1.0
    heat[6, 0] = 0.4
    assert heatmap_bbox(heat, 800, 800) == (500.0, 200.0, 700.0, 400.0)


@pytest.mark.parametrize("precision", ["float32", "int8", "int2", "binary"])
def test_benchmark_grounding_all_precisions(visual_index, layout_corpus, precision):
    pages, queries = layout_corpus
    regions = {r.id: r for p in pages for r in p.regions}
    ok = 0
    for q in queries:
        h = visual_index.search(q.text, k=1, precision=precision)[0]
        gt = regions[q.answer_region].bbox
        ok += h.page_id == q.page_id and h.bbox == gt
    assert ok / len(queries) >= 0.9


def test_crop_and_heatmap_shapes(visual_index):
    h = visual_index.search("Place Order button", k=2)[0]
    crop = visual_index.crop(h)
    x1, y1, x2, y2 = h.bbox
    assert crop.shape[0] >= y2 - y1 and crop.shape[1] >= x2 - x1 and crop.shape[2] == 3
    assert h.heatmap.shape == (32, 32) and 0 <= h.heatmap.min() and h.heatmap.max() <= 1.0
    assert set(h.token_peaks) == {"place", "order", "button"}


def test_save_load_roundtrip(tmp_path, layout_corpus):
    pages, queries = layout_corpus
    idx = VisualIndex()
    idx.add_pages(pages[:4])
    p = idx.save(tmp_path / "idx.sqlite")
    idx2 = VisualIndex.load(p)
    assert len(idx2) == 4 and idx2.images.keys() == idx.images.keys()
    for q in queries[:8]:
        a, b = idx.search(q.text, k=2), idx2.search(q.text, k=2)
        assert [h.page_id for h in a] == [h.page_id for h in b]
        assert [h.bbox for h in a] == [h.bbox for h in b]


def test_invalid_inputs():
    idx = VisualIndex()
    assert idx.search("anything") == []
    page = Page("x", np.zeros((64, 64, 3), np.uint8))
    idx.add_page(page)
    with pytest.raises(ValueError):
        idx.add_page(page)
    with pytest.raises(ValueError):
        idx.search("x", precision="fp4")
    assert idx.search("the of and") == []   # only stopwords


def test_regionless_screenshot_uses_heatmap_grounding():
    img = np.full((256, 256, 3), 255, np.uint8)
    img[200:240, 20:120] = (220, 30, 30)    # red box bottom-left
    idx = VisualIndex(replace(VisionConfig(), grid_rows=16, grid_cols=16))
    idx.add_page(Page("shot", img))
    h = idx.search("red", k=1)[0]
    assert h.grounding == "heatmap"
    x1, y1, x2, y2 = h.bbox
    assert y1 >= 160 and x2 <= 160
