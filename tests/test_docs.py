"""Documentation consistency checks for README.md and its Italian translation README.it.md."""

import re
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EN, IT = ROOT / "README.md", ROOT / "README.it.md"


@pytest.fixture(scope="module")
def docs():
    return {"en": EN.read_text(encoding="utf-8"), "it": IT.read_text(encoding="utf-8")}


def _math_segments(md: str) -> list[str]:
    segs = re.findall(r"```math\n(.*?)```", md, flags=re.S)
    segs += re.findall(r"\$\$(.+?)\$\$", md, flags=re.S)
    segs += re.findall(r"\$`(.+?)`\$", md)
    return segs


def _text_blocks(tex: str) -> list[str]:
    """Contents of every \\text{...} (brace-balanced)."""
    out = []
    for m in re.finditer(r"\\text\{", tex):
        depth, i = 1, m.end()
        while i < len(tex) and depth:
            depth += {"{": 1, "}": -1}.get(tex[i], 0)
            i += 1
        assert depth == 0, f"unbalanced \\text{{ in {tex!r}"
        out.append(tex[m.end(): i - 1])
    return out


def _block(md: str, tag: str) -> str:
    m = re.search(rf"<!-- {tag}:START -->(.*?)<!-- {tag}:END -->", md, flags=re.S)
    assert m, f"missing {tag} block"
    return m.group(1)


def _numbers(text: str, lang: str) -> list[str]:
    if lang == "it":
        text = re.sub(r"(\d),(\d)", r"\1.\2", text)   # Italian decimal comma
    return re.findall(r"\d+(?:\.\d+)*", text)


def test_cross_links_at_top(docs):
    assert "[Italiano](README.it.md)" in "\n".join(docs["en"].splitlines()[:4])
    assert "[English](README.md)" in "\n".join(docs["it"].splitlines()[:4])


def test_latex_text_blocks_have_no_unescaped_underscores(docs):
    found = 0
    for lang, md in docs.items():
        for seg in _math_segments(md):
            assert seg.count("{") == seg.count("}"), (lang, seg)
            for t in _text_blocks(seg):
                found += 1
                assert not re.search(r"(?<!\\)_", t), f"{lang}: unescaped underscore in \\text{{{t}}}"
    assert found > 0


def test_formulas_identical_across_languages(docs):
    fence = lambda md: re.findall(r"```math\n(.*?)```", md, flags=re.S)  # noqa: E731
    assert fence(docs["en"]) == fence(docs["it"]) and len(fence(docs["en"])) == 4


def test_same_section_structure(docs):
    levels = lambda md: [len(h) for h in re.findall(r"^(#{1,4}) ", re.sub(r"```.*?```", "", md, flags=re.S), re.M)]  # noqa: E731
    assert levels(docs["en"]) == levels(docs["it"])


def test_architecture_diagrams_share_one_skeleton(docs):
    def skeleton(md):
        block = next(b for b in re.findall(r"```text\n(.*?)```", md, flags=re.S) if "┌" in b)
        rows = [l for l in block.splitlines() if l.startswith(("┌", "└", "│"))]
        widths = {sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in l) for l in rows}
        assert len(widths) == 1, "box rows must all have the same display width"
        return ["".join(c if c in "┌┐└┘─│▼┬" else " " for c in l) for l in rows]
    assert skeleton(docs["en"]) == skeleton(docs["it"])


def test_generated_blocks_carry_identical_numbers(docs):
    for tag in ("METRICS", "REPEAT"):
        en, it = _block(docs["en"], tag), _block(docs["it"], tag)
        assert _numbers(en, "en") == _numbers(it, "it"), tag


def test_local_links_resolve(docs):
    for lang, md in docs.items():
        slugs = {re.sub(r"[^\w\- ]", "", h.strip().lower()).replace(" ", "-")
                 for h in re.findall(r"^#{1,4} (.+)$", md, re.M)}
        for target in re.findall(r"\]\(([^)]+)\)", md):
            if target.startswith("#"):
                assert target[1:] in slugs, f"{lang}: dangling anchor {target}"
            elif not target.startswith("http"):
                assert (ROOT / target).exists(), f"{lang}: missing file {target}"
