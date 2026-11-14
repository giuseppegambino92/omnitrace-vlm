"""Deterministic text featurisation shared by every subsystem.

* ``tokenize``      - lower-cased word tokens (identifiers like ``P-201`` stay whole).
* ``token_vector``  - signed feature hashing of the word plus its character
                      trigrams (so ``P-201`` ~ ``p201`` and ``voltage`` ~ ``voltages``).
* ``text_vector``   - normalised bag of token vectors.

Hashing uses BLAKE2b (not Python's salted ``hash``) so vectors are identical
across processes, machines and runs.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*")

STOPWORDS = frozenset(
    """a an the of for to in on at by with from and or is are was were be been being this that
    these those it its as into what which who whom where when how does do did show find give
    me my our your their there here than then please value values tell about under over
    whose has have had can could should would will shall may might must""".split()
)

WORD_WEIGHT = 2.0
NGRAM_WEIGHT = 1.0
WORD_SLOTS = 8    # a word is spread over several signed slots: collisions become small,
NGRAM_SLOTS = 2   # near-Gaussian noise instead of rare large spikes


def _h(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def tokenize(text: str, keep_stopwords: bool = False, bind: str | None = None) -> list[str]:
    """Lower-cased tokens.

    ``bind`` enables numeric-qualifier binding ("line 1" -> ``line-1``), which keeps short
    ordinal numbers attached to the noun they qualify:

    * ``"doc"``   - emit the word, the number *and* the bound compound;
    * ``"query"`` - emit the word and the compound, drop the bare number (a lone "1" is
                    not a useful retrieval constraint once it is bound).
    """
    toks = _TOKEN_RE.findall(text.lower())
    out: list[str] = []
    prev_word: str | None = None
    for t in toks:
        t = t.strip("-_./")
        if not t:
            continue
        if not keep_stopwords and t in STOPWORDS:
            prev_word = None
            continue
        if bind and prev_word and t.isdigit() and len(t) <= 2:
            out.append(f"{prev_word}-{t}")
            if bind == "doc":
                out.append(t)
            prev_word = None
            continue
        if len(t) == 1 and not t.isdigit():
            prev_word = None
            continue
        out.append(t)
        prev_word = t if t.isalpha() else None
    return out


def _stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


@lru_cache(maxsize=262_144)
def _token_vector_cached(token: str, dim: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    base = _stem(token)
    ww = WORD_WEIGHT / np.sqrt(WORD_SLOTS)
    for k in range(WORD_SLOTS):
        h = _h(f"w{k}:" + base)
        v[h % dim] += ww * (1.0 if (h >> 40) & 1 else -1.0)
    padded = f"#{base.replace('-', '').replace('_', '')}#"
    grams = [padded[i:i + 3] for i in range(max(1, len(padded) - 2))]
    w = NGRAM_WEIGHT / np.sqrt(len(grams) * NGRAM_SLOTS)
    for g in grams:
        for k in range(NGRAM_SLOTS):
            hg = _h(f"g{k}:" + g)
            v[hg % dim] += w * (1.0 if (hg >> 40) & 1 else -1.0)
    n = float(np.linalg.norm(v))
    if n > 0:
        v /= n
    v.setflags(write=False)
    return v


def token_vector(token: str, dim: int) -> np.ndarray:
    """Unit-norm signed-hash embedding of a single token (read-only, cached)."""
    return _token_vector_cached(token, dim)


def text_vector(text: str, dim: int, bind: str | None = None) -> np.ndarray:
    """Unit-norm bag-of-tokens embedding (zero vector for empty text)."""
    toks = tokenize(text, bind=bind)
    v = np.zeros(dim, dtype=np.float32)
    for t in toks:
        v += token_vector(t, dim)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b) / (na * nb)
