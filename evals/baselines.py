"""Text-only RAG baselines used as comparison points in the benchmark.

* ``FlatChunkBM25``  - the classic pipeline: linearise the page text in reading order
  (top-to-bottom, left-to-right), split into overlapping fixed-size token windows,
  rank with Okapi BM25.  Its "citation" is the union bbox of the regions whose words
  fall in the winning chunk (generous: real chunkers usually cannot cite at all).
* ``RegionBM25``     - structure-aware but context-free: each layout region is its own
  BM25 document (no 2-D relations: a table cell is just "42.5 A").
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from omnitrace.textfeat import tokenize
from omnitrace.vision.patch_encoder import NON_LEAF_ROLES, Page


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.df = Counter(t for d in docs for t in set(d))
        self.avgdl = sum(len(d) for d in docs) / max(1, len(docs))
        self.N = len(docs)

    def idf(self, t: str) -> float:
        n = self.df.get(t, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for d, tf in zip(self.docs, self.tf):
            s, dl = 0.0, len(d)
            for t in query:
                f = tf.get(t, 0)
                if f:
                    s += self.idf(t) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


@dataclass
class TextHit:
    page_id: str
    bbox: tuple[float, float, float, float]
    text: str


def _union(boxes):
    xs1, ys1, xs2, ys2 = zip(*boxes)
    return (min(xs1), min(ys1), max(xs2), max(ys2))


class FlatChunkBM25:
    def __init__(self, pages: list[Page], chunk: int = 40, stride: int = 20):
        self.chunks: list[tuple[str, list[str], list[tuple]]] = []
        for p in pages:
            leaves = [r for r in p.regions if r.role not in NON_LEAF_ROLES and r.text]
            leaves.sort(key=lambda r: (round(r.bbox[1] / 12), r.bbox[0]))   # reading order
            words, owners = [], []
            for r in leaves:
                for t in tokenize(r.text):
                    words.append(t)
                    owners.append(r.bbox)
            for s in range(0, max(1, len(words) - stride), stride):
                w, o = words[s:s + chunk], owners[s:s + chunk]
                if w:
                    self.chunks.append((p.page_id, w, o))
        self.bm25 = BM25([c[1] for c in self.chunks])

    def search(self, query: str) -> TextHit:
        sc = self.bm25.scores(tokenize(query))
        i = max(range(len(sc)), key=sc.__getitem__)
        pid, w, o = self.chunks[i]
        return TextHit(pid, _union(o), " ".join(w))


class RegionBM25:
    def __init__(self, pages: list[Page]):
        self.items = [(p.page_id, r.bbox, r.text) for p in pages for r in p.regions
                      if r.role not in NON_LEAF_ROLES and r.text]
        self.bm25 = BM25([tokenize(t) for _, _, t in self.items])

    def search(self, query: str) -> TextHit:
        sc = self.bm25.scores(tokenize(query))
        i = max(range(len(sc)), key=sc.__getitem__)
        pid, bbox, text = self.items[i]
        return TextHit(pid, bbox, text)
