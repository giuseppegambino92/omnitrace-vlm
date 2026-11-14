"""Semantic action-entropy monitor that halts agentic loops before the token budget burns.

Every tool call is reduced to a *semantic action symbol*:

1. arguments are canonicalised (sorted JSON) and volatile values - UUIDs, timestamps,
   epoch numbers, long hex ids, nonces - are replaced by placeholders;
2. the canonical text is embedded with signed feature hashing;
3. a call joins an existing prototype of the same tool if the argument cosine is above
   ``arg_similarity``, or above the lower ``grounded_similarity`` when it also produced
   the *same observation* (paraphrased retries of the same query are the same action).

Observations are normalised the same way and hashed, so a poll whose result changes,
or a paginated fetch, counts as progress while an identical answer does not.

Halting rules (checked after every step):

* **cycle** - the last ``min_repeats * p`` steps are periodic with period ``p <= max_period``
  in both action symbols *and* observations (A,B,A,B,A,B with nothing new learned);
* **entropy collapse** - over the last ``window`` steps the Shannon entropy of action
  symbols is below ``entropy_threshold`` bits and the fraction of novel
  (action, observation) pairs is below ``novelty_threshold``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from omnitrace.config import LoopConfig
from omnitrace.textfeat import cosine, text_vector

__all__ = ["LoopVerdict", "LoopBreaker", "canonicalize"]

_VOLATILE = [
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I), "<uuid>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?\b"), "<ts>"),
    (re.compile(r"\b1[5-9]\d{8}(\d{3})?\b"), "<epoch>"),
    (re.compile(r"\b[0-9a-f]{16,}\b", re.I), "<hex>"),
    (re.compile(r"(nonce|request_id|req_id|trace_id|cache_bust|_t|ts)=[\w-]+", re.I), r"\1=<v>"),
]


def canonicalize(x) -> str:
    if isinstance(x, (dict, list, tuple)):
        s = json.dumps(x, sort_keys=True, default=str)
    else:
        s = str(x)
    for rx, rep in _VOLATILE:
        s = rx.sub(rep, s)
    return re.sub(r"\s+", " ", s.strip().lower())


def _h(s: str) -> str:
    return hashlib.blake2b(s.encode(), digest_size=8).hexdigest()


@dataclass
class LoopVerdict:
    halt: bool
    reason: str
    step: int
    entropy: float
    novelty: float
    cycle_period: int | None = None
    tokens_used: int = 0
    tokens_saved: int | None = None

    def to_dict(self) -> dict:
        return {"halt": self.halt, "reason": self.reason, "step": self.step, "entropy_bits": round(self.entropy, 3),
                "novelty": round(self.novelty, 3), "cycle_period": self.cycle_period,
                "tokens_used": self.tokens_used, "tokens_saved": self.tokens_saved}


@dataclass
class _Proto:
    tool: str
    vec: np.ndarray
    obs: set[str] = field(default_factory=set)


class LoopBreaker:
    def __init__(self, cfg: LoopConfig | None = None, token_budget: int | None = None):
        self.cfg = cfg or LoopConfig()
        self.token_budget = token_budget
        self.reset()

    def reset(self) -> None:
        self.protos: list[_Proto] = []
        self.symbols: list[int] = []
        self.obs: list[str] = []
        self.tokens = 0
        self.halted: LoopVerdict | None = None

    def _symbol(self, tool: str, args_c: str, obs_h: str) -> int:
        v = text_vector(f"{tool} {args_c}", 256)
        best, best_sim = None, -1.0
        for i, p in enumerate(self.protos):
            if p.tool != tool:
                continue
            sim = cosine(v, p.vec)
            ok = sim >= self.cfg.arg_similarity or (sim >= self.cfg.grounded_similarity and obs_h in p.obs)
            if ok and sim > best_sim:
                best, best_sim = i, sim
        if best is None:
            self.protos.append(_Proto(tool, v, {obs_h}))
            return len(self.protos) - 1
        self.protos[best].obs.add(obs_h)
        return best

    def observe(self, tool: str, args, observation, tokens: int | None = None) -> LoopVerdict:
        cfg = self.cfg
        obs_h = _h(canonicalize(observation))
        sym = self._symbol(tool, canonicalize(args), obs_h)
        self.symbols.append(sym)
        self.obs.append(obs_h)
        self.tokens += tokens if tokens is not None else cfg.tokens_per_step
        n = len(self.symbols)
        w = self.symbols[-cfg.window:]
        pairs = list(zip(w, self.obs[-cfg.window:]))
        cnt = Counter(w)
        ent = max(0.0, -sum(c / len(w) * math.log2(c / len(w)) for c in cnt.values()))
        novelty = len(set(pairs)) / len(pairs)

        def verdict(halt: bool, reason: str, period: int | None = None) -> LoopVerdict:
            saved = None
            if halt and self.token_budget:
                saved = max(0, self.token_budget - self.tokens)
            return LoopVerdict(halt, reason, n, ent, novelty, period, self.tokens, saved)

        for p in range(1, cfg.max_period + 1):
            span = p * cfg.min_repeats
            if n < span:
                break
            s, o = self.symbols[-span:], self.obs[-span:]
            if all(s[i] == s[i - p] and o[i] == o[i - p] for i in range(p, span)):
                if p > 1 and len(set(s[:p])) == 1:
                    continue  # already reported as period 1
                v = verdict(True, f"cycle of period {p} repeated {cfg.min_repeats}x with identical observations", p)
                self.halted = self.halted or v
                return v
        if n >= cfg.window and ent <= cfg.entropy_threshold and novelty <= cfg.novelty_threshold:
            v = verdict(True, f"action entropy {ent:.2f} bits with {novelty:.0%} novel observations over {cfg.window} steps")
            self.halted = self.halted or v
            return v
        return verdict(False, "progressing")

    def run(self, steps) -> LoopVerdict:
        """Feed an iterable of (tool, args, observation[, tokens]); stop at the first halt."""
        last = LoopVerdict(False, "empty", 0, 0.0, 1.0)
        for st in steps:
            last = self.observe(*st)
            if last.halt:
                return last
        return last
