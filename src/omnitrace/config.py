"""Central configuration for every omnitrace subsystem.

All knobs are plain frozen dataclasses so that tests, benchmarks and the CLI
share the exact same defaults.  ``OmnitraceConfig.from_env()`` lets operators
override the most common settings without touching code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path


@dataclass(frozen=True)
class VisionConfig:
    """Spatial patch encoder + late-interaction retrieval."""

    grid_rows: int = 32
    grid_cols: int = 32
    text_dim: int = 976          # signed feature-hashing subspace (dim = 1024 total)
    visual_dim: int = 32         # colour / gradient descriptor subspace
    pos_dim: int = 16            # spatial-coordinate subspace
    context_weight: float = 1.0  # layout-graph context mixed into each region
    visual_weight: float = 0.6   # scale of the visual subspace in patch vectors
    pos_weight: float = 0.35     # scale of the positional subspace
    rotation_seed: int = 1337    # random orthogonal rotation used before quantisation
    heat_eps: float = 0.02       # soft-AND floor for the activation heatmap
    heat_power: float = -1.0     # power-mean exponent of the soft AND (-1 = harmonic)
    token_floor: float = 0.10    # query tokens whose best match is below this are uninformative
    and_feasible: float = 0.25   # if no patch satisfies all tokens at this level, drop the blocking token
    candidate_pages: int = 8     # binary first-stage shortlist size
    match_floor: float = 0.2     # corpus-best similarity a token needs to count as genuinely matched
    calibrated: bool = True      # per-token corpus-calibrated, IDF-weighted MaxSim (False = raw ColBERT sum)

    @property
    def dim(self) -> int:
        return self.text_dim + self.visual_dim + self.pos_dim


@dataclass(frozen=True)
class DedupConfig:
    """Perceptual keyframe extraction."""

    hash_size: int = 8
    work_reduce: int = 3          # integer box-reduction factor before hashing
    tile_rows: int = 18
    tile_cols: int = 32
    tile_threshold: float = 0.9   # mean grey-level change of a tile that counts as "changed"
    hamming_threshold: int = 12   # dHash distance that (with weak tile evidence) flags a diffuse change
    animation_max_frac: float = 0.03  # a change confined to <= this fraction of tiles may be an animation
    animation_min_frames: int = 3 # consecutive confined changes before a region is treated as animated


@dataclass(frozen=True)
class CorrelatorConfig:
    """Cross-modal causal DAG + RCA ranking."""

    max_clock_skew_s: float = 0.35   # tolerated skew between the screen recorder and the browser clock
    dom_clock_skew_s: float = 0.05   # client spans and DOM records share the browser clock
    dom_window_s: float = 2.5        # span end -> DOM mutation causal window
    visual_window_s: float = 3.0     # DOM mutation -> visible anomaly causal window
    stall_threshold_s: float = 3.0   # spinner animating longer than this is "frozen"
    blank_persist_s: float = 1.0     # blank region must persist this long
    temporal_tau_s: float = 4.0      # decay constant for temporal proximity
    propagation_weight: float = 0.9  # child -> parent span edge weight
    origin_penalty: float = 0.3      # multiplier for spans whose anomaly is explained by a child


@dataclass(frozen=True)
class ShieldConfig:
    """Indirect prompt-injection shield."""

    block_threshold: float = 1.0
    encoded_min_len: int = 24
    max_decode_depth: int = 3
    sensitive_tools: tuple[str, ...] = (
        "send_email", "http_post", "http_request", "execute_shell", "run_command",
        "write_file", "delete_file", "transfer_funds", "create_user", "update_permissions",
        "git_push", "upload_file",
    )


@dataclass(frozen=True)
class LoopConfig:
    """Semantic action-entropy loop breaker."""

    window: int = 10
    max_period: int = 4
    min_repeats: int = 3
    entropy_threshold: float = 1.0      # bits
    novelty_threshold: float = 0.2
    arg_similarity: float = 0.85        # cosine above which two calls are the same action
    grounded_similarity: float = 0.45   # lower bar when the observation is identical
    tokens_per_step: int = 1500


@dataclass(frozen=True)
class OmnitraceConfig:
    vision: VisionConfig = field(default_factory=VisionConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    correlator: CorrelatorConfig = field(default_factory=CorrelatorConfig)
    shield: ShieldConfig = field(default_factory=ShieldConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    db_path: Path = Path("omnitrace_index.sqlite")

    @classmethod
    def from_env(cls) -> "OmnitraceConfig":
        cfg = cls()
        if p := os.environ.get("OMNITRACE_DB"):
            cfg = replace(cfg, db_path=Path(p))
        if g := os.environ.get("OMNITRACE_GRID"):
            r, c = (int(x) for x in g.lower().split("x"))
            cfg = replace(cfg, vision=replace(cfg.vision, grid_rows=r, grid_cols=c))
        return cfg


DEFAULT_CONFIG = OmnitraceConfig()
