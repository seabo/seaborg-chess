"""Model configuration for ChessFormer.

The default (`CF_6M`) reproduces the small model from the paper exactly:
8 encoder layers, embedding depth 256, 8 heads, feed-forward depth 256 (~6M params).
Everything is configurable so larger variants (e.g. CF-240M) can be instantiated by
overriding the relevant fields.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Fixed problem-geometry constants (these are properties of chess, not the model)
# ---------------------------------------------------------------------------
N_SQUARES = 64                 # board tokens (one per square)
N_MOVES = N_SQUARES * N_SQUARES  # 4096 (from-square, to-square) move classes
MAX_REL = 7                    # max horizontal/vertical displacement on an 8x8 board
N_REL = (2 * MAX_REL + 1) ** 2  # 15*15 = 225 distinct relative positions


@dataclass
class ChessFormerConfig:
    # --- encoder ---
    n_layer: int = 8
    n_embd: int = 256
    n_head: int = 8
    d_ff: int = 256            # feed-forward hidden depth (paper's CF-6M uses 256, == n_embd)

    # --- input ---
    n_features: int = 18       # per-square feature vector length (see encoding.py)

    # --- output heads ---
    value_hidden: int = 256    # hidden width of the value MLP

    # --- regularization / misc ---
    dropout: float = 0.0

    # --- relative position encoding ---
    use_rel_value: bool = True  # apply Shaw relative encoding to the value path (a^V) too

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def describe(self) -> str:
        return (
            f"ChessFormer(layers={self.n_layer}, d={self.n_embd}, heads={self.n_head}, "
            f"d_ff={self.d_ff}, head_dim={self.head_dim})"
        )


# Paper's small model (~6M params). Default.
CF_6M = ChessFormerConfig(n_layer=8, n_embd=256, n_head=8, d_ff=256)

# ~22M capacity bump: deeper (12) and wider (384), keeping head_dim=32 and the 4x FFN
# ratio. Trains from scratch on an 8GB card at batch ~256 with --compile.
CF_22M = ChessFormerConfig(n_layer=12, n_embd=384, n_head=12, d_ff=1536, value_hidden=384)

# Equal-parameter MLP baseline (~6.8M) for the architecture ablation (used with --arch mlp).
# Reuses the config fields: n_embd = MLP width, d_ff = block hidden, n_layer = #blocks.
MLP_7M = ChessFormerConfig(n_layer=4, n_embd=384, n_head=8, d_ff=1536, value_hidden=384)

# ~100M scaling probe for the cloud (1x H100). Same proportions: head_dim=32, 4x FFN.
CF_100M = ChessFormerConfig(n_layer=14, n_embd=768, n_head=24, d_ff=3072, value_hidden=768)

# Tiny preset for fast pipeline smoke-tests.
CF_TINY = ChessFormerConfig(n_layer=2, n_embd=64, n_head=4, d_ff=64, value_hidden=64)

# Paper's large model (~240M params); included for completeness — will not fit an 8GB GPU
# at a large batch size, but documents the configurable knobs.
CF_240M = ChessFormerConfig(n_layer=15, n_embd=1024, n_head=32, d_ff=4096, value_hidden=1024)
