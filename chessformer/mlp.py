"""Equal-parameter MLP baseline for the architecture ablation.

Flattens the same oriented 64x18 board features into a 1152-vector, runs a residual MLP,
and reads out the *same* outputs as ChessFormer — masked 4096-way policy + WDL + scalar
value — but with a plain linear policy head instead of the transformer's attention head.

Same input features, same targets, same training/eval pipeline as ChessFormer, so the only
variable in a head-to-head is the architecture itself. Reuses ChessFormerConfig (n_embd =
MLP width, d_ff = block hidden width, n_layer = number of residual blocks; n_head unused).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ChessFormerConfig, N_SQUARES, N_MOVES
from .model import RMSNorm


class MLPBlock(nn.Module):
    """Pre-norm residual MLP block: x + fc2(Mish(fc1(norm(x))))."""

    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.norm = RMSNorm(width)
        self.fc1 = nn.Linear(width, hidden)
        self.act = nn.Mish()
        self.fc2 = nn.Linear(hidden, width)

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class ChessMLP(nn.Module):
    def __init__(self, config: ChessFormerConfig):
        super().__init__()
        self.config = config
        in_dim = N_SQUARES * config.n_features
        self.proj = nn.Linear(in_dim, config.n_embd)
        self.blocks = nn.ModuleList(
            [MLPBlock(config.n_embd, config.d_ff) for _ in range(config.n_layer)]
        )
        self.norm_f = RMSNorm(config.n_embd)

        self.policy_head = nn.Linear(config.n_embd, N_MOVES)   # plain 4096-way policy
        self.value_proj = nn.Linear(config.n_embd, config.value_hidden)
        self.value_act = nn.Mish()
        self.wdl = nn.Linear(config.value_hidden, 3)
        self.value = nn.Linear(config.value_hidden, 1)

        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, features, legal_mask=None):
        x = features.flatten(1)               # (B, 64*n_features)
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)

        policy_logits = self.policy_head(x)   # (B, 4096)
        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask, float("-inf"))
        h = self.value_act(self.value_proj(x))
        wdl_logits = self.wdl(h)
        value = torch.tanh(self.value(h)).squeeze(-1)
        return policy_logits, wdl_logits, value

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
