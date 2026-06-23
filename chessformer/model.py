"""ChessFormer model (arXiv:2409.12272).

Faithful to the paper's described architecture:
  * 64 tokens (one per square); token embedding = linear projection of the per-square
    feature vector, then a per-token learned affine ("adding and multiplying by learned
    offset vectors which are separate across tokens and depth").
  * Encoder blocks with Shaw et al. relative position encoding (a^Q, a^K, a^V indexed by
    horizontal/vertical displacement), Mish feed-forward, post-LN with the DeepNorm
    residual scaling + init scheme, and no-centering/no-bias normalization (RMSNorm).
  * Attention-based policy head: query = source-square embedding, key = destination-square
    embedding, move logits = scaled dot product -> [B, 4096]; illegal moves masked.
  * Value heads: WDL (3-way) + scalar (tanh) value, from a pooled board representation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ChessFormerConfig, N_SQUARES, N_MOVES, N_REL
from .encoding import REL_INDEX


class RMSNorm(nn.Module):
    """LayerNorm with centering and bias omitted (per the paper)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class RelativeAttention(nn.Module):
    """Multi-head self-attention with Shaw-style relative position encodings.

    logit_ij = (q_i . k_j + q_i . a^K_ij + k_j . a^Q_ij) / sqrt(d)
    out_i    = sum_j softmax(logit)_ij (v_j + a^V_ij)

    Relative-position embedding tables have shape [N_REL, head_dim] and are shared across
    heads (as in Shaw et al.). The [64, 64] -> rel-id map is a fixed buffer.
    """

    def __init__(self, config: ChessFormerConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.use_rel_value = config.use_rel_value

        # QKV without bias (paper omits QKV biases).
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        self.rel_q = nn.Parameter(torch.zeros(N_REL, self.head_dim))
        self.rel_k = nn.Parameter(torch.zeros(N_REL, self.head_dim))
        self.rel_v = nn.Parameter(torch.zeros(N_REL, self.head_dim))
        nn.init.normal_(self.rel_q, std=0.02)
        nn.init.normal_(self.rel_k, std=0.02)
        nn.init.normal_(self.rel_v, std=0.02)

        self.register_buffer("rel_index", torch.from_numpy(REL_INDEX), persistent=False)

    def forward(self, x):
        B, T, C = x.shape  # T == 64
        H, d = self.n_head, self.head_dim

        q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)  # (B, H, T, d)
        k = self.k_proj(x).view(B, T, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, d).transpose(1, 2)

        # gather relative-position embeddings into [T, T, d]
        a_q = self.rel_q[self.rel_index]   # (T, T, d)
        a_k = self.rel_k[self.rel_index]
        scale = 1.0 / math.sqrt(d)

        content = torch.matmul(q, k.transpose(-2, -1))          # (B, H, T, T)
        rel_k = torch.einsum("bhid,ijd->bhij", q, a_k)          # q_i . a^K_ij
        rel_q = torch.einsum("bhjd,ijd->bhij", k, a_q)          # k_j . a^Q_ij
        att = (content + rel_k + rel_q) * scale
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = torch.matmul(att, v)                              # (B, H, T, d)
        if self.use_rel_value:
            a_v = self.rel_v[self.rel_index]                    # (T, T, d)
            out = out + torch.einsum("bhij,ijd->bhid", att, a_v)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, config: ChessFormerConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, config.d_ff, bias=True)
        self.act = nn.Mish()
        self.fc2 = nn.Linear(config.d_ff, config.n_embd, bias=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class EncoderBlock(nn.Module):
    """Post-LN encoder block with DeepNorm residual scaling.

    DeepNorm: y = Norm(alpha * x + sublayer(x)), with alpha = (2N)^(1/4) for an N-layer
    encoder. The matching init gain beta = (8N)^(-1/4) is applied to the v/out/ffn weights
    in ChessFormer.apply_deepnorm_init.
    """

    def __init__(self, config: ChessFormerConfig, alpha: float):
        super().__init__()
        self.alpha = alpha
        self.attn = RelativeAttention(config)
        self.norm1 = RMSNorm(config.n_embd)
        self.ff = FeedForward(config)
        self.norm2 = RMSNorm(config.n_embd)

    def forward(self, x):
        x = self.norm1(self.alpha * x + self.attn(x))
        x = self.norm2(self.alpha * x + self.ff(x))
        return x


class PolicyHead(nn.Module):
    """Attention-based move scorer: logits[i, j] = <Q(src_i), K(dst_j)> / sqrt(d)."""

    def __init__(self, config: ChessFormerConfig):
        super().__init__()
        self.embed = nn.Linear(config.n_embd, config.n_embd)
        self.act = nn.Mish()
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.scale = 1.0 / math.sqrt(config.n_embd)

    def forward(self, tokens, legal_mask=None):
        p = self.act(self.embed(tokens))          # (B, 64, C)
        q = self.q_proj(p)                         # source squares
        k = self.k_proj(p)                         # destination squares
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, 64, 64)
        logits = logits.reshape(logits.shape[0], N_MOVES)           # (B, 4096)
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, float("-inf"))
        return logits


class ValueHead(nn.Module):
    """Pooled board -> WDL logits (3) + scalar value (tanh)."""

    def __init__(self, config: ChessFormerConfig):
        super().__init__()
        self.proj = nn.Linear(config.n_embd, config.value_hidden)
        self.act = nn.Mish()
        self.wdl = nn.Linear(config.value_hidden, 3)
        self.value = nn.Linear(config.value_hidden, 1)

    def forward(self, tokens):
        pooled = tokens.mean(dim=1)                # (B, C)
        h = self.act(self.proj(pooled))
        wdl_logits = self.wdl(h)                   # (B, 3)
        value = torch.tanh(self.value(h)).squeeze(-1)  # (B,)
        return wdl_logits, value


class ChessFormer(nn.Module):
    def __init__(self, config: ChessFormerConfig):
        super().__init__()
        self.config = config

        # token embedding: linear projection + per-token learned affine
        self.input_proj = nn.Linear(config.n_features, config.n_embd, bias=False)
        self.tok_scale = nn.Parameter(torch.ones(N_SQUARES, config.n_embd))
        self.tok_offset = nn.Parameter(torch.zeros(N_SQUARES, config.n_embd))
        self.embed_drop = nn.Dropout(config.dropout)

        alpha = (2.0 * config.n_layer) ** 0.25
        self.blocks = nn.ModuleList([EncoderBlock(config, alpha) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.n_embd)

        self.policy_head = PolicyHead(config)
        self.value_head = ValueHead(config)

        self.apply(self._base_init)
        self._apply_deepnorm_init()

    # --- initialization ---
    def _base_init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _apply_deepnorm_init(self):
        beta = (8.0 * self.config.n_layer) ** -0.25
        for block in self.blocks:
            for lin in (block.attn.v_proj, block.attn.out_proj, block.ff.fc1, block.ff.fc2):
                nn.init.xavier_normal_(lin.weight, gain=beta)

    # --- forward ---
    def forward(self, features, legal_mask=None):
        """features: (B, 64, n_features). legal_mask: bool (B, 4096) or None.

        Returns: policy_logits (B, 4096), wdl_logits (B, 3), value (B,).
        """
        x = self.input_proj(features)
        x = x * self.tok_scale + self.tok_offset
        x = self.embed_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)

        policy_logits = self.policy_head(x, legal_mask)
        wdl_logits, value = self.value_head(x)
        return policy_logits, wdl_logits, value

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
