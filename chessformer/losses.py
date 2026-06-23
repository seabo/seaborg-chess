"""Combined policy + value loss for ChessFormer.

Mirrors the paper's weighted-sum objective for the heads we can train from this dataset:
    L = c_pol * policy_CE + c_wdl * wdl_CE + c_l2 * value_MSE
Defaults c_pol = c_wdl = c_l2 = 1 (the paper's vanilla-policy / WDL / L2 weights).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    policy: float = 1.0
    wdl: float = 1.0
    value: float = 1.0


def chessformer_loss(policy_logits, wdl_logits, value, batch, weights: LossWeights):
    """Return (total_loss, components_dict). `batch` provides the targets."""
    # policy: hard cross-entropy over the (masked) 4096 move classes
    policy_loss = F.cross_entropy(policy_logits, batch["policy_target"])

    # WDL: cross-entropy against the soft win/draw/loss target
    log_p = F.log_softmax(wdl_logits, dim=-1)
    wdl_loss = -(batch["wdl_target"] * log_p).sum(dim=-1).mean()

    # scalar value: MSE against expected score q
    value_loss = F.mse_loss(value, batch["value_target"])

    total = weights.policy * policy_loss + weights.wdl * wdl_loss + weights.value * value_loss
    return total, {
        "loss": total.detach(),
        "policy": policy_loss.detach(),
        "wdl": wdl_loss.detach(),
        "value": value_loss.detach(),
    }


@torch.no_grad()
def policy_accuracy(policy_logits, policy_target) -> torch.Tensor:
    """Top-1 best-move accuracy (over masked logits)."""
    pred = policy_logits.argmax(dim=-1)
    return (pred == policy_target).float().mean()


# --- soft-policy (action-value) variant -------------------------------------------------
def soft_policy_ce(policy_logits, soft_target):
    """Cross-entropy against a soft policy distribution.

    policy_logits has -inf on illegal moves; soft_target is 0 there (and on legal-but-
    unanalysed moves). We take log-probs only where the target has mass, so the illegal
    -inf entries (target 0) never produce 0 * -inf = NaN. Putting probability on the
    unanalysed/illegal moves is still penalised implicitly via the softmax normalization.
    """
    log_pred = F.log_softmax(policy_logits, dim=-1)
    log_pred = torch.where(soft_target > 0, log_pred, torch.zeros_like(log_pred))
    return -(soft_target * log_pred).sum(dim=-1).mean()


def chessformer_soft_loss(policy_logits, wdl_logits, value, batch, weights: LossWeights):
    """Combined loss with a SOFT policy target (action-value distribution)."""
    policy_loss = soft_policy_ce(policy_logits, batch["soft_target"])
    log_p = F.log_softmax(wdl_logits, dim=-1)
    wdl_loss = -(batch["wdl_target"] * log_p).sum(dim=-1).mean()
    value_loss = F.mse_loss(value, batch["value_target"])
    total = weights.policy * policy_loss + weights.wdl * wdl_loss + weights.value * value_loss
    return total, {
        "loss": total.detach(),
        "policy": policy_loss.detach(),
        "wdl": wdl_loss.detach(),
        "value": value_loss.detach(),
    }


@torch.no_grad()
def soft_policy_accuracy(policy_logits, soft_target) -> torch.Tensor:
    """Top-1: does the model's argmax match the best analysed move (argmax of the target)?"""
    return (policy_logits.argmax(dim=-1) == soft_target.argmax(dim=-1)).float().mean()
