"""Inference engine: load a trained checkpoint and pick moves for a position.

Two move-selection modes:
  * "policy"  — use the policy head directly (argmax, or temperature-sampled). This is the
                model "playing itself" — the most direct view of what it learned.
  * "value"   — 1-ply search: try every legal move, score the resulting position with the
                value head (negamax: our score = -opponent's value), pick the best. Slower
                (one batched forward over all children) but usually stronger, since the
                value head is the better-trained part of this model.

Handles board orientation (the model always sees a White-to-move-like board) and un-mirrors
the chosen move back into the real frame.
"""

from __future__ import annotations

import math
from typing import List, Optional

import chess
import numpy as np
import torch

from .config import ChessFormerConfig
from .encoding import (
    CP_SCALE,
    encode_board,
    index_to_move,
    legal_move_mask,
    mirror_move,
    move_to_index,
    orient,
)
from .model import ChessFormer


def q_to_cp(q: float) -> int:
    """Side-to-move expected score q in (-1, 1) -> centipawns (inverse of the eval->q map)."""
    p = min(max((q + 1.0) / 2.0, 1e-6), 1 - 1e-6)  # win probability
    return int(round(math.log(p / (1.0 - p)) / CP_SCALE))


# piece values for quiescence MVV capture ordering
_PIECE_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
              chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


class ChessFormerEngine:
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = ChessFormerConfig(**ckpt["config"])
        self.step = ckpt.get("step", -1)
        self.arch = ckpt.get("arch", "transformer")
        if self.arch == "mlp":
            from .mlp import ChessMLP
            self.model = ChessMLP(self.config).to(self.device)
        else:
            self.model = ChessFormer(self.config).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    # --- low-level batched forward ---
    @torch.no_grad()
    def _forward(self, boards: List[chess.Board], with_mask: bool):
        feats = np.stack([encode_board(orient(b)[0]) for b in boards])
        x = torch.from_numpy(feats).to(self.device)
        mask = None
        if with_mask:
            m = np.stack([legal_move_mask(orient(b)[0]) for b in boards])
            mask = torch.from_numpy(m).to(self.device)
        return self.model(x, mask)

    @torch.no_grad()
    def evaluate(self, board: chess.Board) -> dict:
        """Return {'q', 'cp', 'wdl'} from the side-to-move's perspective."""
        _, wdl_logits, value = self._forward([board], with_mask=False)
        q = float(value[0])
        wdl = torch.softmax(wdl_logits[0], dim=-1).tolist()
        return {"q": q, "cp": q_to_cp(q), "wdl": wdl}

    # --- move selection ---
    @torch.no_grad()
    def select_move(self, board: chess.Board, mode: str = "policy",
                    temperature: float = 0.0, depth: int = 3, width: int = 4,
                    qdepth: int = 6) -> dict:
        if not list(board.legal_moves):
            return {"move": None, "cp": None, "info": "no legal moves"}
        if mode == "value":
            return self._select_value(board)
        if mode == "search":
            return self._select_search(board, depth, width, qdepth)
        return self._select_policy(board, temperature)

    def _select_policy(self, board, temperature):
        oriented, flipped = orient(board)
        policy_logits, _, value = self._forward([board], with_mask=True)
        logits = policy_logits[0]  # (4096,), illegal == -inf

        if temperature and temperature > 1e-6:
            probs = torch.softmax(logits / temperature, dim=-1)
            idx = int(torch.multinomial(probs, 1))
        else:
            idx = int(torch.argmax(logits))

        oriented_move = index_to_move(idx, oriented)
        if oriented_move is None:  # safety net — fall back to any legal move
            oriented_move = next(iter(oriented.legal_moves))
        move = mirror_move(oriented_move) if flipped else oriented_move
        return {"move": move, "cp": q_to_cp(float(value[0])), "info": "policy"}

    def _select_value(self, board):
        moves = list(board.legal_moves)
        children, scores = [], [None] * len(moves)
        idx_map = []
        for i, m in enumerate(moves):
            board.push(m)
            if board.is_checkmate():
                scores[i] = math.inf            # we deliver mate
            elif board.is_stalemate() or board.is_insufficient_material() \
                    or board.can_claim_draw():
                scores[i] = 0.0                 # forced draw
            else:
                children.append(board.copy(stack=False))
                idx_map.append(i)
            board.pop()

        if children:
            _, _, child_vals = self._forward(children, with_mask=False)
            for j, i in enumerate(idx_map):
                scores[i] = -float(child_vals[j])  # negamax: our score = -opponent value

        best_i = max(range(len(moves)), key=lambda i: scores[i])
        best_q = scores[best_i] if math.isfinite(scores[best_i]) else 0.999
        cp = 100000 if scores[best_i] == math.inf else q_to_cp(best_q)
        return {"move": moves[best_i], "cp": cp, "info": "value-1ply"}

    # --- policy-pruned alpha-beta (negamax) -----------------------------------------
    # Uses the value head at the leaves and the policy head for move ordering AND forward
    # pruning (only the top-`width` policy moves are searched at each node). Not sound
    # alpha-beta — it trusts the policy prior to limit branching — but that's the point:
    # the NN eval is slow, so we lean on the 85%-top-5 policy to search only plausible
    # moves. Sequential (one forward per node), so keep depth/width small.
    def _ordered_moves(self, board, pol, width):
        oriented, flipped = orient(board)

        def oidx(mv):
            om = mirror_move(mv) if flipped else mv
            return move_to_index(om)

        moves = sorted(board.legal_moves, key=lambda mv: pol[oidx(mv)], reverse=True)
        return moves[:width] if width and width > 0 else moves

    @torch.no_grad()
    def _value_of(self, board):
        _, _, value = self._forward([board], with_mask=False)
        return float(value[0])

    def _ordered_captures(self, board):
        """Legal captures, ordered most-valuable-victim first (cheap MVV ordering)."""
        scored = []
        for mv in board.generate_legal_captures():
            victim = 1 if board.is_en_passant(mv) else _PIECE_VAL.get(
                board.piece_at(mv.to_square).piece_type, 0)
            scored.append((victim, mv))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [mv for _, mv in scored]

    @torch.no_grad()
    def _qsearch(self, board, alpha, beta, qdepth):
        """Quiescence: extend only forcing lines (captures; all evasions when in check)
        until quiet, so the value head is evaluated on settled positions — fixes the
        horizon effect where the main search stops mid-capture-sequence."""
        if board.is_checkmate():
            return -1.0
        if not any(board.legal_moves) or board.is_insufficient_material():
            return 0.0
        if board.is_check():
            # in check you can't "pass" — must consider every evasion (no stand-pat)
            if qdepth <= 0:
                return self._value_of(board)
            moves, best = list(board.legal_moves), -math.inf
        else:
            stand_pat = self._value_of(board)        # baseline: you may decline to capture
            if stand_pat >= beta or qdepth <= 0:
                return stand_pat
            alpha = max(alpha, stand_pat)
            moves = self._ordered_captures(board)
            if not moves:
                return stand_pat
            best = stand_pat
        for mv in moves:
            board.push(mv)
            score = -self._qsearch(board, -beta, -alpha, qdepth - 1)
            board.pop()
            best = max(best, score)
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best

    @torch.no_grad()
    def _negamax(self, board, depth, alpha, beta, width, qdepth):
        if board.is_checkmate():
            return -1.0                       # side to move is mated
        if not any(board.legal_moves) or board.is_insufficient_material():
            return 0.0                        # stalemate / dead draw
        # Repetition / 50-move are invisible to the value head (FEN carries no history), so
        # the search must inject them or the engine shuffles won positions into a threefold.
        # is_repetition(2) scores the FIRST repetition in-tree as a draw (strong engines do
        # this): when winning, q>0 beats a 0 draw so it steers away; when losing, 0 beats q<0
        # so it steers toward it. Every _negamax node is >=1 ply below the root, so the root
        # (which may itself already be a twofold) is never wrongly flagged.
        if board.is_repetition(2) or board.halfmove_clock >= 100:
            return 0.0
        if depth <= 0:
            return self._qsearch(board, alpha, beta, qdepth)   # leaf -> quiescence
        policy_logits, _, _ = self._forward([board], with_mask=True)
        pol = policy_logits[0].float().cpu().numpy()
        best = -math.inf
        for mv in self._ordered_moves(board, pol, width):
            board.push(mv)
            score = -self._negamax(board, depth - 1, -beta, -alpha, width, qdepth)
            board.pop()
            best = max(best, score)
            alpha = max(alpha, best)
            if alpha >= beta:
                break                          # cutoff
        return best

    @torch.no_grad()
    def _select_search(self, board, depth, width, qdepth):
        policy_logits, _, _ = self._forward([board], with_mask=True)
        pol = policy_logits[0].float().cpu().numpy()
        best_score, best_move = -math.inf, None
        alpha = -math.inf
        for mv in self._ordered_moves(board, pol, width):
            board.push(mv)
            score = -self._negamax(board, depth - 1, -math.inf, -alpha, width, qdepth)
            board.pop()
            if score > best_score:
                best_score, best_move = score, mv
            alpha = max(alpha, best_score)
        if best_move is None:
            best_move = next(iter(board.legal_moves))
        cp = q_to_cp(best_score) if math.isfinite(best_score) else 0
        return {"move": best_move, "cp": cp, "info": f"search d{depth} w{width} q{qdepth}"}
