"""Encoding of chess positions, moves, and Stockfish evaluations into tensors.

Design notes / deviations from the paper (arXiv:2409.12272), all driven by the fact
that the Lichess dataset gives us *isolated* Stockfish-evaluated positions rather than
the paper's self-play *games* with MCTS visit distributions:

  * Input representation. The paper uses 64 tokens, each a 112-dim vector that stacks
    the current + previous 7 positions plus repetition/clock/castling info. The Lichess
    rows are single positions with no history, so we use a single-position 18-dim
    per-token feature vector (12 piece planes + 4 castling + 1 en-passant + 1 clock).
    The 64-token-per-square structure is identical to the paper.

  * Board orientation. As is standard (and as the paper does), the board is always
    oriented so the side to move is "us" at the bottom. When it is Black to move we
    mirror the board vertically and swap colours (python-chess `board.mirror()`), so
    the network always sees a "white-to-move-like" position. Move targets and the legal
    mask are produced in this oriented frame; callers un-mirror at inference time.

  * Policy target. The paper learns a soft MCTS visit distribution. We only have the
    Stockfish principal variation, so the policy target is the *single best move* (first
    move of `line`) as a hard class over the 4096 (from, to) square pairs scored by the
    attention policy head. Promotions collapse to their (from, to) pair (queen-promotion
    by default); under-promotions therefore share a class with queen-promotion. Illegal
    moves are always masked, as in the paper.

  * Value targets. `cp`/`mate` are converted to a side-to-move expected score q in
    [-1, 1] (logistic in centipawns, ±1 for mate). We train the paper's WDL head
    (soft win/draw/loss target derived from q) and L2 scalar head (predicts q). The
    paper's additional categorical-reward and value-error heads require self-play reward
    data that this dataset does not contain, so they are omitted.

Centipawns in the Lichess dataset are stored from *White's* perspective; we convert to
the side-to-move perspective here.
"""

from __future__ import annotations

import math
from typing import Optional

import chess
import numpy as np

from .config import N_SQUARES, N_MOVES, MAX_REL

# --- feature layout (per square / token) ---------------------------------------------
# 0..5   : our pieces      (P, N, B, R, Q, K)
# 6..11  : their pieces     (P, N, B, R, Q, K)
# 12..15 : castling rights  (our K-side, our Q-side, their K-side, their Q-side)  [broadcast]
# 16     : en-passant target square marker
# 17     : halfmove clock / 100                                                   [broadcast]
N_FEATURES = 18

# piece_type (1..6) -> plane offset (0..5)
_PIECE_PLANE = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
}

# Lichess centipawns -> logistic scale (the conversion Lichess itself uses for win%).
CP_SCALE = 0.00368208
# q value assigned to a forced mate (slightly inside [-1, 1] so a tanh head can hit it).
MATE_Q = 0.9999


# ---------------------------------------------------------------------------
# Relative-position index map (Shaw et al.)
# ---------------------------------------------------------------------------
def build_rel_index() -> np.ndarray:
    """Return an int64 [64, 64] table mapping (square_i, square_j) -> relative-position id.

    Two pairs share an id iff they have the same horizontal AND vertical displacement,
    giving (2*7+1)^2 = 225 distinct relative positions.
    """
    idx = np.zeros((N_SQUARES, N_SQUARES), dtype=np.int64)
    side = 2 * MAX_REL + 1  # 15
    for i in range(N_SQUARES):
        fi, ri = chess.square_file(i), chess.square_rank(i)
        for j in range(N_SQUARES):
            fj, rj = chess.square_file(j), chess.square_rank(j)
            dx = (fj - fi) + MAX_REL   # 0..14
            dy = (rj - ri) + MAX_REL   # 0..14
            idx[i, j] = dy * side + dx
    return idx


REL_INDEX = build_rel_index()


# ---------------------------------------------------------------------------
# Position -> features
# ---------------------------------------------------------------------------
def orient(board: chess.Board):
    """Orient the board so the side to move is White-like. Returns (oriented_board, flipped)."""
    if board.turn == chess.WHITE:
        return board, False
    return board.mirror(), True


def encode_board(oriented: chess.Board) -> np.ndarray:
    """Encode an *already-oriented* board (White to move) into a [64, 18] float32 array."""
    feats = np.zeros((N_SQUARES, N_FEATURES), dtype=np.float32)

    for sq, piece in oriented.piece_map().items():
        plane = _PIECE_PLANE[piece.piece_type]
        if piece.color == chess.WHITE:   # "us" (oriented => side to move is White)
            feats[sq, plane] = 1.0
        else:                            # "them"
            feats[sq, 6 + plane] = 1.0

    # castling rights (broadcast to every token)
    feats[:, 12] = 1.0 if oriented.has_kingside_castling_rights(chess.WHITE) else 0.0
    feats[:, 13] = 1.0 if oriented.has_queenside_castling_rights(chess.WHITE) else 0.0
    feats[:, 14] = 1.0 if oriented.has_kingside_castling_rights(chess.BLACK) else 0.0
    feats[:, 15] = 1.0 if oriented.has_queenside_castling_rights(chess.BLACK) else 0.0

    # en-passant target square
    if oriented.ep_square is not None:
        feats[oriented.ep_square, 16] = 1.0

    # halfmove clock (50-move rule progress), normalised
    feats[:, 17] = min(oriented.halfmove_clock, 100) / 100.0
    return feats


# ---------------------------------------------------------------------------
# Move <-> policy index
# ---------------------------------------------------------------------------
def move_to_index(move: chess.Move) -> int:
    """(from, to) square pair -> class id in [0, 4096). Promotion piece is ignored."""
    return move.from_square * N_SQUARES + move.to_square


def index_to_move(index: int, board: chess.Board) -> Optional[chess.Move]:
    """Inverse of move_to_index against a concrete board (re-attaches a legal promotion)."""
    frm, to = divmod(index, N_SQUARES)
    candidate = chess.Move(frm, to)
    if candidate in board.legal_moves:
        return candidate
    # try promotions (queen first) for pawn-to-last-rank moves
    for promo in (chess.QUEEN, chess.KNIGHT, chess.ROOK, chess.BISHOP):
        m = chess.Move(frm, to, promotion=promo)
        if m in board.legal_moves:
            return m
    return None


def mirror_move(move: chess.Move) -> chess.Move:
    """Vertically mirror a move's squares (for the Black-to-move orientation flip)."""
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )


def legal_move_mask(oriented: chess.Board) -> np.ndarray:
    """Boolean [4096] mask: True where the (from, to) move class is legal in this position."""
    mask = np.zeros(N_MOVES, dtype=bool)
    for m in oriented.legal_moves:
        mask[move_to_index(m)] = True
    return mask


# ---------------------------------------------------------------------------
# Stockfish eval -> value targets
# ---------------------------------------------------------------------------
def eval_to_q(cp: Optional[int], mate: Optional[int], white_to_move: bool) -> Optional[float]:
    """Convert a White-POV cp/mate eval into a side-to-move expected score q in [-1, 1]."""
    if mate is not None:
        q_white = MATE_Q if mate > 0 else -MATE_Q
    elif cp is not None:
        # logistic win-prob in [0,1] -> expected score in [-1,1]
        q_white = 2.0 / (1.0 + math.exp(-CP_SCALE * float(cp))) - 1.0
    else:
        return None
    return q_white if white_to_move else -q_white


def move_eval_to_q(cp: Optional[int], mate: Optional[int]) -> float:
    """Convert an already-side-to-move-POV cp/mate (as stored in the action-value dataset)
    into an expected score q in [-1, 1]. Higher = better for the side to move."""
    if mate is not None:
        return MATE_Q if mate > 0 else -MATE_Q
    if cp is not None:
        return 2.0 / (1.0 + math.exp(-CP_SCALE * float(cp))) - 1.0
    return 0.0


def q_to_wdl(q: float, draw_max: float = 0.5) -> np.ndarray:
    """Soft (win, draw, loss) target from expected score q, side-to-move POV.

    Uses a simple drawiness model: P(draw) = draw_max * (1 - |q|), and
    P(win) - P(loss) = q with P(win) + P(draw) + P(loss) = 1. Always non-negative.
    """
    p_draw = draw_max * (1.0 - abs(q))
    p_win = (1.0 - p_draw + q) / 2.0
    p_loss = (1.0 - p_draw - q) / 2.0
    return np.array([p_win, p_draw, p_loss], dtype=np.float32)
