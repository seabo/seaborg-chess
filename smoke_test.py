"""Quick self-checks for the ChessFormer pipeline. Runs without the full dataset;
if at least one parquet shard is present it also validates real-row encoding and the
centipawn perspective convention.

    python smoke_test.py
"""

from __future__ import annotations

import glob
import os

import chess
import numpy as np
import torch

from chessformer import encoding
from chessformer.config import CF_TINY, N_MOVES
from chessformer.data import encode_row, find_shards
from chessformer.losses import LossWeights, chessformer_loss
from chessformer.model import ChessFormer


def check_encoding():
    print("== encoding ==")
    b = chess.Board()  # startpos, white to move
    oriented, flipped = encoding.orient(b)
    assert not flipped
    feats = encoding.encode_board(oriented)
    assert feats.shape == (64, encoding.N_FEATURES)
    # 32 pieces on the board -> 32 one-hot piece entries in planes 0..11
    assert feats[:, :12].sum() == 32, feats[:, :12].sum()
    # white pawns ("us") on rank 2
    for sq in chess.SquareSet(chess.BB_RANK_2):
        assert feats[sq, 0] == 1.0
    print("  startpos features OK")

    # orientation: black to move mirrors so side-to-move is always "us"
    b2 = chess.Board()
    b2.push_san("e4")  # now black to move
    oriented2, flipped2 = encoding.orient(b2)
    assert flipped2 and oriented2.turn == chess.WHITE
    feats2 = encoding.encode_board(oriented2)
    assert feats2[:, :12].sum() == 32
    print("  black-to-move orientation OK")

    # move round-trip through the policy index in the oriented frame
    for fen, uci in [
        (chess.STARTING_FEN, "e2e4"),
        ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "f1b5"),
    ]:
        board = chess.Board(fen)
        mv = chess.Move.from_uci(uci)
        oriented, flipped = encoding.orient(board)
        om = encoding.mirror_move(mv) if flipped else mv
        idx = encoding.move_to_index(om)
        back = encoding.index_to_move(idx, oriented)
        assert back is not None and back.from_square == om.from_square
        assert encoding.legal_move_mask(oriented)[idx]
    print("  move<->index round-trip OK")

    # value conversions
    assert abs(encoding.eval_to_q(0, None, True)) < 1e-6           # equal -> 0
    assert encoding.eval_to_q(500, None, True) > 0.5               # white better, white to move
    assert encoding.eval_to_q(500, None, False) < -0.5            # white better, black to move -> bad for stm
    wdl = encoding.q_to_wdl(0.0)
    assert np.allclose(wdl.sum(), 1.0) and abs(wdl[0] - wdl[2]) < 1e-6
    wdl_win = encoding.q_to_wdl(0.9)
    assert wdl_win[0] > wdl_win[2]
    print("  cp/mate -> q -> WDL OK")


def check_rel_index():
    print("== relative-position index ==")
    idx = encoding.REL_INDEX
    assert idx.shape == (64, 64)
    # a square relative to itself is the same id everywhere (dx=dy=0)
    diag = np.unique(np.diagonal(idx))
    assert len(diag) == 1
    print(f"  {len(np.unique(idx))} distinct relative positions (expect 225)")
    assert len(np.unique(idx)) == 225


def check_model():
    print("== model forward/backward ==")
    torch.manual_seed(0)
    model = ChessFormer(CF_TINY)
    print(f"  tiny model params: {model.num_params()/1e6:.3f}M")
    B = 8
    feats = torch.randn(B, 64, encoding.N_FEATURES)
    mask = torch.zeros(B, N_MOVES, dtype=torch.bool)
    # give every example a handful of legal moves incl. the target
    targets = torch.randint(0, N_MOVES, (B,))
    for i in range(B):
        legal = torch.randint(0, N_MOVES, (10,))
        mask[i, legal] = True
        mask[i, targets[i]] = True
    batch = {
        "features": feats,
        "legal_mask": mask,
        "policy_target": targets,
        "wdl_target": torch.softmax(torch.randn(B, 3), dim=-1),
        "value_target": torch.tanh(torch.randn(B)),
    }
    pol, wdl, val = model(feats, mask)
    assert pol.shape == (B, N_MOVES) and wdl.shape == (B, 3) and val.shape == (B,)
    # illegal logits must be -inf so they vanish under softmax
    assert torch.isinf(pol[~mask]).all()
    loss, comp = chessformer_loss(pol, wdl, val, batch, LossWeights())
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    print(f"  forward+backward OK, loss={loss.item():.4f}, finite grads={len(grads)} tensors")


def check_real_rows():
    print("== real dataset rows ==")
    shards = find_shards("data/lichess-evals")
    # also accept partially-downloaded blobs
    if not shards:
        shards = glob.glob("data/lichess-evals/**/*.parquet", recursive=True)
    if not shards:
        print("  (no parquet shard available yet — skipping)")
        return
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(shards[0])
    batch = next(pf.iter_batches(batch_size=2000, columns=["fen", "line", "cp", "mate"]))
    d = batch.to_pydict()
    n_ok = 0
    for i in range(len(d["fen"])):
        ex = encode_row(d["fen"][i], d["line"][i], d["cp"][i], d["mate"][i])
        if ex is not None:
            n_ok += 1
            if n_ok == 1:
                print(f"  example: feats{tuple(ex['features'].shape)} "
                      f"target={ex['policy_target']} q={ex['value_target']:.3f}")
    print(f"  encoded {n_ok}/{len(d['fen'])} rows from {os.path.basename(shards[0])}")

    # --- empirically verify the cp perspective (White-POV) on a clear example ---
    # print a few high-|cp| positions with material balance so the sign can be eyeballed.
    from chessformer.data import _normalize_fen
    shown = 0
    for i in range(len(d["fen"])):
        cp = d["cp"][i]
        if cp is None or abs(cp) < 400:
            continue
        b = chess.Board(_normalize_fen(d["fen"][i]))
        mat = material_balance(b)  # +ve => white has more material
        q = encoding.eval_to_q(cp, None, b.turn == chess.WHITE)
        print(f"  perspective sample: cp={cp:+d} (White POV), "
              f"material(white-black)={mat:+d}, to_move={'white' if b.turn else 'black'}, "
              f"stm_q={q:+.3f}")
        shown += 1
        if shown == 4:
            break


def material_balance(board: chess.Board) -> int:
    vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    total = 0
    for piece in board.piece_map().values():
        v = vals.get(piece.piece_type, 0)
        total += v if piece.color == chess.WHITE else -v
    return total


if __name__ == "__main__":
    check_encoding()
    check_rel_index()
    check_model()
    check_real_rows()
    print("\nAll smoke tests passed.")
