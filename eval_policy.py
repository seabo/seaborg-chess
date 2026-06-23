#!/usr/bin/env python3
"""Evaluate a checkpoint's policy move-quality on the action-value val split.

Unlike plain move-match, this uses the per-move q-values in the action-value data to
measure *how much the model's chosen move actually costs* vs the best move — a search-free
proxy for playing strength. Compare a hard-target checkpoint vs a soft-policy one to see
whether the fine-tune helped.

Metrics (over val positions with >= --min-moves analysed moves):
    top1        argmax move == best (highest-q) analysed move
    top5        best move within the model's top-5
    win%-loss   mean expected-score the chosen move concedes vs best (lower = better),
                in win-probability points; computed when the chosen move is on the list
    off-list    fraction where the chosen move isn't among the analysed moves
    blunder     fraction where win%-loss > --blunder-pct, or off-list
    value MSE   value head vs the position's q (best move's q)

    python eval_policy.py --checkpoint checkpoints/softpolicy/chessformer_latest.pt \
        --data-dir data/action-values-raw --min-moves 2 --num-positions 200000

Run with --device cpu (or while the GPU is free) to avoid contending with a live train run.
"""

from __future__ import annotations

import argparse
import glob
import os

import chess
import numpy as np
import torch

import pyarrow.parquet as pq
from chessformer import encoding
from chessformer.data import _normalize_fen
from chessformer.inference import ChessFormerEngine


def iter_val_rows(data_dir, min_moves, limit):
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True))
    n = 0
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=8192, columns=["fen", "moves", "cp", "mate", "is_val"]):
            d = batch.to_pydict()
            for i in range(len(d["fen"])):
                if not d["is_val"][i] or len(d["moves"][i]) < min_moves:
                    continue
                yield d["fen"][i], d["moves"][i], d["cp"][i], d["mate"][i]
                n += 1
                if limit and n >= limit:
                    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default="data/action-values-raw")
    ap.add_argument("--min-moves", type=int, default=2, help="only eval positions with >= this many analysed moves")
    ap.add_argument("--num-positions", type=int, default=200_000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--blunder-pct", type=float, default=10.0, help="win%% loss above which a move counts as a blunder")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    eng = ChessFormerEngine(args.checkpoint, device=args.device)
    model, device = eng.model, eng.device
    amp = device.type == "cuda"
    print(f"checkpoint step {eng.step} | {eng.config.describe()} | device={device}")

    feats_buf, mask_buf, meta = [], [], []
    agg = {"n": 0, "top1": 0, "top5": 0, "onlist": 0, "offlist": 0,
           "winloss_sum": 0.0, "blunder": 0, "vse": 0.0}

    def flush():
        if not feats_buf:
            return
        x = torch.from_numpy(np.stack(feats_buf)).to(device)
        m = torch.from_numpy(np.stack(mask_buf)).to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            pol, _, val = model(x, m)
        top5 = pol.topk(5, dim=-1).indices.cpu().numpy()
        val = val.float().cpu().numpy()
        for i, (idxs, qs) in enumerate(meta):
            best = int(qs.argmax())
            best_idx, q_best = int(idxs[best]), float(qs[best])
            pred = int(top5[i, 0])
            agg["n"] += 1
            agg["top1"] += pred == best_idx
            agg["top5"] += best_idx in top5[i]
            agg["vse"] += (float(val[i]) - q_best) ** 2
            hit = np.where(idxs == pred)[0]
            if len(hit):  # chosen move is on the analysed list
                winloss = (q_best - float(qs[hit[0]])) * 50.0  # Δq -> win-probability points
                agg["onlist"] += 1
                agg["winloss_sum"] += winloss
                agg["blunder"] += winloss > args.blunder_pct
            else:
                agg["offlist"] += 1
                agg["blunder"] += 1  # off-list move counts as a blunder
        feats_buf.clear(); mask_buf.clear(); meta.clear()

    for fen, moves, cps, mates in iter_val_rows(args.data_dir, args.min_moves, args.num_positions):
        try:
            board = chess.Board(_normalize_fen(fen))
        except (ValueError, IndexError):
            continue
        oriented, flipped = encoding.orient(board)
        legal = encoding.legal_move_mask(oriented)
        idxs, qs = [], []
        for mv, cp, mt in zip(moves, cps, mates):
            try:
                mm = chess.Move.from_uci(mv)
            except (ValueError, IndexError):
                continue
            if flipped:
                mm = encoding.mirror_move(mm)
            idx = encoding.move_to_index(mm)
            if not legal[idx]:
                continue
            idxs.append(idx); qs.append(encoding.move_eval_to_q(cp, mt))
        if len(idxs) < args.min_moves:
            continue
        feats_buf.append(encoding.encode_board(oriented))
        mask_buf.append(legal)
        meta.append((np.asarray(idxs), np.asarray(qs, dtype=np.float32)))
        if len(feats_buf) >= args.batch_size:
            flush()
    flush()

    n = max(1, agg["n"])
    onl = max(1, agg["onlist"])
    print(f"\nevaluated {agg['n']:,} val positions (>= {args.min_moves} moves)")
    print(f"  top-1 move-match : {100*agg['top1']/n:5.2f}%")
    print(f"  top-5 move-match : {100*agg['top5']/n:5.2f}%")
    print(f"  win%-loss/move   : {agg['winloss_sum']/onl:5.2f} pts   (mean, on-list moves; lower=better)")
    print(f"  off-list rate    : {100*agg['offlist']/n:5.2f}%   (chose a move outside the analysed set)")
    print(f"  blunder rate     : {100*agg['blunder']/n:5.2f}%   (>{args.blunder_pct:.0f} pts lost, or off-list)")
    print(f"  value-head MSE   : {agg['vse']/n:.4f}")


if __name__ == "__main__":
    main()
