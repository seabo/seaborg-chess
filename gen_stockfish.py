#!/usr/bin/env python3
"""Generate action-value training data: for sampled positions, record Stockfish's
evaluation of each of the top-K moves (MultiPV). These rich targets let the policy learn
*which moves hang material* directly from the labels (the DeepMind "searchless" recipe),
rather than only the single best move we get from the Lichess dataset.

Output: one row per position
    fen   : str                 position (FEN, original frame)
    moves : list[str]           UCI of each analysed move (best-first)
    cp    : list[int16|null]    centipawn eval of that move, side-to-move POV (null if mate)
    mate  : list[int8|null]     mate-in-N of that move, side-to-move POV (null if not mate)

CPU-only (Stockfish). Parallel over positions; each chunk-task opens its own single-thread
Stockfish, labels ~chunk-size positions, and writes a self-contained parquet file (so a
crash loses only in-flight chunks, and re-running skips FENs already on disk).

    python gen_stockfish.py --num-positions 1000000 --nodes 500000 --multipv 16 --workers 10

Throughput (Ryzen 3900XT, measured): ~2.6 pos/s/proc @ 500k nodes (depth ~16),
~6.4 @ 200k (depth ~13). 10 workers @ 500k ≈ 26 pos/s ≈ 1M positions in ~11h.
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from multiprocessing import Pool

import chess
import chess.engine
import pyarrow as pa
import pyarrow.parquet as pq

from chessformer.data import _normalize_fen

SF_DEFAULT = "tools/stockfish-bin"
SCHEMA = pa.schema([
    ("fen", pa.string()),
    ("moves", pa.list_(pa.string())),
    ("cp", pa.list_(pa.int16())),
    ("mate", pa.list_(pa.int8())),
])

# globals set per worker process (via Pool initializer) so we open Stockfish once per worker
_ENG = None
_CFG = None


def _init_worker(cfg):
    global _ENG, _CFG
    _CFG = cfg
    _ENG = chess.engine.SimpleEngine.popen_uci(cfg["sf"])
    _ENG.configure({"Threads": 1, "Hash": cfg["hash_mb"]})


def _label_chunk(task):
    chunk_id, fens = task
    eng, cfg = _ENG, _CFG
    limit = chess.engine.Limit(nodes=cfg["nodes"])
    out_fen, out_moves, out_cp, out_mate = [], [], [], []
    for fen in fens:
        try:
            board = chess.Board(_normalize_fen(fen))
        except (ValueError, IndexError):
            continue
        if board.is_game_over() or not any(board.legal_moves):
            continue  # terminal positions have no policy target
        try:
            infos = eng.analyse(board, limit, multipv=cfg["multipv"])
        except chess.engine.EngineError:
            continue
        moves, cps, mates = [], [], []
        for info in infos:
            pv = info.get("pv")
            if not pv:
                continue
            rel = info["score"].relative  # side-to-move perspective
            moves.append(pv[0].uci())
            cps.append(rel.score())   # int or None (None when mate)
            mates.append(rel.mate())  # int or None
        if not moves:
            continue
        out_fen.append(fen); out_moves.append(moves); out_cp.append(cps); out_mate.append(mates)

    if out_fen:
        table = pa.table(
            {"fen": out_fen, "moves": out_moves, "cp": out_cp, "mate": out_mate}, schema=SCHEMA
        )
        path = os.path.join(cfg["out"], f"av_{cfg['run']}_{chunk_id:06d}.parquet")
        pq.write_table(table, path)
    return len(out_fen)


def sample_fens(src, n, seed):
    """Reservoir-sample ~n distinct, column-pruned FENs from the dataset (read-only, fast)."""
    import duckdb
    con = duckdb.connect()
    con.execute("SET threads=4")
    g = os.path.join(src, "data", "data_*.parquet")
    rows = con.execute(
        f"SELECT fen FROM read_parquet('{g}') USING SAMPLE reservoir({int(n * 1.3)} ROWS) REPEATABLE({seed})"
    ).fetchall()
    seen, out = set(), []
    for (f,) in rows:
        if f not in seen:
            seen.add(f); out.append(f)
        if len(out) >= n:
            break
    return out


def already_done(out_dir):
    done = set()
    for p in glob.glob(os.path.join(out_dir, "*.parquet")):
        try:
            done |= set(pq.read_table(p, columns=["fen"]).column("fen").to_pylist())
        except Exception:
            pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/lichess-evals", help="dataset to sample positions from")
    ap.add_argument("--out", default="data/action-values")
    ap.add_argument("--num-positions", type=int, default=1_000_000)
    ap.add_argument("--nodes", type=int, default=500_000, help="Stockfish node budget per position")
    ap.add_argument("--multipv", type=int, default=16, help="top-K moves to evaluate per position")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--hash-mb", type=int, default=64)
    ap.add_argument("--sf", default=SF_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"sampling {args.num_positions:,} candidate FENs from {args.src} ...", flush=True)
    fens = sample_fens(args.src, args.num_positions, args.seed)
    done = already_done(args.out)
    todo = [f for f in fens if f not in done]
    print(f"  {len(fens):,} sampled, {len(done):,} already labelled, {len(todo):,} to do", flush=True)
    if not todo:
        print("nothing to do."); return

    run = str(os.getpid())  # unique tag so re-runs don't clobber prior shards
    cfg = {"sf": args.sf, "nodes": args.nodes, "multipv": args.multipv,
           "hash_mb": args.hash_mb, "out": args.out, "run": run}
    n_chunks = (len(todo) + args.chunk_size - 1) // args.chunk_size
    tasks = [(i, todo[i * args.chunk_size:(i + 1) * args.chunk_size]) for i in range(n_chunks)]

    print(f"labelling with {args.workers} workers @ {args.nodes:,} nodes, MultiPV={args.multipv} ...", flush=True)
    t0 = time.time()
    n_done = 0
    with Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for cnt in pool.imap_unordered(_label_chunk, tasks):
            n_done += cnt
            el = time.time() - t0
            rate = n_done / el if el > 0 else 0
            eta = (len(todo) - n_done) / rate / 3600 if rate > 0 else 0
            print(f"  {n_done:>8,}/{len(todo):,}  |  {rate:5.1f} pos/s  |  ETA {eta:4.1f}h", flush=True)

    print(f"done: labelled {n_done:,} positions in {(time.time()-t0)/3600:.2f}h -> {args.out}/", flush=True)


if __name__ == "__main__":
    main()
