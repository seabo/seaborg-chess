#!/usr/bin/env python3
"""Build a soft-policy / action-value dataset *for free* from the raw Lichess data.

The raw dataset already stores Stockfish MultiPV — multiple (move, eval) rows per position
(the top-K moves, each with its centipawn eval). We aggregate those into one row per FEN
holding the move distribution, which is exactly the soft-policy target we want — at the
deep search levels Lichess used (depth 24-55), no Stockfish run required.

For each FEN we pick the single richest analysis (the (depth, knodes) group with the most
moves, tie-broken by depth) so the per-move evals are mutually consistent (same search),
and store its moves with side-to-move-POV cp/mate.

Output (one row per FEN) — SAME schema as gen_stockfish.py, so the training consumer is shared:
    fen   : str
    moves : list[str]            UCI, original frame, best-first
    cp    : list[int16|null]     centipawns, SIDE-TO-MOVE POV (null if mate)
    mate  : list[int8|null]      mate-in-N, side-to-move POV (null if not mate)
    depth : int                  search depth of the chosen analysis
    is_val: bool                 leak-free FEN-hash val split (matches clean_data.py)

Memory/disk-bounded via the same hash-partition-by-FEN approach as clean_data.py.

    python build_action_values.py --src data/lichess-evals --out data/action-values-raw
"""

from __future__ import annotations

import argparse
import functools
import os
import shutil

import duckdb

print = functools.partial(print, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/lichess-evals")
    ap.add_argument("--out", default="data/action-values-raw")
    ap.add_argument("--min-moves", type=int, default=1, help="drop positions with fewer analysed moves")
    ap.add_argument("--val-mod", type=int, default=50)
    ap.add_argument("--buckets", type=int, default=32)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--memory-limit", default="10GB")
    args = ap.parse_args()

    src_glob = os.path.join(args.src, "data", "data_*.parquet")
    parts_dir = os.path.join(os.path.dirname(args.out) or ".", "_parts_av")
    tmp_dir = os.path.join(os.path.dirname(args.out) or ".", "_duckdb_tmp")
    for d in (args.out, parts_dir, tmp_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET threads={args.threads}")
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    con.execute(f"SET temp_directory='{tmp_dir}'")
    con.execute("SET preserve_insertion_order=false")

    raw = con.execute(f"SELECT count(*) FROM read_parquet('{src_glob}')").fetchone()[0]
    print(f"raw rows: {raw:,}")

    # Phase 1: hash-partition by FEN so every row for a position lands in one bucket -------
    print(f"phase 1/2: partitioning into {args.buckets} buckets by hash(fen) ...")
    con.execute(
        f"""
        COPY (
          SELECT fen, line, cp, mate, depth, knodes, (hash(fen) % {args.buckets}) AS bucket
          FROM read_parquet('{src_glob}')
          WHERE length(line) > 0
        ) TO '{parts_dir}' (FORMAT parquet, PARTITION_BY (bucket), OVERWRITE_OR_IGNORE);
        """
    )

    # Phase 2: per bucket, gather MultiPV into one row per FEN -----------------------------
    print("phase 2/2: aggregating MultiPV per FEN (pick richest analysis) ...")
    for b in range(args.buckets):
        bdir = os.path.join(parts_dir, f"bucket={b}")
        if not os.path.isdir(bdir):
            continue
        con.execute(
            f"""
            COPY (
              WITH analyses AS (
                -- one row per (position, search): the search's moves packed as aligned structs
                SELECT fen, depth, knodes,
                       list(struct_pack(m := split_part(line, chr(32), 1), c := cp, mt := mate)) AS ml,
                       count(*) AS nmoves
                FROM read_parquet('{bdir}/*.parquet')
                GROUP BY fen, depth, knodes
              ),
              best AS (
                -- per FEN, keep the richest analysis (most moves, then deepest)
                SELECT fen,
                       arg_max(struct_pack(ml := ml, depth := depth, nmoves := nmoves),
                               nmoves * 1000 + depth) AS r
                FROM analyses
                GROUP BY fen
              )
              SELECT
                fen,
                list_transform(r.ml, s -> s.m) AS moves,
                -- convert White-POV eval to side-to-move POV (negate when Black to move)
                CASE WHEN split_part(fen, chr(32), 2) = 'b'
                     THEN list_transform(r.ml, s -> -s.c) ELSE list_transform(r.ml, s -> s.c) END AS cp,
                CASE WHEN split_part(fen, chr(32), 2) = 'b'
                     THEN list_transform(r.ml, s -> -s.mt) ELSE list_transform(r.ml, s -> s.mt) END AS mate,
                r.depth AS depth,
                (hash(fen) % {args.val_mod} = 0) AS is_val
              FROM best
              WHERE r.nmoves >= {args.min_moves}
            ) TO '{os.path.join(args.out, f"av_{b:03d}.parquet")}' (FORMAT parquet);
            """
        )
        print(f"  bucket {b+1}/{args.buckets} done")

    shutil.rmtree(parts_dir)

    n, nval, multi = con.execute(
        f"""SELECT count(*), count(*) FILTER (is_val), count(*) FILTER (len(moves) >= 2)
            FROM read_parquet('{os.path.join(args.out, '*.parquet')}')"""
    ).fetchone()
    print(f"positions: {n:,}  ({100*n/raw:.1f}% of raw rows)")
    print(f"  with a multi-move distribution (>=2): {multi:,} ({100*multi/n:.1f}%)")
    print(f"  val: {nval:,} ({100*nval/n:.2f}%)")
    print(f"output: {args.out}/")


if __name__ == "__main__":
    main()
