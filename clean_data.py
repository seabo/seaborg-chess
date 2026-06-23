#!/usr/bin/env python3
"""Deduplicate + quality-filter the raw Lichess eval dataset into a clean training set.

For each unique position (FEN), keep the single best-analyzed row — the one with the
greatest (depth, knodes) — drop positions whose best depth is still below a floor, and
stamp a leak-free train/val split derived from a hash of the FEN.

  945M rows (~2.4x duplicated)  ->  ~388M unique positions, deepest eval each.

Memory/disk-bounded by hash-partitioning on FEN: a single streaming pass splits the raw
rows into K buckets by hash(fen) % K, then each bucket is deduplicated independently in
RAM. All rows for a given FEN land in the same bucket, so per-bucket dedup is exact (same
result as a global GROUP BY) without the giant out-of-core aggregation that would spill
the whole dataset to disk many times over.

    python clean_data.py --src data/lichess-evals --out data/lichess-clean

NOTE: this is heavy on disk I/O — avoid running it alongside a training job.
"""

from __future__ import annotations

import argparse
import functools
import os
import shutil

import duckdb

print = functools.partial(print, flush=True)  # unbuffered, so nohup logs update live


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/lichess-evals")
    ap.add_argument("--out", default="data/lichess-clean")
    ap.add_argument("--min-depth", type=int, default=12, help="drop positions whose best eval depth is below this")
    ap.add_argument("--val-mod", type=int, default=50, help="hash(fen) %% val_mod == 0 -> validation (50 => 2%%)")
    ap.add_argument("--buckets", type=int, default=32, help="hash partitions (more => less RAM per bucket)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--memory-limit", default="8GB")
    args = ap.parse_args()

    src_glob = os.path.join(args.src, "data", "data_*.parquet")
    parts_dir = os.path.join(os.path.dirname(args.out) or ".", "_parts")
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

    raw_rows = con.execute(f"SELECT count(*) FROM read_parquet('{src_glob}')").fetchone()[0]
    print(f"raw rows: {raw_rows:,}")

    # Phase 1: streaming hash-partition by FEN (bounded memory) ---------------------
    print(f"phase 1/2: partitioning into {args.buckets} buckets by hash(fen) ...")
    con.execute(
        f"""
        COPY (
          SELECT fen, line, cp, mate, depth, knodes, (hash(fen) % {args.buckets}) AS bucket
          FROM read_parquet('{src_glob}')
        ) TO '{parts_dir}' (FORMAT parquet, PARTITION_BY (bucket), OVERWRITE_OR_IGNORE);
        """
    )

    # Phase 2: dedup each bucket independently (fits in RAM) -------------------------
    print("phase 2/2: deduplicating each bucket (keep deepest eval per FEN) ...")
    for b in range(args.buckets):
        bdir = os.path.join(parts_dir, f"bucket={b}")
        if not os.path.isdir(bdir):
            continue
        con.execute(
            f"""
            COPY (
              WITH best AS (
                SELECT fen,
                       arg_max(
                         struct_pack(line := line, cp := cp, mate := mate, depth := depth, knodes := knodes),
                         depth::BIGINT * 10000000000 + coalesce(knodes, 0)
                       ) AS r
                FROM read_parquet('{bdir}/*.parquet')
                GROUP BY fen
              )
              SELECT fen, r.line AS line, r.cp AS cp, r.mate AS mate,
                     r.depth AS depth, r.knodes AS knodes,
                     (hash(fen) % {args.val_mod} = 0) AS is_val
              FROM best
              WHERE r.depth >= {args.min_depth}
            ) TO '{os.path.join(args.out, f"data_{b:03d}.parquet")}' (FORMAT parquet);
            """
        )
        print(f"  bucket {b+1}/{args.buckets} done")

    shutil.rmtree(parts_dir)  # drop the temp partitions
    clean_rows, val_rows = con.execute(
        f"SELECT count(*), count(*) FILTER (is_val) FROM read_parquet('{os.path.join(args.out, '*.parquet')}')"
    ).fetchone()
    print(f"clean rows: {clean_rows:,}  ({100*clean_rows/raw_rows:.1f}% of raw)")
    print(f"  val rows: {val_rows:,}  ({100*val_rows/clean_rows:.2f}%)")
    print(f"  dropped (depth < {args.min_depth} or duplicates): {raw_rows-clean_rows:,}")
    print(f"output: {args.out}/ ({len(os.listdir(args.out))} parquet files)")


if __name__ == "__main__":
    main()
