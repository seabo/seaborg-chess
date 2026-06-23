"""Streaming data pipeline over the Lichess Stockfish-evaluation parquet shards.

Each row (fen, line, cp, mate, depth, knodes) is turned into a training example:
    features      : float32 [64, 18]   per-square input planes (oriented to side-to-move)
    legal_mask    : bool    [4096]      legal (from, to) move classes
    policy_target : int64               class id of Stockfish's best move (first of `line`)
    wdl_target    : float32 [3]         soft win/draw/loss target derived from cp/mate
    value_target  : float32             expected score q in [-1, 1]

Implemented as an IterableDataset so we never materialise the 945M-row dataset in RAM:
pyarrow streams record batches from each parquet shard, shards are split across workers,
and a per-worker shuffle buffer decorrelates consecutive positions.
"""

from __future__ import annotations

import glob
import os
import random
from typing import List, Optional

import chess
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

from . import encoding


def find_shards(data_dir: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(data_dir, "data", "data_*.parquet")))
    if not paths:  # fall back to a flat layout
        paths = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    return paths


def _normalize_fen(fen: str) -> str:
    """Lichess FENs omit the halfmove/fullmove counters; append defaults so python-chess
    can parse them."""
    if fen.count(" ") == 3:
        return fen + " 0 1"
    return fen


def encode_row(fen: str, line: str, cp, mate) -> Optional[dict]:
    """Turn one dataset row into training tensors, or None if it should be skipped."""
    if not line:
        return None
    try:
        board = chess.Board(_normalize_fen(fen))
    except (ValueError, IndexError):
        return None

    q = encoding.eval_to_q(cp, mate, white_to_move=board.turn == chess.WHITE)
    if q is None:
        return None

    try:
        best = chess.Move.from_uci(line.split(" ", 1)[0])
    except (ValueError, IndexError):
        return None

    oriented, flipped = encoding.orient(board)
    if flipped:
        best = encoding.mirror_move(best)

    # policy class is (from, to) only; confirm the move is legal in the oriented frame
    target_idx = encoding.move_to_index(best)
    legal_mask = encoding.legal_move_mask(oriented)
    if not legal_mask[target_idx]:
        return None

    features = encoding.encode_board(oriented)
    wdl = encoding.q_to_wdl(q)

    return {
        "features": torch.from_numpy(features),
        "legal_mask": torch.from_numpy(legal_mask),
        "policy_target": target_idx,
        "wdl_target": torch.from_numpy(wdl),
        "value_target": float(q),
    }


class _ShardIterable(IterableDataset):
    """Shared streaming machinery: split parquet shards across DataLoader workers, stream
    rows through a per-worker shuffle buffer, and (optionally) loop forever for step-based
    training. Subclasses implement `_raw_rows(shards, rng)` yielding example dicts, and set
    `shard_paths`, `shuffle_buffer`, `seed`, `loop`.
    """

    def _shards_for_worker(self) -> List[str]:
        info = get_worker_info()
        if info is None:
            return self.shard_paths
        return self.shard_paths[info.id :: info.num_workers]

    def __iter__(self):
        info = get_worker_info()
        wid = 0 if info is None else info.id
        shards = self._shards_for_worker()
        if not shards:
            return  # this worker was assigned no shards (more workers than shards)

        rng = random.Random(self.seed + wid)
        buffer = []
        epoch = 0
        while True:
            n_pass = 0
            for ex in self._raw_rows(shards, rng):
                n_pass += 1
                if len(buffer) < self.shuffle_buffer:
                    buffer.append(ex)
                    continue
                j = rng.randrange(len(buffer))
                buffer[j], ex = ex, buffer[j]
                yield ex
            if not self.loop or n_pass == 0:
                break  # single pass requested, or shards yielded nothing (avoid spinning)
            epoch += 1
            rng = random.Random(self.seed + wid + 1000 * epoch)
        # drain the buffer on a final pass
        rng.shuffle(buffer)
        yield from buffer


class LichessEvalDataset(_ShardIterable):
    """Hard best-move + value targets from the (clean) Lichess eval dataset."""

    def __init__(
        self,
        shard_paths: List[str],
        shuffle_buffer: int = 16384,
        min_depth: int = 0,
        batch_rows: int = 4096,
        seed: int = 0,
        loop: bool = True,
        split: Optional[str] = None,
    ):
        super().__init__()
        assert shard_paths, "no parquet shards provided"
        assert split in (None, "train", "val"), "split must be None, 'train', or 'val'"
        self.shard_paths = list(shard_paths)
        self.shuffle_buffer = shuffle_buffer
        self.min_depth = min_depth
        self.batch_rows = batch_rows
        self.seed = seed
        self.loop = loop  # if True, iterate forever (step-based training); else one pass
        # split filtering on the clean dataset's `is_val` column (None => use all rows)
        self.split = split

    def _raw_rows(self, shards, rng):
        """Yield encoded examples from the given shards, one pass."""
        order = list(shards)
        rng.shuffle(order)
        cols = ["fen", "line", "cp", "mate", "depth"]
        if self.split is not None:
            cols.append("is_val")
        want_val = self.split == "val"
        for path in order:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=self.batch_rows, columns=cols):
                d = batch.to_pydict()
                fens, lines = d["fen"], d["line"]
                cps, mates, depths = d["cp"], d["mate"], d["depth"]
                is_val = d["is_val"] if self.split is not None else None
                for i in range(len(fens)):
                    if is_val is not None and bool(is_val[i]) != want_val:
                        continue
                    if self.min_depth and (depths[i] or 0) < self.min_depth:
                        continue
                    ex = encode_row(fens[i], lines[i], cps[i], mates[i])
                    if ex is not None:
                        yield ex


def encode_action_value_row(fen, moves, cps, mates, policy_temp: float) -> Optional[dict]:
    """Turn one action-value row (a position + its analysed moves with side-to-move evals)
    into a soft-policy training example, or None to skip.

    soft_target: softmax(q / policy_temp) over the analysed moves, placed on their (from,to)
    classes in the oriented frame (0 elsewhere). value/wdl come from the best move's q.
    """
    if not moves:
        return None
    try:
        board = chess.Board(_normalize_fen(fen))
    except (ValueError, IndexError):
        return None

    oriented, flipped = encoding.orient(board)
    legal_mask = encoding.legal_move_mask(oriented)

    idxs, qs = [], []
    for mv_uci, cp, mate in zip(moves, cps, mates):
        try:
            mv = chess.Move.from_uci(mv_uci)
        except (ValueError, IndexError):
            continue
        if flipped:
            mv = encoding.mirror_move(mv)
        idx = encoding.move_to_index(mv)
        if not legal_mask[idx]:
            continue  # safety: drop anything that isn't legal in the oriented frame
        idxs.append(idx)
        qs.append(encoding.move_eval_to_q(cp, mate))
    if not idxs:
        return None

    qs = np.asarray(qs, dtype=np.float32)
    z = qs / max(policy_temp, 1e-6)
    z -= z.max()
    w = np.exp(z)
    w /= w.sum()

    soft = np.zeros(encoding.N_MOVES, dtype=np.float32)
    for i, idx in enumerate(idxs):
        soft[idx] += w[i]  # collapsed promotions (same from,to) accumulate

    q_pos = float(qs.max())  # position value = best move's expected score
    return {
        "features": torch.from_numpy(encoding.encode_board(oriented)),
        "legal_mask": torch.from_numpy(legal_mask),
        "soft_target": torch.from_numpy(soft),
        "wdl_target": torch.from_numpy(encoding.q_to_wdl(q_pos)),
        "value_target": q_pos,
    }


class ActionValueDataset(_ShardIterable):
    """Soft-policy + value targets from an action-value dataset (raw-MultiPV-derived or
    Stockfish-generated): each row is a position with a list of analysed moves and their
    side-to-move cp/mate."""

    def __init__(
        self,
        shard_paths: List[str],
        shuffle_buffer: int = 16384,
        policy_temp: float = 0.1,
        batch_rows: int = 4096,
        seed: int = 0,
        loop: bool = True,
        split: Optional[str] = None,
    ):
        super().__init__()
        assert shard_paths, "no parquet shards provided"
        assert split in (None, "train", "val"), "split must be None, 'train', or 'val'"
        self.shard_paths = list(shard_paths)
        self.shuffle_buffer = shuffle_buffer
        self.policy_temp = policy_temp
        self.batch_rows = batch_rows
        self.seed = seed
        self.loop = loop
        self.split = split

    def _raw_rows(self, shards, rng):
        order = list(shards)
        rng.shuffle(order)
        cols = ["fen", "moves", "cp", "mate"]
        if self.split is not None:
            cols.append("is_val")
        want_val = self.split == "val"
        for path in order:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=self.batch_rows, columns=cols):
                d = batch.to_pydict()
                fens = d["fen"]
                moves, cps, mates = d["moves"], d["cp"], d["mate"]
                is_val = d["is_val"] if self.split is not None else None
                for i in range(len(fens)):
                    if is_val is not None and bool(is_val[i]) != want_val:
                        continue
                    ex = encode_action_value_row(fens[i], moves[i], cps[i], mates[i], self.policy_temp)
                    if ex is not None:
                        yield ex
