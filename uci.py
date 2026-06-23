#!/usr/bin/env python3
"""Minimal UCI wrapper around a trained ChessFormer checkpoint.

Speaks just enough of the UCI protocol to work with any UCI GUI or with the
`lichess-bot` bridge (so you can play it through the Lichess web interface).

    python uci.py --checkpoint checkpoints/chessformer_latest.pt --mode value

Supported UCI options (settable from the GUI / lichess-bot config):
    Mode         combo  Policy | Value      (move-selection strategy)
    Temperature  string 0.0                 (policy sampling temperature; 0 = greedy)
"""

from __future__ import annotations

import argparse
import sys

import chess

from chessformer.inference import ChessFormerEngine


def parse_position(line: str) -> chess.Board:
    tokens = line.split()
    board = chess.Board()
    i = 1
    if i < len(tokens) and tokens[i] == "startpos":
        i += 1
    elif i < len(tokens) and tokens[i] == "fen":
        fen = " ".join(tokens[i + 1 : i + 7])
        board = chess.Board(fen)
        i += 7
    if i < len(tokens) and tokens[i] == "moves":
        for uci in tokens[i + 1 :]:
            board.push(chess.Move.from_uci(uci))
    return board


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/chessformer_latest.pt")
    ap.add_argument("--mode", choices=["policy", "value", "search"], default="search")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--search-depth", type=int, default=3)
    ap.add_argument("--search-width", type=int, default=4)
    ap.add_argument("--search-qdepth", type=int, default=4, help="quiescence ply cap (0 disables)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    engine = None  # lazy-load on first `isready` so `uci` responds instantly
    board = chess.Board()
    mode = args.mode
    temperature = args.temperature
    depth = args.search_depth
    width = args.search_width
    qdepth = args.search_qdepth

    def out(s: str):
        sys.stdout.write(s + "\n")
        sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if line == "uci":
            out("id name ChessFormer")
            out("id author chess-gpt")
            out("option name Mode type combo default Search var Policy var Value var Search")
            out("option name Temperature type string default 0.0")
            out("option name SearchDepth type spin default 3 min 1 max 6")
            out("option name SearchWidth type spin default 4 min 1 max 20")
            out("option name QDepth type spin default 4 min 0 max 12")
            out("uciok")
        elif line == "isready":
            if engine is None:
                engine = ChessFormerEngine(args.checkpoint, device=args.device)
            out("readyok")
        elif line.startswith("setoption"):
            t = line.split()
            if "Mode" in t:
                mode = t[t.index("value") + 1].lower()
            elif "Temperature" in t:
                temperature = float(t[t.index("value") + 1])
            elif "SearchDepth" in t:
                depth = int(t[t.index("value") + 1])
            elif "SearchWidth" in t:
                width = int(t[t.index("value") + 1])
            elif "QDepth" in t:
                qdepth = int(t[t.index("value") + 1])
        elif line == "ucinewgame":
            board = chess.Board()
        elif line.startswith("position"):
            board = parse_position(line)
        elif line.startswith("go"):
            if engine is None:
                engine = ChessFormerEngine(args.checkpoint, device=args.device)
            res = engine.select_move(board, mode=mode, temperature=temperature,
                                     depth=depth, width=width, qdepth=qdepth)
            mv = res["move"]
            if mv is None:
                out("bestmove (none)")
            else:
                cp = res.get("cp")
                if cp is not None:
                    shown_depth = depth if mode == "search" else 1
                    out(f"info depth {shown_depth} score cp {cp} pv {mv.uci()}")
                out(f"bestmove {mv.uci()}")
        elif line == "quit":
            break


if __name__ == "__main__":
    main()
