#!/usr/bin/env python3
"""Play the trained model in the terminal — the fastest way to sanity-check a checkpoint.

    python play.py --checkpoint checkpoints/chessformer_latest.pt --mode value --color white

Enter moves in UCI (e2e4) or SAN (Nf3). Commands: `eval`, `board`, `undo`, `quit`.
"""

from __future__ import annotations

import argparse

import chess

from chessformer.inference import ChessFormerEngine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/chessformer_latest.pt")
    ap.add_argument("--mode", choices=["policy", "value", "search"], default="search")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--search-depth", type=int, default=3)
    ap.add_argument("--search-width", type=int, default=4)
    ap.add_argument("--search-qdepth", type=int, default=4, help="quiescence ply cap (0 disables)")
    ap.add_argument("--color", choices=["white", "black"], default="white",
                    help="the colour YOU play")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    engine = ChessFormerEngine(args.checkpoint, device=args.device)
    print(f"Loaded checkpoint (step {engine.step}), {engine.config.describe()}, mode={args.mode}")
    board = chess.Board()
    human_white = args.color == "white"

    def show():
        print()
        print(board.unicode(borders=True, invert_color=True))
        ev = engine.evaluate(board)
        w, d, l = ev["wdl"]
        print(f"model eval (side to move): {ev['cp']:+d}cp  W/D/L={w:.2f}/{d:.2f}/{l:.2f}")

    show()
    while not board.is_game_over():
        if board.turn == (chess.WHITE if human_white else chess.BLACK):
            raw = input("\nyour move> ").strip()
            if raw in ("quit", "q"):
                return
            if raw == "board":
                show(); continue
            if raw == "eval":
                continue  # eval already shown by show(); just re-show
            if raw == "undo":
                if len(board.move_stack) >= 2:
                    board.pop(); board.pop()
                show(); continue
            try:
                move = board.parse_san(raw)
            except ValueError:
                try:
                    move = chess.Move.from_uci(raw)
                except ValueError:
                    print("  ?? couldn't parse move"); continue
            if move not in board.legal_moves:
                print("  illegal move"); continue
            board.push(move)
        else:
            res = engine.select_move(board, mode=args.mode, temperature=args.temperature,
                                     depth=args.search_depth, width=args.search_width,
                                     qdepth=args.search_qdepth)
            print(f"\nChessFormer plays: {board.san(res['move'])}  ({res['cp']:+d}cp, {res['info']})")
            board.push(res["move"])
        show()

    print(f"\nGame over: {board.result()} ({board.outcome().termination.name})")


if __name__ == "__main__":
    main()
