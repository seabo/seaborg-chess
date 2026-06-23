#!/usr/bin/env bash
# UCI engine entrypoint for lichess-bot (or any GUI that wants a single executable).
# Uses the project venv so torch/python-chess are on the path. Extra args from the
# GUI/bridge are forwarded after ours.
#
# Toggle behaviour via env vars (so you can A/B raw vs. search without editing files):
#   CF_MODE=policy   ./engine.sh      # raw network (policy argmax)
#   CF_MODE=search   ./engine.sh      # policy-pruned alpha-beta (default)
#   CF_DEPTH / CF_WIDTH                # search depth / top-K width
#   CF_CKPT / CF_DEVICE                # checkpoint path / cpu|cuda
# For lichess-bot, export these before launching `python lichess-bot.py`.
cd "$(dirname "$0")" || exit 1
exec venv/bin/python uci.py \
    --checkpoint "${CF_CKPT:-checkpoints/softpolicy/chessformer_latest.pt}" \
    --mode "${CF_MODE:-search}" \
    --search-depth "${CF_DEPTH:-3}" \
    --search-width "${CF_WIDTH:-4}" \
    --search-qdepth "${CF_QDEPTH:-4}" \
    --device "${CF_DEVICE:-cpu}" \
    "$@"
