#!/usr/bin/env bash
# Turn-key ~100M training run on a fresh Lambda 1x H100 instance.
#
# PREREQ — get this repo onto the instance, then run from its root:
#   git clone <your-repo-url> chess-gpt && cd chess-gpt && bash lambda_run.sh
# (private repo: clone over HTTPS with a PAT, or add a deploy key. data/ + checkpoints/
#  are gitignored and rebuilt/written on the box — or mounted from a Lambda Filesystem.)
#
# Optional (faster dataset download): export HF_TOKEN=hf_xxx before running.
# Tunables:  STEPS=... BATCH=... LR=... WORKERS=...  bash lambda_run.sh
set -euo pipefail
cd "$(dirname "$0")"

STEPS=${STEPS:-200000}      # ~0.5 epoch ceiling; WATCH val acc and stop early once it plateaus
BATCH=${BATCH:-1024}        # H100/80GB; drop to 512 if the attention transient OOMs
LR=${LR:-6e-4}
WORKERS=${WORKERS:-16}
# Set CF_FS to a mounted Lambda Filesystem (e.g. CF_FS=/home/ubuntu/<fs-name>) to keep the
# dataset AND checkpoints on persistent storage — they survive instance termination, and a
# re-run skips the data build. Without CF_FS, everything goes to the instance-local SSD.
CF_FS=${CF_FS:-}
if [ -n "$CF_FS" ]; then
  [ -d "$CF_FS" ] || { echo "ERROR: CF_FS=$CF_FS does not exist — is the filesystem attached/mounted?"; exit 1; }
  RAW_DIR=${RAW_DIR:-$CF_FS/lichess-evals}
  DATA_DIR=${DATA_DIR:-$CF_FS/action-values-raw}
  OUT_DIR=${OUT_DIR:-$CF_FS/cf100m}
  echo "using persistent filesystem at $CF_FS"
else
  RAW_DIR=${RAW_DIR:-data/lichess-evals}
  DATA_DIR=${DATA_DIR:-data/action-values-raw}
  OUT_DIR=${OUT_DIR:-checkpoints/cf100m}
fi

echo "=== GPU ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=== disk ==="; df -h . | tail -1

# --- 1. venv + deps (idempotent) ---
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -q -r requirements-cloud.txt
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), torch.version.cuda)"

# --- 2. data: raw Lichess set -> action-values (skipped if already built) ---
if [ -z "$(ls -A "$DATA_DIR" 2>/dev/null || true)" ]; then
  if [ ! -d "$RAW_DIR/data" ]; then
    echo "=== downloading raw dataset (~41 GB) ==="
    hf download Lichess/chess-position-evaluations --repo-type dataset --local-dir "$RAW_DIR"
  fi
  echo "=== building action-values dataset (~5-15 min) ==="
  python build_action_values.py --src "$RAW_DIR" --out "$DATA_DIR"
else
  echo "$DATA_DIR already present — skipping data build"
fi

# --- 3. launch training in tmux ---
LOG=train_cf100m.log
tmux new-session -d -s train "source venv/bin/activate && \
  python train.py --preset cf100m --arch transformer --soft-policy \
    --precision bf16 --compile \
    --data-dir $DATA_DIR --val-mode column \
    --out-dir $OUT_DIR \
    --batch-size $BATCH --num-workers $WORKERS --steps $STEPS --save-interval 2000 \
    --lr $LR --warmup 2000 --policy-temp 0.1 2>&1 | tee $LOG"

cat <<EOF

Training launched in tmux session 'train'.
  attach:       tmux attach -t train        (detach again with: Ctrl-b then d)
  watch log:    tail -f $LOG
  checkpoints:  $OUT_DIR/chessformer_latest.pt  (written every 2000 steps)

*** COST REMINDER ***
This H100 bills ~\$4.29/hr for as long as the instance EXISTS.
Watch the val acc; once it plateaus, stop training and TERMINATE THE INSTANCE
in the Lambda dashboard. An OS 'shutdown' does NOT reliably stop billing.
EOF
