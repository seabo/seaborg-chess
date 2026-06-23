# ChessFormer (PyTorch)

A faithful PyTorch re-implementation of the transformer architecture from

> **Mastering Chess with a Transformer Model** — Daniel Monroe, Philip A. Chalmers,
> [arXiv:2409.12272](https://arxiv.org/abs/2409.12272)

trained as a **supervised** policy + value model on the
[Lichess Stockfish position-evaluation dataset](https://huggingface.co/datasets/Lichess/chess-position-evaluations).

---

## Architecture (`chessformer/model.py`)

The model reproduces the architecture described in the paper:

| Component | Implementation |
|---|---|
| **Tokens** | 64 — one per board square |
| **Token embedding** | Linear projection of the per-square feature vector, then a per-token learned affine (scale + offset), as in the paper |
| **Position encoding** | **Shaw et al. relative position encoding** — learned `a^Q`, `a^K`, `a^V` vectors keyed by the (horizontal, vertical) displacement between squares (225 distinct relative positions), applied to both attention logits and the value path |
| **Normalization** | RMSNorm (LayerNorm with **centering and bias omitted**, per the paper) |
| **Residuals / init** | **Post-LN with DeepNorm** scaling `α=(2N)^¼` and matching init gain `β=(8N)^-¼` |
| **Feed-forward** | Two linear layers with a **Mish** activation |
| **Policy head** | Attention-based: move `(from, to)` logit = `⟨Q(src), K(dst)⟩ / √d` over the 64×64 = 4096 square pairs; **illegal moves masked** |
| **Value heads** | Pooled board → **WDL** (win/draw/loss, 3-way) + **scalar value** (`tanh`) |

### Presets (`chessformer/config.py`)

| Preset | Layers | `d_model` | Heads | `d_ff` | Params |
|---|---|---|---|---|---|
| `tiny` | 2 | 64 | 4 | 64 | ~0.1M |
| `cf6m` *(default)* | 8 | 256 | 8 | 256 | ~3.6M |
| `cf240m` | 15 | 1024 | 32 | 4096 | ~193M |

> **Note on parameter counts.** This implementation of `cf6m` has ~3.6M params vs the
> paper's stated ~6M (and `cf240m` ~193M vs the paper's 243M). The mechanism-level
> architecture matches the paper; the gap comes from details the paper text does not
> fully pin down (e.g. exact feed-forward width for the small model — the large model
> uses a 4× FFN ratio, so the small model's FFN may be wider than the literal "256" the
> text gives — and whether relative encodings are per-head or shared). All dims are
> configurable: e.g. `--d-ff 1024` brings `cf6m` to ~6.8M, close to the paper's figure.

---

## Adapting the paper to this dataset (`chessformer/encoding.py`)

The paper trains on ~100B **self-play game** positions with MCTS visit-count policy
targets and game outcomes. The Lichess dataset is different — **isolated positions** with
**Stockfish evaluations** (`cp`/`mate`) and a principal variation (`line`). The
*architecture* is unchanged; only the *targets/inputs* are adapted:

- **Input features** — single-position 18-dim per-square vector (12 piece planes +
  4 castling flags + en-passant marker + 50-move clock) instead of the paper's 112-dim
  8-position history stack (the dataset has no move history). Board is oriented so the
  side to move is always "us" (vertical mirror + colour swap when Black to move).
- **Policy target** — Stockfish's best move (first move of `line`), as a hard class over
  the 4096 `(from, to)` pairs. Promotions collapse to their `(from, to)` (queen by
  default); under-promotions therefore share a class with queen-promotion.
- **Value targets** — `cp`/`mate` → side-to-move expected score `q ∈ [-1, 1]`
  (logistic in centipawns; `±1` for mate). Trains the WDL head (soft win/draw/loss
  derived from `q`) and the scalar head (predicts `q`). `cp` is stored White-POV in the
  dataset (verified empirically) and converted to side-to-move POV.
- **Omitted heads** — the paper's categorical-reward and value-error heads need self-play
  reward distributions absent from this dataset, so they are not trained.

Loss = `policy_CE + wdl_CE + value_MSE` (the paper's vanilla-policy / WDL / L2 weights, all 1).

---

## Setup

The project venv already has `torch` (CUDA), `python-chess`, `datasets`, `pyarrow`.

### 1. Download the dataset (~41 GB, 20 parquet shards)

```bash
hf download Lichess/chess-position-evaluations --repo-type dataset \
    --local-dir data/lichess-evals
```

(Already kicked off in the background during setup. Fits comfortably — 41 GB on 640 GB free.)

### 2. Sanity-check the pipeline

```bash
python smoke_test.py        # encoding, relative-index, model fwd/bwd, real-row encoding
```

### 3. Train

```bash
python train.py --preset cf6m --batch-size 512 --num-workers 12 --steps 200000
```

Useful flags: `--preset {tiny,cf6m,cf240m}`, dim overrides `--n-layer/--n-embd/--n-head/--d-ff`,
`--lr`, `--warmup`, `--grad-accum`, `--min-depth N` (skip shallow evals),
`--eval-interval`, `--save-interval`, `--resume <ckpt>`, `--no-amp`, `--compile`.

**Memory:** the Shaw relative-attention term has a transient `(B, heads, 64, 64, head_dim)`
intermediate, so peak VRAM scales ~linearly with batch — measured on the RTX 2070 SUPER
(7.6 GB usable): batch 256 → ~2 GB, **512 → ~4 GB (default)**, 768 → ~6 GB, 1024 → OOM.
Use `--grad-accum` to raise the *effective* batch without more memory (the paper used 2048).
Throughput is typically **data-bound** by python-chess legal-move generation, so scale
`--num-workers` toward your core count.

---

## Playing the model

Move selection has two modes (`--mode`):
- **`policy`** — use the policy head directly (argmax, or `--temperature` to sample). The
  most direct view of what the network learned.
- **`value`** *(default)* — 1-ply search: score every legal move's resulting position with
  the value head (negamax) and pick the best. Usually stronger, since the value head is the
  better-trained part of this model.

### Quick local test (no setup)

```bash
python play.py --checkpoint checkpoints/chessformer_latest.pt --mode value --color white
```
Play in the terminal (UCI `e2e4` or SAN `Nf3`; commands `eval`, `board`, `undo`, `quit`).

### As a UCI engine

`uci.py` speaks UCI, so it drops into any UCI GUI (Cute Chess, BanksiaGUI, …) or the
`lichess-bot` bridge. `engine.sh` is a ready-made entrypoint that runs it via the venv:

```bash
printf 'uci\nisready\nposition startpos\ngo\nquit\n' | ./engine.sh   # smoke test
```

### Play it on Lichess (recommended)

Lichess's [Bot API](https://lichess.org/api#tag/Bot) + the official
[`lichess-bot`](https://github.com/lichess-bot-devs/lichess-bot) bridge let you play this
engine through the real Lichess web UI. (Lichess's "external engine" feature is analysis-only;
the Bot API is the route for actually *playing* a game.)

1. **Create a dedicated Lichess account** for the bot — it must have **zero games played** to
   be upgradeable.
2. **Make an API token** (Preferences → API access) with the **`bot:play`** scope.
3. **Upgrade it to a BOT account** (irreversible):
   ```bash
   curl -d '' https://lichess.org/api/bot/account/upgrade -H "Authorization: Bearer <TOKEN>"
   ```
4. **Install `lichess-bot`** (use a *separate* venv to avoid clashing with this project's deps):
   ```bash
   git clone https://github.com/lichess-bot-devs/lichess-bot && cd lichess-bot
   python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
   ```
5. **Configure** `config.yml`: set `token`, and point the engine at this repo:
   ```yaml
   engine:
     dir: "/home/seabo/chess-gpt"
     name: "engine.sh"
     protocol: "uci"
   ```
   (`engine.sh` already uses the project venv, so torch/python-chess resolve correctly even
   though lichess-bot runs in its own venv.)
6. **Run it:** `python lichess-bot.py` — then from your *normal* Lichess account, challenge the
   bot's username and play it in the web interface.

> Note: the value-mode 1-ply search is fast (one batched forward per move), so it comfortably
> handles real-time games. Strength reflects the checkpoint — a 3.6M model at 10k steps plays
> coherently but weakly; expect blunders.

---

## Files

```
chessformer/
  config.py     model config + presets (CF_6M default, CF_TINY, CF_240M)
  encoding.py   board/move/eval → tensors; relative-position index; perspective handling
  model.py      ChessFormer: relative attention, DeepNorm, Mish, policy + value heads
  losses.py     combined policy + WDL + value loss; policy top-1 accuracy
  data.py       streaming IterableDataset over parquet shards (worker-sharded + shuffle buffer)
  inference.py  ChessFormerEngine: load a checkpoint, evaluate, select moves (policy/value)
train.py        training loop (NAdam, warmup+cosine LR, fp16 AMP, grad-clip, checkpointing)
uci.py          UCI protocol wrapper (for GUIs / lichess-bot)
engine.sh       UCI entrypoint that runs uci.py via the venv (point lichess-bot here)
play.py         terminal play against a checkpoint
smoke_test.py   end-to-end self-checks
```
