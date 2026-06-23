"""Training loop for ChessFormer on the Lichess Stockfish-evaluation dataset.

Example:
    python train.py --preset cf6m --batch-size 512 --steps 200000 --num-workers 12

Trains the attention policy head (best-move prediction) and the value heads (WDL + scalar)
jointly with the paper's weighted-sum loss. Uses NAdam, gradient clipping at 10, a
warmup+cosine LR schedule, and fp16 mixed precision on CUDA.
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from chessformer.config import CF_6M, CF_22M, CF_100M, CF_TINY, CF_240M, MLP_7M
from chessformer.data import ActionValueDataset, LichessEvalDataset, find_shards
from chessformer.mlp import ChessMLP
from chessformer.losses import (
    LossWeights,
    chessformer_loss,
    chessformer_soft_loss,
    policy_accuracy,
    soft_policy_accuracy,
)
from chessformer.model import ChessFormer

PRESETS = {"cf6m": CF_6M, "cf22m": CF_22M, "cf100m": CF_100M, "tiny": CF_TINY,
           "cf240m": CF_240M, "mlp7m": MLP_7M}


def parse_args():
    p = argparse.ArgumentParser(description="Train ChessFormer on Lichess evaluations")
    p.add_argument("--preset", choices=PRESETS, default="cf6m")
    p.add_argument("--arch", choices=["transformer", "mlp"], default="transformer",
                   help="model architecture (use --arch mlp --preset mlp7m for the MLP baseline)")
    # optional dim overrides on top of the preset (handy for matching paper variants)
    p.add_argument("--n-layer", type=int, default=None)
    p.add_argument("--n-embd", type=int, default=None)
    p.add_argument("--n-head", type=int, default=None)
    p.add_argument("--d-ff", type=int, default=None)
    p.add_argument("--data-dir", default="data/lichess-evals")
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--steps", type=int, default=200_000, help="number of optimizer steps")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--shuffle-buffer", type=int, default=16384)
    p.add_argument("--min-depth", type=int, default=0, help="skip rows below this engine depth")
    p.add_argument("--val-shards", type=int, default=1, help="number of shards held out for val (shards mode)")
    p.add_argument("--val-mode", choices=["shards", "column"], default="shards",
                   help="'shards': hold out whole shards; 'column': use the clean dataset's is_val column (leak-free)")
    p.add_argument("--val-batches", type=int, default=50)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--eval-interval", type=int, default=2000)
    p.add_argument("--save-interval", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision (force fp32)")
    p.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16",
                   help="mixed-precision dtype on CUDA. bf16 (H100/Ampere+) needs no GradScaler; "
                        "fp16 (Turing) uses one; fp32 disables AMP")
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    p.add_argument("--resume", default=None, help="resume a run (model+optimizer+step) from a checkpoint")
    p.add_argument("--init-from", default=None,
                   help="warm-start MODEL WEIGHTS ONLY from a checkpoint (fresh optimizer/LR/step) — for fine-tuning")
    # soft-policy (action-value) training
    p.add_argument("--soft-policy", action="store_true",
                   help="train on an action-value dataset with a soft-policy target (use with --data-dir action-values)")
    p.add_argument("--policy-temp", type=float, default=0.1,
                   help="temperature for softmax(q/T) when building the soft-policy target")
    return p.parse_args()


def lr_lambda(step, warmup, total, min_ratio):
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, progress)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def make_loader(shards, args, *, loop, seed, split=None):
    if args.soft_policy:
        ds = ActionValueDataset(
            shards,
            shuffle_buffer=args.shuffle_buffer if loop else 4096,
            policy_temp=args.policy_temp,
            seed=seed,
            loop=loop,
            split=split,
        )
    else:
        ds = LichessEvalDataset(
            shards,
            shuffle_buffer=args.shuffle_buffer if loop else 4096,
            min_depth=args.min_depth,
            seed=seed,
            loop=loop,
            split=split,
        )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, weights, max_batches, amp_dtype, loss_fn, acc_fn):
    model.eval()
    totals = {"loss": 0.0, "policy": 0.0, "wdl": 0.0, "value": 0.0, "acc": 0.0}
    n = 0
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            pol, wdl, val = model(batch["features"], batch["legal_mask"])
            _, comp = loss_fn(pol, wdl, val, batch, weights)
        for k in ("loss", "policy", "wdl", "value"):
            totals[k] += comp[k].item()
        totals["acc"] += acc_fn(pol, batch).item()
        n += 1
        if n >= max_batches:
            break
    model.train()
    return {k: v / max(1, n) for k, v in totals.items()}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")  # TF32 for fp32 matmuls (Ampere+/Hopper)
    _PREC = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}
    prec = "fp32" if args.no_amp else args.precision
    amp_dtype = None if device.type != "cuda" else _PREC[prec]
    print(f"device={device} | precision={prec if device.type=='cuda' else 'fp32 (cpu)'}")

    shards = find_shards(args.data_dir)
    if not shards:
        raise SystemExit(
            f"No parquet shards found under {args.data_dir!r}. Is the download finished?"
        )
    if args.val_mode == "column":
        train_shards = val_shards = shards  # both read all shards; split via is_val column
        print(f"Found {len(shards)} shards; train/val split by is_val column (leak-free)")
    else:
        val_shards = shards[: args.val_shards] if args.val_shards > 0 else []
        train_shards = shards[args.val_shards :] if args.val_shards > 0 else shards
        print(f"Found {len(shards)} shards: {len(train_shards)} train, {len(val_shards)} val")

    import dataclasses
    config = PRESETS[args.preset]
    overrides = {k: v for k, v in (
        ("n_layer", args.n_layer), ("n_embd", args.n_embd),
        ("n_head", args.n_head), ("d_ff", args.d_ff),
    ) if v is not None}
    if overrides:
        config = dataclasses.replace(config, **overrides)
    model = (ChessMLP(config) if args.arch == "mlp" else ChessFormer(config)).to(device)
    print(f"arch={args.arch} | {config.describe()}  |  {model.num_params()/1e6:.2f}M params  |  device={device}")

    # task dispatch: hard best-move target vs soft action-value policy target
    if args.soft_policy:
        loss_fn = chessformer_soft_loss
        acc_fn = lambda pol, batch: soft_policy_accuracy(pol, batch["soft_target"])
        print(f"soft-policy (action-value) training | policy_temp={args.policy_temp}")
    else:
        loss_fn = chessformer_loss
        acc_fn = lambda pol, batch: policy_accuracy(pol, batch["policy_target"])

    weights = LossWeights()
    opt = torch.optim.NAdam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-7,
        weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, args.warmup, args.steps, args.min_lr_ratio)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"]
        print(f"Resumed from {args.resume} at step {start_step}")
    elif args.init_from:
        # warm-start: load model weights only; keep the fresh optimizer/scheduler/step=0
        ckpt = torch.load(args.init_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Warm-started weights from {args.init_from} (was step {ckpt.get('step', '?')}); "
              f"fresh optimizer + LR schedule")

    # Compile AFTER loading weights: torch.compile wraps params under an `_orig_mod.`
    # prefix that wouldn't match the (prefix-free) checkpoint keys. Params are shared,
    # so the already-built optimizer still references the right tensors.
    if args.compile:
        model = torch.compile(model)

    if args.val_mode == "column":
        train_loader = make_loader(train_shards, args, loop=True, seed=args.seed, split="train")
        val_loader = make_loader(val_shards, args, loop=True, seed=args.seed + 7, split="val")
    else:
        train_loader = make_loader(train_shards, args, loop=True, seed=args.seed)
        val_loader = (
            make_loader(val_shards, args, loop=True, seed=args.seed + 7) if val_shards else None
        )

    model.train()
    train_iter = iter(train_loader)
    t0 = time.time()
    seen = 0
    # accumulate metrics on-GPU; only sync (.item()) at log time to avoid a CPU<->GPU
    # stall on every step (which would stop the CPU queuing the next step's kernels)
    running = {k: torch.zeros((), device=device) for k in ("loss", "policy", "wdl", "value", "acc")}
    log_n = 0
    n_skipped = 0  # optimizer steps skipped due to non-finite gradients

    for step in range(start_step, args.steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            batch = to_device(next(train_iter), device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                pol, wdl, val = model(batch["features"], batch["legal_mask"])
                loss, comp = loss_fn(pol, wdl, val, batch, weights)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            seen += batch["features"].size(0)
            for k in ("loss", "policy", "wdl", "value"):
                running[k] += comp[k]
            running["acc"] += acc_fn(pol, batch)
            log_n += 1

        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if torch.isfinite(gnorm):
            prev_scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            # advance LR only if the optimizer actually stepped (fp16 scaler skips on inf)
            if scaler.get_scale() >= prev_scale:
                sched.step()
        else:
            # non-finite gradients: under bf16 there's no GradScaler to catch these, so
            # skip the update — one bad step can't be allowed to corrupt the weights
            scaler.update()
            n_skipped += 1
            print(f"  [warn] step {step}: non-finite grad norm, step skipped (total skipped {n_skipped})", flush=True)

        if step % args.log_interval == 0 and log_n > 0:
            dt = time.time() - t0
            rate = seen / dt
            avg = {k: (v / log_n).item() for k, v in running.items()}
            mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
            print(
                f"step {step:>7} | loss {avg['loss']:.4f} "
                f"(pol {avg['policy']:.4f} wdl {avg['wdl']:.4f} val {avg['value']:.4f}) "
                f"| acc {avg['acc']*100:5.2f}% | lr {sched.get_last_lr()[0]:.2e} "
                f"| {rate:6.0f} pos/s | {mem:.2f}GB",
                flush=True,
            )
            running = {k: torch.zeros((), device=device) for k in running}
            log_n = 0
            seen = 0
            t0 = time.time()

        if val_loader is not None and step > start_step and step % args.eval_interval == 0:
            v = evaluate(model, val_loader, device, weights, args.val_batches, amp_dtype, loss_fn, acc_fn)
            print(
                f"  [val] step {step} | loss {v['loss']:.4f} "
                f"(pol {v['policy']:.4f} wdl {v['wdl']:.4f} val {v['value']:.4f}) "
                f"| acc {v['acc']*100:5.2f}%",
                flush=True,
            )
            t0 = time.time()  # don't count eval time against throughput

        if step > start_step and step % args.save_interval == 0:
            save(model, opt, sched, scaler, step, config, args.out_dir, "latest", args.arch)

    save(model, opt, sched, scaler, args.steps, config, args.out_dir, "final", args.arch)
    print("Training complete.")


def save(model, opt, sched, scaler, step, config, out_dir, tag, arch="transformer"):
    raw = getattr(model, "_orig_mod", model)  # unwrap torch.compile
    path = os.path.join(out_dir, f"chessformer_{tag}.pt")
    torch.save(
        {
            "model": raw.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "config": config.__dict__,
            "arch": arch,
        },
        path,
    )
    print(f"  saved checkpoint -> {path} (step {step})", flush=True)


if __name__ == "__main__":
    main()
