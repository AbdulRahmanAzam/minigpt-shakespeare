"""Train MiniGPT on Tiny Shakespeare.

    python -m src.train --config configs/tiny_gpt.yaml
    python -m src.train --config configs/ablation_base.yaml --set model.n_layers=2 --set train.run_name=depth2
    python -m src.train --config configs/tiny_gpt.yaml --resume checkpoints/tiny_gpt/last.pt

Logs (per run, in <out_dir>/<run_name>/):
    config.json     resolved configuration
    metrics.jsonl   one JSON line per log/eval event: step, epoch, lr, train/val loss, tokens/s, ...
    samples.txt     a sample generation at every evaluation
    summary.json    parameter count, hardware, wall-clock, final/best losses
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from src.checkpoint import load_checkpoint, save_checkpoint
from src.config import Config, load_config
from src.dataset import encode_split, infinite, load_tokenizer, make_loader
from src.gpt import MiniGPT
from src.utils import (
    autocast, environment_info, format_count, pick_device, pick_precision, set_seed, synchronize,
    write_json,
)


# ---------------------------------------------------------------------------- optimisation helpers
def lr_at(step: int, cfg: Config) -> float:
    """Linear warm-up, then cosine decay from lr to lr * min_lr_ratio at max_steps."""
    t = cfg.train
    if t.warmup_steps > 0 and step < t.warmup_steps:
        return t.lr * (step + 1) / t.warmup_steps
    progress = min(1.0, (step - t.warmup_steps) / max(1, t.max_steps - t.warmup_steps))
    min_lr = t.lr * t.min_lr_ratio
    return min_lr + 0.5 * (t.lr - min_lr) * (1 + math.cos(math.pi * progress))


def build_optimizer(model: MiniGPT, cfg: Config) -> torch.optim.Optimizer:
    """AdamW; weight decay only on Linear weight matrices (not biases, LayerNorm or embeddings)."""
    decay, no_decay = [], []
    seen: set[int] = set()
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if id(p) in seen or not p.requires_grad:
                continue
            seen.add(id(p))
            is_linear_weight = isinstance(module, torch.nn.Linear) and name == "weight"
            tied = cfg.model.tie_weights and p is model.tok_emb.weight
            (decay if is_linear_weight and not tied else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.train.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.train.lr, betas=(cfg.train.beta1, cfg.train.beta2))


# ---------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate_loss(
    model: MiniGPT,
    ids: torch.Tensor,
    context_length: int,
    batch_size: int,
    device: torch.device,
    precision: str = "fp32",
    max_batches: int | None = None,
) -> float:
    """Mean next-token cross-entropy over non-overlapping windows (deterministic, no dropout)."""
    was_training = model.training
    model.eval()
    loader = make_loader(ids, context_length, batch_size, stride=context_length, shuffle=False, drop_last=False)
    total, count = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with autocast(device, precision):
            loss = model(x, y).loss
        total += loss.item() * y.numel()
        count += y.numel()
    model.train(was_training)
    return total / max(1, count)


def sample_text(model: MiniGPT, tokenizer, prompt: str, n_tokens: int, device: torch.device,
                temperature: float = 0.8, seed: int = 0) -> str:
    ids = torch.tensor([tokenizer.encode(prompt, strict=False) or [0]], device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    out = model.generate(ids, n_tokens, temperature=temperature, generator=gen)
    return tokenizer.decode(out[0].tolist())


# ---------------------------------------------------------------------------- main loop
def train(cfg: Config, resume: str | None = None, quiet: bool = False,
          return_model: bool = False) -> dict[str, Any] | tuple[dict[str, Any], MiniGPT]:
    t = cfg.train
    set_seed(t.seed)
    device = pick_device(t.device)
    precision = pick_precision(t.precision, device)
    say = (lambda *a, **k: None) if quiet else functools.partial(print, flush=True)

    tokenizer = load_tokenizer(cfg.data.data_dir)
    cfg.model.vocab_size = tokenizer.vocab_size
    train_ids = encode_split(tokenizer, cfg.data.data_dir, "train")
    if cfg.data.max_train_tokens:
        train_ids = train_ids[: cfg.data.max_train_tokens]
    val_ids = encode_split(tokenizer, cfg.data.data_dir, "val")

    model = MiniGPT(cfg.model).to(device)
    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda") if precision == "fp16" else None
    n_params = model.num_parameters()

    start_step = 0
    if resume:
        _, _, state = load_checkpoint(resume, device)
        model.load_state_dict(state["model_state"])
        if "optimizer_state" in state:
            optimizer.load_state_dict(state["optimizer_state"])
        if scaler is not None and "scaler_state" in state:
            scaler.load_state_dict(state["scaler_state"])
        start_step = state["step"]
        say(f"resumed from {resume} at step {start_step}")

    T = cfg.model.context_length
    loader = make_loader(train_ids, T, t.batch_size, stride=cfg.data.train_stride,
                         shuffle=True, seed=t.seed + start_step, num_workers=t.num_workers)
    windows_per_epoch = len(loader.dataset)
    # fixed subset of training windows, evaluated exactly like validation (eval mode, no dropout)
    g = torch.Generator().manual_seed(t.seed)
    sub_start = torch.randint(0, max(1, len(train_ids) - len(val_ids)), (1,), generator=g).item()
    train_eval_ids = train_ids[sub_start: sub_start + len(val_ids)]

    run_dir = Path(t.out_dir) / t.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(t.ckpt_dir) / t.run_name
    write_json(run_dir / "config.json", cfg.to_dict())
    metrics_f = open(run_dir / "metrics.jsonl", "a" if resume else "w", encoding="utf-8")
    samples_f = open(run_dir / "samples.txt", "a" if resume else "w", encoding="utf-8")

    def log_event(rec: dict[str, Any]) -> None:
        metrics_f.write(json.dumps(rec) + "\n")
        metrics_f.flush()

    env = environment_info(device)
    say(f"device={device} ({env.get('gpu', 'cpu')}) precision={precision}")
    say(f"params={n_params:,} ({format_count(n_params)})  vocab={tokenizer.vocab_size}  "
        f"context={T}  train_tokens={len(train_ids):,}  windows/epoch={windows_per_epoch:,}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_val = math.inf
    best_step = -1
    history: list[dict[str, Any]] = []
    run_loss, run_n = 0.0, 0
    tokens_since, time_since = 0, time.perf_counter()
    train_time = 0.0
    t0 = time.perf_counter()
    data_iter = infinite(loader)
    model.train()

    def do_eval(step: int) -> dict[str, Any]:
        nonlocal best_val, best_step
        val_loss = evaluate_loss(model, val_ids, T, t.batch_size, device, precision, t.eval_batches)
        tr_loss = evaluate_loss(model, train_eval_ids, T, t.batch_size, device, precision, t.eval_batches)
        rec = {"event": "eval", "step": step, "train_loss": tr_loss, "val_loss": val_loss,
               "val_ppl": math.exp(val_loss), "elapsed_s": time.perf_counter() - t0}
        log_event(rec)
        history.append(rec)
        improved = val_loss < best_val
        if improved:
            best_val, best_step = val_loss, step
            if t.save_checkpoints:
                save_checkpoint(ckpt_dir / "best.pt", model, tokenizer, full_config=cfg.to_dict(), step=step,
                                metrics={"val_loss": val_loss, "train_loss": tr_loss})
        sample = sample_text(model, tokenizer, t.sample_prompt, t.sample_tokens, device, seed=t.seed)
        samples_f.write(f"===== step {step} | val_loss {val_loss:.4f} =====\n{sample}\n\n")
        samples_f.flush()
        say(f"[eval] step {step:>6} | train {tr_loss:.4f} | val {val_loss:.4f} (ppl {math.exp(val_loss):.2f})"
            f"{'  *best*' if improved else ''}")
        return rec

    step = start_step
    for step in range(start_step, t.max_steps):
        if step % t.eval_interval == 0 and (step > start_step or step == 0):
            synchronize(device)
            train_time += time.perf_counter() - time_since
            do_eval(step)
            time_since = time.perf_counter()

        lr = lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        epoch, (x, y) = next(data_iter)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(device, precision):
            loss = model(x, y).loss
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
            optimizer.step()

        run_loss += loss.item()
        run_n += 1
        tokens_since += x.numel()
        if (step + 1) % t.log_interval == 0:
            synchronize(device)
            now = time.perf_counter()
            dt = now - time_since
            train_time += dt
            rec = {
                "event": "train", "step": step + 1,
                # epoch = target tokens processed / training tokens (windows overlap when stride < T)
                "epoch": (step + 1) * t.batch_size * T / len(train_ids),
                "lr": lr, "train_loss": run_loss / run_n, "grad_norm": float(grad_norm),
                "tokens_per_s": tokens_since / dt, "elapsed_s": now - t0,
            }
            log_event(rec)
            say(f"step {step + 1:>6} | epoch {rec['epoch']:.3f} | lr {lr:.2e} | loss {rec['train_loss']:.4f} "
                f"| {rec['tokens_per_s'] / 1e3:.0f}k tok/s")
            run_loss, run_n, tokens_since, time_since = 0.0, 0, 0, time.perf_counter()

    synchronize(device)
    train_time += time.perf_counter() - time_since
    final = do_eval(t.max_steps)
    total_time = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None

    if t.save_checkpoints:
        save_checkpoint(ckpt_dir / "last.pt", model, tokenizer, full_config=cfg.to_dict(), step=t.max_steps,
                        metrics={"val_loss": final["val_loss"], "train_loss": final["train_loss"]},
                        optimizer=optimizer, scaler=scaler)

    train_steps_run = t.max_steps - start_step
    summary = {
        "run_name": t.run_name,
        "parameters": n_params,
        "parameters_human": format_count(n_params),
        "model": cfg.model.__dict__,
        "steps": t.max_steps,
        "batch_size": t.batch_size,
        "tokens_seen": t.max_steps * t.batch_size * T,
        "epochs": t.max_steps * t.batch_size * T / len(train_ids),
        "precision": precision,
        "hardware": env,
        "train_time_s": train_time,
        "wall_clock_s": total_time,
        "tokens_per_s": train_steps_run * t.batch_size * T / max(train_time, 1e-9),
        "peak_memory_mb": peak_mem,
        "final_train_loss": final["train_loss"],
        "final_val_loss": final["val_loss"],
        "final_val_ppl": math.exp(final["val_loss"]),
        "best_val_loss": best_val,
        "best_step": best_step,
        "checkpoint_best": str(ckpt_dir / "best.pt") if t.save_checkpoints else None,
    }
    write_json(run_dir / "summary.json", summary)
    metrics_f.close()
    samples_f.close()
    say(f"done in {total_time / 60:.1f} min | final val {final['val_loss']:.4f} | best val {best_val:.4f} "
        f"@ step {best_step} | peak mem {peak_mem or 0:.0f} MB")
    if return_model:
        return summary, model.eval()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MiniGPT")
    parser.add_argument("--config", default="configs/tiny_gpt.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="override a config value, e.g. --set model.n_layers=2 (repeatable)")
    parser.add_argument("--resume", default=None, help="path to a last.pt checkpoint to resume from")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
