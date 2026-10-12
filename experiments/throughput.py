"""Controlled speed / memory benchmark for the ablation settings.

Wall-clock numbers taken from full training runs on a laptop GPU are noisy: clocks change with temperature,
and runs happen at different times. Here every setting does the same thing - a full training step
(forward + backward + AdamW) on random tokens, 8192 tokens per step, no data loader - and the settings are
measured round-robin over several rounds, so thermal drift affects all of them equally. Reported: median
step time, tokens/s and peak GPU memory.

    python -m experiments.throughput
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from experiments.ablation import BASE, GROUPS, run_key
from src.config import load_config
from src.gpt import MiniGPT
from src.utils import autocast, pick_device, pick_precision, synchronize, write_json

OUT = Path("outputs/ablations")


def bench(cfg, device, precision, steps: int) -> tuple[list[float], float]:
    torch.manual_seed(0)
    model = MiniGPT(cfg.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    B, T = cfg.train.batch_size, cfg.model.context_length
    x = torch.randint(0, cfg.model.vocab_size, (B, T), device=device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for i in range(steps + 5):
        synchronize(device)
        t0 = time.perf_counter()
        with autocast(device, precision):
            loss = model(x, x).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        synchronize(device)
        if i >= 5:                                   # first steps include kernel warm-up
            times.append(time.perf_counter() - t0)
    mem = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else float("nan")
    del model, opt
    return times, mem


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--steps", type=int, default=40)
    args = p.parse_args()
    device = pick_device("auto")
    precision = pick_precision("auto", device)

    settings = {}
    for g, items in GROUPS.items():
        for label, ov in items:
            settings.setdefault(run_key(ov), (ov, []))[1].append(f"{g}/{label}")
    times: dict[str, list[float]] = {k: [] for k in settings}
    mems: dict[str, float] = {}
    for r in range(args.rounds):
        for key, (ov, _) in settings.items():
            cfg = load_config(BASE, ov)
            cfg.model.vocab_size = 65
            t, m = bench(cfg, device, precision, args.steps)
            times[key] += t
            mems[key] = m
        print(f"round {r + 1}/{args.rounds} done", flush=True)

    results = {}
    lines = ["| setting | batch x context | median step (ms) | tokens/s | peak memory (MB) |", "|---|---|---|---|---|"]
    for key, (ov, labels) in settings.items():
        cfg = load_config(BASE, ov)
        med = statistics.median(times[key])
        tps = cfg.train.batch_size * cfg.model.context_length / med
        results[key] = {"labels": labels, "median_step_ms": med * 1000, "tokens_per_s": tps, "peak_memory_mb": mems[key],
                        "batch": cfg.train.batch_size, "context": cfg.model.context_length}
        lines.append(f"| {', '.join(labels)} | {cfg.train.batch_size} x {cfg.model.context_length} | {med * 1000:.1f} | "
                     f"{tps:,.0f} | {mems[key]:.0f} |")
    text = "\n".join(lines)
    print(text)
    write_json(OUT / "throughput.json", {"device": str(device), "precision": precision, "rounds": args.rounds,
                                         "steps_per_round": args.steps, "results": results})
    (OUT / "throughput.md").write_text(
        f"# Controlled throughput benchmark ({device}, {precision})\n\nFull training step on random tokens, "
        f"{args.rounds} interleaved rounds x {args.steps} steps per setting, median reported.\n\n" + text + "\n",
        encoding="utf-8")
    print(f"saved {OUT}/throughput.md")


if __name__ == "__main__":
    main()
