"""Lesson 01, end-to-end: a recurrent language model vs our Transformer under the same budget.

A 2-layer LSTM character LM with about the same parameter count as the main MiniGPT (~4.8M) is trained on
the same data, batch shape (64 x 256), optimizer, LR schedule and number of steps (3000). We compare
validation loss, throughput and wall-clock, and how loss falls over wall-clock time.

The LSTM uses PyTorch's cuDNN kernel (the fastest recurrent implementation available) - inside, it still has
to process the 256 time steps one after another, while attention processes them in parallel.

    python -m experiments.lstm_baseline
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.config import load_config  # noqa: E402
from src.dataset import encode_split, infinite, load_tokenizer, make_loader  # noqa: E402
from src.gpt import GPTOutput  # noqa: E402
from src.train import evaluate_loss, lr_at  # noqa: E402
from src.utils import autocast, pick_device, pick_precision, set_seed, synchronize, write_json  # noqa: E402

OUT = Path("outputs/lstm_baseline")


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> GPTOutput:
        h, _ = self.lstm(self.drop(self.emb(idx)))
        logits = self.head(self.drop(h))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return GPTOutput(logits, loss, None)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/tiny_gpt.yaml")
    p.add_argument("--hidden", type=int, default=544)          # 544 -> ~4.81M parameters (MiniGPT: 4.77M)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--gpt-run", default="outputs/logs/tiny_gpt")
    args = p.parse_args()

    cfg = load_config(args.config)
    t = cfg.train
    set_seed(t.seed)
    device = pick_device("auto")
    precision = pick_precision("auto", device)
    tok = load_tokenizer("data")
    train_ids, val_ids = encode_split(tok, "data", "train"), encode_split(tok, "data", "val")
    T = cfg.model.context_length

    model = LSTMLanguageModel(tok.vocab_size, args.hidden, args.layers, cfg.model.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=t.lr, betas=(t.beta1, t.beta2), weight_decay=t.weight_decay)
    loader = make_loader(train_ids, T, t.batch_size, shuffle=True, seed=t.seed)
    data = infinite(loader)
    print(f"LSTM: {n_params:,} parameters, hidden {args.hidden}, layers {args.layers}, precision {precision}")

    curve, train_time = [], 0.0
    for step in range(t.max_steps + 1):
        if step % t.eval_interval == 0 or step == t.max_steps:
            vl = evaluate_loss(model, val_ids, T, t.batch_size, device, precision)
            curve.append({"step": step, "train_time_s": train_time, "val_loss": vl})
            print(f"step {step:>5} | {train_time:7.1f}s | val {vl:.4f}", flush=True)
        if step == t.max_steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        _, (x, y) = next(data)
        x, y = x.to(device), y.to(device)
        synchronize(device)
        t0 = time.perf_counter()
        with autocast(device, precision):
            loss = model(x, y).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
        opt.step()
        synchronize(device)
        train_time += time.perf_counter() - t0

    best = min(c["val_loss"] for c in curve)
    result = {"parameters": n_params, "hidden": args.hidden, "layers": args.layers, "steps": t.max_steps,
              "final_val_loss": curve[-1]["val_loss"], "best_val_loss": best, "train_time_s": train_time,
              "tokens_per_s": t.max_steps * t.batch_size * T / train_time, "curve": curve}

    gpt_curve, gpt_summary = [], None
    run = Path(args.gpt_run)
    if (run / "metrics.jsonl").exists():
        gpt_summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        evals = [json.loads(l) for l in open(run / "metrics.jsonl", encoding="utf-8") if '"eval"' in l]
        sec_per_step = gpt_summary["train_time_s"] / gpt_summary["steps"]
        gpt_curve = [{"step": e["step"], "train_time_s": e["step"] * sec_per_step, "val_loss": e["val_loss"]}
                     for e in evals]
        result["gpt"] = {"parameters": gpt_summary["parameters"], "best_val_loss": gpt_summary["best_val_loss"],
                         "final_val_loss": gpt_summary["final_val_loss"], "train_time_s": gpt_summary["train_time_s"],
                         "tokens_per_s": gpt_summary["tokens_per_s"]}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "results.json", result)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for c, name, col in ((curve, f"LSTM ({n_params / 1e6:.2f}M)", "#d9480f"),
                         (gpt_curve, f"MiniGPT ({(gpt_summary or {}).get('parameters', 0) / 1e6:.2f}M)", "#1f5fbf")):
        if c:
            a1.plot([e["step"] for e in c], [e["val_loss"] for e in c], "o-", color=col, label=name)
            a2.plot([e["train_time_s"] for e in c], [e["val_loss"] for e in c], "o-", color=col, label=name)
    a1.set_xlabel("optimizer step")
    a2.set_xlabel("training wall-clock (s), excluding evaluation")
    for a in (a1, a2):
        a.set_ylabel("validation loss (nats/char)")
        a.set_ylim(1.2, 3.0)
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle("Recurrent vs Transformer language model - same data, parameters, batch and steps")
    fig.tight_layout()
    fig.savefig(OUT / "lstm_vs_gpt.png", dpi=150)
    fig.savefig("outputs/figures/fig0b_lstm_vs_gpt.png", dpi=150)

    lines = ["| model | params | best val loss | train time (s) | tokens/s |", "|---|---|---|---|---|",
             f"| LSTM ({args.layers}x{args.hidden}) | {n_params:,} | {best:.4f} | {train_time:.0f} | "
             f"{result['tokens_per_s']:,.0f} |"]
    if gpt_summary:
        g = result["gpt"]
        lines.append(f"| MiniGPT (6x256) | {g['parameters']:,} | {g['best_val_loss']:.4f} | {g['train_time_s']:.0f} | "
                     f"{g['tokens_per_s']:,.0f} |")
    (OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"perplexity LSTM {math.exp(best):.3f}; saved {OUT}/")


if __name__ == "__main__":
    main()
