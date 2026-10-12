"""Figure 6 - training vs validation loss (+ learning-rate schedule) from a run's metrics.jsonl.

    python -m experiments.plot_training --run outputs/logs/tiny_gpt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("outputs/figures")


def read_metrics(run_dir: Path) -> tuple[list[dict], list[dict]]:
    train, evals = [], []
    with open(run_dir / "metrics.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            (train if rec["event"] == "train" else evals).append(rec)
    return train, evals


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="outputs/logs/tiny_gpt")
    p.add_argument("--out", default=str(OUT / "fig6_loss_curves.png"))
    args = p.parse_args()
    run = Path(args.run)
    train, evals = read_metrics(run)

    fig, (ax, ax_lr) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.plot([r["step"] for r in train], [r["train_loss"] for r in train], color="#9ab", lw=1,
            label="train loss (running mean over log interval, dropout on)")
    ax.plot([r["step"] for r in evals], [r["train_loss"] for r in evals], "o-", color="#1f5fbf",
            label="train loss (eval mode, fixed subset)")
    ax.plot([r["step"] for r in evals], [r["val_loss"] for r in evals], "o-", color="#d9480f",
            label="validation loss (full split)")
    best = min(evals, key=lambda r: r["val_loss"])
    ax.axvline(best["step"], color="#d9480f", ls=":", lw=1)
    ax.annotate(f"best val {best['val_loss']:.3f} @ step {best['step']}", (best["step"], best["val_loss"]),
                textcoords="offset points", xytext=(-150, 40), arrowprops={"arrowstyle": "->"})
    ax.axhline(math.log(65), color="gray", ls="--", lw=0.8)
    ax.text(0, math.log(65) + 0.03, "uniform guess ln(65) = 4.17", fontsize=8, color="gray")
    ax.set_ylabel("cross-entropy (nats / char)")
    ax.set_ylim(0.8, 4.4)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(f"Figure 6 - training vs validation loss ({run.name})")

    ax_lr.plot([r["step"] for r in train], [r["lr"] for r in train], color="#555")
    ax_lr.set_ylabel("learning rate")
    ax_lr.set_xlabel("optimizer step")
    ax_lr.grid(alpha=0.3)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    gap = evals[-1]["val_loss"] - evals[-1]["train_loss"]
    print(f"final: train {evals[-1]['train_loss']:.4f}, val {evals[-1]['val_loss']:.4f} (gap {gap:.4f}); "
          f"best val {best['val_loss']:.4f} @ step {best['step']}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
