"""Figure 5 - model summary and parameter count (from a checkpoint if given, else from a config).

    python -m experiments.model_summary --checkpoint checkpoints/minigpt_shakespeare.pt
    python -m experiments.model_summary --config configs/tiny_gpt.yaml
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from src.checkpoint import load_checkpoint
from src.config import load_config
from src.gpt import MiniGPT, analytic_parameter_count
from src.utils import format_count

OUT = Path("outputs/figures")


def summarize(model: MiniGPT) -> str:
    cfg = model.cfg
    lines = ["MiniGPT model summary", "=" * 78]
    for k, v in model.config_dict().items():
        lines.append(f"  {k:<16} {v}")
    lines += ["", f"  {'module':<34}{'shape':<22}{'params':>12}", "  " + "-" * 68]
    groups: dict[str, int] = defaultdict(int)
    for name, shape, n in model.parameter_table():
        if name.startswith("blocks.0.") or not name.startswith("blocks."):
            lines.append(f"  {name:<34}{str(shape):<22}{n:>12,}")
        top = name.split(".")[0] if not name.startswith("blocks.") else "blocks (all layers)"
        groups[top] += n
    lines.append(f"  ... blocks.1 - blocks.{cfg.n_layers - 1} identical to blocks.0")
    lines += ["", "  parameters by component:"]
    total = model.num_parameters()
    for g, n in groups.items():
        lines.append(f"    {g:<24}{n:>12,}  ({100 * n / total:5.1f}%)")
    lines += ["", f"  TOTAL trainable parameters: {total:,} ({format_count(total)})",
              f"  analytic formula agrees:    {analytic_parameter_count(cfg) == total}",
              f"  fp32 weight size:           {total * 4 / 2**20:.1f} MB",
              "", "  architecture:",
              f"    tokens (B,T) -> Embedding({cfg.vocab_size},{cfg.d_model}) * sqrt({cfg.d_model}) + {cfg.pos_encoding} PE",
              f"    -> {cfg.n_layers} x [x + MHA_{cfg.n_heads}heads(LN(x)); x + FFN_{cfg.d_model}->{cfg.ffn_dim}->{cfg.d_model}(LN(x))]",
              f"    -> LayerNorm -> Linear({cfg.d_model},{cfg.vocab_size}) -> logits (B,T,{cfg.vocab_size})"]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--config", default="configs/tiny_gpt.yaml")
    args = p.parse_args()
    if args.checkpoint:
        model, _, _ = load_checkpoint(args.checkpoint)
    else:
        model = MiniGPT(load_config(args.config).model)
    text = summarize(model)
    print(text)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fig5_model_summary.txt").write_text(text + "\n", encoding="utf-8")
    print(f"\nsaved {OUT / 'fig5_model_summary.txt'}")


if __name__ == "__main__":
    main()
