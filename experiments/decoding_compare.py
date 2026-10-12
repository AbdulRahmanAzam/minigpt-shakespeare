"""Optional experiment - generation strategy: greedy vs temperature vs top-k vs top-p.

For each strategy we sample several continuations of fixed prompts and measure
  * valid-word rate         (fraction of words that exist in the training corpus)
  * distinct-1 / distinct-2 (lexical diversity)
  * 4-gram repetition       (how much the text loops)
  * model NLL of the text    (how "typical" the text is under the model itself; low = safe/boring)

    python -m experiments.decoding_compare --checkpoint checkpoints/minigpt_shakespeare.pt
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch

from experiments.text_metrics import all_metrics
from src.checkpoint import load_checkpoint
from src.generate import generate_text, resolve_checkpoint
from src.utils import pick_device, write_json

OUT = Path("outputs/decoding")
PROMPTS = ["ROMEO:", "First Citizen:\n", "To be, or not to be", "KING HENRY VI:\n"]
STRATEGIES = {
    "greedy (T=0)": {"temperature": 0.0},
    "T=0.5": {"temperature": 0.5},
    "T=0.8": {"temperature": 0.8},
    "T=1.0": {"temperature": 1.0},
    "T=1.3": {"temperature": 1.3},
    "T=0.8, top-k=10": {"temperature": 0.8, "top_k": 10},
    "T=1.0, top-p=0.9": {"temperature": 1.0, "top_p": 0.9},
}


@torch.no_grad()
def model_nll(model, tok, text: str) -> float:
    ids = torch.tensor([tok.encode(text, strict=False)], device=next(model.parameters()).device)
    ids = ids[:, -(model.cfg.context_length + 1):]
    return model(ids[:, :-1], ids[:, 1:]).loss.item()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--tokens", type=int, default=300)
    args = p.parse_args()
    device = pick_device("auto")
    model, tok, _ = load_checkpoint(resolve_checkpoint(args.checkpoint), device)

    results, examples = {}, {}
    for name, kw in STRATEGIES.items():
        metrics = []
        for i, prompt in enumerate(PROMPTS):
            for s in range(args.samples if kw["temperature"] > 0 else 1):     # greedy is deterministic
                r = generate_text(model, tok, prompt, args.tokens, seed=1000 * i + s, **kw)
                m = all_metrics(r.completion)
                m["model_nll"] = model_nll(model, tok, r.text)
                metrics.append(m)
                if i == 0 and s == 0:
                    examples[name] = r.text
        results[name] = {k: statistics.fmean(m[k] for m in metrics) for k in metrics[0]}
        print(f"{name:<18} " + "  ".join(f"{k} {v:.3f}" for k, v in results[name].items()), flush=True)

    md = ["# Decoding strategy comparison", "",
          f"{len(PROMPTS)} prompts x {args.samples} samples x {args.tokens} characters (greedy: 1 sample per prompt).", "",
          "| strategy | valid words | distinct-1 | distinct-2 | 4-gram repetition | model NLL |", "|---|---|---|---|---|---|"]
    for name, r in results.items():
        md.append(f"| {name} | {r['valid_word_rate']:.3f} | {r['distinct_1']:.3f} | {r['distinct_2']:.3f} | "
                  f"{r['repetition_4gram']:.3f} | {r['model_nll']:.3f} |")
    md += ["", "## Examples (prompt 'ROMEO:')", ""]
    for name, text in examples.items():
        md += [f"**{name}**", "", "```text", text, "```", ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.md").write_text("\n".join(md), encoding="utf-8")
    write_json(OUT / "results.json", {"metrics": results, "examples": examples})
    print(f"saved {OUT}/results.md")


if __name__ == "__main__":
    main()
