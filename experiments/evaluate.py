"""Final evaluation (Figure 8 + quantitative results).

  * full validation and test loss, perplexity and bits-per-character
  * generations from fixed prompts with a fixed seed (temperature 0.8) + greedy
  * simple text-quality metrics (valid-word rate, distinct-n, repetition)

    python -m experiments.evaluate --checkpoint checkpoints/minigpt_shakespeare.pt
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch

from experiments.text_metrics import all_metrics
from src.checkpoint import load_checkpoint
from src.dataset import encode_split, load_split, make_loader
from src.generate import generate_text, resolve_checkpoint
from src.train import evaluate_loss
from src.utils import autocast, format_count, pick_device, pick_precision, read_json, write_json

OUT_EVAL = Path("outputs/eval")
OUT_SAMPLES = Path("outputs/samples")

FIXED_PROMPTS = [
    "ROMEO:",
    "JULIET:\n",
    "First Citizen:\n",
    "KING RICHARD III:\n",
    "To be, or not to be",
    "The future of artificial intelligence",
]


@torch.no_grad()
def next_char_accuracy(model, ids: torch.Tensor, T: int, device, precision: str, batch_size: int = 64) -> dict:
    """Top-1 / top-5 next-character accuracy over non-overlapping windows (same positions as the loss)."""
    model.eval()
    loader = make_loader(ids, T, batch_size, stride=T, shuffle=False, drop_last=False)
    top1 = top5 = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast(device, precision):
            logits = model(x).logits.float()
        top = logits.topk(5, dim=-1).indices                          # (B, T, 5)
        top1 += (top[..., 0] == y).sum().item()
        top5 += (top == y.unsqueeze(-1)).any(-1).sum().item()
        n += y.numel()
    return {"top1": top1 / n, "top5": top5 / n}


def ngram_accuracy(train: str, text: str, n: int) -> float:
    """Accuracy of always predicting the most frequent next character after the previous n-1 characters."""
    table: dict[str, Counter] = defaultdict(Counter)
    for i in range(n - 1, len(train)):
        table[train[i - n + 1: i]][train[i]] += 1
    fallback = Counter(train).most_common(1)[0][0]
    best = {ctx: c.most_common(1)[0][0] for ctx, c in table.items()}
    hits = sum(best.get(text[i - n + 1: i], fallback) == text[i] for i in range(n - 1, len(text)))
    return hits / (len(text) - n + 1)


def test_segments(model, tok, T: int, device, precision: str) -> dict:
    """Split the test text after the last speaker that also speaks in train/val. Everything after it is a play
    the model never saw (for Tiny Shakespeare: The Tempest, from line 899 'Master:'). Scoring both parts
    separately measures the domain shift."""
    speaker = re.compile(r"^([A-Z][A-Za-z ]+):$", re.M)
    seen = set(speaker.findall(load_split("data", "train") + load_split("data", "val")))
    test = load_split("data", "test")
    tags = [(m.start(), m.group(1)) for m in speaker.finditer(test)]
    seen_idx = [i for i, (_, name) in enumerate(tags) if name in seen]
    if not seen_idx or seen_idx[-1] + 1 >= len(tags):
        return {}
    cut = tags[seen_idx[-1] + 1][0]
    out = {"split_at_char": cut, "split_at_line": test.count("\n", 0, cut) + 1,
           "unseen_speakers": sorted({n for _, n in tags[seen_idx[-1] + 1:] if n.isupper()})}
    for name, text in (("seen_plays", test[:cut]), ("unseen_play", test[cut:])):
        ids = torch.tensor(tok.encode(text), dtype=torch.long)
        loss = evaluate_loss(model, ids, T, 64, device, precision)
        out[name] = {"chars": len(text), "loss": loss, "perplexity": math.exp(loss)}
        print(f"test/{name:<12} {len(text):>6} chars | loss {loss:.4f} | perplexity {math.exp(loss):.3f}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = pick_device(args.device)
    precision = pick_precision("auto", device)
    ckpt = resolve_checkpoint(args.checkpoint)
    model, tok, state = load_checkpoint(ckpt, device)
    T = model.cfg.context_length

    results: dict = {"checkpoint": str(ckpt), "step": state.get("step"), "parameters": model.num_parameters()}
    for split in ("val", "test"):
        ids = encode_split(tok, "data", split)
        loss = evaluate_loss(model, ids, T, 64, device, precision)
        results[split] = {"loss": loss, "perplexity": math.exp(loss), "bits_per_char": loss / math.log(2),
                          "tokens": len(ids)}
        acc = next_char_accuracy(model, ids, T, device, precision)
        results[split].update({"top1_accuracy": acc["top1"], "top5_accuracy": acc["top5"]})
        print(f"{split:<5} loss {loss:.4f} | perplexity {math.exp(loss):.3f} | bits/char {loss / math.log(2):.3f} | "
              f"next-char accuracy top-1 {100 * acc['top1']:.2f}% top-5 {100 * acc['top5']:.2f}%")
    train_text, val_text = load_split("data", "train"), load_split("data", "val")
    results["baseline_accuracy_val"] = {name: ngram_accuracy(train_text, val_text, n)
                                        for n, name in ((1, "most_frequent_char"), (2, "bigram"), (3, "trigram"))}
    print("baseline top-1 accuracy on val: " + ", ".join(
        f"{k} {100 * v:.2f}%" for k, v in results["baseline_accuracy_val"].items()))
    results["test_segments"] = test_segments(model, tok, T, device, precision)
    baselines_path = OUT_EVAL / "baselines.json"
    if baselines_path.exists():
        results["baselines"] = read_json(baselines_path)

    md = [f"# Generation samples - {ckpt.name}", "",
          f"Model: {format_count(model.num_parameters())} parameters, {model.cfg.n_layers} layers, "
          f"{model.cfg.n_heads} heads, d_model {model.cfg.d_model}, context {T}.  ",
          f"Sampling: temperature {args.temperature}, seed {args.seed}, {args.max_new_tokens} new characters.  ",
          f"Validation loss {results['val']['loss']:.4f} (perplexity {results['val']['perplexity']:.2f}); "
          f"test loss {results['test']['loss']:.4f}.", ""]
    samples = []
    for i, prompt in enumerate(FIXED_PROMPTS):
        r = generate_text(model, tok, prompt, args.max_new_tokens, args.temperature, seed=args.seed + i)
        greedy = generate_text(model, tok, prompt, 120, temperature=0.0)
        m = all_metrics(r.completion)
        samples.append({"prompt": prompt, "sampled": r.text, "greedy": greedy.text, "metrics": m,
                        "ignored_chars": r.ignored_chars})
        md += [f"## Prompt {i + 1}: `{prompt!r}`", "",
               f"*valid-word rate {m['valid_word_rate']:.2f}, distinct-2 {m['distinct_2']:.2f}, "
               f"4-gram repetition {m['repetition_4gram']:.2f}*" +
               (f" — dropped out-of-vocabulary characters: {r.ignored_chars}" if r.ignored_chars else ""), "",
               "**Sampled (T = %.1f):**" % args.temperature, "", "```text", r.text, "```", "",
               "**Greedy (T = 0), 120 characters:**", "", "```text", greedy.text, "```", ""]
        print(f"\n--- prompt {prompt!r} ---\n{r.text}")

    results["samples"] = samples
    results["sample_metrics_mean"] = {k: sum(s["metrics"][k] for s in samples) / len(samples)
                                      for k in samples[0]["metrics"]}
    write_json(OUT_EVAL / "final_eval.json", results)
    OUT_SAMPLES.mkdir(parents=True, exist_ok=True)
    (OUT_SAMPLES / "samples.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nmean sample metrics: {results['sample_metrics_mean']}")
    print(f"saved {OUT_EVAL / 'final_eval.json'} and {OUT_SAMPLES / 'samples.md'}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
