"""Reference losses that put the Transformer's numbers in context (nats per character on validation).

    uniform  : every one of the 65 characters equally likely           = ln(65)
    unigram  : character frequencies from the training split
    bigram   : P(c_t | c_{t-1}) with add-one smoothing
    trigram  : P(c_t | c_{t-2}, c_{t-1}) with add-one smoothing

    python -m experiments.baselines
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from src.dataset import load_split, load_tokenizer
from src.utils import write_json

OUT = Path("outputs/eval")


def ngram_loss(train: str, val: str, n: int, V: int) -> float:
    ctx_counts: Counter = Counter()
    counts: Counter = Counter()
    for i in range(n - 1, len(train)):
        ctx = train[i - n + 1: i]
        counts[(ctx, train[i])] += 1
        ctx_counts[ctx] += 1
    total = 0.0
    for i in range(n - 1, len(val)):
        ctx = val[i - n + 1: i]
        p = (counts[(ctx, val[i])] + 1) / (ctx_counts[ctx] + V)
        total -= math.log(p)
    return total / (len(val) - n + 1)


def main() -> None:
    train, val = load_split("data", "train"), load_split("data", "val")
    V = load_tokenizer("data").vocab_size
    results = {"uniform": math.log(V)}
    for n, name in ((1, "unigram"), (2, "bigram"), (3, "trigram"), (5, "5-gram")):
        results[name] = ngram_loss(train, val, n, V)
    print(f"{'model':<10}{'val loss (nats/char)':>22}{'perplexity':>12}{'bits/char':>11}")
    for k, v in results.items():
        print(f"{k:<10}{v:>22.4f}{math.exp(v):>12.2f}{v / math.log(2):>11.3f}")
    write_json(OUT / "baselines.json", results)
    print(f"saved {OUT / 'baselines.json'}")


if __name__ == "__main__":
    main()
