"""Simple, model-free quality metrics for generated text (used by evaluate / ablation / decoding scripts)."""

from __future__ import annotations

import re
from functools import lru_cache

from src.dataset import load_split

WORD = re.compile(r"[A-Za-z']+")


@lru_cache(maxsize=1)
def training_words(data_dir: str = "data") -> frozenset[str]:
    return frozenset(w.lower() for w in WORD.findall(load_split(data_dir, "train")))


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD.findall(text)]


def valid_word_rate(text: str, data_dir: str = "data") -> float:
    """Fraction of generated words that occur in the training corpus (spelling-level coherence)."""
    ws = words(text)
    if not ws:
        return 0.0
    vocab = training_words(data_dir)
    return sum(w in vocab for w in ws) / len(ws)


def distinct_n(text: str, n: int) -> float:
    """Unique word n-grams / total word n-grams (higher = more diverse)."""
    ws = words(text)
    grams = [tuple(ws[i: i + n]) for i in range(len(ws) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 0.0


def repetition_rate(text: str, n: int = 4) -> float:
    """Fraction of word n-grams that already appeared earlier in the same text (higher = loopier)."""
    ws = words(text)
    seen: set[tuple[str, ...]] = set()
    repeats = total = 0
    for i in range(len(ws) - n + 1):
        g = tuple(ws[i: i + n])
        repeats += g in seen
        seen.add(g)
        total += 1
    return repeats / total if total else 0.0


def all_metrics(text: str, data_dir: str = "data") -> dict[str, float]:
    return {
        "valid_word_rate": valid_word_rate(text, data_dir),
        "distinct_1": distinct_n(text, 1),
        "distinct_2": distinct_n(text, 2),
        "repetition_4gram": repetition_rate(text, 4),
    }
