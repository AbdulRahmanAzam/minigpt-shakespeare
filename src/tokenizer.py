"""Deterministic character-level tokenizer.

Design (documented for the report):
  * Vocabulary = the sorted set of unique characters in the *training* split (sorted by Unicode
    code point), so the same corpus always yields the same ids. For Tiny Shakespeare: 65 symbols.
  * Special-token policy: **no special tokens**. The corpus is one continuous stream that we cut into
    fixed-length windows, so BOS/EOS/PAD are never needed. Characters outside the vocabulary
    (e.g. digits other than '3', '"', '(') can only come from user prompts; ``encode(strict=False)``
    drops them and reports which ones were dropped, ``encode(strict=True)`` raises. We deliberately do
    not add an <unk> id because its embedding would never be trained.
"""

from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    def __init__(self, chars: list[str]):
        if len(set(chars)) != len(chars):
            raise ValueError("vocabulary contains duplicates")
        if any(len(c) != 1 for c in chars):
            raise ValueError("every vocabulary entry must be a single character")
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, d: dict) -> "CharTokenizer":
        return cls(d["chars"])

    def to_dict(self) -> dict:
        return {"type": "char", "vocab_size": self.vocab_size, "chars": self.chars}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------ encode / decode
    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def unknown_chars(self, text: str) -> list[str]:
        """Characters in ``text`` that are not in the vocabulary (unique, in order of appearance)."""
        seen: dict[str, None] = {}
        for c in text:
            if c not in self.stoi:
                seen.setdefault(c, None)
        return list(seen)

    def encode(self, text: str, strict: bool = True) -> list[int]:
        if strict:
            unknown = self.unknown_chars(text)
            if unknown:
                raise ValueError(f"characters not in vocabulary: {unknown!r}")
            return [self.stoi[c] for c in text]
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def __len__(self) -> int:
        return self.vocab_size
