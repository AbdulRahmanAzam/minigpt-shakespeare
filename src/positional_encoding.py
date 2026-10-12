"""Positional information.

Self-attention is permutation-equivariant: without position information, shuffling the input tokens
just shuffles the outputs. We therefore add a position-dependent vector to every token embedding.

Sinusoidal encoding (Vaswani et al., 2017, Sec. 3.5):

    PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
    PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))

Each pair of dimensions is a rotation at its own frequency (wavelengths 2*pi ... 10000*2*pi), so
PE[pos + k] is a fixed linear function of PE[pos] and nearby positions have similar encodings.
It has no parameters and is defined for any position.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_table(max_len: int, d_model: int) -> torch.Tensor:
    """(max_len, d_model) table of sinusoidal encodings."""
    if d_model % 2 != 0:
        raise ValueError("sinusoidal encoding needs an even d_model")
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)              # (L, 1)
    i2 = torch.arange(0, d_model, 2, dtype=torch.float32)                      # 2i = 0, 2, 4, ...
    inv_freq = torch.exp(-math.log(10000.0) * i2 / d_model)                    # 1 / 10000^(2i/d)
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * inv_freq)
    pe[:, 1::2] = torch.cos(pos * inv_freq)
    return pe


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.register_buffer("pe", sinusoidal_table(max_len, d_model), persistent=False)

    def forward(self, T: int) -> torch.Tensor:
        if T > self.pe.size(0):
            raise ValueError(f"sequence length {T} exceeds max_len {self.pe.size(0)}")
        return self.pe[:T]                                                     # (T, d_model)


class LearnedPositionalEmbedding(nn.Module):
    """One trainable vector per absolute position (GPT-2 style); used as an ablation."""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, T: int) -> torch.Tensor:
        if T > self.emb.num_embeddings:
            raise ValueError(f"sequence length {T} exceeds max_len {self.emb.num_embeddings}")
        return self.emb(torch.arange(T, device=self.emb.weight.device))


class NoPositionalEncoding(nn.Module):
    """Ablation: no explicit position signal (the causal mask is then the only source of order)."""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.register_buffer("zero", torch.zeros(1), persistent=False)

    def forward(self, T: int) -> torch.Tensor:
        return self.zero.new_zeros(T, self.d_model)


def build_positional_encoding(kind: str, max_len: int, d_model: int) -> nn.Module:
    if kind == "sinusoidal":
        return SinusoidalPositionalEncoding(max_len, d_model)
    if kind == "learned":
        return LearnedPositionalEmbedding(max_len, d_model)
    if kind == "none":
        return NoPositionalEncoding(max_len, d_model)
    raise ValueError(f"unknown positional encoding {kind!r}")
