"""Decoder-style Transformer block (pre-LayerNorm, as in GPT-2).

    x = x + Attention(LayerNorm(x))     # tokens exchange information (causally masked)
    x = x + FFN(LayerNorm(x))           # each position is transformed independently

Residual connections keep an identity path for gradients through every layer; LayerNorm before each
sublayer ("pre-norm") keeps the residual stream un-normalised, which trains stably without the long
learning-rate warm-up that the original post-norm Transformer needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.attention import MultiHeadAttention
from src.config import ModelConfig


class FeedForward(nn.Module):
    """Two linear maps with a GELU in between: d_model -> ffn_dim -> d_model."""

    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ffn_dim, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ffn_dim, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(
            cfg.d_model, cfg.n_heads, cfg.context_length, dropout=cfg.dropout,
            bias=cfg.bias, causal=cfg.causal, attn_impl=cfg.attn_impl,
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.ffn_dim, cfg.dropout, cfg.bias)

    def forward(
        self, x: torch.Tensor, return_attn: bool = False, trace: list[str] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_out, weights = self.attn(self.ln1(x), return_attn=return_attn, trace=trace)
        x = x + attn_out                       # residual around attention
        x = x + self.ffn(self.ln2(x))          # residual around the MLP
        if trace is not None:
            trace.append(f"block output (after FFN)    {tuple(x.shape)}  (B, T, d_model)")
        return x, weights
