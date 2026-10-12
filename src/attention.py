"""Scaled dot-product attention and multi-head attention, written out explicitly.

Shapes (B = batch, T = sequence length, d = d_model, H = heads, d_h = d / H):

    x                (B, T, d)
    Q, K, V = xW     (B, T, d)          three learned projections
    split_heads      (B, H, T, d_h)     each head gets its own d_h-dim slice
    scores = QK^T    (B, H, T, T)       / sqrt(d_h)
    mask             (T, T)             True above the diagonal = "future" -> -inf
    weights          (B, H, T, T)       softmax over the last axis (each row sums to 1)
    weights @ V      (B, H, T, d_h)
    merge_heads      (B, T, d)          concatenate heads back together
    out = W_o(...)   (B, T, d)          final output projection
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(T: int, device: torch.device | str | None = None) -> torch.Tensor:
    """Boolean (T, T) mask, True where attention is *forbidden* (key position j > query position i).

    Example T=4:
        [[F, T, T, T],
         [F, F, T, T],
         [F, F, F, T],
         [F, F, F, F]]
    """
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """softmax(Q K^T / sqrt(d_k) + M) V.

    Args:
        q: (..., T_q, d_k)
        k: (..., T_k, d_k)
        v: (..., T_k, d_v)
        mask: boolean tensor broadcastable to (..., T_q, T_k); True = blocked (score set to -inf).
        dropout_p: dropout applied to the attention weights (training only).

    Returns:
        output (..., T_q, d_v) and the attention weights (..., T_q, T_k) before dropout.
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)          # (..., T_q, T_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))       # exp(-inf) = 0 after softmax
    weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)  # softmax in fp32 for stability
    attn = F.dropout(weights, p=dropout_p, training=training) if dropout_p > 0 else weights
    return attn @ v, weights


def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """(B, T, d) -> (B, H, T, d/H): reshape the feature axis into heads, then move heads before time."""
    B, T, d = x.shape
    return x.view(B, T, n_heads, d // n_heads).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """(B, H, T, d_h) -> (B, T, H*d_h): inverse of split_heads (concatenates the heads)."""
    B, H, T, d_h = x.shape
    return x.transpose(1, 2).contiguous().view(B, T, H * d_h)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        context_length: int,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = True,
        attn_impl: str = "manual",
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        self.attn_impl = attn_impl
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)
        # Precomputed mask for the maximum context; sliced to the actual length in forward().
        self.register_buffer("mask", causal_mask(context_length), persistent=False)

    def _mask(self, T: int, device: torch.device) -> torch.Tensor | None:
        if not self.causal:
            return None
        if T <= self.mask.size(0):
            return self.mask[:T, :T]
        return causal_mask(T, device)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
        trace: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T, d = x.shape
        log = trace.append if trace is not None else (lambda _s: None)
        log(f"input x                     {tuple(x.shape)}  (B, T, d_model)")

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        log(f"Q = x W_q, K, V             {tuple(q.shape)}  (B, T, d_model)")

        q, k, v = split_heads(q, self.n_heads), split_heads(k, self.n_heads), split_heads(v, self.n_heads)
        log(f"split_heads(Q)              {tuple(q.shape)}  (B, H, T, d_head)")

        mask = self._mask(T, x.device)
        if self.attn_impl == "sdpa" and not return_attn and trace is None:
            # PyTorch's fused kernel: mathematically identical (checked in tests), used only for speed.
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=self.causal, dropout_p=self.dropout if self.training else 0.0
            )
            weights = None
        else:
            out, weights = scaled_dot_product_attention(
                q, k, v, mask=mask, dropout_p=self.dropout, training=self.training
            )
            log(f"scores / weights per head   {tuple(weights.shape)}  (B, H, T, T)")
        log(f"weights @ V  (per head)     {tuple(out.shape)}  (B, H, T, d_head)")

        out = merge_heads(out)
        log(f"merge_heads (concat)        {tuple(out.shape)}  (B, T, d_model)")

        out = self.resid_dropout(self.out_proj(out))
        log(f"output projection W_o       {tuple(out.shape)}  (B, T, d_model)")
        return out, (weights if return_attn else None)
