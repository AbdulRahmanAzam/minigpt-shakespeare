"""MiniGPT: a decoder-only causal language model.

    token ids (B, T)
      -> token embedding * sqrt(d_model)            (B, T, d)
      +  positional encoding                         (T, d)  broadcast over the batch
      -> dropout
      -> n_layers x TransformerBlock (causal)        (B, T, d)
      -> final LayerNorm                             (B, T, d)
      -> LM head (Linear d -> vocab)                 (B, T, V)   logits for the next token at every position

Training loss: cross-entropy between logits[:, t] and targets[:, t] = ids[t + 1] (see src/dataset.py).
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig
from src.positional_encoding import build_positional_encoding
from src.transformer_block import TransformerBlock


class GPTOutput(NamedTuple):
    logits: torch.Tensor
    loss: torch.Tensor | None
    attentions: list[torch.Tensor] | None   # per layer: (B, H, T, T)


class MiniGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_enc = build_positional_encoding(cfg.pos_encoding, cfg.context_length, cfg.d_model)
        self.emb_scale = math.sqrt(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight      # weight tying (optional experiment)
        self.apply(self._init_weights)
        # GPT-2: scale the init of projections that write into the residual stream by 1/sqrt(2 * n_layers),
        # so the variance of the residual stream does not grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("attn.out_proj.weight") or name.endswith("ffn.fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_attn: bool = False,
        trace: list[str] | None = None,
    ) -> GPTOutput:
        B, T = idx.shape
        if T > self.cfg.context_length:
            raise ValueError(f"sequence length {T} > context_length {self.cfg.context_length}")
        log = trace.append if trace is not None else (lambda _s: None)
        log(f"token ids                   {tuple(idx.shape)}  (B, T)")

        x = self.tok_emb(idx) * self.emb_scale                # (B, T, d)
        log(f"token embedding * sqrt(d)   {tuple(x.shape)}  (B, T, d_model)")
        pe = self.pos_enc(T)                                  # (T, d)
        log(f"positional encoding         {tuple(pe.shape)}  (T, d_model) broadcast over batch")
        x = self.drop(x + pe)

        attentions: list[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            # only trace the first block in detail; the others are identical
            x, w = block(x, return_attn=return_attn, trace=trace if i == 0 else None)
            if return_attn:
                attentions.append(w)
        x = self.ln_f(x)
        logits = self.lm_head(x)                              # (B, T, V)
        log(f"final LayerNorm -> LM head  {tuple(logits.shape)}  (B, T, vocab_size)")

        loss = None
        if targets is not None:
            # Token-level cross-entropy, averaged over all B*T positions. ignore_index=-100 lets the
            # MLM experiment score only masked positions with the same code path.
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return GPTOutput(logits, loss, attentions if return_attn else None)

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = None,
        top_p: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling. temperature == 0 -> greedy (argmax) decoding.

        At every step only the last ``context_length`` tokens are fed to the model (sliding window),
        the logits of the final position give the next-token distribution, and the sampled token is
        appended to the sequence.
        """
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.context_length:]
            logits = self(idx_cond).logits[:, -1, :].float()     # (B, V) next-token logits
            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(filter_logits(logits / temperature, top_k, top_p), dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat([idx, next_id], dim=1)
        self.train(was_training)
        return idx

    # ------------------------------------------------------------------ introspection
    def num_parameters(self) -> int:
        # parameters() de-duplicates tied weights, so this is the true trainable count
        return sum(p.numel() for p in self.parameters())

    def parameter_table(self) -> list[tuple[str, tuple[int, ...], int]]:
        return [(n, tuple(p.shape), p.numel()) for n, p in self.named_parameters()]

    def config_dict(self) -> dict:
        return asdict(self.cfg)


def filter_logits(logits: torch.Tensor, top_k: int | None = None, top_p: float | None = None) -> torch.Tensor:
    """Top-k and/or nucleus (top-p) filtering: disallowed tokens get -inf."""
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # drop a token if the mass *before* it already reaches top_p (always keeps the top token)
        drop = (cum - torch.softmax(sorted_logits, dim=-1)) >= top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return logits


def analytic_parameter_count(cfg: ModelConfig) -> int:
    """Closed-form parameter count, used by tests to check the implementation and in the report."""
    d, f, V, L, b = cfg.d_model, cfg.ffn_dim, cfg.vocab_size, cfg.n_layers, int(cfg.bias)
    emb = V * d
    pos = cfg.context_length * d if cfg.pos_encoding == "learned" else 0
    attn = 4 * (d * d + b * d)                     # Q, K, V, O projections
    ffn = d * f + b * f + f * d + b * d
    norms = 2 * (2 * d)                            # two LayerNorms (gamma, beta)
    per_block = attn + ffn + norms
    head = 0 if cfg.tie_weights else d * V
    return emb + pos + L * per_block + 2 * d + head    # + final LayerNorm
