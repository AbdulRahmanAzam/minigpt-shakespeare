"""NumPy-only MiniGPT inference (no PyTorch), used by the deployed REST API.

The forward pass mirrors src/gpt.py exactly (pre-LayerNorm blocks, exact-erf GELU, sinusoidal or learned
positions, causal attention) and runs in float32. Weights come from a ``.npz`` file written by
``scripts/export_numpy.py``, which keeps the server image small: numpy instead of a ~200 MB torch wheel.

Sampling uses NumPy's random generator, so a given seed is reproducible here but does not produce the same
text as the PyTorch sampler; greedy decoding (temperature 0) matches.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.tokenizer import CharTokenizer

META_KEY = "__meta__"


def _erf(x: np.ndarray) -> np.ndarray:
    """Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7, below float32 resolution of GELU's output).

    Written with in-place float32 ops: GELU is the most expensive part of the forward pass otherwise.
    """
    a = np.abs(x)
    t = 1.0 + np.float32(0.3275911) * a
    np.reciprocal(t, out=t)
    y = np.float32(1.061405429) * t
    for c in (-1.453152027, 1.421413741, -0.284496736, 0.254829592):    # Horner's rule
        y += np.float32(c)
        y *= t
    np.multiply(a, a, out=a)
    np.negative(a, out=a)
    np.exp(a, out=a)
    y *= a
    np.subtract(1.0, y, out=y)                     # erf(|x|)
    return np.copysign(y, x, out=y)


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def sinusoidal_table(max_len: int, d_model: int) -> np.ndarray:
    pos = np.arange(max_len, dtype=np.float32)[:, None]
    inv_freq = np.exp(-math.log(10000.0) * np.arange(0, d_model, 2, dtype=np.float32) / d_model)
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(pos * inv_freq)
    pe[:, 1::2] = np.cos(pos * inv_freq)
    return pe


class NumpyGPT:
    def __init__(self, weights: dict[str, np.ndarray], config: dict[str, Any]):
        self.cfg = dict(config)
        self.w = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}
        c = self.cfg
        if c["pos_encoding"] == "sinusoidal":
            self.pe = sinusoidal_table(c["context_length"], c["d_model"])
        elif c["pos_encoding"] == "learned":
            self.pe = self.w["pos_enc.emb.weight"]
        else:
            self.pe = np.zeros((c["context_length"], c["d_model"]), dtype=np.float32)
        self.lm_head = self.w["tok_emb.weight"] if c["tie_weights"] else self.w["lm_head.weight"]

    def _linear(self, x: np.ndarray, name: str) -> np.ndarray:
        y = x @ self.w[f"{name}.weight"].T
        bias = self.w.get(f"{name}.bias")
        return y if bias is None else y + bias

    def forward(self, ids: np.ndarray, cache: list[tuple[np.ndarray, np.ndarray]] | None = None) -> np.ndarray:
        """ids (T,) -> logits (T, vocab_size).

        ``cache`` holds per-layer keys/values (H, T_past, d_head) of the tokens already processed; the new
        tokens take positions T_past, T_past + 1, ... and the cache is extended in place. Pass ``[]`` to start
        one, or None for a plain full forward pass.
        """
        c = self.cfg
        past = cache[0][0].shape[1] if cache else 0
        T, H, d = len(ids), c["n_heads"], c["d_model"]
        if past + T > c["context_length"]:
            raise ValueError(f"sequence length {past + T} > context_length {c['context_length']}")
        dh = d // H
        x = self.w["tok_emb.weight"][ids] * np.float32(math.sqrt(d)) + self.pe[past:past + T]
        # query i sits at absolute position past + i and may not see keys after it
        blocked = np.triu(np.ones((T, past + T), dtype=bool), k=past + 1) if c["causal"] else None
        for i in range(c["n_layers"]):
            p = f"blocks.{i}."
            h = layer_norm(x, self.w[p + "ln1.weight"], self.w[p + "ln1.bias"])
            q, k, v = (self._linear(h, p + f"attn.{n}_proj").reshape(T, H, dh).transpose(1, 0, 2)
                       for n in ("q", "k", "v"))
            if cache is not None:
                if len(cache) > i:
                    k = np.concatenate([cache[i][0], k], axis=1)
                    v = np.concatenate([cache[i][1], v], axis=1)
                    cache[i] = (k, v)
                else:
                    cache.append((k, v))
            scores = q @ k.transpose(0, 2, 1) / np.float32(math.sqrt(dh))     # (H, T, past + T)
            if blocked is not None:
                scores = np.where(blocked, -np.inf, scores)
            out = (softmax(scores) @ v).transpose(1, 0, 2).reshape(T, d)
            x = x + self._linear(out, p + "attn.out_proj")
            h = layer_norm(x, self.w[p + "ln2.weight"], self.w[p + "ln2.bias"])
            x = x + self._linear(gelu(self._linear(h, p + "ffn.fc1")), p + "ffn.fc2")
        x = layer_norm(x, self.w["ln_f.weight"], self.w["ln_f.bias"])
        return (x @ self.lm_head.T).astype(np.float32)

    def generate(self, ids: list[int], max_new_tokens: int, temperature: float = 0.8,
                 top_k: int | None = None, top_p: float | None = None,
                 rng: np.random.Generator | None = None) -> list[int]:
        """Autoregressive sampling with a sliding window of ``context_length`` tokens (as MiniGPT.generate)."""
        rng = rng or np.random.default_rng()
        ids = list(ids)
        ctx = self.cfg["context_length"]
        cache: list[tuple[np.ndarray, np.ndarray]] = []
        fed = 0                                     # tokens of ``ids`` already in the cache
        for _ in range(max_new_tokens):
            if len(ids) > ctx:
                # the window slid, so every position changed: recompute it (exactly what MiniGPT does)
                cache, fed = [], len(ids) - ctx
            logits = self.forward(np.array(ids[fed:]), cache)[-1]
            fed = len(ids)
            if temperature == 0:
                nxt = int(logits.argmax())
            else:
                probs = softmax(filter_logits(logits / temperature, top_k, top_p).astype(np.float64))
                nxt = int(rng.choice(len(probs), p=probs))
            ids.append(nxt)
        return ids

    def num_parameters(self) -> int:
        tied = self.cfg["tie_weights"]           # a tied lm_head is the embedding, count it once
        return sum(v.size for k, v in self.w.items() if not (tied and k == "lm_head.weight"))

    def config_dict(self) -> dict[str, Any]:
        return dict(self.cfg)


def filter_logits(logits: np.ndarray, top_k: int | None = None, top_p: float | None = None) -> np.ndarray:
    """Top-k and/or nucleus filtering for one (V,) logit vector; same rules as src.gpt.filter_logits."""
    logits = logits.copy()
    if top_k is not None and top_k > 0:
        kth = np.sort(logits)[-min(top_k, logits.size)]
        logits[logits < kth] = -np.inf
    if top_p is not None and 0.0 < top_p < 1.0:
        order = np.argsort(-logits, kind="stable")
        p = softmax(logits[order])
        drop = (np.cumsum(p) - p) >= top_p            # always keeps the top token
        logits[order[drop]] = -np.inf
    return logits


# ---------------------------------------------------------------------- files
def save_npz(path: str | Path, weights: dict[str, np.ndarray], config: dict[str, Any],
             tokenizer: CharTokenizer, step: int | None = None, metrics: dict[str, Any] | None = None) -> None:
    meta = {"model_config": config, "tokenizer": tokenizer.to_dict(), "step": step, "metrics": metrics or {}}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{META_KEY: np.array(json.dumps(meta))}, **weights)


def load_npz(path: str | Path) -> tuple[NumpyGPT, CharTokenizer, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f[META_KEY]))
        weights = {k: f[k] for k in f.files if k != META_KEY}
    return NumpyGPT(weights, meta["model_config"]), CharTokenizer.from_dict(meta["tokenizer"]), meta


def from_torch_checkpoint(path: str | Path) -> tuple[NumpyGPT, CharTokenizer, dict[str, Any]]:
    """Load a .pt training checkpoint (needs torch; only for local use, the server image ships .npz)."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    weights = {k: v.float().numpy() for k, v in state["model_state"].items()}
    return NumpyGPT(weights, state["model_config"]), CharTokenizer.from_dict(state["tokenizer"]), state


def load_model(path: str | Path) -> tuple[NumpyGPT, CharTokenizer, dict[str, Any]]:
    return load_npz(path) if Path(path).suffix == ".npz" else from_torch_checkpoint(path)


# ---------------------------------------------------------------------- text in, text out
@dataclass
class GenerationResult:
    prompt: str
    completion: str
    text: str
    ignored_chars: list[str]
    new_tokens: int
    seconds: float


def generate_text(model: NumpyGPT, tokenizer: CharTokenizer, prompt: str, max_new_tokens: int = 200,
                  temperature: float = 0.8, top_k: int | None = None, top_p: float | None = None,
                  seed: int | None = None) -> GenerationResult:
    ignored = tokenizer.unknown_chars(prompt)
    ids = tokenizer.encode(prompt, strict=False)
    clean_prompt = tokenizer.decode(ids)
    if not ids:                       # empty prompt: start from a newline, like the start of a speech
        ids = tokenizer.encode("\n")
    start = time.perf_counter()
    out = model.generate(ids, max_new_tokens, temperature, top_k, top_p, np.random.default_rng(seed))
    completion = tokenizer.decode(out[len(ids):])
    return GenerationResult(clean_prompt, completion, clean_prompt + completion, ignored, max_new_tokens,
                            time.perf_counter() - start)
