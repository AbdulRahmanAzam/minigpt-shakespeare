"""Tiny overfit gate (acceptance test 3): loss must collapse on a tiny repeated batch.

If target shifting, masking or gradients were broken, the model could not memorise the batch.
"""

import math

import torch

from src.config import ModelConfig
from src.gpt import MiniGPT


def test_model_memorises_a_tiny_repeated_batch():
    torch.manual_seed(0)
    text = "To be, or not to be, that is the question."
    chars = sorted(set(text))
    ids = torch.tensor([chars.index(c) for c in text])
    T = 24
    x = torch.stack([ids[i: i + T] for i in range(0, len(ids) - T - 1, 4)])
    y = torch.stack([ids[i + 1: i + T + 1] for i in range(0, len(ids) - T - 1, 4)])   # shifted by one

    cfg = ModelConfig(vocab_size=len(chars), context_length=T, d_model=64, n_heads=4, n_layers=2, dropout=0.0)
    model = MiniGPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(200):
        loss = model(x, y).loss
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    final = model(x, y).loss.item()

    assert abs(first - math.log(len(chars))) < 0.5           # starts near a uniform guess
    assert final < 0.1 * first, f"loss only went {first:.3f} -> {final:.3f}"

    # and the memorised model reproduces the sentence greedily from its first character
    model.eval()
    out = model.generate(ids[:1].unsqueeze(0), max_new_tokens=len(ids) - 1, temperature=0)
    generated = "".join(chars[i] for i in out[0].tolist())
    assert generated[:20] == text[:20]
