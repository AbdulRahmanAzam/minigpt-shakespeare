"""Causal masking: future positions can never receive attention or influence earlier predictions."""

import torch

from src.attention import MultiHeadAttention, causal_mask, scaled_dot_product_attention
from src.config import ModelConfig
from src.gpt import MiniGPT


def test_mask_is_strictly_upper_triangular():
    m = causal_mask(5)
    assert m.shape == (5, 5) and m.dtype == torch.bool
    for i in range(5):
        for j in range(5):
            assert bool(m[i, j]) == (j > i)


def test_future_positions_get_exactly_zero_weight():
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 8, 4), torch.randn(2, 8, 4), torch.randn(2, 8, 4)
    _, w = scaled_dot_product_attention(q, k, v, mask=causal_mask(8))
    assert torch.all(w.masked_select(causal_mask(8).expand_as(w)) == 0)
    assert torch.all(w[:, 0, 0] == 1)      # the first token can only attend to itself


def test_multihead_weights_are_lower_triangular():
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=16, n_heads=4, context_length=10).eval()
    _, w = mha(torch.randn(3, 10, 16), return_attn=True)
    assert w.shape == (3, 4, 10, 10)
    assert torch.all(torch.triu(w, diagonal=1) == 0)


def test_changing_future_tokens_does_not_change_past_logits(tiny_model):
    """Causal correctness at the model level (acceptance test 6)."""
    torch.manual_seed(1)
    V, T = tiny_model.cfg.vocab_size, tiny_model.cfg.context_length
    x = torch.randint(0, V, (2, T))
    base = tiny_model(x).logits
    for t in range(T - 1):
        x2 = x.clone()
        x2[:, t + 1:] = torch.randint(0, V, (2, T - t - 1))      # rewrite the entire future
        out = tiny_model(x2).logits
        torch.testing.assert_close(out[:, : t + 1], base[:, : t + 1], atol=1e-6, rtol=0)


def test_gradient_from_position_t_never_reaches_future_inputs(tiny_model):
    T, d = tiny_model.cfg.context_length, tiny_model.cfg.d_model
    x = torch.randint(0, tiny_model.cfg.vocab_size, (1, T))
    emb = (tiny_model.tok_emb(x) * tiny_model.emb_scale).detach().requires_grad_(True)
    h = emb + tiny_model.pos_enc(T)
    for block in tiny_model.blocks:
        h, _ = block(h)
    logits = tiny_model.lm_head(tiny_model.ln_f(h))
    t = 5
    logits[0, t].sum().backward()
    grad_per_pos = emb.grad[0].abs().sum(-1)                     # (T,)
    assert torch.all(grad_per_pos[t + 1:] == 0)
    assert torch.all(grad_per_pos[: t + 1] > 0)


def test_without_mask_future_tokens_leak():
    """Sanity check of the test itself: a bidirectional model DOES change when the future changes."""
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=11, context_length=8, d_model=16, n_heads=2, n_layers=1, dropout=0.0, causal=False)
    model = MiniGPT(cfg).eval()
    x = torch.randint(0, 11, (1, 8))
    x2 = x.clone()
    x2[0, -1] = (x2[0, -1] + 1) % 11
    assert not torch.allclose(model(x).logits[:, 0], model(x2).logits[:, 0])
