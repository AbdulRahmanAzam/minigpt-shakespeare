"""Tensor shapes through every component, head split/merge, positional encoding, parameter count."""

import math

import pytest
import torch

from src.attention import MultiHeadAttention, merge_heads, split_heads
from src.config import ModelConfig
from src.gpt import MiniGPT, analytic_parameter_count
from src.positional_encoding import build_positional_encoding, sinusoidal_table
from src.transformer_block import FeedForward, TransformerBlock


def test_split_merge_heads_round_trip_and_layout():
    x = torch.randn(2, 5, 12)
    h = split_heads(x, 3)
    assert h.shape == (2, 3, 5, 4)
    # head 1 must hold feature slice 4:8 of every position
    torch.testing.assert_close(h[:, 1], x[..., 4:8])
    torch.testing.assert_close(merge_heads(h), x)


def test_multihead_attention_shapes_and_trace():
    mha = MultiHeadAttention(d_model=32, n_heads=4, context_length=16)
    trace: list[str] = []
    out, w = mha(torch.randn(2, 10, 32), return_attn=True, trace=trace)
    assert out.shape == (2, 10, 32)
    assert w.shape == (2, 4, 10, 10)
    text = "\n".join(trace)
    for expected in ["(2, 10, 32)", "(2, 4, 10, 8)", "(2, 4, 10, 10)"]:
        assert expected in text


def test_heads_are_computed_independently():
    """Each head's output only depends on its own Q/K/V slice."""
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=8, n_heads=2, context_length=6, causal=True).eval()
    x = torch.randn(1, 6, 8)
    q, k, v = (split_heads(p(x), 2) for p in (mha.q_proj, mha.k_proj, mha.v_proj))
    from src.attention import causal_mask, scaled_dot_product_attention
    head0, _ = scaled_dot_product_attention(q[:, 0], k[:, 0], v[:, 0], mask=causal_mask(6))
    head1, _ = scaled_dot_product_attention(q[:, 1], k[:, 1], v[:, 1], mask=causal_mask(6))
    expected = mha.out_proj(torch.cat([head0, head1], dim=-1))
    out, _ = mha(x)
    torch.testing.assert_close(out, expected)


def test_d_model_must_divide_by_heads():
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=10, n_heads=3, context_length=4)
    with pytest.raises(ValueError):
        ModelConfig(d_model=10, n_heads=3)


def test_sdpa_impl_matches_manual():
    torch.manual_seed(0)
    a = MultiHeadAttention(32, 4, 16, attn_impl="manual").eval()
    b = MultiHeadAttention(32, 4, 16, attn_impl="sdpa").eval()
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 16, 32)
    torch.testing.assert_close(a(x)[0], b(x)[0], atol=1e-5, rtol=1e-5)


def test_feedforward_and_block_preserve_shape(tiny_cfg):
    x = torch.randn(3, 7, tiny_cfg.d_model)
    assert FeedForward(tiny_cfg.d_model, tiny_cfg.ffn_dim)(x).shape == x.shape
    out, w = TransformerBlock(tiny_cfg)(x, return_attn=True)
    assert out.shape == x.shape and w.shape == (3, tiny_cfg.n_heads, 7, 7)


def test_gpt_logits_loss_and_attention_shapes(tiny_model, tiny_cfg):
    x = torch.randint(0, tiny_cfg.vocab_size, (2, 9))
    out = tiny_model(x, targets=x, return_attn=True)
    assert out.logits.shape == (2, 9, tiny_cfg.vocab_size)
    assert out.loss.dim() == 0
    assert len(out.attentions) == tiny_cfg.n_layers
    assert all(a.shape == (2, tiny_cfg.n_heads, 9, 9) for a in out.attentions)


def test_initial_loss_is_close_to_uniform(tiny_model, tiny_cfg):
    x = torch.randint(0, tiny_cfg.vocab_size, (8, tiny_cfg.context_length))
    loss = tiny_model(x, targets=x).loss.item()
    assert abs(loss - math.log(tiny_cfg.vocab_size)) < 0.3


def test_sequence_longer_than_context_is_rejected(tiny_model, tiny_cfg):
    with pytest.raises(ValueError):
        tiny_model(torch.zeros(1, tiny_cfg.context_length + 1, dtype=torch.long))


@pytest.mark.parametrize("pos", ["sinusoidal", "learned", "none"])
@pytest.mark.parametrize("tie", [False, True])
def test_parameter_count_matches_formula(pos, tie):
    cfg = ModelConfig(vocab_size=65, context_length=32, d_model=48, n_heads=4, n_layers=3,
                      pos_encoding=pos, tie_weights=tie)
    assert MiniGPT(cfg).num_parameters() == analytic_parameter_count(cfg)


def test_sinusoidal_values_match_closed_form():
    pe = sinusoidal_table(50, 16)
    for pos, i in [(0, 0), (3, 1), (17, 5), (49, 7)]:
        angle = pos / (10000 ** (2 * i / 16))
        assert pe[pos, 2 * i].item() == pytest.approx(math.sin(angle), abs=1e-5)
        assert pe[pos, 2 * i + 1].item() == pytest.approx(math.cos(angle), abs=1e-5)
    assert torch.all(pe[0, 0::2] == 0) and torch.all(pe[0, 1::2] == 1)


@pytest.mark.parametrize("kind", ["sinusoidal", "learned", "none"])
def test_positional_modules_shape(kind):
    assert build_positional_encoding(kind, 20, 8)(12).shape == (12, 8)


def test_position_changes_representation(tiny_model):
    """Same token at different positions must give different first-layer inputs."""
    x = torch.full((1, 6), 3, dtype=torch.long)
    h = tiny_model.tok_emb(x) * tiny_model.emb_scale + tiny_model.pos_enc(6)
    assert not torch.allclose(h[0, 0], h[0, 1])
