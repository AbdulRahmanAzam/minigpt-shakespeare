"""Scaled dot-product attention: hand-calculated values, softmax properties, reference equivalence."""

import math

import pytest
import torch
import torch.nn.functional as F

from src.attention import causal_mask, scaled_dot_product_attention
from tests.conftest import HAND_K, HAND_Q, HAND_V

Q, K, V = (torch.tensor(m, dtype=torch.float64) for m in (HAND_Q, HAND_K, HAND_V))


def test_hand_example_unmasked_matches_manual_calculation():
    # Every intermediate value below was computed by hand (see experiments/hand_attention.py).
    scores = Q @ K.T
    assert torch.equal(scores, torch.tensor([[1.0, 1, 0], [0, 1, 1], [1, 2, 1]], dtype=torch.float64))
    scaled = scores / math.sqrt(2)
    e = torch.exp(scaled)
    weights_by_hand = e / e.sum(dim=1, keepdim=True)
    out, weights = scaled_dot_product_attention(Q, K, V)
    torch.testing.assert_close(weights, weights_by_hand)
    torch.testing.assert_close(
        weights, torch.tensor([[0.4011, 0.4011, 0.1978], [0.1978, 0.4011, 0.4011], [0.2483, 0.5035, 0.2483]],
                              dtype=torch.float64), atol=1e-4, rtol=0)
    torch.testing.assert_close(
        out, torch.tensor([[0.5989, 1.0000], [0.5989, 1.2033], [0.4965, 1.2552]], dtype=torch.float64),
        atol=1e-4, rtol=0)


def test_hand_example_causal_matches_manual_calculation():
    out, weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask(3))
    torch.testing.assert_close(
        weights, torch.tensor([[1.0, 0, 0], [0.3302, 0.6698, 0], [0.2483, 0.5035, 0.2483]], dtype=torch.float64),
        atol=1e-4, rtol=0)
    torch.testing.assert_close(
        out, torch.tensor([[1.0, 0.0], [0.3302, 1.3395], [0.4965, 1.2552]], dtype=torch.float64),
        atol=1e-4, rtol=0)
    # the last token already sees every earlier token, so masking does not change it
    full_out, _ = scaled_dot_product_attention(Q, K, V)
    torch.testing.assert_close(out[2], full_out[2])


@pytest.mark.parametrize("shape", [(5, 8), (2, 7, 4), (2, 3, 6, 16)])
def test_output_and_weight_shapes(shape):
    *lead, T, d = shape
    q, k = torch.randn(*shape), torch.randn(*shape)
    v = torch.randn(*lead, T, d + 3)          # d_v may differ from d_k
    out, w = scaled_dot_product_attention(q, k, v)
    assert out.shape == (*lead, T, d + 3)
    assert w.shape == (*lead, T, T)


def test_rows_are_probability_distributions():
    q, k, v = torch.randn(3, 10, 8), torch.randn(3, 10, 8), torch.randn(3, 10, 8)
    for mask in (None, causal_mask(10)):
        _, w = scaled_dot_product_attention(q, k, v, mask=mask)
        assert (w >= 0).all()
        torch.testing.assert_close(w.sum(-1), torch.ones(3, 10))


def test_cross_attention_with_different_lengths():
    q, k, v = torch.randn(2, 4, 8), torch.randn(2, 9, 8), torch.randn(2, 9, 5)
    out, w = scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 4, 5) and w.shape == (2, 4, 9)


@pytest.mark.parametrize("causal", [False, True])
def test_matches_pytorch_reference(causal):
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 4, 12, 16), torch.randn(2, 4, 12, 16), torch.randn(2, 4, 12, 16)
    out, _ = scaled_dot_product_attention(q, k, v, mask=causal_mask(12) if causal else None)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_scaling_by_sqrt_dk():
    # identical keys -> uniform weights regardless of scale; distinct keys -> scaling softens softmax
    q = torch.tensor([[4.0, 0.0]])
    k = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    v = torch.eye(2)
    _, w = scaled_dot_product_attention(q, k, v)
    expected = torch.softmax(torch.tensor([4.0, 0.0]) / math.sqrt(2), dim=0)
    torch.testing.assert_close(w[0], expected)
