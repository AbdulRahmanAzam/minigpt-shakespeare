"""The NumPy inference path (used by the deployed API) must reproduce the PyTorch model."""

import numpy as np
import torch

from src.gpt import MiniGPT, filter_logits
from src.config import ModelConfig
from src.numpy_gpt import NumpyGPT, filter_logits as np_filter_logits, load_npz, save_npz
from src.tokenizer import CharTokenizer


def as_numpy(model: MiniGPT) -> NumpyGPT:
    return NumpyGPT({k: v.numpy() for k, v in model.state_dict().items()}, model.config_dict())


def test_logits_match_torch(tiny_model):
    ids = torch.randint(0, tiny_model.cfg.vocab_size, (12,), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        expected = tiny_model(ids[None]).logits[0].numpy()
    np.testing.assert_allclose(as_numpy(tiny_model).forward(ids.numpy()), expected, atol=1e-5)


def test_kv_cache_matches_full_forward(tiny_model):
    m = as_numpy(tiny_model)
    ids = np.arange(10) % tiny_model.cfg.vocab_size
    cache: list = []
    step_by_step = np.concatenate([m.forward(ids[:4], cache)] + [m.forward(ids[i:i + 1], cache) for i in range(4, 10)])
    np.testing.assert_allclose(step_by_step, m.forward(ids), atol=1e-5)


def test_greedy_generation_matches_torch_past_the_context_window(tiny_model):
    n = 2 * tiny_model.cfg.context_length          # forces the sliding window to move
    expected = tiny_model.generate(torch.tensor([[1, 2, 3]]), n, temperature=0)[0].tolist()
    assert as_numpy(tiny_model).generate([1, 2, 3], n, temperature=0) == expected


def test_learned_positions_and_tied_weights():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=11, context_length=8, d_model=16, n_heads=2, n_layers=1, dropout=0.0,
                      pos_encoding="learned", tie_weights=True)
    model = MiniGPT(cfg).eval()
    ids = torch.tensor([3, 1, 4, 1, 5])
    with torch.no_grad():
        expected = model(ids[None]).logits[0].numpy()
    m = as_numpy(model)
    np.testing.assert_allclose(m.forward(ids.numpy()), expected, atol=1e-5)
    assert m.num_parameters() == model.num_parameters()


def test_filter_logits_matches_torch():
    logits = torch.randn(20, generator=torch.Generator().manual_seed(1))
    for top_k, top_p in [(5, None), (None, 0.7), (8, 0.5), (None, None)]:
        expected = filter_logits(logits[None], top_k, top_p)[0].numpy()
        np.testing.assert_array_equal(np_filter_logits(logits.numpy(), top_k, top_p), expected)


def test_npz_roundtrip(tiny_model, tmp_path):
    m = as_numpy(tiny_model)
    tok = CharTokenizer(list("abcdefghijklmnopq"))
    save_npz(tmp_path / "m.npz", m.w, m.config_dict(), tok, step=7, metrics={"val_loss": 1.2})
    m2, tok2, meta = load_npz(tmp_path / "m.npz")
    assert tok2.chars == tok.chars and meta["step"] == 7 and m2.config_dict() == m.config_dict()
    ids = np.array([0, 5, 9])
    np.testing.assert_array_equal(m2.forward(ids), m.forward(ids))
