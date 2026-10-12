"""Autoregressive generation, sampling filters and checkpoint reload."""

import subprocess
import sys

import torch

from src.checkpoint import load_checkpoint, save_checkpoint
from src.gpt import filter_logits
from src.tokenizer import CharTokenizer
from tests.conftest import ROOT


def test_generate_length_and_prefix(tiny_model):
    prompt = torch.tensor([[1, 2, 3]])
    out = tiny_model.generate(prompt, max_new_tokens=10, temperature=1.0, generator=torch.Generator().manual_seed(0))
    assert out.shape == (1, 13)
    assert torch.equal(out[:, :3], prompt)


def test_generation_beyond_context_uses_sliding_window(tiny_model, tiny_cfg):
    out = tiny_model.generate(torch.tensor([[0]]), max_new_tokens=3 * tiny_cfg.context_length, temperature=0)
    assert out.shape[1] == 1 + 3 * tiny_cfg.context_length


def test_seeded_sampling_is_reproducible(tiny_model):
    p = torch.tensor([[4, 5]])
    a = tiny_model.generate(p, 20, temperature=0.8, generator=torch.Generator().manual_seed(7))
    b = tiny_model.generate(p, 20, temperature=0.8, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_greedy_equals_argmax(tiny_model):
    p = torch.tensor([[1, 2, 3, 4]])
    out = tiny_model.generate(p, 1, temperature=0)
    assert out[0, -1].item() == tiny_model(p).logits[0, -1].argmax().item()


def test_top_k_and_top_p_filtering():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    k2 = filter_logits(logits, top_k=2)
    assert torch.isinf(k2[0, 2:]).all() and torch.isfinite(k2[0, :2]).all()
    p = filter_logits(logits, top_p=0.7)                     # 0.5 + 0.3 >= 0.7 -> keep two tokens
    assert torch.isfinite(p[0, :2]).all() and torch.isinf(p[0, 2:]).all()
    assert torch.isfinite(filter_logits(logits, top_p=0.01)[0, 0])   # top token always kept


def test_checkpoint_reload_in_fresh_process_gives_identical_output(tiny_model, tmp_path):
    """Acceptance test 4: a fresh process can load the saved weights and generate the same text."""
    tok = CharTokenizer([chr(ord("a") + i) for i in range(tiny_model.cfg.vocab_size)])
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, tiny_model, tok, step=1)

    model2, tok2, state = load_checkpoint(path)
    assert tok2.chars == tok.chars and state["step"] == 1
    x = torch.randint(0, tiny_model.cfg.vocab_size, (2, 8))
    torch.testing.assert_close(model2(x).logits, tiny_model(x).logits)

    expected = tok.decode(tiny_model.generate(torch.tensor([tok.encode("abc")]), 12, temperature=0)[0].tolist())
    code = (
        "import sys, torch; sys.path.insert(0, r'%s');"
        "from src.checkpoint import load_checkpoint;"
        "m, t, _ = load_checkpoint(r'%s');"
        "print(t.decode(m.generate(torch.tensor([t.encode('abc')]), 12, temperature=0)[0].tolist()), end='')"
    ) % (ROOT, path)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout == expected
