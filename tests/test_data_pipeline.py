"""Tokenizer determinism / round trip and next-token dataset construction."""

import pytest
import torch

from scripts.prepare_data import split_text
from src.config import apply_overrides, load_config
from src.dataset import CharWindowDataset
from src.tokenizer import CharTokenizer


def test_tokenizer_is_deterministic_and_round_trips():
    text = "ROMEO:\nBut, soft! what light through yonder window breaks?"
    a, b = CharTokenizer.from_text(text), CharTokenizer.from_text(text[::-1])
    assert a.chars == b.chars == sorted(set(text))           # order of appearance does not matter
    assert a.decode(a.encode(text)) == text


def test_tokenizer_unknown_character_policy():
    tok = CharTokenizer.from_text("abc ")
    with pytest.raises(ValueError):
        tok.encode("abz")
    assert tok.decode(tok.encode("abz c", strict=False)) == "ab c"
    assert tok.unknown_chars("abzzy") == ["z", "y"]


def test_tokenizer_save_load(tmp_path):
    tok = CharTokenizer.from_text("hello world\n")
    tok.save(tmp_path / "v.json")
    assert CharTokenizer.load(tmp_path / "v.json").chars == tok.chars


def test_targets_are_inputs_shifted_by_one():
    ids = torch.arange(100)
    ds = CharWindowDataset(ids, context_length=8, stride=1)
    assert len(ds) == 100 - 9 + 1
    for i in (0, 5, len(ds) - 1):
        x, y = ds[i]
        assert x.shape == y.shape == (8,)
        assert torch.equal(y[:-1], x[1:])                     # y[t] = x[t + 1]
        assert y[-1].item() == x[-1].item() + 1               # the extra token after the window
    x, _ = ds[-1]
    assert x[-1].item() == 98                                 # last window ends at the second-to-last id


def test_window_stride():
    ds = CharWindowDataset(torch.arange(50), context_length=10, stride=10)
    assert len(ds) == 4
    assert ds[1][0][0].item() == 10
    with pytest.raises(IndexError):
        ds[4]


def test_split_matches_hugging_face_rule():
    text = "x" * 1000
    tr, va, te = split_text(text)
    assert (len(tr), len(va), len(te)) == (900, 50, 50)
    assert tr + va + te == text


def test_config_overrides():
    cfg = load_config(None, ["model.n_layers=2", "train.lr=1e-3", "model.pos_encoding=learned"])
    assert cfg.model.n_layers == 2 and cfg.train.lr == pytest.approx(1e-3)
    assert cfg.model.pos_encoding == "learned"
    assert cfg.model.ffn_dim == 4 * cfg.model.d_model
    with pytest.raises(KeyError):
        load_config(None, ["model.not_a_key=1"])
    with pytest.raises(ValueError):
        apply_overrides({}, ["no_dot=1"])
