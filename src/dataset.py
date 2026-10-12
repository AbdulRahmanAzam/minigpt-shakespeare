"""Next-token prediction dataset.

The corpus is encoded into one long stream of token ids. A training example is a contiguous window
of ``context_length + 1`` ids, split into

    x = ids[i     : i + T]        (model input)
    y = ids[i + 1 : i + T + 1]    (targets: the token *after* each input position)

so ``y[t]`` is the next token after ``x[t]``. With a causal mask, the prediction at position t can only
use ``x[0..t]``, which gives T next-token training signals per window in a single forward pass.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.tokenizer import CharTokenizer


class CharWindowDataset(Dataset):
    def __init__(self, ids: torch.Tensor, context_length: int, stride: int = 1):
        if ids.dim() != 1:
            raise ValueError("ids must be a 1-D tensor")
        if len(ids) < context_length + 1:
            raise ValueError(f"need at least {context_length + 1} tokens, got {len(ids)}")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.ids = ids.long()
        self.context_length = context_length
        self.stride = stride
        self.n_windows = (len(ids) - (context_length + 1)) // stride + 1

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0:
            idx += self.n_windows
        if not 0 <= idx < self.n_windows:
            raise IndexError(idx)
        start = idx * self.stride
        chunk = self.ids[start : start + self.context_length + 1]
        return chunk[:-1], chunk[1:]


def load_split(data_dir: str | Path, split: str) -> str:
    path = Path(data_dir) / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python scripts/prepare_data.py` first")
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def load_tokenizer(data_dir: str | Path) -> CharTokenizer:
    """Tokenizer built from the training split (vocab.json is written by prepare_data.py)."""
    vocab_path = Path(data_dir) / "vocab.json"
    if vocab_path.exists():
        return CharTokenizer.load(vocab_path)
    return CharTokenizer.from_text(load_split(data_dir, "train"))


def encode_split(tokenizer: CharTokenizer, data_dir: str | Path, split: str) -> torch.Tensor:
    return torch.tensor(tokenizer.encode(load_split(data_dir, split)), dtype=torch.long)


def make_loader(
    ids: torch.Tensor,
    context_length: int,
    batch_size: int,
    stride: int = 1,
    shuffle: bool = True,
    seed: int = 0,
    num_workers: int = 0,
    drop_last: bool = True,
) -> DataLoader:
    ds = CharWindowDataset(ids, context_length, stride)
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, generator=gen,
        num_workers=num_workers, drop_last=drop_last, pin_memory=torch.cuda.is_available(),
    )


def infinite(loader: DataLoader):
    """Cycle a DataLoader forever, counting completed epochs."""
    epoch = 0
    while True:
        for batch in loader:
            yield epoch, batch
        epoch += 1
