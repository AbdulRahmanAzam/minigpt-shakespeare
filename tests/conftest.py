import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ModelConfig  # noqa: E402
from src.gpt import MiniGPT  # noqa: E402

# The hand-calculated example from the report (Section 6 of the assignment), d_k = d_v = 2.
HAND_Q = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
HAND_K = [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
HAND_V = [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]


@pytest.fixture
def tiny_cfg() -> ModelConfig:
    return ModelConfig(vocab_size=17, context_length=16, d_model=32, n_heads=4, n_layers=2, dropout=0.0)


@pytest.fixture
def tiny_model(tiny_cfg) -> MiniGPT:
    torch.manual_seed(0)
    return MiniGPT(tiny_cfg).eval()
