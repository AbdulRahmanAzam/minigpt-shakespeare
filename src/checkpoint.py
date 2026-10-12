"""Checkpoint save / load.

A checkpoint is self-contained: weights + model config + tokenizer vocabulary (+ training metadata),
so a fresh process can rebuild the exact model and generate text from the file alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.config import ModelConfig
from src.gpt import MiniGPT
from src.tokenizer import CharTokenizer
from src.utils import git_hash

FORMAT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    model: MiniGPT,
    tokenizer: CharTokenizer,
    *,
    full_config: dict[str, Any] | None = None,
    step: int = 0,
    metrics: dict[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format_version": FORMAT_VERSION,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": model.config_dict(),
        "tokenizer": tokenizer.to_dict(),
        "config": full_config,
        "step": step,
        "metrics": metrics or {},
        "git_hash": git_hash(),
    }
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if scaler is not None:
        state["scaler_state"] = scaler.state_dict()
    if extra:
        state.update(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)          # atomic: never leaves a half-written checkpoint behind


def load_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[MiniGPT, CharTokenizer, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"checkpoint {path} not found - train one (python -m src.train) "
            f"or download the released one (python scripts/download_checkpoint.py)"
        )
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**state["model_config"])
    model = MiniGPT(cfg)
    model.load_state_dict(state["model_state"])
    model.to(device).eval()
    tokenizer = CharTokenizer.from_dict(state["tokenizer"])
    return model, tokenizer, state


def export_inference_checkpoint(src: str | Path, dst: str | Path) -> None:
    """Strip optimizer / scaler state so the released file only contains what inference needs."""
    state = torch.load(src, map_location="cpu", weights_only=False)
    for key in ("optimizer_state", "scaler_state"):
        state.pop(key, None)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, dst)


def load_as_numpy(path: str | Path) -> tuple[Any, CharTokenizer, dict[str, Any]]:
    """Load a checkpoint as a NumPy-only model (src/numpy_gpt.py), e.g. to serve a .pt file locally."""
    from src.numpy_gpt import NumpyGPT

    state = torch.load(path, map_location="cpu", weights_only=False)
    weights = {k: v.float().numpy() for k, v in state["model_state"].items()}
    return NumpyGPT(weights, state["model_config"]), CharTokenizer.from_dict(state["tokenizer"]), state
