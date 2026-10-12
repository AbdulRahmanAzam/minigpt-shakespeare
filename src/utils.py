"""Small shared helpers: seeding, device / precision selection, timing, run metadata."""

from __future__ import annotations

import contextlib
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    """Seed every RNG we use so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cuda_usable() -> bool:
    # device_count() also guards against CUDA_VISIBLE_DEVICES="" where is_available() can still be True
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def pick_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if cuda_usable() else "cpu")
    if requested == "cuda" and not cuda_usable():
        raise RuntimeError("device=cuda requested but CUDA is not available")
    return torch.device(requested)


def pick_precision(requested: str, device: torch.device) -> str:
    """bf16 where the GPU supports it (Ampere+), fp16 on older GPUs (e.g. Colab T4), fp32 on CPU."""
    if requested != "auto":
        if device.type == "cpu" and requested != "fp32":
            return "fp32"
        return requested
    if device.type != "cuda":
        return "fp32"
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"


def autocast(device: torch.device, precision: str) -> contextlib.AbstractContextManager:
    if precision == "fp32" or device.type != "cuda":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def format_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def environment_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": str(device),
        "cpu_threads": torch.get_num_threads(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info["gpu"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / 2**30, 2)
    return info


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextlib.contextmanager
def timer() -> Iterator[dict[str, float]]:
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start


def write_json(path: str | os.PathLike, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path: str | os.PathLike) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
