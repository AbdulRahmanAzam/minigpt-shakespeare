"""Typed configuration loaded from YAML, with command-line overrides.

Every hyper-parameter listed in Section 11 of the assignment lives here, so any run can be
reproduced from a single YAML file plus (optionally) a few ``--set section.key=value`` overrides.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 65            # overwritten by the tokenizer at train time
    context_length: int = 128       # maximum number of tokens the model attends over
    d_model: int = 128              # width of the residual stream
    n_heads: int = 4                # attention heads per layer (d_model % n_heads == 0)
    n_layers: int = 4               # number of Transformer blocks
    ffn_dim: int | None = None      # hidden size of the MLP; None -> 4 * d_model
    dropout: float = 0.1
    pos_encoding: str = "sinusoidal"  # "sinusoidal" | "learned" | "none"
    tie_weights: bool = False       # share token-embedding and LM-head weights
    attn_impl: str = "manual"       # "manual" (our implementation) | "sdpa" (PyTorch fused, for speed checks)
    causal: bool = True             # False only for the MLM / leakage experiments
    bias: bool = True               # bias terms in Linear layers

    def __post_init__(self) -> None:
        if self.ffn_dim is None:
            self.ffn_dim = 4 * self.d_model
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.pos_encoding not in {"sinusoidal", "learned", "none"}:
            raise ValueError(f"unknown pos_encoding {self.pos_encoding!r}")
        if self.attn_impl not in {"manual", "sdpa"}:
            raise ValueError(f"unknown attn_impl {self.attn_impl!r}")


@dataclass
class DataConfig:
    data_dir: str = "data"
    train_stride: int = 1           # step between consecutive training windows (1 = every offset)
    max_train_tokens: int | None = None  # truncate training stream (used by the overfit gate)


@dataclass
class TrainConfig:
    seed: int = 1337
    device: str = "auto"            # "auto" | "cuda" | "cpu"
    precision: str = "auto"         # "auto" | "bf16" | "fp16" | "fp32"
    batch_size: int = 64
    max_steps: int = 3000
    lr: float = 1e-3                # peak learning rate
    min_lr_ratio: float = 0.1       # cosine decays to lr * min_lr_ratio
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_batches: int = 50          # batches used for the periodic validation estimate
    log_interval: int = 50
    sample_prompt: str = "ROMEO:"
    sample_tokens: int = 200
    out_dir: str = "outputs/logs"
    ckpt_dir: str = "checkpoints"
    run_name: str = "run"
    save_checkpoints: bool = True
    num_workers: int = 0


@dataclass
class GenerationConfig:
    max_new_tokens: int = 300
    temperature: float = 0.8
    top_k: int | None = None
    top_p: float | None = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTIONS = {"model": ModelConfig, "data": DataConfig, "train": TrainConfig, "generation": GenerationConfig}


def _build(raw: dict[str, Any]) -> Config:
    kwargs = {}
    for name, cls in _SECTIONS.items():
        section = raw.get(name) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(section) - known
        if unknown:
            raise KeyError(f"unknown keys in '{name}' config: {sorted(unknown)}")
        section = dict(section)
        for f in fields(cls):
            # PyYAML (YAML 1.1) reads "1e-3" as a string; coerce declared floats explicitly.
            if str(f.type).startswith("float") and isinstance(section.get(f.name), str):
                section[f.name] = float(section[f.name])
        kwargs[name] = cls(**section)
    extra = set(raw) - set(_SECTIONS)
    if extra:
        raise KeyError(f"unknown config sections: {sorted(extra)}")
    return Config(**kwargs)


def apply_overrides(raw: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``section.key=value`` overrides; values are parsed as YAML (so 1e-3, true, null work)."""
    raw = copy.deepcopy(raw)
    for item in overrides or []:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError(f"override must look like section.key=value, got {item!r}")
        dotted, value = item.split("=", 1)
        section, key = dotted.split(".", 1)
        raw.setdefault(section, {})
        raw[section][key] = yaml.safe_load(value)
    return raw


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    raw: dict[str, Any] = {}
    if path is not None:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    return _build(apply_overrides(raw, overrides))


def config_from_dict(raw: dict[str, Any]) -> Config:
    return _build(raw)
