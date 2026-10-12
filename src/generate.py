"""Command-line text generation from a trained MiniGPT checkpoint.

    python -m src.generate --prompt "ROMEO:" --max-new-tokens 200
    python -m src.generate --prompt "To be, or not to be" --temperature 0.8 --top-k 40 --seed 7
    python -m src.generate --prompt "JULIET:" --temperature 0          # greedy decoding
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from src.checkpoint import load_checkpoint
from src.gpt import MiniGPT
from src.tokenizer import CharTokenizer
from src.utils import format_count, pick_device

DEFAULT_CHECKPOINT = Path("checkpoints/minigpt_shakespeare.pt")


@dataclass
class GenerationResult:
    prompt: str
    completion: str
    text: str
    ignored_chars: list[str]
    new_tokens: int
    seconds: float


def generate_text(
    model: MiniGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
) -> GenerationResult:
    device = next(model.parameters()).device
    ignored = tokenizer.unknown_chars(prompt)
    ids = tokenizer.encode(prompt, strict=False)
    clean_prompt = tokenizer.decode(ids)
    if not ids:                       # empty prompt: start from a newline, like the start of a speech
        ids = tokenizer.encode("\n")
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    start = time.perf_counter()
    out = model.generate(torch.tensor([ids], device=device), max_new_tokens, temperature=temperature,
                         top_k=top_k, top_p=top_p, generator=generator)
    seconds = time.perf_counter() - start
    completion = tokenizer.decode(out[0, len(ids):].tolist())
    return GenerationResult(clean_prompt, completion, clean_prompt + completion, ignored, max_new_tokens, seconds)


def resolve_checkpoint(path: str | None) -> Path:
    if path:
        return Path(path)
    for candidate in (DEFAULT_CHECKPOINT, Path("checkpoints/tiny_gpt/best.pt")):
        if candidate.exists():
            return candidate
    return DEFAULT_CHECKPOINT


def main() -> None:
    p = argparse.ArgumentParser(description="Generate text with a trained MiniGPT")
    p.add_argument("--checkpoint", default=None,
                   help=f"default: {DEFAULT_CHECKPOINT} (released weights), else checkpoints/tiny_gpt/best.pt")
    p.add_argument("--prompt", default="ROMEO:")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8, help="0 = greedy decoding")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = pick_device(args.device)
    ckpt = resolve_checkpoint(args.checkpoint)
    model, tokenizer, state = load_checkpoint(ckpt, device)

    for i in range(args.num_samples):
        seed = None if args.seed is None else args.seed + i
        r = generate_text(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature,
                          args.top_k, args.top_p, seed)
        if args.num_samples > 1:
            print(f"----- sample {i + 1} -----")
        print(f"Prompt: {r.prompt}")
        print(f"Generated: {r.text}")
        print()
        if r.ignored_chars:
            print(f"note: characters not in the vocabulary were dropped from the prompt: {r.ignored_chars}")

    cfg = model.cfg
    val = state.get("metrics", {}).get("val_loss")
    print(f"model parameters: {format_count(model.num_parameters())} ({model.num_parameters():,})")
    print(f"architecture: {cfg.n_layers} layers, {cfg.n_heads} heads, d_model {cfg.d_model}, "
          f"{cfg.pos_encoding} positions, vocab {cfg.vocab_size} (character-level)")
    print(f"context length: {cfg.context_length}")
    print(f"temperature: {args.temperature}" + (f", top-k: {args.top_k}" if args.top_k else "")
          + (f", top-p: {args.top_p}" if args.top_p else ""))
    print(f"seed: {args.seed}")
    print(f"checkpoint: {ckpt} (step {state.get('step')}" + (f", val loss {val:.4f})" if val else ")"))
    print(f"device: {device} | {r.new_tokens} new tokens in {r.seconds:.2f}s")


if __name__ == "__main__":
    main()
