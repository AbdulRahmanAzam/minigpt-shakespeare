"""Convert a PyTorch checkpoint into the .npz file served by the NumPy-only API (src/numpy_gpt.py).

    python scripts/export_numpy.py                       # checkpoints/minigpt_shakespeare.pt -> .npz
    python scripts/export_numpy.py --src checkpoints/tiny_gpt/best.pt --out checkpoints/tiny.npz

Checks that the NumPy forward pass reproduces the PyTorch logits before writing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.checkpoint import load_checkpoint  # noqa: E402
from src.numpy_gpt import NumpyGPT, save_npz  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(ROOT / "checkpoints" / "minigpt_shakespeare.pt"))
    p.add_argument("--out", default=None, help="default: same path with .npz")
    args = p.parse_args()
    out = Path(args.out) if args.out else Path(args.src).with_suffix(".npz")

    model, tok, state = load_checkpoint(args.src, "cpu")
    weights = {k: v.float().numpy() for k, v in model.state_dict().items()}
    ids = torch.tensor(tok.encode("ROMEO:\nBut soft, what light through yonder window breaks?", strict=False))
    with torch.no_grad():
        expected = model(ids[None]).logits[0].numpy()
    diff = np.abs(NumpyGPT(weights, model.config_dict()).forward(ids.numpy()) - expected).max()
    if diff > 1e-3:
        raise SystemExit(f"NumPy forward pass disagrees with PyTorch (max |logit diff| {diff:.2e})")

    save_npz(out, weights, model.config_dict(), tok, step=state.get("step"), metrics=state.get("metrics"))
    print(f"ok: {out} ({out.stat().st_size / 2**20:.1f} MB, max |logit diff| vs torch {diff:.1e})")


if __name__ == "__main__":
    main()
