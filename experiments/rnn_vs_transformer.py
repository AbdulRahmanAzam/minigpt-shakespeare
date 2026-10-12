"""Lesson 01 - why Transformers: sequential depth, speed and memory, measured.

Hand analysis (one layer, sequence length N, width d):

                         sequential steps   FLOPs              activation memory
    RNN  h_t = f(h_{t-1}, x_t)    O(N)      O(N d^2)           O(N d)
    self-attention                O(1)      O(N^2 d + N d^2)   O(N^2)  (the T x T score matrix)

An RNN must finish step t before starting t+1, so N steps run one after another no matter how many cores
the GPU has. Attention computes all N x N scores with a few large matrix multiplies, so its
sequential depth does not grow with N - at the price of N^2 memory.

Measurement: forward + backward time and peak memory for
  * an Elman RNN cell unrolled in a Python loop (explicit sequential dependency)
  * PyTorch's cuDNN GRU (same recurrence, heavily optimised kernel - still sequential inside)
  * one causal multi-head self-attention layer from src/attention.py

    python -m experiments.rnn_vs_transformer
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from src.attention import MultiHeadAttention  # noqa: E402
from src.utils import pick_device, synchronize, write_json  # noqa: E402

OUT = Path("outputs/rnn_vs_transformer")


class LoopRNN(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.cell = nn.RNNCell(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.zeros(x.size(0), x.size(2), device=x.device)
        outs = []
        for t in range(x.size(1)):           # step t needs h from step t-1: inherently sequential
            h = self.cell(x[:, t], h)
            outs.append(h)
        return torch.stack(outs, 1)


class GRUWrap(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.gru = nn.GRU(d, d, batch_first=True)

    def forward(self, x):
        return self.gru(x)[0]


class AttnWrap(nn.Module):
    def __init__(self, d: int, max_len: int):
        super().__init__()
        self.attn = MultiHeadAttention(d, 4, max_len, dropout=0.0)

    def forward(self, x):
        return self.attn(x)[0]


def measure(module: nn.Module, N: int, d: int, B: int, device, repeats: int) -> tuple[float, float]:
    x = torch.randn(B, N, d, device=device, requires_grad=True)
    for _ in range(2):                                   # warm-up
        module(x).sum().backward()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        base = torch.cuda.memory_allocated(device)
    times = []
    for _ in range(repeats):
        synchronize(device)
        t0 = time.perf_counter()
        module(x).sum().backward()
        synchronize(device)
        times.append(time.perf_counter() - t0)
    mem = (torch.cuda.max_memory_allocated(device) - base) / 2**20 if device.type == "cuda" else float("nan")
    return sorted(times)[len(times) // 2] * 1000, mem


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lengths", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = pick_device(args.device)
    torch.manual_seed(0)
    models = {
        "RNN (python loop)": LoopRNN(args.d).to(device),
        "GRU (cuDNN)": GRUWrap(args.d).to(device),
        "self-attention (ours)": AttnWrap(args.d, max(args.lengths)).to(device),
    }
    rows = []
    print(f"device {device}, d={args.d}, batch={args.batch}; forward+backward, median of {args.repeats}")
    print(f"{'N':>6} | " + " | ".join(f"{k:>24}" for k in models) + " |  sequential steps RNN vs attn")
    for N in args.lengths:
        row = {"N": N}
        cells = []
        for name, m in models.items():
            ms, mb = measure(m, N, args.d, args.batch, device, args.repeats)
            row[name] = {"ms": ms, "peak_mb": mb}
            cells.append(f"{ms:9.2f} ms {mb:8.1f} MB")
        rows.append(row)
        print(f"{N:>6} | " + " | ".join(f"{c:>24}" for c in cells) + f" |  {N:>5} vs 1")

    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "results.json", {"device": str(device), "d": args.d, "batch": args.batch, "rows": rows})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    Ns = [r["N"] for r in rows]
    for name in models:
        a1.plot(Ns, [r[name]["ms"] for r in rows], "o-", label=name)
        a2.plot(Ns, [r[name]["peak_mb"] for r in rows], "o-", label=name)
    for a, yl, t in ((a1, "forward + backward time (ms)",
                      "python-loop RNN: ~linear in N (N serial steps); cuDNN GRU hides the\n"
                      "loop inside one fused kernel; attention jumps once N^2 scores exceed GPU memory"),
                     (a2, "extra peak GPU memory (MB)", "recurrence memory ~N, attention memory ~N^2 (B x H x N x N scores)")):
        a.set_xscale("log", base=2)
        a.set_yscale("log")
        a.set_xlabel("sequence length N")
        a.set_ylabel(yl)
        a.set_title(t, fontsize=10)
        a.grid(alpha=0.3, which="both")
        a.legend()
    fig.suptitle(f"Recurrence vs self-attention (d={args.d}, batch={args.batch}, {device})")
    fig.tight_layout()
    fig.savefig(OUT / "rnn_vs_attention.png", dpi=150)
    fig.savefig("outputs/figures/fig0_rnn_vs_attention.png", dpi=150)
    print(f"saved {OUT}/results.json and rnn_vs_attention.png")


if __name__ == "__main__":
    main()
