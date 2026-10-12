"""Figure 7 - attention maps from the trained model, plus per-head statistics over validation text.

    python -m experiments.attention_visualization --checkpoint checkpoints/minigpt_shakespeare.pt

Outputs:
    fig7_attention_heads.png     every layer x head on a short prompt
    fig7_attention_detail.png    two contrasting heads, large, with character labels
    fig7_attention_stats.txt     mean attention distance / previous-token / first-token share per head
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from src.checkpoint import load_checkpoint  # noqa: E402
from src.dataset import encode_split  # noqa: E402
from src.generate import resolve_checkpoint  # noqa: E402

OUT = Path("outputs/figures")
TEXT = "ROMEO:\nBut soft, what light through yonder window breaks?"


def label(c: str) -> str:
    return {"\n": "\\n", " ": "_"}.get(c, c)


@torch.no_grad()
def head_statistics(model, ids: torch.Tensor, T: int, n_windows: int = 64) -> dict[str, torch.Tensor]:
    """Average, over validation windows, of per-head attention behaviour (layers x heads tensors)."""
    starts = torch.linspace(0, len(ids) - T - 1, n_windows).long()
    x = torch.stack([ids[s: s + T] for s in starts]).to(next(model.parameters()).device)
    att = torch.stack(model(x, return_attn=True).attentions).float()     # (L, B, H, T, T)
    i = torch.arange(T, device=att.device)
    dist = (i[:, None] - i[None, :]).clamp(min=0).float()                  # i - j for j <= i
    rows = slice(16, T)   # skip the first positions, which have almost nothing to attend to
    return {
        "mean_distance": (att * dist).sum(-1)[..., rows].mean(dim=(1, 3)),
        "prev_token": att.diagonal(offset=-1, dim1=-2, dim2=-1)[..., 15:].mean(dim=(1, 3)),
        "self": att.diagonal(dim1=-2, dim2=-1)[..., rows].mean(dim=(1, 3)),
        "first_token": att[..., rows, 0].mean(dim=(1, 3)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--text", default=TEXT)
    args = p.parse_args()

    model, tok, _ = load_checkpoint(resolve_checkpoint(args.checkpoint), "cpu")
    cfg = model.cfg
    ids = torch.tensor([tok.encode(args.text, strict=False)])
    chars = [label(c) for c in tok.decode(ids[0].tolist())]
    with torch.no_grad():
        att = [a[0] for a in model(ids, return_attn=True).attentions]      # L x (H, T, T)

    OUT.mkdir(parents=True, exist_ok=True)
    # --- grid of every head
    L, H, T = cfg.n_layers, cfg.n_heads, ids.shape[1]
    fig, axes = plt.subplots(L, H, figsize=(3.2 * H, 3.0 * L))
    for l in range(L):
        for h in range(H):
            ax = axes[l][h] if L > 1 else axes[h]
            ax.imshow(att[l][h].numpy(), cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if h == 0:
                ax.set_ylabel(f"layer {l}")
            if l == 0:
                ax.set_title(f"head {h}")
    fig.suptitle(f"Figure 7a - attention weights of every head (query = row, key = column)\ntext: {args.text!r}")
    fig.tight_layout()
    fig.savefig(OUT / "fig7_attention_heads.png", dpi=120)

    # --- statistics over validation text
    stats = head_statistics(model, encode_split(tok, "data", "val"), cfg.context_length)
    lines = ["Per-head attention statistics, averaged over 64 validation windows "
             f"(context {cfg.context_length}, query positions >= 16)",
             "  mean_dist  = expected distance (i - j) to the attended key",
             "  prev       = weight on the immediately preceding character",
             "  self       = weight on the current character",
             "  first      = weight on the first position of the window", ""]
    lines.append(f"{'layer':>5} {'head':>4} {'mean_dist':>10} {'prev':>6} {'self':>6} {'first':>6}")
    for l in range(L):
        for h in range(H):
            lines.append(f"{l:>5} {h:>4} {stats['mean_distance'][l, h]:>10.2f} {stats['prev_token'][l, h]:>6.2f} "
                         f"{stats['self'][l, h]:>6.2f} {stats['first_token'][l, h]:>6.2f}")
    text = "\n".join(lines)
    print(text)
    (OUT / "fig7_attention_stats.txt").write_text(text + "\n", encoding="utf-8")

    # --- detail: the most "local" head vs the most "long-range" head
    md = stats["mean_distance"]
    local = divmod(int(md.argmin()), H)
    longr = divmod(int(md.argmax()), H)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5))
    for ax, (l, h), kind in zip(axes, (local, longr), ("most local", "longest range")):
        ax.imshow(att[l][h].numpy(), cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(T), chars, fontsize=7)
        ax.set_yticks(range(T), chars, fontsize=7)
        ax.set_title(f"layer {l}, head {h} ({kind}: mean distance {md[l, h]:.1f} chars, "
                     f"prev-token weight {stats['prev_token'][l, h]:.2f})", fontsize=10)
        ax.set_xlabel("key (attended-to character)")
        ax.set_ylabel("query (current character)")
    fig.suptitle("Figure 7b - two contrasting heads of the trained model; upper triangle is zero (causal)")
    fig.tight_layout()
    fig.savefig(OUT / "fig7_attention_detail.png", dpi=130)
    upper = max(torch.triu(a, 1).abs().max().item() for a in att)
    print(f"\nmax attention weight on a future position across all heads: {upper}")
    print(f"saved {OUT}/fig7_attention_heads.png, fig7_attention_detail.png, fig7_attention_stats.txt")


if __name__ == "__main__":
    main()
