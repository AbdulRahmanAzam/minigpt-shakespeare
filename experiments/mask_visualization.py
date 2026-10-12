"""Figure 2 - causal (GPT) mask vs bidirectional (BERT MLM) attention constraints.

    python -m experiments.mask_visualization
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.attention import causal_mask, scaled_dot_product_attention  # noqa: E402

OUT = Path("outputs/figures")
TOKENS = list("ROMEO:")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    T = len(TOKENS)
    causal = causal_mask(T)
    additive = torch.zeros(T, T).masked_fill(causal, float("-inf"))

    lines = ["Causal mask M (added to scores before softmax), tokens = " + " ".join(repr(t) for t in TOKENS)]
    for i in range(T):
        lines.append(f"  {TOKENS[i]!r:>5} " + " ".join(" -inf" if v == float('-inf') else "  0  " for v in additive[i]))
    lines.append("Row i = query position, column j = key position; -inf above the diagonal blocks the future.")
    lines.append("")
    lines.append("BERT-style bidirectional mask: all zeros (every position sees every other position);")
    lines.append("instead of hiding the future, MLM hides ~15% of the *input tokens* with [MASK] and predicts them.")

    # attention weights from random Q/K with each mask
    torch.manual_seed(0)
    q, k, v = torch.randn(T, 8), torch.randn(T, 8), torch.randn(T, 8)
    _, w_causal = scaled_dot_product_attention(q, k, v, mask=causal)
    _, w_bidir = scaled_dot_product_attention(q, k, v)
    lines.append("")
    lines.append(f"sum of weights above the diagonal: causal = {torch.triu(w_causal, 1).sum():.6f}, "
                 f"bidirectional = {torch.triu(w_bidir, 1).sum():.4f}")
    text = "\n".join(lines)
    print(text)
    (OUT / "fig2_causal_mask.txt").write_text(text + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    allowed = (~causal).float().numpy()
    panels = [
        (allowed, "GPT causal mask\n(1 = may attend, 0 = blocked)", "Greens"),
        (w_causal.numpy(), "GPT attention weights\n(random Q, K; rows sum to 1)", "Blues"),
        (np.ones((T, T)), "BERT bidirectional mask\n(every pair allowed)", "Greens"),
        (w_bidir.numpy(), "BERT attention weights\n(random Q, K)", "Blues"),
    ]
    for ax, (m, title, cmap) in zip(axes, panels):
        ax.imshow(m, cmap=cmap, vmin=0, vmax=1)
        for i in range(T):
            for j in range(T):
                ax.text(j, i, f"{m[i, j]:.2f}" if cmap == "Blues" else f"{int(m[i, j])}",
                        ha="center", va="center", fontsize=8, color="white" if m[i, j] > 0.6 else "black")
        ax.set_xticks(range(T), TOKENS)
        ax.set_yticks(range(T), TOKENS)
        ax.set_xlabel("key position j (attended to)")
        ax.set_ylabel("query position i")
        ax.set_title(title, fontsize=10)
    fig.suptitle("Figure 2 - causal mask blocks future tokens (GPT) vs bidirectional attention (BERT)")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_causal_mask.png", dpi=150)
    print(f"saved {OUT / 'fig2_causal_mask.png'}")


if __name__ == "__main__":
    main()
