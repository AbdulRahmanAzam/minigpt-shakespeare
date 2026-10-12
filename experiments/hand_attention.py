"""Figure 1 - the hand-calculated attention example (Section 6 of the assignment).

3 tokens, d_k = d_v = 2. Every step is printed with the arithmetic written out, then checked against
our implementation in src/attention.py.

    python -m experiments.hand_attention
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from src.attention import causal_mask, scaled_dot_product_attention  # noqa: E402

OUT = Path("outputs/figures")

Q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
K = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
V = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=torch.float64)


def fmt(m: torch.Tensor) -> str:
    rows = []
    for r in m.tolist():
        rows.append("  [" + ", ".join("  -inf" if math.isinf(v) else f"{v:6.4f}" for v in r) + "]")
    return "\n".join(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    say = lines.append
    d_k = Q.shape[1]

    say("HAND-CALCULATED SCALED DOT-PRODUCT ATTENTION  (3 tokens, d_k = d_v = 2)")
    say("=" * 72)
    say("Q (one query row per token)\n" + fmt(Q))
    say("K (one key row per token)\n" + fmt(K))
    say("V (one value row per token)\n" + fmt(V))

    scores = Q @ K.T
    say("\nStep 1  S = Q K^T     (S[i, j] = q_i . k_j)")
    for i in range(3):
        parts = [f"q{i + 1}.k{j + 1} = {Q[i, 0]:.0f}*{K[j, 0]:.0f} + {Q[i, 1]:.0f}*{K[j, 1]:.0f} = {scores[i, j]:.0f}"
                 for j in range(3)]
        say("  " + " | ".join(parts))
    say(fmt(scores))

    scaled = scores / math.sqrt(d_k)
    say(f"\nStep 2  S / sqrt(d_k) = S / sqrt(2) = S / {math.sqrt(d_k):.4f}")
    say(fmt(scaled))

    say("\nStep 3  row-wise softmax: w_ij = exp(s_ij) / sum_j exp(s_ij)")
    e = torch.exp(scaled)
    for i in range(3):
        say(f"  row {i + 1}: exp = [{', '.join(f'{v:.4f}' for v in e[i].tolist())}]  sum = {e[i].sum():.4f}")
    weights = e / e.sum(dim=1, keepdim=True)
    say(fmt(weights))
    say(f"  row sums = {[round(v, 4) for v in weights.sum(1).tolist()]}")

    out = weights @ V
    say("\nStep 4  output = W V   (each output row is a weighted average of the value rows)")
    for i in range(3):
        w = weights[i]
        say(f"  out{i + 1} = {w[0]:.4f}*v1 + {w[1]:.4f}*v2 + {w[2]:.4f}*v3 = [{out[i, 0]:.4f}, {out[i, 1]:.4f}]")
    say(fmt(out))

    mask = causal_mask(3)
    masked = scaled.masked_fill(mask, float("-inf"))
    say("\nStep 5  causal mask: M[i, j] = -inf if j > i (future), else 0;  S_masked = S/sqrt(d_k) + M")
    say(fmt(torch.zeros(3, 3).masked_fill(mask, float("-inf"))))
    say("  masked scores:")
    say(fmt(masked))

    em = torch.exp(masked)
    w_m = em / em.sum(dim=1, keepdim=True)
    say("\nStep 6  softmax of masked scores (exp(-inf) = 0)")
    for i in range(3):
        say(f"  row {i + 1}: exp = [{', '.join(f'{v:.4f}' for v in em[i].tolist())}]  sum = {em[i].sum():.4f}")
    say(fmt(w_m))
    out_m = w_m @ V
    say("  causal output = W_masked V")
    say(fmt(out_m))

    say("\nStep 7  what changed for each position")
    say("  token 1: may only attend to itself -> weight 1.0 on v1, output = v1 = [1, 0].")
    say("           Without the mask it mixed in v2 and v3, i.e. it 'saw the future'.")
    say("  token 2: sees tokens 1-2 only; the softmax renormalises over 2 scores, so the weights")
    say("           [0.3302, 0.6698] differ from the unmasked [0.1978, 0.4011] - v3 is removed.")
    say("  token 3: is the last token; it already only had past/present to see, so its row is unchanged.")
    say("  => Masking removes information about later tokens. This is what lets a GPT train on every")
    say("     position of a sequence in parallel without the answer (the next token) leaking in.")

    # cross-check against our implementation
    impl_out, impl_w = scaled_dot_product_attention(Q, K, V)
    impl_out_m, impl_w_m = scaled_dot_product_attention(Q, K, V, mask=mask)
    ok = all(torch.allclose(a, b) for a, b in [(impl_out, out), (impl_w, weights), (impl_out_m, out_m), (impl_w_m, w_m)])
    say(f"\nCheck vs src.attention.scaled_dot_product_attention: {'MATCH' if ok else 'MISMATCH'}")

    text = "\n".join(lines)
    print(text)
    (OUT / "fig1_hand_attention.txt").write_text(text + "\n", encoding="utf-8")

    # visual version
    panels = [("Q K^T / sqrt(d_k)", scaled), ("softmax weights", weights), ("output = W V", out),
              ("masked scores", masked), ("masked softmax weights", w_m), ("causal output", out_m)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (title, m) in zip(axes.flat, panels):
        shown = m.clone()
        shown[torch.isinf(shown)] = float("nan")
        ax.imshow(shown, cmap="Blues")
        for (i, j), v in _enum(m):
            ax.text(j, i, "-inf" if math.isinf(v) else f"{v:.3f}", ha="center", va="center", fontsize=10)
        ax.set_title(title)
        ax.set_yticks(range(m.shape[0]), [f"tok {i + 1}" for i in range(m.shape[0])])
        ax.set_xticks(range(m.shape[1]), [f"tok {j + 1}" for j in range(m.shape[1])] if m.shape[1] == 3
                      else [f"dim {j + 1}" for j in range(m.shape[1])])
    fig.suptitle("Figure 1 - hand-calculated attention (top: no mask, bottom: causal mask)")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_hand_attention.png", dpi=150)
    print(f"\nsaved {OUT / 'fig1_hand_attention.png'} and .txt")


def _enum(m: torch.Tensor):
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            yield (i, j), m[i, j].item()


if __name__ == "__main__":
    main()
