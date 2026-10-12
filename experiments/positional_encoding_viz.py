"""Figure 4 - sinusoidal positional encoding.

Panels: (a) the PE matrix, (b) a few dimensions as functions of position (different frequencies),
(c) dot-product similarity between positions (depends only on the distance |i - j|),
(d) the same token at different positions becomes a different input vector.

    python -m experiments.positional_encoding_viz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from src.config import load_config  # noqa: E402
from src.positional_encoding import sinusoidal_table  # noqa: E402

OUT = Path("outputs/figures")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/tiny_gpt.yaml")
    args = p.parse_args()
    cfg = load_config(args.config).model
    L, d = cfg.context_length, cfg.d_model
    pe = sinusoidal_table(L, d)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    im = ax.imshow(pe.numpy(), aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xlabel("embedding dimension")
    ax.set_ylabel("position")
    ax.set_title(f"(a) PE matrix ({L} positions x {d} dims)")
    fig.colorbar(im, ax=ax)

    ax = axes[0, 1]
    for dim in (16, 64, 96, 128):
        if dim < d:
            ax.plot(pe[:, dim].numpy(), label=f"dim {dim} (wavelength {2 * 3.1416 * 10000 ** (dim / d):.0f})")
    ax.set_xlabel("position")
    ax.set_ylabel("value")
    ax.set_title("(b) each dimension is a sinusoid of a different frequency")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    sim = pe @ pe.T
    im = ax.imshow(sim.numpy(), cmap="viridis")
    ax.set_xlabel("position j")
    ax.set_ylabel("position i")
    ax.set_title("(c) PE[i] . PE[j]: high near the diagonal, depends on |i - j|")
    fig.colorbar(im, ax=ax)

    ax = axes[1, 1]
    torch.manual_seed(0)
    token_vec = torch.randn(d) * 0.02 * d ** 0.5            # a token embedding at our init scale (x sqrt(d))
    inputs = token_vec + pe                                  # same token placed at every position
    cos = torch.nn.functional.cosine_similarity(inputs[:1], inputs, dim=-1)
    ax.plot(cos.numpy())
    ax.set_xlabel("position of the token")
    ax.set_ylabel("cosine similarity to the same token at position 0")
    ax.set_title("(d) same token, different position -> different input vector")
    ax.set_ylim(-0.2, 1.05)

    fig.suptitle("Figure 4 - sinusoidal positional encoding: PE[pos,2i]=sin(pos/10000^(2i/d)), "
                 "PE[pos,2i+1]=cos(...)")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig4_positional_encoding.png", dpi=150)

    # numeric evidence: similarity is (almost) a function of distance only
    d1 = torch.stack([sim[i, i + 5] for i in range(L - 5)])
    print(f"PE . PE at distance 5 across all positions: mean {d1.mean():.4f}, std {d1.std():.2e} "
          f"(constant -> encodes relative offset)")
    print(f"||PE[pos]||^2 = d/2 = {d / 2:.0f} for every pos: {torch.allclose((pe ** 2).sum(-1), torch.full((L,), d / 2))}")
    print(f"saved {OUT / 'fig4_positional_encoding.png'}")


if __name__ == "__main__":
    main()
