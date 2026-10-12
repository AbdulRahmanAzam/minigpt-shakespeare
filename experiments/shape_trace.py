"""Figure 3 - tensor-shape trace through multi-head attention and the full model.

    python -m experiments.shape_trace                       # main config (configs/tiny_gpt.yaml)
    python -m experiments.shape_trace --batch 2 --seq 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.attention import causal_mask, merge_heads, scaled_dot_product_attention, split_heads
from src.config import load_config
from src.gpt import MiniGPT

OUT = Path("outputs/figures")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/tiny_gpt.yaml")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq", type=int, default=8)
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg.model.dropout = 0.0
    torch.manual_seed(0)
    model = MiniGPT(cfg.model).eval()
    B, T, d, H = args.batch, args.seq, cfg.model.d_model, cfg.model.n_heads
    lines = [f"MULTI-HEAD ATTENTION SHAPE TRACE  (B={B}, T={T}, d_model={d}, n_heads={H}, d_head={d // H})",
             "=" * 78, "", "[A] full model forward pass (first block traced in detail)"]

    trace: list[str] = []
    x = torch.randint(0, cfg.model.vocab_size, (B, T))
    out = model(x, return_attn=True, trace=trace)
    lines += ["  " + s for s in trace]
    lines.append(f"  attention maps returned: {len(out.attentions)} layers x {tuple(out.attentions[0].shape)}")

    # [B] explicit per-head computation, concatenation and projection, checked against the module
    mha = model.blocks[0].attn
    h = model.blocks[0].ln1(model.tok_emb(x) * model.emb_scale + model.pos_enc(T))
    q, k, v = (split_heads(proj(h), H) for proj in (mha.q_proj, mha.k_proj, mha.v_proj))
    lines += ["", "[B] attention inside each head, then concatenation + output projection",
              f"  Q, K, V after split_heads: {tuple(q.shape)}"]
    head_outputs = []
    for i in range(H):
        o, w = scaled_dot_product_attention(q[:, i], k[:, i], v[:, i], mask=causal_mask(T))
        head_outputs.append(o)
        lines.append(f"  head {i}: Q_i {tuple(q[:, i].shape)} -> weights {tuple(w.shape)} -> out {tuple(o.shape)}"
                     f"   (row sums = {w.sum(-1).mean():.3f})")
    concat = torch.cat(head_outputs, dim=-1)
    lines.append(f"  concat heads along features: {H} x {tuple(head_outputs[0].shape)} -> {tuple(concat.shape)}")
    stacked = torch.stack(head_outputs, dim=1)
    same = torch.allclose(merge_heads(stacked), concat)
    lines.append(f"  merge_heads(stack(heads)) == concat: {same}")
    projected = mha.out_proj(concat)
    lines.append(f"  W_o projection: {tuple(concat.shape)} @ {tuple(mha.out_proj.weight.T.shape)} -> {tuple(projected.shape)}")
    module_out, _ = mha(h)
    lines.append(f"  equals MultiHeadAttention module output: {torch.allclose(module_out, projected, atol=1e-6)}")

    text = "\n".join(lines)
    print(text)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fig3_shape_trace.txt").write_text(text + "\n", encoding="utf-8")
    print(f"\nsaved {OUT / 'fig3_shape_trace.txt'}")


if __name__ == "__main__":
    main()
