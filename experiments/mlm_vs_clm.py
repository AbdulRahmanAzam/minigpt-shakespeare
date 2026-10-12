"""Lesson 06 - why GPT (causal LM) and BERT (masked LM) use different attention constraints.

Three models with the SAME Transformer code, size, data and step budget:

  CLM          causal mask      + next-token targets           (GPT)
  leaky CLM    NO mask          + next-token targets           (what happens if GPT forgot the mask)
  MLM          NO mask          + 15% masked-token targets     (BERT; 80% [MASK] / 10% random / 10% keep)

Measurements:
  * training / validation loss on each model's own objective
  * "honest" next-token loss: predict token t+1 when tokens > t are NOT available
    (we feed only the prefix, as at generation time) - exposes the leaky model
  * masked-token loss for CLM vs MLM at the same positions (does seeing the right context help?)
  * a left-to-right generation sample from each model

    python -m experiments.mlm_vs_clm --steps 1500
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from src.config import ModelConfig  # noqa: E402
from src.dataset import encode_split, load_tokenizer  # noqa: E402
from src.gpt import MiniGPT  # noqa: E402
from src.utils import autocast, pick_device, pick_precision, set_seed, write_json  # noqa: E402

OUT = Path("outputs/mlm_vs_clm")
T, B = 128, 64


def mlm_corrupt(x: torch.Tensor, vocab: int, mask_id: int, gen: torch.Generator, p: float = 0.15):
    """BERT masking: choose 15% of positions; of those 80% -> [MASK], 10% -> random token, 10% unchanged."""
    chosen = torch.rand(x.shape, generator=gen, device=x.device) < p
    labels = torch.where(chosen, x, torch.full_like(x, -100))
    r = torch.rand(x.shape, generator=gen, device=x.device)
    inp = x.clone()
    inp[chosen & (r < 0.8)] = mask_id
    rand_pos = chosen & (r >= 0.8) & (r < 0.9)
    inp[rand_pos] = torch.randint(0, vocab, (int(rand_pos.sum()),), generator=gen, device=x.device)
    return inp, labels


def batches(ids: torch.Tensor, gen: torch.Generator, device):
    while True:
        s = torch.randint(0, len(ids) - T - 1, (B,), generator=gen)
        x = torch.stack([ids[i: i + T] for i in s])
        y = torch.stack([ids[i + 1: i + T + 1] for i in s])
        yield x.to(device), y.to(device)


def train_variant(kind: str, train_ids, val_ids, vocab: int, steps: int, device, precision, log_every=100):
    set_seed(0)
    mask_id = vocab                                   # extra id for [MASK] (MLM only)
    cfg = ModelConfig(vocab_size=vocab + (kind == "mlm"), context_length=T, d_model=128, n_heads=4,
                      n_layers=4, dropout=0.1, causal=(kind == "clm"))
    model = MiniGPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
    gen = torch.Generator().manual_seed(0)
    mgen = torch.Generator(device=device).manual_seed(0)
    data = batches(train_ids, gen, device)
    curve = []

    def step_loss(x, y, g):
        if kind == "mlm":
            inp, labels = mlm_corrupt(x, vocab, mask_id, g)
            return model(inp, labels).loss
        return model(x, y).loss

    for step in range(steps + 1):
        lr = 1e-3 * min(1.0, (step + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for gr in opt.param_groups:
            gr["lr"] = lr
        if step % log_every == 0:
            model.eval()
            with torch.no_grad(), autocast(device, precision):
                vgen = torch.Generator().manual_seed(123)
                vmgen = torch.Generator(device=device).manual_seed(123)
                vdata = batches(val_ids, vgen, device)
                vl = sum(step_loss(*next(vdata), vmgen).item() for _ in range(10)) / 10
            curve.append((step, vl))
            model.train()
        if step == steps:
            break
        x, y = next(data)
        with autocast(device, precision):
            loss = step_loss(x, y, mgen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    return model, curve


@torch.no_grad()
def honest_next_token_loss(model, ids, device, n=512) -> float:
    """Feed only the prefix x[:t+1] and score the prediction of x[t+1] at the LAST position.
    No future token is present in the input, exactly like autoregressive generation."""
    gen = torch.Generator().manual_seed(7)
    starts = torch.randint(0, len(ids) - T - 1, (n,), generator=gen)
    x = torch.stack([ids[s: s + T] for s in starts]).to(device)
    y = torch.stack([ids[s + T] for s in starts]).to(device)
    logits = model(x).logits[:, -1, :].float()
    return torch.nn.functional.cross_entropy(logits, y).item()


@torch.no_grad()
def masked_position_loss(model, ids, device, kind: str, vocab: int, n=256) -> float:
    """Loss of predicting a hidden character at position 64 of a 128-char window.
    MLM sees both sides ([MASK] in the middle); CLM sees only the 64 characters to the left."""
    gen = torch.Generator().manual_seed(11)
    starts = torch.randint(0, len(ids) - T - 1, (n,), generator=gen)
    x = torch.stack([ids[s: s + T] for s in starts]).to(device)
    target = x[:, 64].clone()
    if kind == "mlm":
        inp = x.clone()
        inp[:, 64] = vocab
        logits = model(inp).logits[:, 64, :vocab].float()
    else:
        logits = model(x[:, :64]).logits[:, -1, :vocab].float()     # prefix only, predicts position 64
    return torch.nn.functional.cross_entropy(logits, target).item()


@torch.no_grad()
def generate_left_to_right(model, tok, kind: str, vocab: int, prompt="ROMEO:", n=200) -> str:
    """CLM / leaky: standard sampling. MLM: append [MASK] and fill it in, one character at a time."""
    gen = torch.Generator(device=next(model.parameters()).device).manual_seed(0)
    ids = torch.tensor([tok.encode(prompt)], device=gen.device)
    for _ in range(n):
        ctx = ids[:, -(T - 1):] if kind == "mlm" else ids[:, -T:]
        if kind == "mlm":
            ctx = torch.cat([ctx, torch.full((1, 1), vocab, device=ctx.device)], dim=1)
        logits = model(ctx).logits[:, -1, :vocab].float() / 0.8
        nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=gen)
        ids = torch.cat([ids, nxt], dim=1)
    return tok.decode(ids[0].tolist())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1500)
    args = p.parse_args()
    device = pick_device("auto")
    precision = pick_precision("auto", device)
    tok = load_tokenizer("data")
    V = tok.vocab_size
    train_ids, val_ids = encode_split(tok, "data", "train"), encode_split(tok, "data", "val")

    results, curves, lines = {}, {}, []
    for kind, name in (("clm", "CLM (causal mask)"), ("leaky", "leaky CLM (no mask)"), ("mlm", "MLM (bidirectional)")):
        print(f"training {name} ...", flush=True)
        model, curve = train_variant(kind, train_ids, val_ids, V, args.steps, device, precision)
        curves[name] = curve
        honest = honest_next_token_loss(model, val_ids, device) if kind != "mlm" else float("nan")
        masked = masked_position_loss(model, val_ids, device, kind if kind == "mlm" else "clm", V) \
            if kind != "leaky" else float("nan")
        sample = generate_left_to_right(model, tok, kind, V)
        results[name] = {"objective_val_loss": curve[-1][1], "honest_next_token_loss": honest,
                         "hidden_char_loss_at_pos64": masked, "sample": sample}
        print(f"  objective val loss {curve[-1][1]:.4f} | honest next-token {honest:.4f} | "
              f"hidden-char loss {masked:.4f}", flush=True)

    lines += ["# CLM vs leaky CLM vs MLM", "",
              f"Same architecture (d_model 128, 4 layers, 4 heads, context {T}), same data, {args.steps} steps each.", "",
              "| model | attention mask | objective val loss | honest next-token loss | hidden-char loss (pos 64) |",
              "|---|---|---|---|---|"]
    masks = {"CLM (causal mask)": "causal", "leaky CLM (no mask)": "none", "MLM (bidirectional)": "none"}
    for name, r in results.items():
        f = lambda v: "n/a" if math.isnan(v) else f"{v:.4f}"  # noqa: E731
        lines.append(f"| {name} | {masks[name]} | {r['objective_val_loss']:.4f} | {f(r['honest_next_token_loss'])} | "
                     f"{f(r['hidden_char_loss_at_pos64'])} |")
    lines += ["", "*objective val loss* = each model's own training objective on validation text.  ",
              "*honest next-token loss* = predict the character after a 128-char prefix with no future input "
              "(the generation setting).  ",
              "*hidden-char loss* = predict a character in the middle of a window: MLM sees both sides, "
              "CLM only the left side.", ""]
    for name, r in results.items():
        lines += [f"**{name}** - left-to-right sample:", "", "```text", r["sample"], "```", ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(OUT / "results.json", {"results": results, "curves": curves})
    print("\n".join(lines))

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, c in curves.items():
        ax.plot([s for s, _ in c], [v for _, v in c], label=name)
    ax.set_xlabel("step")
    ax.set_ylabel("validation loss on own objective (nats/char)")
    ax.set_title("Causal vs bidirectional attention: the leaky model 'solves' next-token\n"
                 "prediction by copying the answer, which is impossible at generation time")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "mlm_vs_clm_curves.png", dpi=150)
    fig.savefig("outputs/figures/fig2b_mlm_vs_clm.png", dpi=150)
    print(f"saved {OUT}/results.md, results.json, mlm_vs_clm_curves.png")


if __name__ == "__main__":
    main()
