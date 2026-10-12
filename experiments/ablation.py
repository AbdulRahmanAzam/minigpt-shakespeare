"""Controlled ablation study (Section 13) - Figure 9.

Every run starts from configs/ablation_base.yaml and changes exactly ONE factor:

    depth     n_layers       2 | 4 | 6
    heads     n_heads        2 | 4 | 8          (d_model fixed at 128 -> d_head 64 | 32 | 16)
    context   context_length 64 | 128 | 256    (batch 128 | 64 | 32, so tokens/step = 8192 is constant)
    position  pos_encoding   sinusoidal | learned | none

Held fixed: data, steps (3000), LR schedule, optimizer, dropout, tokens per step, seed set.
The baseline (4 layers, 4 heads, context 128, sinusoidal) is shared by all groups.

Measured per run: parameter count, final/best validation loss, wall-clock training time, throughput,
peak GPU memory, and generation metrics (valid-word rate, distinct-2, 4-gram repetition) on fixed prompts.

    python -m experiments.ablation                       # all groups, seeds 1337 1338 1339
    python -m experiments.ablation --groups depth --seeds 1337
    python -m experiments.ablation --report-only         # rebuild table + figure from saved runs
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.text_metrics import all_metrics  # noqa: E402
from src.config import load_config  # noqa: E402
from src.dataset import load_tokenizer  # noqa: E402
from src.generate import generate_text  # noqa: E402
from src.train import train  # noqa: E402
from src.utils import read_json, write_json  # noqa: E402

BASE = "configs/ablation_base.yaml"
OUT = Path("outputs/ablations")
RUNS = OUT / "runs"
PROMPTS = ["ROMEO:", "First Citizen:\n", "To be, or not to be"]

# group -> list of (label, overrides)
GROUPS: dict[str, list[tuple[str, list[str]]]] = {
    "depth": [("2 layers", ["model.n_layers=2"]), ("4 layers", []), ("6 layers", ["model.n_layers=6"])],
    "heads": [("2 heads", ["model.n_heads=2"]), ("4 heads", []), ("8 heads", ["model.n_heads=8"])],
    "context": [("ctx 64", ["model.context_length=64", "train.batch_size=128"]), ("ctx 128", []),
                ("ctx 256", ["model.context_length=256", "train.batch_size=32"])],
    "position": [("sinusoidal", []), ("learned", ["model.pos_encoding=learned"]),
                 ("none", ["model.pos_encoding=none"])],
}


def run_key(overrides: list[str]) -> str:
    if not overrides:
        return "baseline"
    return "_".join(o.split(".", 1)[1].replace("=", "") for o in overrides if o.startswith("model."))


def run_one(overrides: list[str], seed: int, force: bool) -> dict:
    name = f"{run_key(overrides)}_seed{seed}"
    summary_path = RUNS / name / "result.json"
    if summary_path.exists() and not force:
        return read_json(summary_path)
    cfg = load_config(BASE, overrides + [f"train.seed={seed}", f"train.run_name={name}"])
    print(f"\n=== {name} ===", flush=True)
    summary, model = train(cfg, return_model=True)
    tok = load_tokenizer(cfg.data.data_dir)
    gens = [generate_text(model, tok, pr, 300, temperature=0.8, seed=seed + i).completion
            for i, pr in enumerate(PROMPTS)]
    metrics = [all_metrics(g) for g in gens]
    result = {
        "name": name, "seed": seed, "overrides": overrides,
        "parameters": summary["parameters"],
        "final_val_loss": summary["final_val_loss"], "best_val_loss": summary["best_val_loss"],
        "final_train_loss": summary["final_train_loss"],
        "train_time_s": summary["train_time_s"], "tokens_per_s": summary["tokens_per_s"],
        "peak_memory_mb": summary["peak_memory_mb"],
        **{k: sum(m[k] for m in metrics) / len(metrics) for k in metrics[0]},
        "sample": PROMPTS[0] + gens[0],
    }
    write_json(summary_path, result)
    return result


def mean_std(xs: list[float]) -> tuple[float, float]:
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def report(groups: list[str], seeds: list[int]) -> None:
    rows, agg = [], defaultdict(dict)
    for g in groups:
        for label, ov in GROUPS[g]:
            results = [read_json(RUNS / f"{run_key(ov)}_seed{s}" / "result.json") for s in seeds
                       if (RUNS / f"{run_key(ov)}_seed{s}" / "result.json").exists()]
            if not results:
                continue
            for r in results:
                rows.append({"group": g, "setting": label, **{k: v for k, v in r.items() if k not in ("overrides", "sample")}})
            agg[g][label] = {k: mean_std([r[k] for r in results]) for k in
                             ("best_val_loss", "final_val_loss", "final_train_loss", "train_time_s", "tokens_per_s",
                              "peak_memory_mb", "valid_word_rate", "distinct_2", "repetition_4gram")}
            agg[g][label]["parameters"] = results[0]["parameters"]
            agg[g][label]["n_seeds"] = len(results)
            agg[g][label]["sample"] = results[0]["sample"]
    if not rows:
        print("no results yet")
        return

    with open(OUT / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def pm(v: tuple[float, float], fmt: str = ".4f") -> str:
        return f"{v[0]:{fmt}} ± {v[1]:{fmt}}" if v[1] else f"{v[0]:{fmt}}"

    md = ["# Ablation results", "",
          f"Base: `{BASE}` (d_model 128, 4 layers, 4 heads, context 128, sinusoidal, 3000 steps, 8192 tokens/step). "
          f"Values are mean ± std over seeds {seeds}. Losses are validation cross-entropy in nats/char.", ""]
    for g, settings in agg.items():
        md += [f"## {g}", "",
               "| setting | params | best val loss | final val loss | train loss | train time (s) | tokens/s | "
               "peak mem (MB) | valid words | distinct-2 | 4-gram rep. |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for label, a in settings.items():
            md.append(f"| {label} | {a['parameters']:,} | {pm(a['best_val_loss'])} | {pm(a['final_val_loss'])} | "
                      f"{pm(a['final_train_loss'])} | {pm(a['train_time_s'], '.0f')} | {a['tokens_per_s'][0]:,.0f} | "
                      f"{a['peak_memory_mb'][0] or 0:.0f} | {a['valid_word_rate'][0]:.3f} | {a['distinct_2'][0]:.3f} | "
                      f"{a['repetition_4gram'][0]:.3f} |")
        md.append("")
    md += ["## Sample generations (seed 1, prompt 'ROMEO:')", ""]
    for g, settings in agg.items():
        for label, a in settings.items():
            md += [f"**{g} / {label}**", "", "```text", a["sample"][:300], "```", ""]
    (OUT / "results.md").write_text("\n".join(md), encoding="utf-8")

    fig, axes = plt.subplots(1, len(agg), figsize=(4.6 * len(agg), 4.6), squeeze=False)
    for ax, (g, settings) in zip(axes[0], agg.items()):
        labels = list(settings)
        means = [settings[l]["best_val_loss"][0] for l in labels]
        stds = [settings[l]["best_val_loss"][1] for l in labels]
        bars = ax.bar(labels, means, yerr=stds, capsize=4, color=["#8fb3e6", "#1f5fbf", "#0b2f6b"])
        for b, l in zip(bars, labels):
            s = settings[l]
            extra = f"{s['parameters'] / 1e6:.2f}M p\n{s['train_time_s'][0]:.0f}s"
            if g == "context":
                extra = f"{s['peak_memory_mb'][0] or 0:.0f} MB\n{s['tokens_per_s'][0] / 1e3:.0f}k tok/s"
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.3f}\n{extra}",
                    ha="center", va="bottom", fontsize=8)
        lo = min(means) - 0.1
        ax.set_ylim(lo, max(means) + 0.12)
        ax.set_title(g)
        ax.set_ylabel("best validation loss (nats/char)")
    fig.suptitle("Figure 9 - controlled ablations (one factor changed per panel; mean ± std over seeds)")
    fig.tight_layout()
    fig.savefig(OUT / "fig9_ablation.png", dpi=150)
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("outputs/figures/fig9_ablation.png", dpi=150)
    print("\n".join(md[: 6 + 6 * len(agg)]))
    print(f"\nsaved {OUT / 'results.csv'}, {OUT / 'results.md'}, {OUT / 'fig9_ablation.png'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--groups", nargs="+", default=list(GROUPS), choices=list(GROUPS))
    p.add_argument("--seeds", nargs="+", type=int, default=[1337, 1338, 1339])
    p.add_argument("--force", action="store_true", help="re-run even if a result exists")
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()
    if not args.report_only:
        for seed in args.seeds:                 # seed-major order: one full pass first, then repeats
            done: set[str] = set()
            for g in args.groups:
                for _, ov in GROUPS[g]:
                    key = run_key(ov)
                    if key not in done:
                        run_one(ov, seed, args.force)
                        done.add(key)
            report(args.groups, args.seeds)
    report(args.groups, args.seeds)


if __name__ == "__main__":
    main()
