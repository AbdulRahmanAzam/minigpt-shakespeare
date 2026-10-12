"""Short hyper-parameter sweep on the main config (evidence for the chosen learning rate / dropout).

    python -m experiments.tune                        # default sweep: lr x dropout, 1500 steps each
    python -m experiments.tune --steps 800 --lrs 3e-4 1e-3 --dropouts 0.1

Writes outputs/tuning/results.csv and outputs/tuning/results.md.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.config import load_config
from src.train import train

OUT = Path("outputs/tuning")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/tiny_gpt.yaml")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lrs", type=float, nargs="+", default=[3e-4, 6e-4, 1e-3, 2e-3])
    p.add_argument("--dropouts", type=float, nargs="+", default=[0.2])
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for dropout in args.dropouts:
        for lr in args.lrs:
            name = f"lr{lr:g}_drop{dropout:g}"
            cfg = load_config(args.config, [
                f"train.lr={lr}", f"model.dropout={dropout}", f"train.max_steps={args.steps}",
                f"train.warmup_steps={min(200, args.steps // 10)}", "train.eval_interval=500",
                f"train.out_dir={OUT / 'runs'}", f"train.run_name={name}", "train.save_checkpoints=false",
            ])
            print(f"\n=== {name} ===")
            s = train(cfg)
            rows.append({"lr": lr, "dropout": dropout, "steps": args.steps,
                         "final_train_loss": round(s["final_train_loss"], 4),
                         "final_val_loss": round(s["final_val_loss"], 4),
                         "best_val_loss": round(s["best_val_loss"], 4),
                         "train_time_s": round(s["train_time_s"], 1)})

    # merge with earlier sweeps so the table accumulates (same lr/dropout/steps -> replaced)
    if (OUT / "results.csv").exists():
        with open(OUT / "results.csv", encoding="utf-8") as f:
            old = [{k: float(v) if k != "steps" else int(v) for k, v in r.items()} for r in csv.DictReader(f)]
        key = lambda r: (float(r["lr"]), float(r["dropout"]), int(r["steps"]))  # noqa: E731
        new_keys = {key(r) for r in rows}
        rows = sorted([r for r in old if key(r) not in new_keys] + rows, key=key)
    with open(OUT / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["| lr | dropout | steps | train loss | val loss | best val | time (s) |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['lr']:g} | {r['dropout']:g} | {r['steps']} | {r['final_train_loss']} | {r['final_val_loss']} "
              f"| {r['best_val_loss']} | {r['train_time_s']} |" for r in rows]
    (OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
