"""Run the submission acceptance tests of Section 16 in one go.

    python scripts/acceptance.py                      # uses checkpoints/minigpt_shakespeare.pt
    python scripts/acceptance.py --checkpoint checkpoints/tiny_gpt/best.pt

  1. dependencies importable          5. generate from three fixed prompts
  2. unit tests pass                  6. causal check on the trained model (future never changes the past)
  3. tiny overfit run                 7. no secrets / large files tracked by git
  4. final checkpoint loads in a fresh process
"""

from __future__ import annotations

import argparse
import importlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROMPTS = ["ROMEO:", "First Citizen:", "To be, or not to be"]


def step(n: int, title: str) -> None:
    print(f"\n[{n}/7] {title}", flush=True)


def run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"FAILED: {' '.join(cmd)}")
    return r.stdout


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/minigpt_shakespeare.pt")
    args = p.parse_args()
    py = sys.executable

    step(1, "dependencies")
    for mod in ("torch", "numpy", "yaml", "matplotlib", "fastapi", "uvicorn", "pydantic", "pytest"):
        importlib.import_module(mod)
    print("  all imports ok")

    step(2, "unit tests")
    out = run([py, "-m", "pytest", "-p", "no:warnings"])
    print("  " + next(l for l in reversed(out.splitlines()) if "passed" in l))

    step(3, "tiny overfit run (configs/overfit.yaml)")
    out = run([py, "-m", "src.train", "--config", "configs/overfit.yaml"])
    losses = [float(m) for m in re.findall(r"\[eval\] step +\d+ \| train (\d+\.\d+)", out)]
    print(f"  loss on the tiny training set: {losses[0]:.3f} (step 0) -> {losses[-1]:.3f} (step 300)")
    assert losses[-1] < 0.1 * losses[0], "overfit gate failed"

    step(4, "load final checkpoint in a fresh process")
    code = ("import sys; sys.path.insert(0, '.'); from src.checkpoint import load_checkpoint; "
            f"m, t, s = load_checkpoint(r'{args.checkpoint}'); "
            "print(f'loaded step {s[\"step\"]}, {m.num_parameters():,} parameters, val loss {s[\"metrics\"].get(\"val_loss\")}')")
    print("  " + run([py, "-c", code]).strip())

    step(5, "generate from three fixed prompts")
    for prompt in PROMPTS:
        out = run([py, "-m", "src.generate", "--checkpoint", args.checkpoint, "--prompt", prompt,
                   "--max-new-tokens", "120", "--seed", "0"])
        gen = out.split("Generated: ", 1)[1].split("\nmodel parameters")[0].strip()
        print(f"  --- {prompt!r}\n  " + gen.replace("\n", "\n  "))

    step(6, "causal correctness on the trained model")
    import torch

    from src.checkpoint import load_checkpoint
    model, tok, _ = load_checkpoint(ROOT / args.checkpoint)
    T, V = model.cfg.context_length, model.cfg.vocab_size
    torch.manual_seed(0)
    x = torch.randint(0, V, (4, T))
    with torch.no_grad():
        base = model(x).logits
        worst = 0.0
        for t in range(0, T - 1, max(1, T // 16)):
            x2 = x.clone()
            x2[:, t + 1:] = torch.randint(0, V, (4, T - t - 1))
            worst = max(worst, (model(x2).logits[:, : t + 1] - base[:, : t + 1]).abs().max().item())
        att = model(x, return_attn=True).attentions
    future = max(torch.triu(a, 1).abs().max().item() for a in att)
    print(f"  max change of past logits when the future is rewritten: {worst:.2e}")
    print(f"  max attention weight on a future position: {future:.2e}")
    assert worst < 1e-4 and future == 0.0, "causal check failed"

    step(7, "no secrets or large artifacts tracked by git")
    files = run(["git", "ls-files"]).split()
    big = [f for f in files if (ROOT / f).exists() and (ROOT / f).stat().st_size > 5 * 2**20]
    bad = [f for f in files if f.endswith((".pt", ".env")) or Path(f).name == ".env"]
    secret_pat = re.compile(r"(sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})")
    leaks = [f for f in files if (ROOT / f).is_file() and (ROOT / f).stat().st_size < 2**20
             and secret_pat.search((ROOT / f).read_text(encoding="utf-8", errors="ignore"))]
    print(f"  tracked files: {len(files)} | >5 MB: {big or 'none'} | weights/.env: {bad or 'none'} | "
          f"secret-like strings: {leaks or 'none'}")
    assert not (big or bad or leaks), "repository hygiene check failed"

    print("\nALL ACCEPTANCE CHECKS PASSED")


if __name__ == "__main__":
    main()
