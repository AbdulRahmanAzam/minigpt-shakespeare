"""Build notebooks/MiniGPT_Colab.ipynb (the Colab showcase), optionally executing it so outputs are saved.

    python scripts/build_notebook.py            # write the notebook (no outputs)
    python scripts/build_notebook.py --execute  # write + run every cell locally, store the outputs in the file
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "MiniGPT_Colab.ipynb"
REPO = "https://github.com/AbdulRahmanAzam/minigpt-shakespeare"

cells = []


def md(text: str) -> None:
    cells.append(new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(new_code_cell(text.strip("\n")))


md(f"""
# MiniGPT — a GPT language model built from scratch

**AI Engineering · Assignment 1 — Transformer & GPT from scratch**

A decoder-only Transformer (GPT) written from first principles in PyTorch — scaled dot-product attention,
multi-head split/merge, sinusoidal positional encoding, pre-norm Transformer blocks, causal masking, training loop
and sampling — trained on **Tiny Shakespeare** (character level). No pretrained weights and no `transformers` /
`nn.Transformer` / `nn.MultiheadAttention` classes are used.

| | |
|---|---|
| Model | 4.77 M parameters · 6 layers · 4 heads · d_model 256 · context 256 · 65-character vocabulary |
| Validation | loss **1.353** nats/char · perplexity **3.87** · next-character accuracy **60.0 %** (top-5 **87.0 %**) |
| Baselines | trigram table 38.5 % accuracy · most-frequent character 15.2 % |
| Repository | [{REPO}]({REPO}) |

**How to run:** `Runtime → Run all`. A GPU runtime (`Runtime → Change runtime type → T4 GPU`) is faster, but every
cell also works on CPU. Outputs from a full run are already saved in this notebook, so it can be read without
running it.

**Contents**
1. Setup · 2. Dataset · 3. Unit tests · 4. Architecture checks (hand-calculated attention, causal mask,
multi-head shapes, positional encoding, model summary) · 5. Training pipeline (overfit gate, loss curves) ·
6. **Accuracy and loss** · 7. Text generation · 8. Attention inspection · 9. Ablation study ·
10. Concept experiments · 11. REST API · 12. (Optional) train from scratch · 13. Limitations
""")

md("## 1. Setup")
code(f"""
import os, sys, json
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not Path("minigpt-shakespeare").exists():
        !git clone -q {REPO}.git
    %cd minigpt-shakespeare
    !pip install -q -r requirements.txt
elif Path.cwd().name == "notebooks":          # running locally from the repository
    os.chdir("..")

os.environ["PYTHONIOENCODING"] = "utf-8"     # child processes print unicode safely on every OS
PY = f'"{{sys.executable}}"'                  # python of this kernel, quoted (paths may contain spaces)

import torch
print("working directory:", Path.cwd())
print("python", sys.version.split()[0], "| torch", torch.__version__,
      "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU)")
""")
code("""
from IPython.display import Image, Markdown, display

def show(path, width=900):
    display(Image(filename=str(path), width=width))

def show_md(path, max_chars=None):
    text = Path(path).read_text(encoding="utf-8")
    display(Markdown(text if max_chars is None else text[:max_chars]))
""")

md("""
## 2. Dataset — Tiny Shakespeare

Source: [Hugging Face `karpathy/tiny_shakespeare`](https://huggingface.co/datasets/karpathy/tiny_shakespeare)
(1,115,394 characters, 40,000 lines). The HF entry is a loading-script dataset that recent `datasets` versions no
longer run, so `prepare_data.py` downloads the same file, verifies its SHA-256 and applies the identical
**90 / 5 / 5** character split. The tokenizer is character level: 65 symbols, no special tokens.
""")
code("""
!{PY} scripts/prepare_data.py
print()
print(Path("data/train.txt").read_text(encoding="utf-8")[:400])
""")

md("""
## 3. Unit tests

59 tests: hand-calculated attention values, softmax properties, equality with PyTorch's reference attention,
causal masking (future tokens never change past outputs, gradients never flow from the future), tensor shapes,
parameter count formula, positional encoding values, tokenizer and target shifting, generation, checkpoint reload
in a fresh process, a tiny overfit gate, and the REST API.
""")
code("""
!{PY} -m pytest -p no:warnings --color=no
""")

md("""
## 4. Architecture checks

### 4.1 Self-attention by hand (Figure 1)
Three tokens, d_k = d_v = 2. Every step — QKᵀ, scaling by √d_k, row-wise softmax, weighted sum of V, then the same
with a causal mask — is printed and finally checked against our implementation in `src/attention.py`.
""")
code("""
!{PY} -m experiments.hand_attention
show("outputs/figures/fig1_hand_attention.png")
""")

md("""
### 4.2 Causal mask vs bidirectional attention (Figure 2)
GPT adds −∞ above the diagonal so position *i* can only attend to positions ≤ *i*; BERT lets every position see
every other position and instead hides input tokens with `[MASK]`.
""")
code("""
!{PY} -m experiments.mask_visualization
show("outputs/figures/fig2_causal_mask.png", 1100)
""")

md("""
### 4.3 Multi-head attention — tensor shapes (Figure 3)
Q, K, V are split into heads, attention runs inside each head, heads are concatenated and projected; the trace
also recomputes every head by hand and checks it equals the module output.
""")
code("""
!{PY} -m experiments.shape_trace
""")

md("""
### 4.4 Sinusoidal positional encoding (Figure 4)
`PE[pos, 2i] = sin(pos / 10000^(2i/d))`, `PE[pos, 2i+1] = cos(pos / 10000^(2i/d))`.
""")
code("""
!{PY} -m experiments.positional_encoding_viz
show("outputs/figures/fig4_positional_encoding.png")
""")

md("""
### 4.5 The trained model — summary and parameter count (Figure 5)
The trained weights are published as a GitHub Release asset and downloaded here (SHA-256 checked).
""")
code("""
CKPT = "checkpoints/minigpt_shakespeare.pt"
if not Path(CKPT).exists():
    !{PY} scripts/download_checkpoint.py
!{PY} -m experiments.model_summary --checkpoint {CKPT}
""")

md("""
## 5. Training pipeline

### 5.1 Overfit gate (Milestone 4)
Before long training, the model must memorise a tiny slice of text. Loss should fall from ln 65 ≈ 4.17 (a uniform
guess) to near zero — proof that gradients, the causal mask and the shifted targets are wired correctly.
""")
code("""
!{PY} -m src.train --config configs/overfit.yaml
""")

md("""
### 5.2 How the final model was trained (Figure 6)
AdamW, peak LR 2e-3 chosen by a 6-point sweep, warm-up + cosine decay, dropout 0.2, batch 64 × 256 characters,
3,000 steps (~9 minutes on an RTX 3050 laptop GPU). A 5,000-step run (second figure) shows the model starts to
overfit after ~3,000 steps, which is why training stops there.
""")
code("""
summary = json.loads(Path("outputs/logs/tiny_gpt/summary.json").read_text())
for k in ["parameters", "steps", "batch_size", "tokens_seen", "precision", "train_time_s", "tokens_per_s",
          "peak_memory_mb", "final_train_loss", "final_val_loss", "best_val_loss", "best_step"]:
    v = summary[k]
    print(f"{k:<18} {v:,.4f}" if isinstance(v, float) else f"{k:<18} {v:,}" if isinstance(v, int) else f"{k:<18} {v}")
print("hardware          ", summary["hardware"].get("gpu", "cpu"))
show("outputs/figures/fig6_loss_curves.png", 800)
show("outputs/figures/fig6b_overfitting_5000steps.png", 800)
show_md("outputs/tuning/results.md")
""")

md("""
## 6. Accuracy and loss of the final model

* **Loss** — mean next-character cross-entropy (nats/char); **perplexity** = e^loss; **bits/char** = loss / ln 2.
* **Accuracy** — how often the model's most likely next character is the true next character (top-1), or is among
  its 5 most likely characters (top-5). Baselines: always predicting the most frequent character, and the best
  bigram / trigram lookup tables built from the training text.
* The test split is reported in two parts: its last 62 % is *The Tempest*, a play that never occurs in training.

This cell also generates the fixed-prompt samples used in the report.
""")
code("""
!{PY} -m experiments.evaluate --checkpoint {CKPT}
""")
code("""
ev = json.loads(Path("outputs/eval/final_eval.json").read_text())
bl, ba = ev["baselines"], ev["baseline_accuracy_val"]
rows = [
    ("uniform guess", bl["uniform"], 1 / 65),
    ("most frequent character", bl["unigram"], ba["most_frequent_char"]),
    ("bigram table", bl["bigram"], ba["bigram"]),
    ("trigram table", bl["trigram"], ba["trigram"]),
    ("MiniGPT (validation)", ev["val"]["loss"], ev["val"]["top1_accuracy"]),
    ("MiniGPT (test)", ev["test"]["loss"], ev["test"]["top1_accuracy"]),
]
lines = ["| model | loss (nats/char) | perplexity | top-1 accuracy |", "|---|---|---|---|"]
lines += [f"| {n} | {l:.3f} | {2.718281828 ** l:.2f} | {100 * a:.1f} % |" for n, l, a in rows]
lines += ["", f"MiniGPT top-5 accuracy: validation **{100 * ev['val']['top5_accuracy']:.1f} %**, "
              f"test {100 * ev['test']['top5_accuracy']:.1f} %.",
          f"Test split by play: already-seen play loss **{ev['test_segments']['seen_plays']['loss']:.3f}**, "
          f"unseen *The Tempest* loss **{ev['test_segments']['unseen_play']['loss']:.3f}**."]
display(Markdown("\\n".join(lines)))
""")

md("""
## 7. Text generation

Autoregressive sampling: feed the prompt, take the next-character distribution at the last position, sample
(temperature 0.8), append, repeat — with a sliding 256-character window.
""")
code("""
!{PY} -m src.generate --checkpoint {CKPT} --prompt "ROMEO:" --max-new-tokens 300 --seed 42
""")
code("""
!{PY} -m src.generate --checkpoint {CKPT} --prompt "To be, or not to be" --max-new-tokens 200 --seed 7
!{PY} -m src.generate --checkpoint {CKPT} --prompt "The future of artificial intelligence" --max-new-tokens 200 --seed 3
""")
md("**Try your own prompt** — edit the variables and re-run this cell.")
code("""
from src.checkpoint import load_checkpoint
from src.generate import generate_text

model, tokenizer, _ = load_checkpoint(CKPT, "cuda" if torch.cuda.is_available() else "cpu")

PROMPT = "KING HENRY:"
TEMPERATURE = 0.8      # 0 = greedy (deterministic, tends to repeat); higher = more diverse
MAX_NEW_TOKENS = 300
SEED = 1

r = generate_text(model, tokenizer, PROMPT, MAX_NEW_TOKENS, TEMPERATURE, seed=SEED)
print(r.text)
if r.ignored_chars:
    print("\\n(characters outside the 65-character vocabulary were dropped:", r.ignored_chars, ")")
""")

md("""
## 8. What the trained model attends to (Figure 7)

Per-head statistics over validation text: early layers contain heads that look almost only at the previous
character (local spelling patterns); the last layers spread attention 30–50 characters back (words, line and
speaker structure). Every weight above the diagonal is exactly zero — the causal mask holds in the trained model.
""")
code("""
!{PY} -m experiments.attention_visualization --checkpoint {CKPT}
show("outputs/figures/fig7_attention_detail.png", 1100)
""")

md("""
## 9. Ablation study (Figure 9)

A small base model (d_model 128, 4 layers, 4 heads, context 128) with **one factor changed at a time**,
3 seeds each, 3,000 steps. Results were produced by `python -m experiments.ablation` (~1 hour on a laptop GPU),
so they are loaded here rather than re-run.
""")
code("""
show("outputs/figures/fig9_ablation.png", 1100)
text = Path("outputs/ablations/results.md").read_text(encoding="utf-8")
display(Markdown(text.split("## Sample generations")[0]))
""")

md("""
## 10. Concept experiments

* **GPT vs BERT** — the same block trained three ways: causal mask (GPT), no mask with next-token targets
  ("leaky"), and masked-language-modelling (BERT).
* **Why Transformers** — recurrence vs attention speed/memory, and an LSTM language model with the same parameter
  count trained on the same data.
* **Decoding strategies** — greedy vs temperature vs top-k / top-p.
""")
code("""
show_md("outputs/mlm_vs_clm/results.md", 1800)
show("outputs/figures/fig2b_mlm_vs_clm.png", 700)
""")
code("""
show("outputs/figures/fig0_rnn_vs_attention.png", 1100)
show_md("outputs/lstm_baseline/results.md")
show("outputs/figures/fig0b_lstm_vs_gpt.png", 1000)
""")
code("""
text = Path("outputs/decoding/results.md").read_text(encoding="utf-8")
display(Markdown(text.split("## Examples")[0]))
""")

md("""
## 11. REST API (FastAPI)

The same model is served by `src/api.py` (also deployed on Render). Here we start the server in the background,
call its endpoints, and stop it.
""")
code("""
import subprocess, time, urllib.request

server = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api:app", "--port", "8000"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
def get(path):
    return json.loads(urllib.request.urlopen("http://127.0.0.1:8000" + path, timeout=10).read())
def post(path, payload):
    req = urllib.request.Request("http://127.0.0.1:8000" + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())

for _ in range(120):
    try:
        print("GET /health ->", get("/health")); break
    except OSError:
        time.sleep(1)
info = get("/model-info")
print("GET /model-info ->", info["parameters"], "parameters, val loss", round(info["metrics"]["val_loss"], 4))
out = post("/generate", {"prompt": "JULIET:", "max_new_tokens": 200, "temperature": 0.8, "seed": 1})
print("POST /generate ->")
print(out["text"])
server.terminate()
""")

md("""
## 12. (Optional) Train the model from scratch

Set `TRAIN = True` to reproduce the final model (~9 min on an RTX 3050; a Colab T4 uses fp16 automatically).
The new checkpoint is written to `checkpoints/tiny_gpt/best.pt`.
""")
code("""
TRAIN = False
if TRAIN:
    !{PY} -m src.train --config configs/tiny_gpt.yaml
    !{PY} -m experiments.plot_training --run outputs/logs/tiny_gpt
    !{PY} -m experiments.evaluate --checkpoint checkpoints/tiny_gpt/best.pt
else:
    print("skipped (set TRAIN = True to run)")
""")

md("""
## 13. Known limitations

1. **Data-limited.** ~1 M training characters: beyond ~3,000 steps the model memorises (train 0.92 vs val 1.39 at
   5,000 steps); a 2.2× larger model or more dropout did not improve validation loss.
2. **Domain shift.** *The Tempest* (62 % of the test split) never appears in training — loss 1.69 there vs 1.26 on
   the already-seen play.
3. **Locally fluent, globally incoherent.** Real words and correct play formatting, but no consistent meaning.
4. **Greedy decoding loops**; use temperature ≥ 0.5.
5. **65-character vocabulary** — digits other than 3, quotes, brackets etc. are dropped from prompts.
6. **256-character context**, no KV cache (generation re-processes the window for every character).

Full details, commands and citations: see the repository README.
""")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--execute", action="store_true")
    p.add_argument("--timeout", type=int, default=1800)
    args = p.parse_args()
    nb = new_notebook(cells=cells, metadata={
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    if args.execute:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=args.timeout, kernel_name="python3",
                       resources={"metadata": {"path": str(NB_PATH.parent)}}).execute()
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, NB_PATH)
    print(f"wrote {NB_PATH} ({len(cells)} cells{', executed' if args.execute else ''})")


if __name__ == "__main__":
    main()
