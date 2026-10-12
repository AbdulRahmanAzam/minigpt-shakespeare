# MiniGPT — a GPT-style language model built from scratch

A small decoder-only Transformer (GPT) **implemented from first principles in PyTorch** — attention,
multi-head split/merge, sinusoidal positions, pre-norm blocks, causal masking, training loop, sampling —
trained on **Tiny Shakespeare**, evaluated with controlled ablations, and served through a **CLI** and a
**FastAPI** REST API. No pretrained weights and no `transformers` / `nn.Transformer` / `nn.MultiheadAttention`
classes are used anywhere.

| | |
|---|---|
| Model | 4.77 M parameters · 6 layers · 4 heads · d_model 256 · context 256 · character vocabulary 65 |
| Data | Tiny Shakespeare (1.1 M characters), 90/5/5 split identical to the Hugging Face dataset |
| Result | validation loss **1.353** nats/char (perplexity 3.87), next-char accuracy **60.0 %** (top-5 87.0 %) · test loss 1.530 |
| Training | 3,000 steps, ~9 min on one RTX 3050 Laptop GPU (4 GB), bf16 |
| Live demo | [minigpt-shakespeare.containers.snapdeploy.app](https://minigpt-shakespeare.containers.snapdeploy.app) — [`/docs`](https://minigpt-shakespeare.containers.snapdeploy.app/docs), [`/model-info`](https://minigpt-shakespeare.containers.snapdeploy.app/model-info); `POST /generate` needs an `X-API-Key` header (sleeps when idle, first request wakes it in ~60–90 s) |
| Colab | [`notebooks/MiniGPT_Colab.ipynb`](notebooks/MiniGPT_Colab.ipynb) (outputs of a full run are saved in the file; regenerate with `python scripts/build_notebook.py --execute`) — [open in Colab](https://colab.research.google.com/github/AbdulRahmanAzam/minigpt-shakespeare/blob/main/notebooks/MiniGPT_Colab.ipynb) |

```text
$ python -m src.generate --prompt "ROMEO:" --max-new-tokens 200 --seed 42
Prompt: ROMEO:
Generated: ROMEO:
As foul shadows not be great as to say,
As much as your mother like a common to death.
God grant you, to this uncle is commended:
I am not like a little fall for the prince,
Which is but unite as a d

model parameters: 4.77M (4,772,352)
architecture: 6 layers, 4 heads, d_model 256, sinusoidal positions, vocab 65 (character-level)
context length: 256
temperature: 0.8
seed: 42
checkpoint: checkpoints/minigpt_shakespeare.pt (step 2750, val loss 1.3531)
device: cpu | 200 new tokens in 5.58s
```

---

## 1. Where everything lives

| What | Where |
|---|---|
| **Model** (core) | [`src/gpt.py`](src/gpt.py) → [`src/transformer_block.py`](src/transformer_block.py) → [`src/attention.py`](src/attention.py), [`src/positional_encoding.py`](src/positional_encoding.py) |
| **Data pipeline** | [`scripts/prepare_data.py`](scripts/prepare_data.py) (download + split), [`src/tokenizer.py`](src/tokenizer.py), [`src/dataset.py`](src/dataset.py) |
| **Training** | [`src/train.py`](src/train.py), configs in [`configs/`](configs/) |
| **Inference** | CLI [`src/generate.py`](src/generate.py), REST API [`src/api.py`](src/api.py), checkpoint I/O [`src/checkpoint.py`](src/checkpoint.py) |
| **Tests** | [`tests/`](tests/) — attention, mask, shapes, overfit, data pipeline, generation/checkpoint, API |
| **Experiments** | [`experiments/`](experiments/) — evidence figures, ablations, evaluation, concept experiments |
| **Results** | [`outputs/`](outputs/) — figures, eval JSON, samples, ablation tables, training logs |
| **Deployment** | [`Dockerfile`](Dockerfile), [`render.yaml`](render.yaml) |
| **Report** | [`report/`](report/) |

## 2. Setup

Python 3.10–3.12 (3.12 recommended; it is what Colab uses).

```bash
git clone https://github.com/AbdulRahmanAzam/minigpt-shakespeare.git
cd minigpt-shakespeare
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
# optional, for an NVIDIA GPU: pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts/prepare_data.py                            # downloads + splits the dataset (~1 MB)
python -m pytest                                          # 59 tests, ~20 s on CPU
```

## 3. Dataset

[Tiny Shakespeare](https://huggingface.co/datasets/karpathy/tiny_shakespeare) (Karpathy, char-rnn): 1,115,394
characters / 40,000 lines from Shakespeare's plays. The HF entry is a *loading-script* dataset, which
`datasets >= 4` can no longer execute, so [`scripts/prepare_data.py`](scripts/prepare_data.py) downloads the same
file the script uses, checks its SHA-256 and applies the same 90/5/5 character split:
**train 1,003,854 · val 55,770 · test 55,770**. Details: [`data/README.md`](data/README.md).

**Tokenizer** — deterministic character level: vocabulary = sorted unique characters of the training split
= **65 symbols** (`\n`, space, `!$&',-.3:;?`, A–Z, a–z). **No special tokens**: the corpus is one continuous
stream cut into fixed windows, so BOS/EOS/PAD are unnecessary. Characters outside the vocabulary can only
come from user prompts; they are dropped and reported (`ignored_chars`) instead of mapped to an untrained `<unk>`.

## 4. Architecture

```text
token ids (B,T) ─► Embedding(65,256)·√256 ─┐
                                           + ─► dropout ─► 6 × Block ─► LayerNorm ─► Linear(256,65) ─► logits (B,T,65)
sinusoidal PE (T,256) ─────────────────────┘

Block (pre-norm):   x = x + MultiHeadAttention(LayerNorm(x))        # causal, 4 heads × 64 dims
                    x = x + FFN(LayerNorm(x))                         # 256 → 1024 → GELU → 256
MultiHeadAttention: Q,K,V = xW_q, xW_k, xW_v → split_heads (B,4,T,64) → softmax(QKᵀ/√64 + M)·V
                    → merge_heads (B,T,256) → W_o           M[i,j] = −∞ if j > i (future), else 0
Loss:               cross-entropy(logits[:, t], ids[t+1]) averaged over all positions
```

Design choices and why: pre-LayerNorm (stable without long warm-up), GPT-2 init with residual projections scaled
by 1/√(2·layers), token embedding × √d_model so it is not drowned by the ±1 sinusoidal PE, AdamW (β = 0.9/0.95,
weight decay 0.1 on matrices only), linear warm-up + cosine decay, gradient clipping 1.0, bf16/fp16 autocast with
an fp32 softmax. Alternatives are configurable for experiments: `pos_encoding: learned | none`,
`tie_weights: true`, `attn_impl: sdpa` (PyTorch's fused kernel, verified equal to ours in the tests).

## 5. Training

```bash
python -m src.train --config configs/overfit.yaml     # debugging gate: memorise 65 characters (loss 4.2 → ~0.03)
python -m src.train --config configs/tiny_gpt.yaml    # main model (~~9 min on an RTX 3050; any CUDA GPU or CPU works)
python -m src.train --config configs/tiny_gpt.yaml --set train.max_steps=1000 --set model.n_layers=4   # overrides
python -m src.train --config configs/tiny_gpt.yaml --resume checkpoints/tiny_gpt/last.pt
```

Each run writes `outputs/logs/<run>/{config.json, metrics.jsonl, samples.txt, summary.json}` (step, epoch, learning
rate, train/val loss, throughput, a sample generation at every evaluation) and `checkpoints/<run>/{best,last}.pt`.

Final run ([`configs/tiny_gpt.yaml`](configs/tiny_gpt.yaml), log in [`outputs/logs/tiny_gpt/`](outputs/logs/tiny_gpt/)):

| | |
|---|---|
| parameters | 4,772,352 (4.77 M; 99.3 % in the 6 Transformer blocks) |
| optimizer | AdamW, peak LR 2e-3 (from a 6-point sweep, [`outputs/tuning/results.md`](outputs/tuning/results.md)), β 0.9/0.95, weight decay 0.1, 200 warm-up steps, cosine to 2e-4, grad-clip 1.0 |
| batch | 64 windows × 256 characters = 16,384 tokens per step |
| steps | 3,000 (49 M tokens ≈ 49 passes over the 1 M-character training split) |
| regularisation | dropout 0.2; best-validation checkpointing |
| hardware | NVIDIA RTX 3050 Laptop GPU (4 GB), bf16 autocast, PyTorch 2.11, Python 3.12, Windows 11 |
| time | 531 s of training (643 s wall-clock including evaluations), 92.6 k tokens/s, 1.8 GB peak GPU memory |
| loss | final train 1.065 · final val 1.355 · **best val 1.353 at step 2750** (the released checkpoint) |

Why 3,000 steps: an earlier 5,000-step run ([`outputs/logs/tiny_gpt_5000steps/`](outputs/logs/tiny_gpt_5000steps/),
[`fig6b_overfitting_5000steps.png`](outputs/figures/fig6b_overfitting_5000steps.png)) reached its best validation
loss (1.358) at step 3,000 and then overfit (train 0.917, val 1.391 at step 5,000), so the schedule was shortened so
that the cosine decay ends where validation loss bottoms out.

## 6. Inference

**Released checkpoint** (GitHub Release asset, 18 MB):

```bash
python scripts/download_checkpoint.py                 # -> checkpoints/minigpt_shakespeare.pt
```

**CLI**

```bash
python -m src.generate --prompt "ROMEO:" --max-new-tokens 200 --temperature 0.8 --seed 42
python -m src.generate --prompt "To be, or not to be" --top-k 10 --num-samples 3
python -m src.generate --prompt "JULIET:" --temperature 0          # greedy
```

A fixed `--seed` reproduces the same text on the same device; CPU and GPU use different random-number
generators, so the sampled text differs between them (greedy decoding, `--temperature 0`, normally matches).

**REST API**

```bash
uvicorn src.api:app --port 8000          # then open http://127.0.0.1:8000 (web page) or /docs (Swagger)
curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" \
     -d '{"prompt": "ROMEO:", "max_new_tokens": 200, "temperature": 0.8, "seed": 1}'
```

| endpoint | purpose |
|---|---|
| `GET /` | minimal web page to try the model |
| `GET /health` | liveness |
| `GET /model-info` | parameter count, config, training metrics |
| `POST /generate` | `{prompt, max_new_tokens ≤ 1000, temperature 0–2, top_k?, top_p?, seed?}` → `{text, completion, ignored_chars, seconds, …}` |

**API keys** — set `MINIGPT_API_KEYS=key1,key2` and `POST /generate` requires an `X-API-Key: key1` header
(401 otherwise); `/`, `/health`, `/model-info` stay open. Without the variable the API is open (local use).

**Docker / SnapDeploy / Render** — `docker build -t minigpt . && docker run -p 8000:8000 minigpt`. The server runs
inference in pure NumPy ([`src/numpy_gpt.py`](src/numpy_gpt.py), tested to reproduce the PyTorch logits and greedy
output), so the image has no PyTorch; it downloads `minigpt_shakespeare.npz` (made by
`python scripts/export_numpy.py`) from the GitHub Release at build time. Any Docker host works; free tiers sleep after
inactivity, so the first request can take ~30–60 s.
SnapDeploy rejects repositories that depend on PyTorch, so deploy it from the `deploy` branch, which holds only
the serving files: refresh it with `bash scripts/publish_deploy_branch.sh`.

## 7. Reproducing the evidence

| Figure / result | Command | Output |
|---|---|---|
| 0 RNN vs attention (speed, memory) | `python -m experiments.rnn_vs_transformer` | `outputs/figures/fig0_rnn_vs_attention.png` |
| 1 hand-calculated attention | `python -m experiments.hand_attention` | `fig1_hand_attention.{png,txt}` |
| 2 causal vs bidirectional mask | `python -m experiments.mask_visualization` | `fig2_causal_mask.{png,txt}` |
| 2b CLM vs MLM vs leaky CLM | `python -m experiments.mlm_vs_clm` | `outputs/mlm_vs_clm/`, `fig2b_mlm_vs_clm.png` |
| 3 multi-head shape trace | `python -m experiments.shape_trace` | `fig3_shape_trace.txt` |
| 4 positional encoding | `python -m experiments.positional_encoding_viz` | `fig4_positional_encoding.png` |
| 5 model summary / parameters | `python -m experiments.model_summary --checkpoint checkpoints/minigpt_shakespeare.pt` | `fig5_model_summary.txt` |
| 6 loss curves | `python -m experiments.plot_training --run outputs/logs/tiny_gpt` | `fig6_loss_curves.png` |
| 7 attention of the trained model | `python -m experiments.attention_visualization` | `fig7_attention_{heads,detail}.png`, `fig7_attention_stats.txt` |
| 8 generation samples + test loss | `python -m experiments.evaluate` | `outputs/samples/samples.md`, `outputs/eval/final_eval.json` |
| 9 ablations | `python -m experiments.ablation` | `outputs/ablations/{results.md,results.csv,fig9_ablation.png}` |
| decoding strategies | `python -m experiments.decoding_compare` | `outputs/decoding/results.md` |
| n-gram baselines | `python -m experiments.baselines` | `outputs/eval/baselines.json` |
| LR sweep | `python -m experiments.tune` | `outputs/tuning/results.md` |
| everything, in order | `bash scripts/reproduce.sh` | |
| submission acceptance tests (§16) | `python scripts/acceptance.py` | |

## 8. Results

All numbers below are produced by the scripts in section 7 and stored under [`outputs/`](outputs/).
Losses are cross-entropy in **nats per character** (perplexity = e^loss, bits/char = loss / ln 2).

### 8.1 Final model vs baselines

| model | val loss | perplexity | bits/char |
|---|---|---|---|
| uniform guess (1/65) | 4.174 | 65.0 | 6.02 |
| unigram (character frequencies) | 3.333 | 28.0 | 4.81 |
| bigram, add-one smoothing | 2.474 | 11.9 | 3.57 |
| trigram, add-one smoothing | 2.037 | 7.7 | 2.94 |
| 5-gram, add-one smoothing (sparser → worse) | 2.123 | 8.4 | 3.06 |
| 2-layer LSTM, 4.81 M, same data / batch / steps (best at step 1,250) | 1.360 | 3.90 | 1.96 |
| **MiniGPT (this repo), 4.77 M** | **1.353** | **3.87** | **1.95** |

**Next-character accuracy** (how often the model's most likely next character is the true one):
**60.0 % top-1 / 87.0 % top-5** on validation, 56.7 % / 85.8 % on test — vs 15.2 % for always predicting the most
frequent character (space), 26.9 % for a bigram and 38.5 % for a trigram table.

Test split: **1.530** overall — **1.261** on the part from an already-seen play, **1.693** on *The Tempest*
(never seen in training; see limitation 2). Fixed-prompt samples: [`outputs/samples/samples.md`](outputs/samples/samples.md)
(≈97 % of generated words are real words from the corpus, 0 % repeated 4-grams at temperature 0.8).

### 8.2 Model selection (can the main model be improved?)

The final model's limit is overfitting (train 1.065 vs val 1.353), so two standard fixes were tried with
everything else unchanged (3,000 steps, same data and schedule). Logs: `outputs/logs/{tiny_gpt,cand_drop03,cand_d384}`.

| candidate | params | dropout | peak LR | best val loss | train loss | training time |
|---|---|---|---|---|---|---|
| **final: d 256, 6 layers, 4 heads** | **4.77 M** | 0.2 | 2e-3 | **1.353** | 1.065 | 9 min |
| more regularisation | 4.77 M | 0.3 | 2e-3 | 1.363 | 1.166 | 8 min |
| bigger: d 384, 6 layers, 6 heads | 10.70 M | 0.3 | 1e-3 | 1.360 | 1.104 | 14 min |

*Observation:* all three land within 0.01 nats; dropout 0.3 closes the train/val gap (0.29 → 0.20) but
learns more slowly, and 2.2× more parameters do not help. *Interpretation:* with ~1 M training characters the
model is data-limited — extra capacity is spent memorising, extra regularisation slows learning by about the same
amount — so the smaller, faster model was kept.

### 8.3 Ablations (Figure 9)

Small base model (d_model 128, 4 layers, 4 heads, context 128, ~0.8 M parameters), 3,000 steps, 8,192 tokens/step,
**one factor changed at a time**, 3 seeds each (mean ± std). Step time / tokens/s from the controlled benchmark
[`experiments/throughput.py`](experiments/throughput.py). Full table: [`outputs/ablations/results.md`](outputs/ablations/results.md).

| experiment | setting | params | best val loss | step (ms) | peak mem (MB) |
|---|---|---|---|---|---|
| depth | 2 / **4** / 6 layers | 0.41 / 0.81 / 1.21 M | 1.630 ± 0.005 / 1.533 ± 0.001 / **1.495 ± 0.006** | 27 / 45 / 71 | 185 / 331 / 476 |
| heads | 2 / **4** / 8 heads | 0.81 M | 1.540 ± 0.003 / **1.533 ± 0.001** / 1.548 ± 0.008 | 45 / 45 / 40 | 275 / 331 / 472 |
| context | 64 / **128** / 256 | 0.81 M | **1.529 ± 0.004** / 1.533 ± 0.001 / 1.573 ± 0.007 | 50 / 45 / 45 | 275 / 331 / 472 |
| position | **sinusoidal** / learned / none | 0.81 / 0.83 / 0.81 M | **1.533 ± 0.001** / 1.551 ± 0.005 / 1.720 ± 0.006 | 45 / 49 / 45 | 331 |

Observations → interpretations (kept separate on purpose):

- **Depth.** Validation loss fell 1.630 → 1.533 → 1.495 as layers went 2 → 4 → 6, while step time grew ~linearly
  (27 → 71 ms). *Interpretation:* more layers compose more attention "hops" (layer-0 heads attend to the previous
  character, deep heads look 30–50 characters back, Fig. 7), but returns diminish: the 2nd pair of layers gained
  0.097, the 3rd pair 0.038.
- **Heads.** With d_model fixed at 128 the three settings differ by < 0.015 (2–3 std). 8 heads were slightly
  worse. *Interpretation:* splitting a 128-dim model into 8 heads leaves only 16 dimensions per head for the
  query–key dot product; more, narrower heads trade per-head capacity for diversity, and at this size 4 × 32
  is the sweet spot. Same FLOPs, so similar speed; memory grows with heads because each head stores a T × T map.
- **Context.** At equal tokens per step, 64 and 128 are tied and 256 is worse (1.573). Peak memory rises with
  context (275 → 331 → 472 MB) because the attention maps are B·H·T² = (tokens/step)·H·T. *Interpretation:* this
  small model, trained for 3,000 steps, does not yet exploit dependencies beyond ~64 characters (most of the
  signal in character LM is local), while longer windows mean fewer independent windows per batch (32 vs 128)
  and noisier gradients. Longer context needs a larger model / longer training to pay off (the main 6-layer
  model does use context 256).
- **Position.** Removing positional information costs **+0.19** nats — the largest effect of any ablation — and
  the valid-word rate drops from 0.91 to 0.78. *Interpretation:* without positions attention is order-blind
  (a bag of previous characters); the causal mask alone still gives some implicit order (a token can count how
  many predecessors it has), which is why it still learns something. Learned positions were slightly worse
  (+0.018) — 16 K extra parameters that must be learned from 1 MB of text, while sinusoidal positions come for free.

### 8.4 Concept experiments

- **Why Transformers ([`outputs/rnn_vs_transformer`](outputs/rnn_vs_transformer), Fig. 0):** a Python-loop RNN
  needs N sequential steps, so its time grows with N (≈0.1 s at N = 64 → ≈6 s at N = 2048; absolute values vary run to run because the loop is CPU-bound); our attention layer
  has constant sequential depth but O(N²) memory (5 MB → 4.1 GB), which exceeded the 4 GB GPU at N = 2048.
  cuDNN's fused GRU kernel is fastest at this tiny width — it hides the loop in one kernel, but stays sequential
  inside. End to end ([`outputs/lstm_baseline`](outputs/lstm_baseline/results.md),
  Fig. 0b), a 2-layer cuDNN LSTM with the same parameter count, data, batch and schedule reached **1.360** at step
  1,250 (122 s) and then overfit badly (1.466 at step 3,000), while MiniGPT reached **1.353** at step 2,750 and stayed
  stable. *Interpretation:* at this tiny scale (256 steps, 4.8 M parameters, one laptop GPU) recurrence is not the
  bottleneck — cuDNN fuses the loop and our un-fused, Python-dispatched attention runs at about half the tokens/s.
  The Transformer still gave the best and most stable validation loss; its parallel-depth advantage grows with
  sequence length, width and hardware parallelism, which this experiment does not reach.
- **GPT vs BERT ([`outputs/mlm_vs_clm`](outputs/mlm_vs_clm/results.md)):** same block, same data, 1,500 steps.
  A model trained on next-token prediction **without** the causal mask reaches a "loss" of **0.019** — it copies
  the answer from position t + 1 — yet its honest next-token loss (no future in the input) is **2.32**, worse
  than the causal model's **1.69**, and it generates `ROMEO:::::::…`. The BERT-style MLM model uses both sides
  but cannot generate left-to-right (gibberish sample). Filling a hidden middle character, MLM (1.780) did not
  beat the causal model (1.762) at equal steps: MLM only receives a training signal on 15 % of positions.
- **Decoding ([`outputs/decoding/results.md`](outputs/decoding/results.md)):** greedy decoding repeats itself
  (35 % repeated 4-grams, distinct-2 0.49); temperature 0.5–0.8 gives fluent, non-repeating text (valid words
  0.98–1.00); temperature 1.3 becomes diverse but misspelled (valid words 0.90). Top-k / top-p at 0.8–1.0 sit in between.

### 8.5 Engineering notes

- **Tuning:** peak LR swept over {3e-4, 6e-4, 1e-3, 2e-3, 3e-3, 4e-3} for 1,500 steps → 1.889, 1.624, 1.480, 1.390,
  1.373, 1.368. Gains flatten beyond 2e-3, so 2e-3 was used for the 2× longer final run (margin against instability).
- **Speed:** these small models are CPU-bound (Python launches thousands of tiny GPU kernels; GPU utilisation
  ≈ 25–30 %). Running 4 ablations in parallel brought GPU utilisation to ~100 % but total throughput rose only
  ~1.25× (after capping CPU threads per process with `OMP_NUM_THREADS`), so timings in the table come from a
  separate controlled benchmark, not from the parallel runs.


## 9. Known limitations

1. **Overfitting on a 1 MB corpus.** Beyond ~3,000 steps the 4.8 M-parameter model memorises the training text
   (5,000-step run: train 0.92 vs val 1.39). Dropout 0.2 and early stopping limit this; more data would fix it.
2. **Domain shift on the test split.** The last 62 % of the test split (from line 899, `Master:`) is *The Tempest*,
   a play that never appears in train or validation (Prospero, Miranda, Ariel, Caliban …). Test loss there is
   **1.693** vs **1.261** on the test lines from the already-seen play, so the overall test loss (1.530) is higher
   than validation (1.353, which is entirely *The Taming of the Shrew*, whose beginning is in the training split).
3. **Locally fluent, globally incoherent.** ~97 % of generated words are real Shakespeare words and the format
   (speaker tags, verse lines) is right, but sentences carry no plot or consistent meaning; the model is a
   style imitator, not a source of facts.
4. **Greedy decoding loops** (35 % of 4-grams repeated vs 0 % with temperature 0.8 sampling,
   [`outputs/decoding/results.md`](outputs/decoding/results.md)) — use temperature ≥ 0.5.
5. **Character vocabulary.** Only the 65 characters of the corpus are known: digits other than `3`, quotes,
   brackets, accents, emoji … are dropped from prompts (reported as `ignored_chars`). Modern words are spelled
   letter by letter from Shakespearean statistics.
6. **Fixed 256-character context.** Longer generations use a sliding window, so the model forgets anything
   more than 256 characters back. Absolute sinusoidal/learned positions are not trained beyond 256.
7. **No KV cache.** Each generated character re-runs the whole window (~25–40 characters/s on a laptop CPU).
   Fine for a demo; a KV cache would make generation O(T) per token instead of O(T²).
8. **Free hosting cold starts.** The free SnapDeploy container sleeps when idle; the first request wakes it in ~60–90 s.

## 10. References

- Vaswani et al., *Attention Is All You Need*, 2017 — scaled dot-product attention, multi-head attention, sinusoidal positional encoding.
- Radford et al., *Improving Language Understanding by Generative Pre-Training* (GPT-1), 2018; *Language Models are Unsupervised Multitask Learners* (GPT-2), 2019 — decoder-only causal LM, pre-norm placement and initialisation scheme.
- Devlin et al., *BERT*, 2018 — masked language modelling and the 80/10/10 masking rule used in `experiments/mlm_vs_clm.py`.
- Xiong et al., *On Layer Normalization in the Transformer Architecture*, 2020 — pre-LN vs post-LN.
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW), 2019.
- Karpathy, [char-rnn](https://github.com/karpathy/char-rnn) — the Tiny Shakespeare corpus; [nanoGPT](https://github.com/karpathy/nanoGPT) was consulted as a reference for common GPT training practice (weight-decay groups, LR schedule). No code was copied.
- Course curriculum: [ai-engineering-from-scratch, phase 07 — Transformers deep dive](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive).
