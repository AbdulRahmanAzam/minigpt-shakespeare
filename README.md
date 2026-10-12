# MiniGPT — a GPT-style language model built from scratch

A small decoder-only Transformer (GPT) **implemented from first principles in PyTorch** — attention,
multi-head split/merge, sinusoidal positions, pre-norm blocks, causal masking, training loop, sampling —
trained on **Tiny Shakespeare**, evaluated with controlled ablations, and served through a **CLI** and a
**FastAPI** REST API. No pretrained weights and no `transformers` / `nn.Transformer` / `nn.MultiheadAttention`
classes are used anywhere.

| | |
|---|---|
| Model | __PARAMS__ parameters · 6 layers · 4 heads · d_model 256 · context 256 · character vocabulary 65 |
| Data | Tiny Shakespeare (1.1 M characters), 90/5/5 split identical to the Hugging Face dataset |
| Result | validation loss **__VAL__** nats/char (perplexity __PPL__), test loss **__TEST__** · bigram baseline 2.474 |
| Training | __TIME__ on one RTX 3050 Laptop GPU (4 GB), bf16 |
| Live demo | __DEMO_URL__ |
| Colab | [`notebooks/MiniGPT_Colab.ipynb`](notebooks/MiniGPT_Colab.ipynb) — [open in Colab](https://colab.research.google.com/github/AbdulRahmanAzam/minigpt-shakespeare/blob/main/notebooks/MiniGPT_Colab.ipynb) |

```text
$ python -m src.generate --prompt "ROMEO:" --max-new-tokens 200 --seed 42
__SAMPLE__
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
python -m pytest                                          # 60+ tests, ~15 s on CPU
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
python -m src.train --config configs/tiny_gpt.yaml    # main model (~__TIME__ on an RTX 3050; any CUDA GPU or CPU works)
python -m src.train --config configs/tiny_gpt.yaml --set train.max_steps=1000 --set model.n_layers=4   # overrides
python -m src.train --config configs/tiny_gpt.yaml --resume checkpoints/tiny_gpt/last.pt
```

Each run writes `outputs/logs/<run>/{config.json, metrics.jsonl, samples.txt, summary.json}` (step, epoch, learning
rate, train/val loss, throughput, a sample generation at every evaluation) and `checkpoints/<run>/{best,last}.pt`.

__TRAINING_TABLE__

## 6. Inference

**Released checkpoint** (GitHub Release asset, __CKPT_SIZE__):

```bash
python scripts/download_checkpoint.py                 # -> checkpoints/minigpt_shakespeare.pt
```

**CLI**

```bash
python -m src.generate --prompt "ROMEO:" --max-new-tokens 200 --temperature 0.8 --seed 42
python -m src.generate --prompt "To be, or not to be" --top-k 10 --num-samples 3
python -m src.generate --prompt "JULIET:" --temperature 0          # greedy
```

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

**Docker / Render** — `docker build -t minigpt . && docker run -p 8000:8000 minigpt`. The image installs CPU-only
PyTorch and downloads the released checkpoint at build time. On Render: *New → Blueprint* → select this repo
(`render.yaml`), or *New → Web Service → Docker*. The free tier sleeps after inactivity, so the first request can
take ~30–60 s.

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

__RESULTS__

## 9. Known limitations

__LIMITATIONS__

## 10. References

- Vaswani et al., *Attention Is All You Need*, 2017 — scaled dot-product attention, multi-head attention, sinusoidal positional encoding.
- Radford et al., *Improving Language Understanding by Generative Pre-Training* (GPT-1), 2018; *Language Models are Unsupervised Multitask Learners* (GPT-2), 2019 — decoder-only causal LM, pre-norm placement and initialisation scheme.
- Devlin et al., *BERT*, 2018 — masked language modelling and the 80/10/10 masking rule used in `experiments/mlm_vs_clm.py`.
- Xiong et al., *On Layer Normalization in the Transformer Architecture*, 2020 — pre-LN vs post-LN.
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW), 2019.
- Karpathy, [char-rnn](https://github.com/karpathy/char-rnn) — the Tiny Shakespeare corpus; [nanoGPT](https://github.com/karpathy/nanoGPT) was consulted as a reference for common GPT training practice (weight-decay groups, LR schedule). No code was copied.
- Course curriculum: [ai-engineering-from-scratch, phase 07 — Transformers deep dive](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive).
- AI-assisted development: code was written with the help of an AI coding assistant (allowed by the course); every component is documented and tested, and the author is responsible for it.
