# MiniGPT — Original Implementation Plan (Phase 0: research + design)

> Written before implementation. Final values differ where experiments said so: LR 2e-3 (sweep), dropout 0.2,
> 3000 steps (the 5000-step run overfit after ~3000), 59 tests. See README for the final numbers.

Source of truth: `A1_AIE.pdf` (11 pages). This plan maps every graded requirement to a concrete file,
command, and pass condition. Report / Medium article / demo video / presentation are **out of scope for now**;
we only build the code + the artifacts (figures, logs, tables) those later documents will need.

---

## 1. What the assignment actually asks (decoded)

Build a GPT-style **decoder-only causal LM from scratch in PyTorch**, train it on **Tiny Shakespeare**,
prove each component works (tests + figures), run **≥3 controlled ablations**, expose it via **CLI + FastAPI**,
**deploy** it (Render or similar), and ship a **Colab notebook**. "Definition of done": a stranger clones the
repo, installs, runs tests, reproduces architecture checks, and generates text with **no undocumented steps**.

Hard rules (Section 8):
- No pretrained checkpoints. No `transformers` model classes. No `nn.Transformer*`, no `nn.MultiheadAttention`.
- Allowed: tensors, autograd, `nn.Linear`, `nn.LayerNorm`, `nn.Embedding`, softmax, AdamW, DataLoader, GPU.
- Everything reproducible from documented commands. No secrets / big artifacts in git.

Grading weights → where effort goes:

| Category | % | Covered by |
|---|---|---|
| Conceptual understanding | 15 | evidence scripts + docstrings (report later) |
| Attention implementation | 15 | `src/attention.py` + `tests/test_attention.py`, `test_mask.py` |
| Transformer architecture | 15 | `src/positional_encoding.py`, `src/transformer_block.py`, `src/gpt.py`, `test_shapes.py` |
| LM pipeline | 15 | `src/tokenizer.py`, `src/dataset.py`, `src/train.py`, checkpointing, `test_overfit.py` |
| Experiments & analysis | 15 | `experiments/ablation.py` (depth, heads, context, +position, +decoding) |
| Generation quality & eval | 10 | `experiments/evaluate.py` (val/test loss, perplexity, bits-per-char, fixed prompts) |
| Software engineering | 10 | configs, tests, README, CLI, API, Docker/Render, Colab |
| Report & demo | 5 | later |

---

## 2. Research findings that change the implementation

1. **The HF dataset cannot be loaded with modern `datasets`.** `karpathy/tiny_shakespeare` is a *loading-script*
   dataset (`tiny_shakespeare.py`, no parquet branch). `datasets>=4` dropped script support, so
   `load_dataset("karpathy/tiny_shakespeare")` fails. The script itself just downloads
   `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt` and splits
   **90 / 5 / 5 by character** (`int(len*0.9)`, then half of the rest).
   → `scripts/prepare_data.py` downloads that exact file, verifies SHA-256, and reproduces the **identical split**.
   We cite the HF card as the source and explain the equivalence. No `datasets` dependency.
2. **Corpus facts (measured):** 1,115,394 chars, 40,000 lines, 65 unique chars,
   SHA-256 `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
   Split: train 1,003,854 / val 55,770 / test 55,770. Val+test contain **no chars missing from train**.
   ~263k word-ish tokens, 13.3k unique words → too little data for a large subword vocab (sparse, overfits).
3. **Hardware (measured on this laptop):** RTX 3050 Laptop 4 GB, bf16 supported, torch 2.11+cu128, Python 3.12.
   Prototype timings (bf16 autocast):
   - d=128, L=4, H=4, T=128, B=64 → 0.81M params, ~25–35 ms/step, ~300 MB  ⇒ ablation base, ~2–3 min / 3k steps.
   - d=256, L=6, H=4, T=256, B=64 → 4.77M params, ~136 ms/step, ~1.6 GB ⇒ main model, ~12 min / 5k steps.
   - d=384, L=6, H=6, T=256, B=64 → 10.7M params, ~230 ms/step, ~2.4 GB ⇒ optional "bigger" run.
   Laptop timings are noisy (boost clocks) → timing measurements need warm-up + averaging.
4. **Colab default GPU is T4: no fast bf16.** Training code must auto-select precision:
   bf16 if supported → else fp16 + `GradScaler` → CPU fp32.
5. **Reference curriculum** (rohitg00/ai-engineering-from-scratch, phase 07) conventions we follow:
   additive `-inf` upper-triangular mask, shift-by-one targets, MLM 15 % with 80/10/10 rule and `-100` ignore labels,
   "serial depth" framing for RNN vs Transformer. We cite it.

---

## 3. Design decisions (each must be defendable in the viva)

| Decision | Choice | Why |
|---|---|---|
| Tokenizer | **Own character-level tokenizer**, vocab = 65 (sorted by codepoint, built from *train* split only, saved as JSON next to the checkpoint) | Deterministic, zero OOV on val/test, fully explainable, matches 1 MB corpus. BPE would need a library or extra code and gives sparse stats on 263k words. |
| Special tokens | **None** in the vocab. Corpus is one continuous stream, so no BOS/EOS/PAD needed (fixed-length windows, no padding). Unknown prompt chars → `encode(strict=False)` drops them and reports them (`ignored_chars` in CLI/API output); `strict=True` raises. | Avoids an untrained `<unk>` embedding polluting generation; policy is explicit & documented. |
| Architecture | **Pre-LN decoder-only** (GPT-2 style): `x + Attn(LN(x))`, `x + FFN(LN(x))`, final LN, LM head | Pre-LN trains stably without long warm-up; we can explain vs Post-LN of original paper. |
| Embedding scale | token embedding × √d_model, then + sinusoidal PE (Vaswani §3.4) | Keeps token signal comparable to PE amplitude (±1). |
| Positional encoding | **Sinusoidal (required)**, `learned` and `none` selectable via config | `none` / `learned` power the position ablation (extra credit). |
| Attention | Own `scaled_dot_product_attention(q,k,v,mask)` → Q/K/V projections → **explicit `split_heads` / `merge_heads`** functions; returns weights for visualization | Requirement: split/merge explicit in code; weights needed for Fig 7. |
| Fused attention | Optional `attn_impl: manual | sdpa` flag; tests assert manual == `F.scaled_dot_product_attention` output to 1e-5 | Proves correctness against PyTorch reference; `manual` is the default & graded path. |
| FFN | Linear(d, 4d) → GELU → Linear(4d, d) → dropout | Spec recommendation. |
| Mask | Boolean upper-triangular (`torch.triu(..., diagonal=1)`) registered as buffer, applied via `masked_fill(-inf)` before softmax | Spec: "upper-triangular future-token mask". |
| LM head | Linear(d, vocab), **weight tying optional** (`tie_weights` flag, default on? decided by quick check) | Spec lists tying as optional experiment. |
| Init | N(0, 0.02); residual output projections scaled by 1/√(2·n_layers) (GPT-2) | Stable deep residual stream; explainable. |
| Loss | Token-level cross-entropy over logits `(B·T, V)` vs targets shifted by one | Required. Also report perplexity = e^loss and bits/char = loss/ln2. |
| Optimizer | AdamW (β=0.9/0.95), weight decay 0.1 on 2-D matrices only (not biases / LayerNorm / embeddings), grad-clip 1.0 | Standard GPT practice; justify in report. |
| LR schedule | Linear warm-up (≈100–200 steps) → cosine decay to 10 % of peak. Peak chosen by small sweep {3e-4, 6e-4, 1e-3} | Spec: "start around 3e-4 and tune" — sweep is the evidence of tuning. |
| Data pipeline | `CharWindowDataset`: contiguous windows `x = ids[i:i+T]`, `y = ids[i+1:i+T+1]`, stride configurable (default 1 for train, T for eval); PyTorch `DataLoader(shuffle=True)` with seeded generator | Matches "contiguous token sequences of fixed context length"; epochs are well-defined. |
| Evaluation | Periodic estimate on fixed val batches; **final full deterministic pass** over val and test (stride = T) | Reproducible numbers; test used once at the end only. |
| Generation | Autoregressive loop, context cropped to last `context_length` tokens; `temperature` (0 ⇒ greedy), `top_k`, `top_p`, `seed`, `max_new_tokens` | Required + optional extras. |
| Config | YAML files in `configs/` → dataclasses; CLI overrides `--set model.n_layers=2` | "Must be configurable" table in §11. |
| Reproducibility | Global seeding (python/numpy/torch/cuda), config + git hash + env saved with each run | Definition of done. |
| Precision | bf16 autocast if available → fp16+GradScaler → fp32 on CPU | Laptop (bf16) and Colab T4 (fp16) both work. |

### Main model config (to confirm with LR/dropout sweep)

```
vocab_size 65 | context_length 256 | d_model 256 | n_heads 4 (head_dim 64) | n_layers 6
ffn_dim 1024 | dropout 0.1–0.2 | batch 64 | AdamW lr ~1e-3 peak | ~5000 steps
≈ 4.8M parameters, ≈ 12 min on RTX 3050
```
Deviation from §11 starting values (ctx 128 → 256, layers 4 → 6) justified by measured hardware headroom;
dropout above 0.1 only if the val curve shows overfitting (evidence-based). Expected val loss ≈ 1.5 nats/char
(context: uniform = ln 65 = 4.17, unigram ≈ 3.3, bigram ≈ 2.5 — we compute these baselines ourselves).

### Ablation base config (small & fast, for controlled experiments)

```
d_model 128 | n_heads 4 | n_layers 4 | context 128 | dropout 0.1 | 3000 steps | tokens/step fixed = 8192
```

---

## 4. Repository layout (follows §10, plus packaging)

```
minigpt-shakespeare/
├── README.md                     setup, dataset, train, inference, expected output, limitations
├── requirements.txt              torch, numpy, pyyaml, matplotlib, fastapi, uvicorn, pydantic, pytest, httpx
├── requirements-deploy.txt       CPU-only torch + API deps (small Render image)
├── pyproject.toml                pytest config, package metadata
├── .gitignore / .env.example
├── configs/
│   ├── tiny_gpt.yaml             main model
│   ├── ablation_base.yaml        ablation baseline
│   └── overfit.yaml              tiny-batch debugging gate
├── data/README.md, .gitkeep      (raw text downloaded, not committed)
├── scripts/
│   ├── prepare_data.py           download + sha256 + 90/5/5 split + vocab
│   └── download_checkpoint.py    fetch released weights (+ sha256)
├── src/
│   ├── config.py                 dataclasses + YAML + --set overrides
│   ├── tokenizer.py              CharTokenizer (build/encode/decode/save/load)
│   ├── dataset.py                CharWindowDataset, loaders, split loading
│   ├── attention.py              scaled_dot_product_attention, causal_mask, split/merge heads, MultiHeadAttention
│   ├── positional_encoding.py    Sinusoidal / Learned / None
│   ├── transformer_block.py      FeedForward, TransformerBlock (pre-LN, residuals)
│   ├── gpt.py                    MiniGPT (embed → PE → blocks → LN → head), generate(), param count/summary
│   ├── train.py                  training loop, logging, eval, sampling, checkpoints, resume
│   ├── generate.py               CLI inference (prints prompt, output, params, ctx, temp)
│   ├── api.py                    FastAPI: GET /health, GET /model-info, POST /generate, GET / (tiny HTML page)
│   └── utils.py                  seeding, device/precision pick, timers, logging, checkpoint io
├── tests/
│   ├── test_attention.py         hand-calc example equality, softmax rows sum 1, vs F.sdpa
│   ├── test_mask.py              mask shape/values, future weights == 0, perturb-future invariance (model level)
│   ├── test_shapes.py            every module's I/O shapes, param count formula, head split/merge round-trip
│   ├── test_overfit.py           tiny repeated batch: loss drops >90 % on CPU in <30 s
│   ├── test_tokenizer.py         round-trip, determinism, unknown-char policy
│   ├── test_dataset.py           target == input shifted by one, window counts
│   ├── test_generate.py          length, determinism with seed, greedy == temp 0, context cropping
│   ├── test_checkpoint.py        save → fresh load → identical logits
│   └── test_api.py               FastAPI TestClient on a tiny random model
├── experiments/
│   ├── hand_attention.py         Fig 1: prints every step of the 3-token example (+ masked)
│   ├── mask_visualization.py     Fig 2: causal vs bidirectional (BERT) mask heatmaps
│   ├── shape_trace.py            Fig 3: multi-head tensor-shape printout
│   ├── positional_encoding_viz.py Fig 4: PE heatmap, sin curves, position-similarity matrix
│   ├── model_summary.py          Fig 5: per-module parameter table
│   ├── plot_training.py          Fig 6: train vs val loss + LR curve from logs
│   ├── attention_visualization.py Fig 7: trained-model attention maps per layer/head
│   ├── rnn_vs_transformer.py     Lesson 01: measured serial-depth comparison (RNN loop vs attention), time & memory vs N
│   ├── mlm_vs_clm.py             Lesson 06: CLM vs MLM (bidirectional) vs "leaky" CLM (no mask) experiment
│   ├── baselines.py              uniform / unigram / bigram loss for context
│   ├── ablation.py               Fig 9: depth / heads / context / position runs → CSV + plots
│   ├── decoding_compare.py       greedy vs temperature vs top-k/top-p: distinct-n, repetition rate
│   └── evaluate.py               Fig 8: final val/test loss, perplexity, bpc, 5+ fixed-prompt samples
├── outputs/{figures,logs,samples,ablations}/   (small text/PNG artifacts committed; raw logs optional)
├── checkpoints/.gitkeep          (weights published as GitHub Release asset, not in git)
├── notebooks/MiniGPT_Colab.ipynb
├── Dockerfile, render.yaml       deployment
└── report/                       (later)
```

---

## 5. Implementation phases (each ends with a gate that must pass before moving on)

### Phase 1 — Scaffold & data  (M0)
1. `git init` this folder as its **own** repo (it currently sits inside the parent `PROJECTS` repo — keep separate).
2. `.gitignore` (.venv, data/*.txt, checkpoints/*.pt, outputs/logs/raw, __pycache__, .env, A1_AIE.pdf).
3. `requirements.txt`, `pyproject.toml`, configs, `src/config.py`, `src/utils.py`.
4. `scripts/prepare_data.py` → `data/{train,val,test}.txt` + `data/meta.json` (sizes, sha256, vocab).
5. `src/tokenizer.py`, `src/dataset.py` + tests.
**Gate:** split sizes 1,003,854 / 55,770 / 55,770; vocab 65; tokenizer round-trip exact; target-shift test passes.

### Phase 2 — Attention engine  (M1)
1. `experiments/hand_attention.py` with the fixed example below; numbers printed at every step.
2. `scaled_dot_product_attention(q, k, v, mask=None) -> (out, weights)` — supports any leading batch dims.
3. `causal_mask(T)`.
**Gate:** `test_attention.py` reproduces hand numbers to 1e-4; masked weights above diagonal exactly 0;
rows sum to 1; matches `F.scaled_dot_product_attention`.

Hand example (verified numerically, d_k = d_v = 2):
```
Q = [[1,0],[0,1],[1,1]]   K = [[1,0],[1,1],[0,1]]   V = [[1,0],[0,2],[1,1]]
QKᵀ        = [[1,1,0],[0,1,1],[1,2,1]]
/√2        = [[.7071,.7071,0],[0,.7071,.7071],[.7071,1.4142,.7071]]
softmax    = [[.4011,.4011,.1978],[.1978,.4011,.4011],[.2483,.5035,.2483]]
out        = [[.5989,1.0000],[.5989,1.2033],[.4965,1.2552]]
masked     = [[.7071,-inf,-inf],[0,.7071,-inf],[.7071,1.4142,.7071]]
softmax_m  = [[1,0,0],[.3302,.6698,0],[.2483,.5035,.2483]]
out_m      = [[1,0],[.3302,1.3395],[.4965,1.2552]]
```
Teaching points: token 1 can only copy its own value; token 2 re-normalises over 2 positions;
token 3 is **unchanged** (it already could see everything before it) — masking only removes the future.

### Phase 3 — Multi-head attention + positional encoding  (M2)
1. `MultiHeadAttention`: fused or separate Q/K/V `nn.Linear(d, d)`, `split_heads (B,T,d)→(B,H,T,d/H)`,
   per-head SDPA, `merge_heads (B,H,T,d/H)→(B,T,d)`, output projection, dropout; optional `trace=True`
   that logs every shape (feeds Fig 3).
2. `SinusoidalPositionalEncoding` (buffer, max_len configurable), `LearnedPositionalEmbedding`, `NoPosition`.
**Gate:** `test_shapes.py` (split∘merge = identity, d % H assertion, per-head weights shape (B,H,T,T));
PE values match the closed form at chosen (pos, i); figure script runs.

### Phase 4 — Transformer block + full GPT  (M3)
1. `FeedForward`, `TransformerBlock` (pre-LN, two residuals, dropout).
2. `MiniGPT`: embedding×√d + PE → dropout → N blocks → final LN → LM head (optional tying);
   `forward(idx, targets=None, return_attn=False) -> (logits, loss, attn)`; `generate(...)`; `num_parameters()`.
3. Causal-correctness tests at **model level**: change tokens at positions > t ⇒ logits at ≤ t unchanged
   (exact in fp32); gradient of logit[t] w.r.t. input embedding at positions > t is zero.
4. `experiments/mask_visualization.py`, `experiments/mlm_vs_clm.py`:
   - CLM (causal mask, next-token) vs MLM (no mask, 15 % masking 80/10/10, loss only on masked positions)
     vs **leaky CLM** (no mask, next-token target) — same block code, same steps.
   - Expected evidence: leaky CLM train loss collapses toward ~0 (it copies the answer from position t+1)
     but generates garbage autoregressively ⇒ shows *why* GPT needs the mask; MLM fills blanks using both sides
     but has no left-to-right generative factorisation.
**Gate:** all shape + causal tests green; analytic param count == `sum(p.numel())`.

### Phase 5 — Training pipeline + overfit gate  (M4)
1. `src/train.py`: config load, seed, data, model, AdamW param groups, warm-up+cosine, AMP, grad-clip,
   eval every N steps, sample generation every eval, JSONL + CSV logging
   (step, epoch, lr, train_loss, val_loss, tokens/s, elapsed, peak memory), best + last checkpoints, `--resume`.
2. Checkpoint = `{model_state, model_config, train_config, tokenizer vocab, step, val_loss, git hash}`;
   last also stores optimizer/scaler for resume. Inference export strips optimizer (~19 MB fp32).
3. `configs/overfit.yaml` + `tests/test_overfit.py`: 1–4 short sequences, no dropout, ~200 steps ⇒ loss < 10 % of start.
**Gate:** overfit test passes on CPU; 300-step smoke run on real data shows val loss falling; resume reproduces.

### Phase 6 — Main MiniGPT training  (M5)
1. `experiments/baselines.py` (uniform / unigram / bigram).
2. Short LR sweep (3 values × ~1500 steps) and dropout check (0.1 vs 0.2) on main config → pick & document.
3. Full run (~5000 steps). Record: param count, config, hardware, wall-clock, final train/val loss, best step.
4. `experiments/evaluate.py`: full val + test loss, perplexity, bpc; ≥5 fixed prompts
   (`"ROMEO:"`, `"JULIET:\n"`, `"First Citizen:\n"`, `"KING RICHARD III:\n"`, `"To be, or not to be"`,
   `"The future of artificial intelligence"`) at temp 0.8, fixed seed → `outputs/samples/`.
5. `plot_training.py` (Fig 6), `attention_visualization.py` (Fig 7), `model_summary.py` (Fig 5).
**Gate:** val loss well below bigram baseline (~2.5) and near ~1.5; samples show speaker tags, line breaks,
mostly real words; checkpoint reload in a fresh process gives identical greedy output.

### Phase 7 — Experiments  (M6 part 1)
Controlled: one factor changes, everything else fixed (same seed, steps, LR schedule, **tokens per step = 8192**).
| Experiment | Values | Measured |
|---|---|---|
| Depth | 2 / 4 / 6 layers | val loss, params, wall-clock, tokens/s |
| Heads | 2 / 4 / 8 (d=128 fixed) | val loss, sample quality (distinct-n, repetition) |
| Context | 64 / 128 / 256 (batch 128/64/32) | val loss, peak GPU memory, tokens/s |
| Position (optional) | sinusoidal / learned / none | val loss, samples |
| Decoding (optional) | greedy / T=0.5 / 0.8 / 1.0 / top-k / top-p | distinct-1/2/3, repetition rate, loss of generated text under model |
| Why Transformers (lesson 01) | RNN-loop vs attention, N = 64…2048 | fwd+bwd time, peak memory, serial depth |
- 3 seeds per config if time allows (≈ 75 min total) → mean ± std; else 1 seed and state it.
- Output: `outputs/ablations/results.csv`, markdown table, plots (Fig 9).
**Gate:** every run logged with config; table auto-generated from CSV (no hand-copied numbers).

### Phase 8 — Packaging & deployment  (M6 part 2)
1. CLI `python -m src.generate --checkpoint ... --prompt ... --max-new-tokens 200 --temperature 0.8 --top-k ... --seed ...`
   prints prompt, generated text, model parameters, context length, temperature, device, ignored chars (matches §12.1).
2. FastAPI `src/api.py` (pydantic-validated, max_new_tokens capped, model loaded once at startup, CPU),
   `uvicorn src.api:app`. Tests via `TestClient`.
3. Checkpoint distribution: GitHub Release asset + `scripts/download_checkpoint.py` (sha256-checked).
4. `Dockerfile` (python:3.12-slim, CPU torch) + `render.yaml`; deploy to Render free tier
   (512 MB RAM is enough: ~5M params). Note cold-start sleep in README.
**Gate:** `curl` to deployed `/generate` returns text; local Docker build works.

### Phase 9 — Colab + README + clean-room acceptance (§16)
1. `notebooks/MiniGPT_Colab.ipynb`: clone → install → tests → download data → (A) download checkpoint & generate,
   (B) optional quick training run → show figures. Must run top-to-bottom on a fresh T4 runtime.
2. README: setup, dataset source + split, train/eval/infer/API commands, expected output, figure index,
   ablation table, known limitations, citations (Vaswani 2017, Radford 2018/2019, Devlin 2018, Karpathy char-rnn,
   reference curriculum).
3. Clean-room run in a fresh clone + fresh venv:
   install → `pytest` → overfit test → load final checkpoint → generate from 3 fixed prompts →
   causal test → README commands verified → secret scan (`git grep` for keys, no `.env`).
**Gate:** all 7 acceptance tests of §16 pass from a fresh clone (also in Colab).

---

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| `datasets` can't load HF script dataset | own downloader, identical split, sha256 |
| Overfitting (1 MB corpus) | dropout, weight decay, best-val checkpoint, early-stop by val |
| Noisy laptop timings in ablations | warm-up steps excluded, CUDA sync, median of steps, same machine for all runs |
| Colab T4 lacks bf16 | auto precision fallback fp16+GradScaler |
| Render free tier RAM/cold start | CPU torch wheel, single small model, lazy warm-up, documented |
| Python 3.14 default on machine (no torch wheels) | project venv pinned to 3.12 (same as Colab) |
| Folder is inside parent git repo | own `git init`, own GitHub repo |
| Prompt has chars outside vocab | documented drop policy, reported in output |

---

## 7. Out of scope now (later, by you)
Technical report / Medium article, screenshots, demo video, class presentation.
All their raw material (figures, tables, samples, logs, numbers) is produced by the scripts above.
