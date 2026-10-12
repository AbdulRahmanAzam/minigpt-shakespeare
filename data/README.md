# Data

The corpus is **not** committed. Create it with:

```bash
python scripts/prepare_data.py
```

| item | value |
|---|---|
| Dataset | Tiny Shakespeare — [Hugging Face `karpathy/tiny_shakespeare`](https://huggingface.co/datasets/karpathy/tiny_shakespeare) |
| Raw file | `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt` (the file the HF loading script downloads) |
| SHA-256 | `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed` |
| Size | 1,115,394 characters, 40,000 lines |
| Split | 90 / 5 / 5 by character, same arithmetic as the HF script: train 1,003,854 · val 55,770 · test 55,770 |
| Vocabulary | 65 characters (built from the train split; val/test contain no unseen characters) |

Why not `datasets.load_dataset("karpathy/tiny_shakespeare")`? The HF dataset is a *loading-script* dataset,
and `datasets >= 4.0` no longer runs loading scripts. `prepare_data.py` reproduces the script exactly
(same file, verified by hash, same split), so the splits are identical.

Files written: `input.txt`, `train.txt`, `val.txt`, `test.txt`, `vocab.json`, `meta.json`.
