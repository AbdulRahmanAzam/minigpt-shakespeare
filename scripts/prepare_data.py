"""Download Tiny Shakespeare and create the train / validation / test splits.

Dataset source: Hugging Face `karpathy/tiny_shakespeare`
(https://huggingface.co/datasets/karpathy/tiny_shakespeare).

That dataset is a *loading-script* dataset: its `tiny_shakespeare.py` downloads Karpathy's char-rnn
`input.txt` and splits it 90% / 5% / 5% by character. Recent versions of the `datasets` library
(>= 4.0) no longer execute loading scripts, so we reproduce the script exactly instead:
same file (verified by SHA-256), same split arithmetic. The resulting splits are byte-identical to the
ones the Hugging Face script produces.

Usage:
    python scripts/prepare_data.py              # download + split into data/
    python scripts/prepare_data.py --source path/to/input.txt
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tokenizer import CharTokenizer  # noqa: E402
from src.utils import write_json  # noqa: E402

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
HF_CARD = "https://huggingface.co/datasets/karpathy/tiny_shakespeare"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_text(text: str) -> tuple[str, str, str]:
    """Exactly the arithmetic used by the HF loading script (90/5/5 by character)."""
    i = int(len(text) * 0.9)
    train, rest = text[:i], text[i:]
    j = int(len(rest) * 0.5)
    return train, rest[:j], rest[j:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--source", default=None, help="use a local copy of input.txt instead of downloading")
    parser.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_path = data_dir / "input.txt"

    if args.source:
        raw_path.write_bytes(Path(args.source).read_bytes())
    elif args.force or not raw_path.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, raw_path)

    digest = sha256_of(raw_path)
    if digest != SHA256:
        raise SystemExit(f"SHA-256 mismatch for {raw_path}: got {digest}, expected {SHA256}")

    text = raw_path.read_text(encoding="utf-8")
    train, val, test = split_text(text)
    for name, part in (("train", train), ("val", val), ("test", test)):
        # newline="" keeps "\n" as-is on Windows so the files are identical on every OS
        with open(data_dir / f"{name}.txt", "w", encoding="utf-8", newline="") as f:
            f.write(part)

    tok = CharTokenizer.from_text(train)
    tok.save(data_dir / "vocab.json")
    unseen = sorted(set(val + test) - set(train))

    meta = {
        "source": HF_CARD,
        "raw_url": URL,
        "sha256": SHA256,
        "total_chars": len(text),
        "total_lines": text.count("\n"),
        "split_rule": "90/5/5 by character (identical to the HF loading script)",
        "train_chars": len(train),
        "val_chars": len(val),
        "test_chars": len(test),
        "vocab_size": tok.vocab_size,
        "vocab": "".join(tok.chars),
        "chars_in_val_test_not_in_train": unseen,
    }
    write_json(data_dir / "meta.json", meta)

    print(f"sha256 ok      : {digest}")
    print(f"total chars    : {len(text):,}  lines: {meta['total_lines']:,}")
    print(f"train/val/test : {len(train):,} / {len(val):,} / {len(test):,}")
    print(f"vocab size     : {tok.vocab_size}  -> {tok.chars!r}")
    print(f"unseen in val/test: {unseen}")
    print(f"wrote {data_dir}/train.txt, val.txt, test.txt, vocab.json, meta.json")


if __name__ == "__main__":
    main()
