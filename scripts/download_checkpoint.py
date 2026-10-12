"""Download the released MiniGPT checkpoint (GitHub Release asset) and verify its SHA-256.

    python scripts/download_checkpoint.py
    python scripts/download_checkpoint.py --url <other-url> --out checkpoints/minigpt_shakespeare.pt
"""

from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "AbdulRahmanAzam/minigpt-shakespeare"
TAG = "v1.0"
FILENAME = "minigpt_shakespeare.pt"
URL = os.environ.get("MINIGPT_CHECKPOINT_URL", f"https://github.com/{REPO}/releases/download/{TAG}/{FILENAME}")
SHA256 = "18cb70df0421b7f9b982e984c27e75a9687187c932dbe0158a443bb64e746291"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=URL)
    p.add_argument("--out", default=str(ROOT / "checkpoints" / FILENAME))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"{out} already exists (use --force to re-download)")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {args.url}")
        tmp = out.with_suffix(".part")
        urllib.request.urlretrieve(args.url, tmp)
        tmp.replace(out)
    digest = sha256_of(out)
    if args.url == URL and SHA256 != "TO_BE_FILLED_AFTER_TRAINING" and digest != SHA256:
        raise SystemExit(f"SHA-256 mismatch: got {digest}, expected {SHA256}")
    print(f"ok: {out} ({out.stat().st_size / 2**20:.1f} MB, sha256 {digest})")


if __name__ == "__main__":
    main()
