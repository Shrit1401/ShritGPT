#!/usr/bin/env python3
"""Shuffle Substack posts + Shakespeare into a mixed corpus file."""

import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpt.paths import DATA_DIR

INPUT_PATH = DATA_DIR / "input.txt"
OUTPUT_PATH = DATA_DIR / "input_shuffled.txt"
SEPARATOR = "\n\n" + "=" * 80 + "\n\n"
SEED = 42


def split_documents(text: str) -> list[str]:
    chunks = re.split(r"\n={80}\n", text)
    return [c.strip() for c in chunks if c.strip()]


def shuffle_corpus(text: str, seed: int = SEED) -> str:
    docs = split_documents(text)
    rng = random.Random(seed)
    rng.shuffle(docs)
    return SEPARATOR.join(docs) + "\n"


def main() -> None:
    if not INPUT_PATH.exists():
        raise SystemExit(f"Missing {INPUT_PATH}")

    raw = INPUT_PATH.read_text(encoding="utf-8")
    docs = split_documents(raw)
    mixed = shuffle_corpus(raw)

    OUTPUT_PATH.write_text(mixed, encoding="utf-8")
    print(f"Shuffled {len(docs)} chunks -> {OUTPUT_PATH} ({len(mixed):,} chars)")


if __name__ == "__main__":
    main()
