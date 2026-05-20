#!/usr/bin/env python3
"""Smoke-test ShritGPT on key prompts after training."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpt.chat import ShritGPT
from gpt.paths import CHECKPOINT, TEST_RESULTS

TEST_PROMPTS = [
    "who is shrit",
    "who even are you",
    "what is shunya",
    "i am starting a finance agency with someone new",
    "what hackathons u won",
    "what is paper",
    "hey",
    "where do you live",
    "do u freelance",
]


def main():
    if not CHECKPOINT.exists():
        raise SystemExit(f"Missing {CHECKPOINT} — run: python train.py")

    bot = ShritGPT()
    lines = ["ShritGPT test run", "=" * 60, ""]

    print("Running tests...\n")
    for prompt in TEST_PROMPTS:
        reply = bot.reply(
            prompt,
            history=[],
            max_tokens=180,
            temperature=0.5,
            top_k=25,
        )
        block = f"them: {prompt}\nme: {reply}\n"
        lines.append(block)
        lines.append("-" * 60)
        print(block)
        print("-" * 60)

    TEST_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    TEST_RESULTS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {TEST_RESULTS}")


if __name__ == "__main__":
    main()
