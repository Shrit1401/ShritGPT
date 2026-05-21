"""ShritGPT — chat wrapper around the char-level model."""

from __future__ import annotations

import re
from pathlib import Path

import torch

from gpt.model import load_checkpoint
from gpt.paths import CHECKPOINT

VOICE_PREFIX = (
    "yoyo if you're new here hi i'm shrit\n\n"
    "i'm 18, in bangalore, building random stuff"
)

DEFAULT_TEMPERATURE = 0.5
DEFAULT_TOP_K = 25
DEFAULT_MAX_TOKENS = 180

# training corpus has repeated "(pass N — same facts new mood)" footers; cut them off
STOP_SEQUENCES = (
    "\nthem:",
    "\nthem\n",
    "\nme:",
    "\n\n##",
    "\n\n---",
    "\n(pass",
    "(pass ",
    "\n\nok random life dump",
    "\n\n## dump",
)

_ROLE_TAIL = re.compile(r"(\n|\s)+(them|me):?\s*$", re.IGNORECASE)


def clean_reply(text: str) -> str:
    reply = text
    for stop in STOP_SEQUENCES:
        if stop in reply:
            reply = reply.split(stop)[0]
    reply = re.sub(r"\n+#\s*$", "", reply)
    reply = re.sub(r"\(pass\s+\d+.*$", "", reply, flags=re.DOTALL)
    reply = _ROLE_TAIL.sub("", reply)
    reply = re.sub(r"\s+", " ", reply)
    return reply.strip()


class ShritGPT:
    def __init__(self, checkpoint: Path = CHECKPOINT):
        self.model, self.encode, self.decode, self.cfg, self.device, stoi = (
            load_checkpoint(checkpoint)
        )
        self.allowed_chars = set(stoi.keys())
        self.block_size = self.cfg.block_size

    def filter_input(self, s: str) -> str:
        return "".join(c for c in s if c in self.allowed_chars)

    def build_prompt(self, user_message: str, history: list[dict]) -> str:
        """Format like training data: them:/me: dm style."""
        parts = [VOICE_PREFIX, "\n\n"]
        for turn in history[-8:]:
            role = turn.get("role", "user")
            content = self.filter_input(turn.get("content", ""))
            if not content:
                continue
            if role == "user":
                parts.append(f"them: {content}\n")
            else:
                parts.append(f"me: {content}\n")
        msg = self.filter_input(user_message)
        if not msg.strip():
            msg = "hey"
        parts.append(f"them: {msg}\nme: ")
        prompt = "".join(parts)
        if len(prompt) > self.block_size:
            prompt = prompt[-self.block_size :]
        return prompt

    @torch.no_grad()
    def reply(
        self,
        user_message: str,
        history: list[dict] | None = None,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
    ) -> str:
        history = history or []
        prompt = self.build_prompt(user_message, history)
        tokens = self.encode(prompt)
        if not tokens:
            return "idk what to say to that lol try normal letters"

        ctx = torch.tensor([tokens], dtype=torch.long, device=self.device)
        out_tokens = self.model.generate(
            ctx,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )[0].tolist()
        full = self.decode(out_tokens)
        marker = "me: "
        idx = full.rfind(marker)
        reply = full[idx + len(marker) :] if idx >= 0 else full[len(prompt) :]

        reply = clean_reply(reply)
        return reply or "hmm give me a sec lol"

    @torch.no_grad()
    def reply_stream(
        self,
        user_message: str,
        history: list[dict] | None = None,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
    ):
        """Yield reply text incrementally (character by character)."""
        history = history or []
        prompt = self.build_prompt(user_message, history)
        tokens = self.encode(prompt)
        if not tokens:
            yield "idk what to say to that lol try normal letters"
            return

        idx = torch.tensor([tokens], dtype=torch.long, device=self.device)
        marker = "me: "
        seen = 0

        for token_id in self.model.generate_stream(
            idx,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        ):
            idx = torch.cat(
                (
                    idx,
                    torch.tensor([[token_id]], dtype=torch.long, device=self.device),
                ),
                dim=1,
            )
            full = self.decode(idx[0].tolist())
            midx = full.rfind(marker)
            if midx < 0:
                continue
            reply = full[midx + len(marker) :]
            if any(stop in reply for stop in STOP_SEQUENCES):
                cleaned = clean_reply(reply)
                if len(cleaned) > seen:
                    yield cleaned[seen:]
                return
            cleaned = clean_reply(reply)
            if len(cleaned) > seen:
                yield cleaned[seen:]
                seen = len(cleaned)

        if seen == 0:
            yield "hmm give me a sec lol"


_bot: ShritGPT | None = None


def get_bot() -> ShritGPT:
    global _bot
    if _bot is None:
        if not CHECKPOINT.exists():
            raise FileNotFoundError(
                f"Missing {CHECKPOINT} — run: python train.py"
            )
        _bot = ShritGPT()
    return _bot
