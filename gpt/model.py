"""GPT model + checkpoint loading (no training data required)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    block_size: int
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float
    vocab_size: int


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sample_next(logits, temperature=1.0, top_k=None):
    logits = logits / max(temperature, 1e-8)
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, k)
        logits = logits.masked_fill(logits < v[-1], float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def build_model(cfg: ModelConfig, device: str) -> nn.Module:
    block_size = cfg.block_size
    n_embd = cfg.n_embd
    n_head = cfg.n_head
    n_layer = cfg.n_layer
    dropout = cfg.dropout
    vocab_size = cfg.vocab_size

    class Head(nn.Module):
        def __init__(self, head_size):
            super().__init__()
            self.key = nn.Linear(n_embd, head_size, bias=False)
            self.query = nn.Linear(n_embd, head_size, bias=False)
            self.value = nn.Linear(n_embd, head_size, bias=False)
            self.register_buffer(
                "tril", torch.tril(torch.ones(block_size, block_size))
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            B, T, C = x.shape
            k = self.key(x)
            q = self.query(x)
            wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            return wei @ self.value(x)

    class MultiHeadAttention(nn.Module):
        def __init__(self, num_heads, head_size):
            super().__init__()
            self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
            self.proj = nn.Linear(head_size * num_heads, n_embd)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            return self.dropout(self.proj(out))

    class FeedForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.ReLU(),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            return self.net(x)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            head_size = n_embd // n_head
            self.sa = MultiHeadAttention(n_head, head_size)
            self.ffwd = FeedForward()
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

        def forward(self, x):
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
            return x

    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
            self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(n_embd)
            self.lm_head = nn.Linear(n_embd, vocab_size)

        def forward(self, idx, targets=None):
            T = idx.shape[1]
            x = self.token_embedding_table(idx) + self.position_embedding_table(
                torch.arange(T, device=idx.device)
            )
            x = self.ln_f(self.blocks(x))
            logits = self.lm_head(x)
            if targets is None:
                return logits, None
            B, T, C = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, C), targets.view(B * T)
            )
            return logits, loss

        @torch.no_grad()
        def generate(self, idx, max_new_tokens, temperature=0.65, top_k=40):
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -block_size:]
                logits, _ = self(idx_cond)
                nxt = sample_next(
                    logits[0, -1], temperature=temperature, top_k=top_k
                )
                idx = torch.cat((idx, nxt.view(1, 1)), dim=1)
            return idx

        @torch.no_grad()
        def generate_stream(self, idx, max_new_tokens, temperature=0.65, top_k=40):
            """Yield one character at a time."""
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -block_size:]
                logits, _ = self(idx_cond)
                nxt = sample_next(
                    logits[0, -1], temperature=temperature, top_k=top_k
                )
                idx = torch.cat((idx, nxt.view(1, 1)), dim=1)
                yield nxt.item()

    return GPT().to(device)


def load_checkpoint(path: Path | str, device: str | None = None):
    path = Path(path)
    device = device or get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    stoi = ckpt["stoi"]
    itos = ckpt["itos"]

    def encode(s: str) -> list[int]:
        return [stoi[c] for c in s if c in stoi]

    def decode(tokens: list[int]) -> str:
        return "".join(itos[i] for i in tokens)

    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, encode, decode, cfg, device, stoi
