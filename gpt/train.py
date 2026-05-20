"""Train the char-level GPT on shrit.txt."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from gpt.model import ModelConfig, build_model, get_device
from gpt.paths import (
    CHECKPOINT,
    CHECKPOINT_DIR,
    LOSS_HISTORY,
    OUTPUT_DIR,
    SAMPLE_OUTPUT,
    TRAINING_DATA,
)

# M2 MacBook Air 8GB — larger context + model for chat Q&A
batch_size = 6
grad_accum_steps = 2
block_size = 384
max_iters = 25000
finetune_iters = 8000
eval_interval = 250
learning_rate = 3e-4
finetune_lr = 8e-5
eval_iters = 50
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.1
gen_prompt = "them: who is shrit\nme: "
max_new_tokens = 200
temperature = 0.5
top_k = 25

device = get_device()

if device == "mps":
    torch.set_num_threads(4)

torch.manual_seed(1337)


def load_corpus(path: Path = TRAINING_DATA):
    if not path.exists():
        raise SystemExit(f"Missing {path} — run: python scripts/build_shrit.py")

    text = path.read_text(encoding="utf-8")
    print(f"Loaded {len(text):,} chars from {path}")

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return text, stoi, itos, encode, decode, train_data, val_data, vocab_size


def get_batch(train_data, val_data, split):
    data_split = train_data if split == "train" else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i : i + block_size] for i in ix])
    y = torch.stack([data_split[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train_data, val_data, split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def init_loss_log(path: Path = LOSS_HISTORY):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "step",
                "train_loss",
                "val_loss",
                "lr",
                "elapsed_min",
                "phase",
            ]
        )


def append_loss_log(
    step: int,
    losses: dict[str, float],
    lr: float,
    elapsed_min: float,
    phase: str,
    path: Path = LOSS_HISTORY,
):
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                step,
                f"{losses['train']:.6f}",
                f"{losses['val']:.6f}",
                f"{lr:.8f}",
                f"{elapsed_min:.2f}",
                phase,
            ]
        )


def build_prompt(text, stoi):
    if all(c in stoi for c in gen_prompt):
        return gen_prompt
    return text[:block_size]


def run_generation(model, decode, text, stoi, label="sample"):
    model.eval()
    start = build_prompt(text, stoi)
    context = torch.tensor([[stoi[c] for c in start]], dtype=torch.long, device=device)
    out = decode(
        model.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )[0].tolist()
    )
    print(f"\n--- {label} (temp={temperature}, top_k={top_k}) ---\n")
    print(out)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_OUTPUT.write_text(out, encoding="utf-8")
    print(f"\nWrote {len(out):,} chars to {SAMPLE_OUTPUT}")
    return out


def save_checkpoint(model, optimizer, step, stoi, itos, vocab_size, phase: str = "train"):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "phase": phase,
            "stoi": stoi,
            "itos": itos,
            "config": {
                "block_size": block_size,
                "n_embd": n_embd,
                "n_head": n_head,
                "n_layer": n_layer,
                "dropout": dropout,
                "vocab_size": vocab_size,
            },
        },
        CHECKPOINT,
    )
    print(f"Saved {CHECKPOINT} (phase={phase}, step={step})")


def train_loop(
    model,
    optimizer,
    train_data,
    val_data,
    *,
    start_iter: int,
    total_iters: int,
    base_lr: float,
    phase: str,
    log_offset: int = 0,
    t0: float | None = None,
):
    t0 = t0 or time.time()
    model.train()

    for local in range(total_iters):
        iter_num = start_iter + local
        progress = local / max(total_iters - 1, 1)
        if phase == "finetune":
            lr = base_lr * (0.1 + 0.9 * (1.0 - progress))
        else:
            lr = base_lr * (1.0 - local / max(total_iters, 1))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if local % eval_interval == 0 or local == total_iters - 1:
            losses = estimate_loss(model, train_data, val_data)
            elapsed = (time.time() - t0) / 60
            global_step = log_offset + local
            append_loss_log(global_step, losses, lr, elapsed, phase)
            print(
                f"[{phase}] step {iter_num}: train {losses['train']:.4f}, "
                f"val {losses['val']:.4f} ({elapsed:.1f} min)"
            )

        for _ in range(grad_accum_steps):
            xb, yb = get_batch(train_data, val_data, "train")
            logits, loss = model(xb, yb)
            loss = loss / grad_accum_steps
            loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return time.time() - t0


def plot_training_curves():
    try:
        from scripts.plot_training import main as plot_main
    except ImportError:
        import sys

        root = Path(__file__).resolve().parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from scripts.plot_training import main as plot_main

    plot_main()


def main():
    global temperature, top_k, max_new_tokens, gen_prompt

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Skip training; load checkpoint and generate",
    )
    parser.add_argument(
        "--finetune-only",
        action="store_true",
        help="Load checkpoint and run fine-tune phase only",
    )
    parser.add_argument("--no-finetune", action="store_true", help="Skip fine-tune phase")
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--tokens", type=int, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    args = parser.parse_args()

    if args.temperature is not None:
        temperature = args.temperature
    if args.top_k is not None:
        top_k = args.top_k
    if args.tokens is not None:
        max_new_tokens = args.tokens
    if args.prompt is not None:
        gen_prompt = args.prompt

    text, stoi, itos, encode, decode, train_data, val_data, vocab_size = load_corpus()

    cfg = ModelConfig(
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
        vocab_size=vocab_size,
    )
    model = build_model(cfg, device)

    if args.sample_only:
        if not CHECKPOINT.exists():
            raise SystemExit(f"No {CHECKPOINT} — run training first.")
        ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded checkpoint (step {ckpt.get('step', '?')}, phase {ckpt.get('phase', '?')})")
        run_generation(model, decode, text, stoi, label="sample from checkpoint")
        return

    if args.finetune_only:
        if not CHECKPOINT.exists():
            raise SystemExit(f"No {CHECKPOINT} — run training first.")
        ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start = ckpt.get("step", 0) + 1
        optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_lr)
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                pass
        print(f"Fine-tuning from step {start} for {finetune_iters} iters...")
        train_loop(
            model,
            optimizer,
            train_data,
            val_data,
            start_iter=start,
            total_iters=finetune_iters,
            base_lr=finetune_lr,
            phase="finetune",
            log_offset=start,
        )
        save_checkpoint(
            model, optimizer, start + finetune_iters - 1, stoi, itos, vocab_size, "finetune"
        )
        run_generation(model, decode, text, stoi, label="sample after fine-tune")
        if not args.no_plots:
            plot_training_curves()
        return

    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters on {device}")
    print(
        f"block_size={block_size} max_iters={max_iters} finetune_iters={finetune_iters}"
    )

    init_loss_log()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    t0 = time.time()

    train_loop(
        model,
        optimizer,
        train_data,
        val_data,
        start_iter=0,
        total_iters=max_iters,
        base_lr=learning_rate,
        phase="train",
        log_offset=0,
        t0=t0,
    )

    save_checkpoint(model, optimizer, max_iters - 1, stoi, itos, vocab_size, "train")

    if not args.no_finetune:
        print(f"\n=== Fine-tune phase ({finetune_iters} iters @ lr~{finetune_lr}) ===\n")
        for pg in optimizer.param_groups:
            pg["lr"] = finetune_lr
        train_loop(
            model,
            optimizer,
            train_data,
            val_data,
            start_iter=max_iters,
            total_iters=finetune_iters,
            base_lr=finetune_lr,
            phase="finetune",
            log_offset=max_iters,
            t0=t0,
        )
        save_checkpoint(
            model,
            optimizer,
            max_iters + finetune_iters - 1,
            stoi,
            itos,
            vocab_size,
            "finetune",
        )

    run_generation(model, decode, text, stoi, label="sample after training")

    if not args.no_plots:
        print("\nGenerating training figures...")
        plot_training_curves()


if __name__ == "__main__":
    main()
