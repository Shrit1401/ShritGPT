#!/usr/bin/env python3
"""
ShritGPT training figures — Transformer Circuits / Distill design theme.

Typography and palette match https://transformer-circuits.pub/ (distill.template.v2):
  - Body/axes: system UI sans stack (same as the emotions article)
  - Figure titles: terracotta brown, centered
  - Minimal spines, no grid, light zero-line, dashed references
  - Stats annotations: monospace

Only the visual theme is borrowed — plots show ShritGPT loss data, not emotion probes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpt.paths import FIGURES_DIR, LOSS_HISTORY

# ---------------------------------------------------------------------------
# Transformer Circuits / Distill v2 theme (from distill.template + figures)
# ---------------------------------------------------------------------------
BG = "#ffffff"
TEXT = "#333333"  # rgba(0,0,0,0.8) on site
MUTED = "#666666"  # rgba(0,0,0,0.6) captions / ticks
CAPTION = "#888888"
TITLE_COLOR = "#a6614e"  # terracotta figure titles (emotions paper figures)
LINK_BLUE = "#004276"  # distill anchor blue — primary series
ZERO_LINE = "#e6e6e6"
REF_DASH = "#888888"
BORDER = "#1a1a1a"

# Categorical palette (emotion-paper figure colors, reused for train/val/etc.)
C_RED = "#c44e52"
C_ORANGE = "#e8913a"
C_GREEN = "#4daf7a"
C_CYAN = "#4db8d8"
C_BLUE = "#4a7fd4"
C_PURPLE = "#9b7ed9"
C_GREY = "#8a8a8a"
C_BROWN = "#a67c52"
C_LIGHT_BLUE = "#8ebad9"  # scatter point fill on site

TRAIN_COLOR = C_CYAN
VAL_COLOR = LINK_BLUE
FINETUNE_COLOR = C_PURPLE
IMPROVE_COLOR = C_BLUE
WORSE_COLOR = C_RED

# Distill base font stack (distill.template.v2-relative.js → html { font-family })
TC_SANS = [
    "-apple-system",
    "BlinkMacSystemFont",
    "Segoe UI",
    "Roboto",
    "Helvetica Neue",
    "Arial",
    "sans-serif",
]
TC_MONO = [
    "SF Mono",
    "Menlo",
    "Consolas",
    "Monaco",
    "Liberation Mono",
    "monospace",
]


def _pick_font(candidates: list[str]) -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return candidates[-1]


def apply_theme() -> tuple[str, str]:
    sans = _pick_font(TC_SANS)
    mono = _pick_font(TC_MONO)
    mpl.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "axes.edgecolor": BORDER,
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.family": "sans-serif",
            "font.sans-serif": [sans, "DejaVu Sans", "sans-serif"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "figure.dpi": 160,
            "savefig.facecolor": BG,
            "savefig.edgecolor": BG,
            "axes.unicode_minus": False,
        }
    )
    mpl.rcParams["font.sans-serif"].insert(0, sans)
    return sans, mono


def _style_axes(ax, sans: str):
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)
    ax.tick_params(colors=MUTED, labelsize=10, direction="in")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(sans)


def _figure_header(fig, title: str, subtitle: str, sans: str):
    """Centered terracotta title + muted subtitle (Distill figure style)."""
    fig.text(
        0.5,
        0.97,
        title,
        ha="center",
        va="top",
        fontsize=15,
        color=TITLE_COLOR,
        fontfamily=sans,
        fontweight="500",
    )
    if subtitle:
        fig.text(
            0.5,
            0.915,
            subtitle,
            ha="center",
            va="top",
            fontsize=10,
            color=MUTED,
            fontfamily=sans,
            fontweight="400",
        )


def _direct_line_labels(ax, lines_data: list[tuple], sans: str):
    """End-of-line labels like Transformer Circuits multi-panel line charts."""
    xlim = ax.get_xlim()
    x_pad = (xlim[1] - xlim[0]) * 0.012
    for x, y, color, label in lines_data:
        ax.text(
            x[-1] + x_pad,
            y[-1],
            label,
            color=color,
            fontsize=10,
            fontfamily=sans,
            va="center",
            ha="left",
        )


def _load_history(path: Path = LOSS_HISTORY) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"No loss log at {path} — run training first.")
    df = pd.read_csv(path)
    df["train_loss"] = pd.to_numeric(df["train_loss"], errors="coerce")
    df["val_loss"] = pd.to_numeric(df["val_loss"], errors="coerce")
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["lr"] = pd.to_numeric(df["lr"], errors="coerce")
    df["perplexity_train"] = np.exp(df["train_loss"])
    df["perplexity_val"] = np.exp(df["val_loss"])
    return df.dropna(subset=["step", "train_loss", "val_loss"])


def plot_loss_journey(df: pd.DataFrame, out: Path, sans: str, mono: str):
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    fig.subplots_adjust(top=0.78, left=0.11, right=0.82, bottom=0.14)
    _figure_header(
        fig,
        "ShritGPT Training Loss Over Time",
        "Character-level transformer · train and fine-tune phases",
        sans,
    )

    steps = df["step"].values
    train_y = df["train_loss"].values
    val_y = df["val_loss"].values

    ax.axhline(0, color=ZERO_LINE, lw=0.8, zorder=0)  # subtle baseline
    ax.plot(steps, train_y, color=TRAIN_COLOR, lw=2, marker="o", markersize=3,
            markevery=max(1, len(steps) // 12), zorder=2)
    ax.plot(steps, val_y, color=VAL_COLOR, lw=2.2, marker="o", markersize=3,
            markevery=max(1, len(steps) // 12), zorder=3)

    finetune = df[df["phase"] == "finetune"]
    if not finetune.empty:
        boundary = finetune["step"].min()
        ax.axvline(boundary, color=REF_DASH, ls="--", lw=1, alpha=0.75, zorder=1)

    best = df.loc[df["val_loss"].idxmin()]
    ax.scatter([best["step"]], [best["val_loss"]], s=28, color=C_ORANGE, zorder=4, edgecolors="none")
    ax.text(
        0.04,
        0.96,
        f"best val = {best['val_loss']:.3f}",
        transform=ax.transAxes,
        fontsize=9.5,
        color=MUTED,
        fontfamily=mono,
        va="top",
    )

    _direct_line_labels(
        ax,
        [
            (steps, train_y, TRAIN_COLOR, "Train"),
            (steps, val_y, VAL_COLOR, "Validation"),
        ],
        sans,
    )

    ax.set_xlabel("Training step", fontfamily=sans, color=TEXT)
    ax.set_ylabel("Cross-entropy loss", fontfamily=sans, color=TEXT)
    _style_axes(ax, sans)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def plot_loss_decrease(df: pd.DataFrame, out: Path, sans: str, mono: str):
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    fig.subplots_adjust(top=0.78, left=0.11, right=0.96, bottom=0.14)
    _figure_header(
        fig,
        "Validation Loss Decrease Per Checkpoint",
        "Positive bars = validation loss dropped since prior checkpoint",
        sans,
    )

    smooth = df["val_loss"].rolling(window=3, min_periods=1).mean()
    delta = -smooth.diff().fillna(0)
    width = max(float(df["step"].diff().median() * 0.85), 40)
    colors = [IMPROVE_COLOR if d >= 0 else WORSE_COLOR for d in delta]

    ax.bar(df["step"], delta, width=width, color=colors, edgecolor="none", alpha=0.9)
    ax.axhline(0, color=BORDER, lw=0.9, zorder=2)

    ax.set_xlabel("Training step", fontfamily=sans, color=TEXT)
    ax.set_ylabel("Val loss improvement (Δ)", fontfamily=sans, color=TEXT)
    _style_axes(ax, sans)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def plot_perplexity(df: pd.DataFrame, out: Path, sans: str, mono: str):
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    fig.subplots_adjust(top=0.78, left=0.11, right=0.82, bottom=0.14)
    _figure_header(
        fig,
        "ShritGPT Perplexity During Training",
        "Lower perplexity = model more confident on next character",
        sans,
    )

    steps = df["step"].values
    p_train = df["perplexity_train"].values
    p_val = df["perplexity_val"].values

    ax.plot(steps, p_train, color=TRAIN_COLOR, lw=2, marker="o", markersize=3,
            markevery=max(1, len(steps) // 12))
    ax.plot(steps, p_val, color=VAL_COLOR, lw=2.2, marker="o", markersize=3,
            markevery=max(1, len(steps) // 12))
    ax.set_yscale("log")

    r = np.corrcoef(steps, p_val)[0, 1] if len(steps) > 2 else float("nan")
    ax.text(
        0.04,
        0.96,
        f"r = {r:.2f}" if not np.isnan(r) else "",
        transform=ax.transAxes,
        fontsize=9.5,
        color=MUTED,
        fontfamily=mono,
        va="top",
    )

    _direct_line_labels(
        ax,
        [
            (steps, p_train, TRAIN_COLOR, "Train"),
            (steps, p_val, VAL_COLOR, "Validation"),
        ],
        sans,
    )

    ax.set_xlabel("Training step", fontfamily=sans, color=TEXT)
    ax.set_ylabel("Perplexity (log scale)", fontfamily=sans, color=TEXT)
    _style_axes(ax, sans)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def plot_dashboard(df: pd.DataFrame, out: Path, sans: str, mono: str):
    fig = plt.figure(figsize=(10.5, 5.6))
    fig.subplots_adjust(top=0.72, left=0.08, right=0.96, bottom=0.14, wspace=0.32)

    final = df.iloc[-1]
    best = df.loc[df["val_loss"].idxmin()]
    _figure_header(
        fig,
        "ShritGPT Training Report",
        f"Final val {final['val_loss']:.3f} · best {best['val_loss']:.3f} @ step {int(best['step'])} · "
        f"{len(df)} checkpoints",
        sans,
    )

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    steps = df["step"].values
    ax1.plot(steps, df["val_loss"], color=VAL_COLOR, lw=2.2, marker="o", markersize=2.5,
             markevery=max(1, len(steps) // 10))
    ax1.plot(steps, df["train_loss"], color=TRAIN_COLOR, lw=1.8, alpha=0.9, marker="o",
             markersize=2.5, markevery=max(1, len(steps) // 10))
    finetune = df[df["phase"] == "finetune"]
    if not finetune.empty:
        ax1.axvline(finetune["step"].min(), color=REF_DASH, ls="--", lw=1, alpha=0.7)
    ax1.set_xlabel("Training step", fontfamily=sans)
    ax1.set_ylabel("Loss", fontfamily=sans)
    ax1.set_title("Loss", fontsize=12, color=TEXT, fontfamily=sans, loc="left", pad=10)
    _style_axes(ax1, sans)

    ax2.plot(steps, df["lr"], color=C_ORANGE, lw=2)
    ax2.set_xlabel("Training step", fontfamily=sans)
    ax2.set_ylabel("Learning rate", fontfamily=sans)
    ax2.set_title("Schedule", fontsize=12, color=TEXT, fontfamily=sans, loc="left", pad=10)
    _style_axes(ax2, sans)

    fig.savefig(out, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def main():
    sans, mono = apply_theme()
    df = _load_history()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plots = {
        "loss_journey.png": lambda d, p: plot_loss_journey(d, p, sans, mono),
        "loss_decrease.png": lambda d, p: plot_loss_decrease(d, p, sans, mono),
        "perplexity.png": lambda d, p: plot_perplexity(d, p, sans, mono),
        "training_dashboard.png": lambda d, p: plot_dashboard(d, p, sans, mono),
    }

    for name, fn in plots.items():
        out = FIGURES_DIR / name
        fn(df, out)
        print(f"Wrote {out}")

    print(f"\nFigures saved to {FIGURES_DIR}/")
    print(f"Font stack: {sans} (Transformer Circuits / Distill body font)")


if __name__ == "__main__":
    main()
