#!/usr/bin/env python3
"""Presentation-ready training curves — Anthropic research paper aesthetic."""

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
# Anthropic-style theme (layout / typography / palette — not their data)
# ---------------------------------------------------------------------------
BG = "#ffffff"
TEXT = "#1a1a1a"
MUTED = "#4a4a4a"
TITLE_SERIF_COLOR = "#6b3e2e"  # brownish-red serif title

# Categorical palette (Anthropic research figures)
C_GREEN = "#4daf7a"
C_CYAN = "#4db8d8"
C_BLUE = "#4a7fd4"  # primary series
C_PURPLE = "#9b7ed9"
C_GREY = "#8a8a8a"
C_BROWN = "#a67c52"
C_ORANGE = "#e8913a"
C_RED = "#d64545"

TRAIN_COLOR = C_CYAN
VAL_COLOR = C_BLUE
FINETUNE_COLOR = C_PURPLE
REF_LINE = "#333333"

SERIF = ["Georgia", "Times New Roman", "DejaVu Serif", "serif"]
SANS = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans", "sans-serif"]


def _pick_font(candidates: list[str]) -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return candidates[-1]


def apply_theme():
    serif = _pick_font(SERIF)
    sans = _pick_font(SANS)
    mpl.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "axes.edgecolor": TEXT,
            "axes.labelcolor": MUTED,
            "text.color": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.family": sans,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.dpi": 160,
            "savefig.facecolor": BG,
            "savefig.edgecolor": BG,
        }
    )
    return serif, sans


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TEXT)
    ax.spines["bottom"].set_color(TEXT)


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


def _fig_title(fig, text: str, serif: str, y: float = 0.98):
    fig.suptitle(
        text,
        fontsize=18,
        color=TITLE_SERIF_COLOR,
        fontfamily=serif,
        fontweight="normal",
        y=y,
        ha="center",
    )


def _legend_dots(ax, loc="upper right"):
    leg = ax.legend(
        loc=loc,
        frameon=False,
        labelcolor=TEXT,
        handlelength=1.2,
        handletextpad=0.6,
    )
    for handle in leg.legend_handles:
        handle.set_linewidth(0)
    return leg


def plot_loss_journey(df: pd.DataFrame, out: Path, serif: str, sans: str):
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    fig.subplots_adjust(top=0.82, left=0.1, right=0.96, bottom=0.14)
    _fig_title(fig, "ShritGPT Training Loss Over Time", serif)

    finetune = df[df["phase"] == "finetune"]
    ax.plot(df["step"], df["train_loss"], color=TRAIN_COLOR, lw=2, label="Train", zorder=2)
    ax.plot(df["step"], df["val_loss"], color=VAL_COLOR, lw=2.2, label="Validation", zorder=3)

    if not finetune.empty:
        boundary = finetune["step"].min()
        ax.axvline(boundary, color=REF_LINE, ls="--", lw=1, alpha=0.65, zorder=1)
        ymax = ax.get_ylim()[1]
        ax.text(
            boundary + (df["step"].max() - df["step"].min()) * 0.01,
            ymax * 0.96,
            "fine-tune",
            fontsize=9,
            color=MUTED,
            va="top",
        )

    best = df.loc[df["val_loss"].idxmin()]
    ax.scatter([best["step"]], [best["val_loss"]], s=36, color=C_ORANGE, zorder=4, edgecolors="none")
    ax.annotate(
        f"best val = {best['val_loss']:.3f}",
        xy=(best["step"], best["val_loss"]),
        xytext=(12, 14),
        textcoords="offset points",
        fontsize=9,
        color=MUTED,
        arrowprops=dict(arrowstyle="-", color=C_GREY, lw=0.8),
    )

    ax.set_xlabel("Training step", fontfamily=sans)
    ax.set_ylabel("Cross-entropy loss", fontfamily=sans)
    _despine(ax)
    _legend_dots(ax, "upper right")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def plot_loss_decrease(df: pd.DataFrame, out: Path, serif: str, sans: str):
    """Val loss improvement per checkpoint — bar style like Anthropic deltas."""
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    fig.subplots_adjust(top=0.82, left=0.1, right=0.96, bottom=0.14)
    _fig_title(fig, "Validation Loss Decrease Per Checkpoint", serif)

    smooth = df["val_loss"].rolling(window=5, min_periods=1).mean()
    delta = -smooth.diff().fillna(0)
    width = max(float(df["step"].diff().median() * 0.85), 50)
    colors = [C_BLUE if d >= 0 else C_RED for d in delta]

    ax.bar(df["step"], delta, width=width, color=colors, edgecolor="none", alpha=0.92)
    ax.axhline(0, color=REF_LINE, lw=0.9, zorder=2)

    ax.set_xlabel("Training step", fontfamily=sans)
    ax.set_ylabel("Val loss improvement (Δ)", fontfamily=sans)
    _despine(ax)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def plot_perplexity(df: pd.DataFrame, out: Path, serif: str, sans: str):
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    fig.subplots_adjust(top=0.82, left=0.1, right=0.96, bottom=0.14)
    _fig_title(fig, "ShritGPT Perplexity During Training", serif)

    ax.plot(df["step"], df["perplexity_train"], color=TRAIN_COLOR, lw=2, label="Train")
    ax.plot(df["step"], df["perplexity_val"], color=VAL_COLOR, lw=2.2, label="Validation")

    ax.set_xlabel("Training step", fontfamily=sans)
    ax.set_ylabel("Perplexity", fontfamily=sans)
    ax.set_yscale("log")
    _despine(ax)
    _legend_dots(ax)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def plot_dashboard(df: pd.DataFrame, out: Path, serif: str, sans: str):
    """Two-panel Anthropic-style layout: loss + learning rate."""
    fig = plt.figure(figsize=(11, 6.2))
    fig.subplots_adjust(top=0.78, left=0.08, right=0.97, bottom=0.12, wspace=0.28)
    _fig_title(fig, "ShritGPT Training Report", serif, y=0.97)

    final = df.iloc[-1]
    best = df.loc[df["val_loss"].idxmin()]
    fig.text(
        0.5,
        0.905,
        f"Final validation loss {final['val_loss']:.3f}  ·  "
        f"Best {best['val_loss']:.3f} at step {int(best['step'])}  ·  "
        f"{len(df)} checkpoints",
        ha="center",
        fontsize=10,
        color=MUTED,
        fontfamily=sans,
    )

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(df["step"], df["val_loss"], color=VAL_COLOR, lw=2.2, label="Validation")
    ax1.plot(df["step"], df["train_loss"], color=TRAIN_COLOR, lw=1.8, alpha=0.85, label="Train")
    finetune = df[df["phase"] == "finetune"]
    if not finetune.empty:
        ax1.axvline(finetune["step"].min(), color=REF_LINE, ls="--", lw=1, alpha=0.6)
    ax1.set_xlabel("Training step", fontfamily=sans)
    ax1.set_ylabel("Loss", fontfamily=sans)
    ax1.set_title("Loss", fontsize=13, color=TEXT, fontfamily=sans, loc="left", pad=8)
    _despine(ax1)
    _legend_dots(ax1)

    ax2.plot(df["step"], df["lr"], color=C_ORANGE, lw=2)
    ax2.set_xlabel("Training step", fontfamily=sans)
    ax2.set_ylabel("Learning rate", fontfamily=sans)
    ax2.set_title("Schedule", fontsize=13, color=TEXT, fontfamily=sans, loc="left", pad=8)
    _despine(ax2)

    fig.savefig(out, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def main():
    serif, sans = apply_theme()
    df = _load_history()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plots = {
        "loss_journey.png": lambda d, p: plot_loss_journey(d, p, serif, sans),
        "loss_decrease.png": lambda d, p: plot_loss_decrease(d, p, serif, sans),
        "perplexity.png": lambda d, p: plot_perplexity(d, p, serif, sans),
        "training_dashboard.png": lambda d, p: plot_dashboard(d, p, serif, sans),
    }

    for name, fn in plots.items():
        out = FIGURES_DIR / name
        fn(df, out)
        print(f"Wrote {out}")

    print(f"\nFigures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
