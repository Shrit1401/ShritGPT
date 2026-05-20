"""Shared paths — all relative to project root."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "output"

TRAINING_DATA = DATA_DIR / "shrit.txt"
CHECKPOINT = CHECKPOINT_DIR / "checkpoint.pt"
SAMPLE_OUTPUT = OUTPUT_DIR / "more.txt"
LOSS_HISTORY = OUTPUT_DIR / "loss_history.csv"
FIGURES_DIR = OUTPUT_DIR / "figures"
TEST_RESULTS = OUTPUT_DIR / "test_results.txt"
