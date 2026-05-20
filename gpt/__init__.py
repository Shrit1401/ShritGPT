"""ShritGPT — char-level GPT core."""

from gpt.chat import ShritGPT, get_bot
from gpt.model import ModelConfig, build_model, get_device, load_checkpoint
from gpt.paths import CHECKPOINT, DATA_DIR, ROOT, TRAINING_DATA

__all__ = [
    "ShritGPT",
    "get_bot",
    "ModelConfig",
    "build_model",
    "get_device",
    "load_checkpoint",
    "CHECKPOINT",
    "DATA_DIR",
    "ROOT",
    "TRAINING_DATA",
]
