"""Shared helpers: config loading, repo-relative paths, seeding, logging, hardware string."""
import json
import logging
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = REPO_ROOT / "task2_sentiment"
MEMBER_DIR = TASK_DIR / "member_chaitanya"
ASWIN_DIR = TASK_DIR / "member_aswin"
MEMBER = "chaitanya"
TASK = "task2_sentiment"

SPLITS_DIR = MEMBER_DIR / "splits"            # committed HF row indices (shared with Aswin)
PROCESSED_DIR = MEMBER_DIR / "data_processed"  # regenerable caches (gitignored)
OUTPUTS_DIR = MEMBER_DIR / "outputs"
CHECKPOINTS_DIR = MEMBER_DIR / "checkpoints"
LOG_DIR = REPO_ROOT / "reproducibility" / "raw_logs" / TASK / MEMBER
MANIFEST_DIR = REPO_ROOT / "reproducibility" / "manifests" / TASK / MEMBER

EVAL_SETS = ("val", "test5k", "test33k")


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = p.relative_to(REPO_ROOT).as_posix()
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(log_file: Path, name: str) -> logging.Logger:
    """Log to stdout and append to the raw log file (never rewritten)."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="a", encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def hardware_string() -> str:
    cpu = platform.processor() or platform.machine()
    if torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_name(0)}; CPU: {cpu}"
    return f"CPU only: {cpu}"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
