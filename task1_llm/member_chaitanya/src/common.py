"""Shared helpers: config loading, repo-relative paths, seeding, logging."""
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
MEMBER_DIR = REPO_ROOT / "task1_llm" / "member_chaitanya"
MEMBER = "chaitanya"
TASK = "task1_llm"


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = p.relative_to(REPO_ROOT).as_posix()
    return cfg


def repo_path(rel: str) -> Path:
    return REPO_ROOT / rel


def run_paths(run_id: str) -> dict:
    paths = {
        "processed": MEMBER_DIR / "data_processed",
        "checkpoints": MEMBER_DIR / "checkpoints" / run_id,
        "outputs": MEMBER_DIR / "outputs" / run_id,
        "raw_log": REPO_ROOT / "reproducibility" / "raw_logs" / TASK / MEMBER / f"{run_id}.log",
    }
    for key in ("processed", "checkpoints", "outputs"):
        paths[key].mkdir(parents=True, exist_ok=True)
    paths["raw_log"].parent.mkdir(parents=True, exist_ok=True)
    return paths


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(log_file: Path, name: str) -> logging.Logger:
    """Log to stdout and append to the raw log file (never rewritten)."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="a", encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def hardware_string() -> str:
    import platform

    cpu = platform.processor() or platform.machine()
    if torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_name(0)}; CPU: {cpu}"
    return f"CPU only: {cpu}"


def load_vocab(processed_dir: Path) -> tuple[dict, dict]:
    vocab = json.loads((processed_dir / "vocab.json").read_text(encoding="utf-8"))
    char_to_idx = vocab["char_to_idx"]
    idx_to_char = {int(k): v for k, v in vocab["idx_to_char"].items()}
    return char_to_idx, idx_to_char
