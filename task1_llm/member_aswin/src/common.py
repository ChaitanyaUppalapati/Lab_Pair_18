"""Shared, path-safe utilities for Aswin's Task 1 implementation."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

MEMBER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def output_paths(run_id: str) -> dict[str, Path]:
    paths = {
        "processed": MEMBER_DIR / "data_processed" / run_id,
        "checkpoints": MEMBER_DIR / "checkpoints" / run_id,
        "plots": MEMBER_DIR / "outputs" / "plots",
        "generated": MEMBER_DIR / "outputs" / "generated_text",
        "failures": MEMBER_DIR / "outputs" / "failure_cases",
        "logs": REPO_ROOT / "reproducibility" / "raw_logs" / "task1_llm" / "aswin",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def make_logger(run_id: str, log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"task1.{run_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def hardware_info(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return f"Apple {platform.machine()} (MPS)"
    return platform.processor() or platform.machine()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024**2)
    if sys.platform != "win32":
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports KiB.
        return peak / (1024**2) if sys.platform == "darwin" else peak / 1024
    try:
        import psutil

        info = psutil.Process(os.getpid()).memory_info()
        return getattr(info, "peak_wset", info.rss) / (1024**2)
    except ImportError:
        return float("nan")
