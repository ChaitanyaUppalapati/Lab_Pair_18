"""Utility functions for Task 2 Yelp Polarity Sentiment Classification.
Includes seeding, logging, hardware tracking, metrics, bootstrap CI, and plotting.
"""

import os
import sys
import time
import math
import random
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import psutil
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
    matthews_corrcoef,
    brier_score_loss,
)
import statsmodels.stats.contingency_tables as ct


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(requested_device: str = "auto") -> torch.device:
    """Select the best available compute device (CUDA > MPS > CPU)."""
    if requested_device == "cpu":
        return torch.device("cpu")
    if requested_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested_device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    # Auto-detect best hardware
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_peak_memory_mb() -> float:
    """Get current process peak memory usage in megabytes."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return float(mem_info.rss / (1024 * 1024))


def setup_logger(log_paths: List[str], logger_name: str = "task2_sentiment") -> logging.Logger:
    """Setup logger writing unedited raw logs to files and stdout."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handlers
    for p in log_paths:
        path = Path(p)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(path), mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) with equal-width probability bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Compute 95% bootstrap confidence intervals for Accuracy, Macro-F1, and MCC."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    indices = np.arange(n)

    accs = []
    f1s = []
    mccs = []

    for _ in range(n_bootstrap):
        boot_idx = rng.choice(indices, size=n, replace=True)
        boot_true = y_true[boot_idx]
        boot_pred = y_pred[boot_idx]

        accs.append(accuracy_score(boot_true, boot_pred))
        f1s.append(f1_score(boot_true, boot_pred, average="macro", zero_division=0))
        mccs.append(matthews_corrcoef(boot_true, boot_pred))

    alpha = (1.0 - ci) / 2.0
    low_pct = alpha * 100.0
    high_pct = (1.0 - alpha) * 100.0

    return {
        "accuracy": (float(np.percentile(accs, low_pct)), float(np.percentile(accs, high_pct))),
        "f1_macro": (float(np.percentile(f1s, low_pct)), float(np.percentile(f1s, high_pct))),
        "mcc": (float(np.percentile(mccs, low_pct)), float(np.percentile(mccs, high_pct))),
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    compute_ci: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics for binary sentiment classification."""
    y_pred = (y_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    p_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    r_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    p_micro = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
    r_micro = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
    f1_micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))

    p_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    r_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    roc_auc = float(roc_auc_score(y_true, y_prob))
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(recall_curve, precision_curve))

    mcc = float(matthews_corrcoef(y_true, y_pred))
    brier = float(brier_score_loss(y_true, y_prob))
    ece = float(compute_expected_calibration_error(y_true, y_prob))

    metrics = {
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "precision_micro": p_micro,
        "recall_micro": r_micro,
        "f1_micro": f1_micro,
        "precision_weighted": p_weighted,
        "recall_weighted": r_weighted,
        "f1_weighted": f1_weighted,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "mcc": mcc,
        "brier": brier,
        "ece": ece,
    }

    if compute_ci:
        ci_dict = compute_bootstrap_ci(y_true, y_pred, n_bootstrap=1000, seed=seed)
        metrics["acc_ci_low"] = ci_dict["accuracy"][0]
        metrics["acc_ci_high"] = ci_dict["accuracy"][1]
        metrics["f1_macro_ci_low"] = ci_dict["f1_macro"][0]
        metrics["f1_macro_ci_high"] = ci_dict["f1_macro"][1]
        metrics["mcc_ci_low"] = ci_dict["mcc"][0]
        metrics["mcc_ci_high"] = ci_dict["mcc"][1]
    else:
        metrics["acc_ci_low"] = acc
        metrics["acc_ci_high"] = acc
        metrics["f1_macro_ci_low"] = f1_macro
        metrics["f1_macro_ci_high"] = f1_macro
        metrics["mcc_ci_low"] = mcc
        metrics["mcc_ci_high"] = mcc

    return metrics


def run_mcnemar_test(
    y_true: np.ndarray,
    y_pred_baseline: np.ndarray,
    y_pred_model: np.ndarray,
) -> Tuple[float, float, List[List[int]], bool]:
    """Perform paired McNemar test with continuity correction."""
    # Contingency table:
    #                 Model correct  Model wrong
    # Baseline corr       n00           n01
    # Baseline wrong      n10           n11
    correct_base = (y_pred_baseline == y_true)
    correct_model = (y_pred_model == y_true)

    n00 = int(np.sum(correct_base & correct_model))
    n01 = int(np.sum(correct_base & (~correct_model)))  # base correct, model wrong
    n10 = int(np.sum((~correct_base) & correct_model))  # base wrong, model correct
    n11 = int(np.sum((~correct_base) & (~correct_model)))

    table = [[n00, n01], [n10, n11]]
    result = ct.mcnemar(table, exact=False, correction=True)
    stat = float(result.statistic)
    p_val = float(result.pvalue)
    is_significant = bool(p_val < 0.05)

    return stat, p_val, table, is_significant
