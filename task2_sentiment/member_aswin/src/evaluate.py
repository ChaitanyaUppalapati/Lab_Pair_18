"""Evaluation and metrics generation script for Task 2 Yelp Polarity models.
Generates predictions, comprehensive metrics, bootstrap confidence intervals, and visualization plots.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from utils import (
    set_seed,
    get_device,
    compute_all_metrics,
    setup_logger,
)
from dataset import prepare_yelp_dataloaders
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Task 2 Sentiment Classification Model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return parser.parse_args()


def plot_confusion_matrix(tn: int, fp: int, fn: int, tp: int, output_path: Path, model_name: str) -> None:
    """Plot and save confusion matrix heatmap."""
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["Negative (0)", "Positive (1)"],
        yticklabels=["Negative (0)", "Positive (1)"],
        title=f"{model_name} — Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_roc_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, output_dir: Path, model_name: str) -> None:
    """Generate ROC and Precision-Recall curve plots."""
    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2980b9", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#7f8c8d", lw=1.5, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(f"{model_name} — ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "plots" / f"{model_name}_roc_curve.png", dpi=300)
    plt.close(fig)

    # PR Curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color="#27ae60", lw=2, label=f"PR curve (AUC = {pr_auc:.4f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{model_name} — Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "plots" / f"{model_name}_pr_curve.png", dpi=300)
    plt.close(fig)


def plot_reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path, model_name: str, n_bins: int = 10) -> None:
    """Generate calibration reliability curve and confidence histogram."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (y_prob >= low) & (y_prob <= high) if i == n_bins - 1 else (y_prob >= low) & (y_prob < high)
        count = int(np.sum(in_bin))
        bin_counts.append(count)
        if count > 0:
            bin_accs.append(np.mean(y_true[in_bin]))
            bin_confs.append(np.mean(y_prob[in_bin]))
        else:
            bin_accs.append(0.0)
            bin_confs.append((low + high) / 2.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Reliability curve
    axes[0].plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    axes[0].plot(bin_confs, bin_accs, "s-", color="#e67e22", label=f"{model_name}")
    axes[0].set_xlabel("Mean Predicted Confidence")
    axes[0].set_ylabel("Fraction of Positives (Empirical Accuracy)")
    axes[0].set_title(f"{model_name} — Reliability Diagram")
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right")

    # Confidence distribution
    axes[1].bar(bin_boundaries[:-1], bin_counts, width=1.0 / n_bins, align="edge", color="#8e44ad", alpha=0.7, edgecolor="black")
    axes[1].set_xlabel("Predicted Probability Bins")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{model_name} — Confidence Histogram")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def update_metrics_csv(report_file: Path, row_data: Dict[str, Any]) -> None:
    """Append or update a model's row in metrics_report.csv."""
    columns = [
        "model", "run_id", "checkpoint", "accuracy", "acc_ci_low", "acc_ci_high",
        "precision_macro", "recall_macro", "f1_macro", "f1_macro_ci_low", "f1_macro_ci_high",
        "precision_micro", "recall_micro", "f1_micro", "precision_weighted", "recall_weighted", "f1_weighted",
        "tn", "fp", "fn", "tp", "roc_auc", "pr_auc", "mcc", "mcc_ci_low", "mcc_ci_high",
        "brier", "ece", "mcnemar_vs_baseline_stat", "mcnemar_vs_baseline_p",
        "param_count", "train_time_s", "examples_per_sec", "peak_memory_mb", "hardware"
    ]

    report_file.parent.mkdir(parents=True, exist_ok=True)
    if report_file.exists():
        df = pd.read_csv(report_file)
    else:
        df = pd.DataFrame(columns=columns)

    # Filter out previous row for this model/run_id if re-evaluating
    df = df[df["run_id"] != row_data["run_id"]]

    new_row = {col: row_data.get(col, "") for col in columns}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(report_file, index=False)


def main():
    args = parse_args()
    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    set_seed(seed)

    log_files = [config["paths"]["log_file"], config["paths"]["local_log"]]
    logger = setup_logger(log_files, logger_name=f"eval_{config['run_id']}")
    logger.info("=" * 60)
    logger.info(f"Starting Evaluation for Model: {config['model_name']} (Run ID: {config['run_id']})")

    device_name = config["training"].get("device", "mps")
    device = get_device(device_name)

    # Load dataloaders
    _, _, test_loader, vocab, _ = prepare_yelp_dataloaders(config, logger)

    # Load checkpoint
    checkpoint_path = Path(config["paths"]["best_checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}. Train the model first.")

    logger.info(f"Loading checkpoint weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_model(config, len(vocab))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load hardware and training metadata from history JSON
    history_file = checkpoint_path.parent / "training_history.json"
    train_metadata = {}
    if history_file.exists():
        with open(history_file, "r", encoding="utf-8") as f:
            train_metadata = json.load(f)

    # Inference loop on test set
    logger.info("Executing inference on the untouched test set...")
    all_y_true = []
    all_y_prob = []
    all_raw_texts = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["length"].to(device)
            labels = batch["label"].cpu().numpy()
            raw_texts = batch["raw_text"]

            logits = model(input_ids, lengths)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_y_true.extend(labels)
            all_y_prob.extend(probs)
            all_raw_texts.extend(raw_texts)

    y_true = np.array(all_y_true, dtype=int)
    y_prob = np.array(all_y_prob, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)

    # Compute comprehensive evaluation metrics
    logger.info("Computing metrics, bootstrap confidence intervals, and calibration statistics...")
    metrics = compute_all_metrics(y_true, y_prob, threshold=0.5, compute_ci=True, seed=seed)

    # Log metrics
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f} [95% CI: {metrics['acc_ci_low']:.4f} - {metrics['acc_ci_high']:.4f}]")
    logger.info(f"Macro-F1:      {metrics['f1_macro']:.4f} [95% CI: {metrics['f1_macro_ci_low']:.4f} - {metrics['f1_macro_ci_high']:.4f}]")
    logger.info(f"ROC-AUC:       {metrics['roc_auc']:.4f}")
    logger.info(f"PR-AUC:        {metrics['pr_auc']:.4f}")
    logger.info(f"MCC:           {metrics['mcc']:.4f} [95% CI: {metrics['mcc_ci_low']:.4f} - {metrics['mcc_ci_high']:.4f}]")
    logger.info(f"Brier Score:   {metrics['brier']:.4f}")
    logger.info(f"ECE:           {metrics['ece']:.4f}")
    logger.info(f"Confusion Matrix: TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, TP={metrics['tp']}")

    # Save test predictions
    output_dir = Path(config["paths"]["output_dir"])
    preds_dir = output_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    preds_file = preds_dir / f"{config['model_name']}_test_predictions.csv"

    preds_df = pd.DataFrame({
        "review_id": np.arange(len(y_true)),
        "true_label": y_true,
        "predicted_label": y_pred,
        "prob_positive": y_prob,
        "review_text": all_raw_texts,
    })
    preds_df.to_csv(preds_file, index=False)
    logger.info(f"Saved test predictions to: {preds_file}")

    # Generate plots
    cm_path = output_dir / "confusion_matrices" / f"{config['model_name']}_confusion_matrix.png"
    plot_confusion_matrix(metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"], cm_path, config["model_name"])

    plot_roc_pr_curves(y_true, y_prob, output_dir, config["model_name"])

    cal_path = output_dir / "calibration" / f"{config['model_name']}_reliability_diagram.png"
    plot_reliability_diagram(y_true, y_prob, cal_path, config["model_name"])

    # Update metrics_report.csv
    row_data = {
        "model": config["model_name"],
        "run_id": config["run_id"],
        "checkpoint": str(checkpoint_path),
        **metrics,
        "param_count": train_metadata.get("param_count", checkpoint.get("param_count", 0)),
        "train_time_s": round(train_metadata.get("total_training_time_s", 0.0), 2),
        "examples_per_sec": round(train_metadata.get("examples_per_sec", 0.0), 2),
        "peak_memory_mb": round(train_metadata.get("peak_memory_mb", 0.0), 2),
        "hardware": "Apple M4 (MPS)",
        "mcnemar_vs_baseline_stat": "",
        "mcnemar_vs_baseline_p": "",
    }

    report_path = Path("task2_sentiment/member_aswin/metrics_report.csv")
    update_metrics_csv(report_path, row_data)
    logger.info(f"Successfully recorded metrics in {report_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
