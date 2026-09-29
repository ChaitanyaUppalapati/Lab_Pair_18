"""Reproducible training script for Task 2 Yelp Polarity models.
Supports Baseline (Mean-Pooling), Experimental 1 (BiGRU), and Experimental 2 (TextCNN).
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any

import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import set_seed, get_device, get_peak_memory_mb, setup_logger
from dataset import prepare_yelp_dataloaders
from models import build_model, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(description="Train Task 2 Sentiment Classification Model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return parser.parse_args()


def plot_training_curves(history: Dict[str, list], output_path: Path, model_name: str) -> None:
    """Plot and save training and validation loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#e74c3c")
    axes[0].plot(epochs, history["val_loss"], "s-", label="Val Loss", color="#3498db")
    axes[0].set_title(f"{model_name} — Loss vs. Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Accuracy curve
    axes[1].plot(epochs, history["train_acc"], "o-", label="Train Acc", color="#2ecc71")
    axes[1].plot(epochs, history["val_acc"], "s-", label="Val Acc", color="#9b59b6")
    axes[1].set_title(f"{model_name} — Accuracy vs. Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    criterion,
    device: torch.device,
) -> tuple[float, float, int]:
    """Execute one training epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["length"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, lengths)

        # NaN detection
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise ValueError("NaN or Inf detected in model forward logits!")

        loss = criterion(logits, labels)
        if torch.isnan(loss):
            raise ValueError("NaN detected in training loss!")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total_loss += loss.item() * len(labels)
        total_samples += len(labels)

    epoch_loss = total_loss / max(total_samples, 1)
    epoch_acc = correct / max(total_samples, 1)
    return epoch_loss, epoch_acc, total_samples


def validate_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    device: torch.device,
) -> tuple[float, float]:
    """Execute validation evaluation."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["length"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)

            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total_loss += loss.item() * len(labels)
            total_samples += len(labels)

    return total_loss / max(total_samples, 1), correct / max(total_samples, 1)


def main():
    args = parse_args()
    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Setup seed
    seed = config.get("seed", 42)
    set_seed(seed)

    # Setup logging
    log_files = [
        config["paths"]["log_file"],
        config["paths"]["local_log"],
    ]
    logger = setup_logger(log_files, logger_name=f"train_{config['run_id']}")
    logger.info("=" * 60)
    logger.info(f"Starting Training for Model: {config['model_name']} (Run ID: {config['run_id']})")
    logger.info(f"Loaded config from: {config_path}")
    logger.info(f"Random seed: {seed}")

    # Determine device
    device_name = config["training"].get("device", "mps")
    device = get_device(device_name)
    logger.info(f"Hardware compute device selected: {device}")

    # Data pipeline
    train_loader, val_loader, test_loader, vocab, eda_stats = prepare_yelp_dataloaders(config, logger)

    # Build model
    model = build_model(config, len(vocab))
    model.to(device)
    param_count = count_parameters(model)
    logger.info(f"Model architecture: {config['architecture']['model_type']}")
    logger.info(f"Total trainable parameters: {param_count:,}")

    # Optimization
    lr = float(config["training"].get("learning_rate", 1e-3))
    weight_decay = float(config["training"].get("weight_decay", 1e-4))
    optimizer_type = config["training"].get("optimizer", "AdamW")
    if optimizer_type == "AdamW":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["training"].get("scheduler_factor", 0.5),
        patience=config["training"].get("scheduler_patience", 1),
    )

    epochs = config["training"].get("epochs", 5)
    best_checkpoint_path = Path(config["paths"]["best_checkpoint"])
    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_epoch = -1

    start_time = time.time()
    total_trained_samples = 0

    logger.info("Beginning training loop...")
    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        train_loss, train_acc, n_samples = train_epoch(model, train_loader, optimizer, criterion, device)
        total_trained_samples += n_samples

        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        ep_duration = time.time() - ep_start
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} [{ep_duration:.1f}s] - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "config": config,
                    "vocab_size": len(vocab),
                    "param_count": param_count,
                },
                best_checkpoint_path,
            )
            logger.info(f"Saved new best checkpoint at epoch {epoch} (Val Loss: {val_loss:.4f}) -> {best_checkpoint_path}")

    total_training_time = time.time() - start_time
    examples_per_sec = total_trained_samples / max(total_training_time, 1e-4)
    peak_mem_mb = get_peak_memory_mb()

    logger.info(f"Training Complete in {total_training_time:.2f} seconds.")
    logger.info(f"Throughput: {examples_per_sec:.2f} examples/sec | Peak Memory: {peak_mem_mb:.2f} MB")
    logger.info(f"Best validation loss {best_val_loss:.4f} achieved at epoch {best_epoch}.")

    # Save training curves
    plots_dir = Path(config["paths"]["output_dir"]) / "plots"
    curve_path = plots_dir / f"{config['model_name']}_training_curves.png"
    plot_training_curves(history, curve_path, config["model_name"])
    logger.info(f"Saved training curves to {curve_path}")

    # Save training history JSON
    history_file = best_checkpoint_path.parent / "training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": config["model_name"],
                "run_id": config["run_id"],
                "history": history,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "total_training_time_s": total_training_time,
                "examples_per_sec": examples_per_sec,
                "peak_memory_mb": peak_mem_mb,
                "param_count": param_count,
                "device": str(device),
            },
            f,
            indent=2,
        )

    # Capture reproducibility manifest
    try:
        manifest_cmd = [
            sys.executable,
            "reproducibility/capture_manifest.py",
            "--member", config.get("member", "aswin"),
            "--task", config.get("task", "task2_sentiment"),
            "--run-id", config["run_id"],
            "--checkpoint", str(best_checkpoint_path),
            "--config", str(config_path),
            "--notes", f"Training {config['model_name']} for {epochs} epochs",
        ]
        res = subprocess.run(manifest_cmd, capture_output=True, text=True)
        if res.returncode == 0:
            logger.info("Successfully recorded reproducibility manifest.")
        else:
            logger.warning(f"Manifest capture warning: {res.stderr}")
    except Exception as e:
        logger.warning(f"Failed to run capture_manifest.py: {e}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
