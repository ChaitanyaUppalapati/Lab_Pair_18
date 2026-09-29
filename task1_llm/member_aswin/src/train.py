"""Train and record all Task 1 optimization and system metrics."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from common import MEMBER_DIR, REPO_ROOT, hardware_info, load_config, make_logger, output_paths, peak_memory_mb, select_device, set_seed, write_json
from model import CharacterGPT, parameter_count

METRIC_COLUMNS = [
    "run_id", "run_type", "checkpoint", "train_ce_loss", "val_ce_loss", "perplexity", "bits_per_char",
    "generalization_gap", "top1_next_char_acc", "distinct_1", "distinct_2", "distinct_3",
    "repeated_4gram_rate", "max_grad_norm", "mean_grad_norm", "loss_spikes", "nan_count",
    "param_count", "train_tokens_per_sec", "gen_tokens_per_sec", "peak_memory_mb",
    "total_train_time_s", "hardware",
]


def load_split(path: Path) -> TensorDataset:
    sequences = torch.load(path, map_location="cpu", weights_only=True)["sequences"]
    return TensorDataset(sequences[:, :-1], sequences[:, 1:])


@torch.no_grad()
def evaluate(model, loader, device, pad_id):
    model.eval()
    loss_sum = correct = valid = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=pad_id, reduction="sum").item()
        mask = y.ne(pad_id)
        correct += (logits.argmax(-1)[mask] == y[mask]).sum().item()
        valid += mask.sum().item()
    return loss_sum / max(valid, 1), correct / max(valid, 1)


def plot_history(history: list[dict], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [row["train_loss"] for row in history], marker="o", label="train")
    plt.plot(epochs, [row["val_loss"] for row in history], marker="o", label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Character GPT loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_metrics(row: dict) -> None:
    path = MEMBER_DIR / "metrics_report.csv"
    rows = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as handle:
            rows = [existing for existing in csv.DictReader(handle) if existing["run_id"] != row["run_id"]]
    rows.append({column: row.get(column, "") for column in METRIC_COLUMNS})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    run_id = config["run_id"]
    paths = output_paths(run_id)
    metadata_path = paths["processed"] / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Run preprocessing first; missing {metadata_path}")
    import json
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    seed = int(config["data"]["seed"])
    set_seed(seed)
    device = select_device(config["training"]["device"])
    log_path = paths["logs"] / f"{run_id}.log"
    logger = make_logger(run_id, log_path)
    config_path = Path(args.config).resolve()
    try:
        displayed_config = config_path.relative_to(REPO_ROOT)
    except ValueError:
        displayed_config = config_path.name
    logger.info("run_id=%s config=%s device=%s", run_id, displayed_config, device)
    logger.info("data smoke_only=%s train=%s val=%s vocab=%s", metadata["smoke_only"], metadata["train_examples"], metadata["val_examples"], metadata["vocab_size"])

    train_ds = load_split(paths["processed"] / "train.pt")
    val_ds = load_split(paths["processed"] / "val.pt")
    batch_size = int(config["training"]["batch_size"])
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=loader_generator, num_workers=int(config["training"]["num_workers"]))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=int(config["training"]["num_workers"]))
    model = CharacterGPT(metadata["vocab_size"], metadata["context_length"], config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    epochs = int(config["training"]["epochs"])
    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = max(1, int(total_steps * float(config["training"]["warmup_ratio"])))
    min_ratio = float(config["training"]["min_lr_ratio"])

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    pad_id = metadata["char_to_idx"]["<PAD>"]
    history, grad_norms, recent_losses = [], [], []
    spikes = nan_count = tokens_seen = global_step = 0
    best_val = float("inf")
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_sum = epoch_valid = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=pad_id)
            if not torch.isfinite(loss):
                nan_count += 1
                logger.error("non-finite loss step=%d", global_step)
                continue
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip"])).item())
            optimizer.step()
            scheduler.step()
            valid = int(y.ne(pad_id).sum().item())
            value = float(loss.item())
            if len(recent_losses) >= 20:
                mean = sum(recent_losses[-20:]) / 20
                std = (sum((v - mean) ** 2 for v in recent_losses[-20:]) / 20) ** 0.5
                if std > 0 and value > mean + 3 * std:
                    spikes += 1
            recent_losses.append(value)
            grad_norms.append(norm)
            epoch_loss_sum += value * valid
            epoch_valid += valid
            tokens_seen += valid
            global_step += 1
            if global_step % int(config["training"]["log_every"]) == 0:
                logger.info("epoch=%d step=%d loss=%.6f lr=%.8f grad_norm=%.6f", epoch, global_step, value, optimizer.param_groups[0]["lr"], norm)
        train_loss = epoch_loss_sum / max(epoch_valid, 1)
        val_loss, val_accuracy = evaluate(model, val_loader, device, pad_id)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_accuracy, "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        logger.info("epoch_complete=%d train_loss=%.6f val_loss=%.6f val_accuracy=%.6f", epoch, train_loss, val_loss, val_accuracy)
        payload = {"model_state": model.state_dict(), "config": config, "metadata": metadata, "epoch": epoch, "history": history}
        if val_loss < best_val:
            best_val = val_loss
            torch.save(payload, paths["checkpoints"] / "best_model.pt")
    torch.save(payload, paths["checkpoints"] / "final_model.pt")
    elapsed = time.perf_counter() - started
    plot_history(history, paths["plots"] / f"{run_id}_loss_curves.png")
    write_json(paths["checkpoints"] / "training_history.json", history)
    checkpoint_rel = str((paths["checkpoints"] / "best_model.pt").relative_to(REPO_ROOT))
    metric_row = {
        "run_id": run_id,
        "run_type": "smoke" if metadata["smoke_only"] else "full",
        "checkpoint": checkpoint_rel,
        "train_ce_loss": history[-1]["train_loss"],
        "val_ce_loss": history[-1]["val_loss"],
        "perplexity": math.exp(min(history[-1]["val_loss"], 50)),
        "bits_per_char": history[-1]["val_loss"] / math.log(2),
        "generalization_gap": history[-1]["val_loss"] - history[-1]["train_loss"],
        "top1_next_char_acc": history[-1]["val_accuracy"],
        "max_grad_norm": max(grad_norms, default=float("nan")),
        "mean_grad_norm": sum(grad_norms) / max(len(grad_norms), 1),
        "loss_spikes": spikes,
        "nan_count": nan_count,
        "param_count": parameter_count(model),
        "train_tokens_per_sec": tokens_seen / max(elapsed, 1e-9),
        "peak_memory_mb": peak_memory_mb(device),
        "total_train_time_s": elapsed,
        "hardware": hardware_info(device),
    }
    write_metrics(metric_row)
    shutil.copyfile(args.config, paths["checkpoints"] / "config.yaml")
    logger.info("finished elapsed_s=%.3f tokens_per_s=%.2f checkpoint=%s", elapsed, metric_row["train_tokens_per_sec"], checkpoint_rel)


if __name__ == "__main__":
    main()
