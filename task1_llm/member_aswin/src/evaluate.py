"""Recompute held-out CE, perplexity, BPC, gap, and next-character accuracy."""

from __future__ import annotations

import argparse
import csv
import json
import math

import torch
from torch.utils.data import DataLoader

from common import MEMBER_DIR, load_config, output_paths, select_device
from model import CharacterGPT
from train import load_split, evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="best_model.pt")
    args = parser.parse_args()
    config = load_config(args.config); run_id = config["run_id"]; paths = output_paths(run_id)
    checkpoint = torch.load(paths["checkpoints"] / args.checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]; device = select_device(config["training"]["device"])
    model = CharacterGPT(metadata["vocab_size"], metadata["context_length"], checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model_state"]); model.to(device)
    loader = DataLoader(load_split(paths["processed"] / "val.pt"), batch_size=int(config["training"]["batch_size"]))
    val_loss, accuracy = evaluate(model, loader, device, metadata["char_to_idx"]["<PAD>"])
    path = MEMBER_DIR / "metrics_report.csv"
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle); fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        if row["run_id"] == run_id:
            train_loss = float(row["train_ce_loss"])
            row.update({"val_ce_loss": val_loss, "perplexity": math.exp(min(val_loss, 50)), "bits_per_char": val_loss / math.log(2), "generalization_gap": val_loss - train_loss, "top1_next_char_acc": accuracy})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"run_id": run_id, "val_ce_loss": val_loss, "perplexity": math.exp(min(val_loss, 50)), "bits_per_char": val_loss / math.log(2), "top1_next_char_acc": accuracy}, indent=2))


if __name__ == "__main__":
    main()
