"""Generate greedy and temperature samples and update diversity metrics."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter

import torch

from common import MEMBER_DIR, REPO_ROOT, load_config, output_paths, select_device, set_seed
from model import CharacterGPT


def decode(ids: list[int], inverse: dict[int, str]) -> str:
    special = {"<PAD>", "<UNK>", "<BOS>", "<EOS>"}
    return "".join(inverse[i] for i in ids if inverse[i] not in special)


def ngrams(text: str, n: int) -> list[str]:
    return [text[i : i + n] for i in range(max(0, len(text) - n + 1))]


def diversity(texts: list[str]) -> dict[str, float]:
    joined = "\n".join(texts)
    result = {}
    for n in (1, 2, 3):
        grams = ngrams(joined, n)
        result[f"distinct_{n}"] = len(set(grams)) / max(len(grams), 1)
    grams4 = ngrams(joined, 4)
    counts = Counter(grams4)
    result["repeated_4gram_rate"] = sum(count - 1 for count in counts.values() if count > 1) / max(len(grams4), 1)
    return result


def update_metrics(run_id: str, values: dict) -> None:
    path = MEMBER_DIR / "metrics_report.csv"
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames, rows = reader.fieldnames, list(reader)
    found = False
    for row in rows:
        if row["run_id"] == run_id:
            row.update({key: value for key, value in values.items() if key in fieldnames})
            found = True
    if not found:
        raise ValueError(f"No metrics row for {run_id}; run train.py first")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="best_model.pt")
    args = parser.parse_args()
    config = load_config(args.config)
    run_id = config["run_id"]
    paths = output_paths(run_id)
    checkpoint_path = paths["checkpoints"] / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    device = select_device(config["training"]["device"])
    set_seed(int(config["data"]["seed"]) + 2)
    model = CharacterGPT(metadata["vocab_size"], metadata["context_length"], checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model_state"]); model.to(device).eval()
    vocab = metadata["char_to_idx"]
    inverse = {int(k): value for k, value in metadata["idx_to_char"].items()}
    records, generated_only = [], []
    modes = [("greedy", None)] + [(f"temperature_{t}", float(t)) for t in config["generation"]["temperatures"]]
    total_tokens = 0
    started = time.perf_counter()
    for prompt in config["generation"]["prompts"]:
        prompt_ids = [vocab["<BOS>"]] + [vocab.get(char, vocab["<UNK>"]) for char in prompt]
        for mode, temperature in modes:
            before = time.perf_counter()
            result = model.generate(torch.tensor([prompt_ids], device=device), int(config["generation"]["max_new_tokens"]), temperature, vocab["<EOS>"])
            elapsed = time.perf_counter() - before
            new_count = result.shape[1] - len(prompt_ids)
            text = decode(result[0].tolist(), inverse)
            generated = text[len(prompt) :]
            generated_only.append(generated)
            total_tokens += new_count
            records.append({"run_id": run_id, "prompt": prompt, "mode": mode, "temperature": temperature, "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)), "generated_text": text, "new_tokens": new_count, "generation_seconds": elapsed, "tokens_per_second": new_count / max(elapsed, 1e-9)})
    output = paths["generated"] / f"{run_id}_samples.jsonl"
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n", encoding="utf-8")
    readable = paths["generated"] / f"{run_id}_samples.txt"
    readable.write_text("\n\n".join(f"[{r['mode']}] Prompt: {r['prompt']}\n{r['generated_text']}" for r in records) + "\n", encoding="utf-8")
    values = diversity(generated_only)
    values["gen_tokens_per_sec"] = total_tokens / max(time.perf_counter() - started, 1e-9)
    update_metrics(run_id, values)
    print(json.dumps({"output": str(output.relative_to(REPO_ROOT)), **values}, indent=2))


if __name__ == "__main__":
    main()
