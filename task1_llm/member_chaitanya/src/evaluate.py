"""Task 1 evaluation: loss metrics, generation, diversity, plots -> metrics_report.csv.

Usage:
    python task1_llm/member_chaitanya/src/evaluate.py --config task1_llm/member_chaitanya/configs/full.yaml [--checkpoint final]

Metric definitions
- train / val CE loss: mean next-char cross-entropy (nats) in eval mode over
  non-overlapping block_size windows of the full train / val stream.
- perplexity = exp(val loss); bits-per-character = val loss / ln 2.
- generalization gap = val loss - train loss.
- top-1 accuracy: fraction of val positions where argmax(logits) == next char.
- distinct-n: unique word n-grams / total word n-grams, pooled over all samples
  generated at one decoding setting (words = lowercase alphabetic tokens).
- repeated 4-gram rate: per sample, fraction of word 4-grams that already
  occurred earlier in the same sample; averaged over samples.
- generation tokens/sec: generated characters / wall time, batch size 1.
"""
import argparse
import csv
import json
import math
import re
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from common import MEMBER_DIR, load_config, load_vocab, run_paths, set_seed
from dataset import CharWindowDataset
from model import CharGPT
from train import evaluate_loss

CSV_COLUMNS = ["run_id", "checkpoint", "train_ce_loss", "val_ce_loss", "perplexity", "bits_per_char", "generalization_gap",
               "top1_next_char_acc", "distinct_1", "distinct_2", "distinct_3", "repeated_4gram_rate", "max_grad_norm",
               "mean_grad_norm", "loss_spikes", "nan_count", "param_count", "train_tokens_per_sec", "gen_tokens_per_sec",
               "peak_memory_mb", "total_train_time_s", "hardware"]


def words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts: list[str], n: int) -> float:
    grams = [g for t in texts for g in ngrams(words(t), n)]
    return len(set(grams)) / len(grams) if grams else float("nan")


def repeated_4gram_rate(texts: list[str]) -> float:
    rates = []
    for t in texts:
        grams = ngrams(words(t), 4)
        if not grams:
            continue
        seen, repeats = set(), 0
        for g in grams:
            repeats += g in seen
            seen.add(g)
        rates.append(repeats / len(grams))
    return float(np.mean(rates)) if rates else float("nan")


def plot_curves(history: dict, out_dir, run_id: str) -> None:
    steps = history["steps"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.plot([s["step"] for s in steps], [s["loss"] for s in steps], lw=0.6, alpha=0.6, label="train loss (per step)")
    if history["evals"]:
        ax.plot([e["step"] for e in history["evals"]], [e["val_loss_partial"] for e in history["evals"]], "o-", ms=2,
                label="val loss (partial eval)")
    ax.set(xlabel="step", ylabel="cross-entropy (nats)", title=f"{run_id}: loss vs step")
    ax.legend()
    ax = axes[1]
    ep = history["epochs"]
    ax.plot([e["epoch"] for e in ep], [e["train_loss_eval"] for e in ep], "o-", label="train loss (eval mode)")
    ax.plot([e["epoch"] for e in ep], [e["val_loss"] for e in ep], "s-", label="val loss")
    ax.set(xlabel="epoch", ylabel="cross-entropy (nats)", title=f"{run_id}: loss vs epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot([s["step"] for s in steps], [s["grad_norm"] for s in steps], lw=0.6)
    axes[0].set(xlabel="step", ylabel="grad norm (pre-clip)", title="gradient norm", yscale="log")
    axes[1].plot([s["step"] for s in steps], [s["lr"] for s in steps])
    axes[1].set(xlabel="step", ylabel="learning rate", title="LR schedule (warm-up + cosine)")
    fig.tight_layout()
    fig.savefig(out_dir / "stability.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="final", choices=["final", "best"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    d_cfg, g_cfg, t_cfg = cfg["data"], cfg["generation"], cfg["training"]
    run_id = cfg["run_id"]
    paths = run_paths(run_id)
    set_seed(int(cfg["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = paths["checkpoints"] / f"{args.checkpoint}.pt"
    state = torch.load(ckpt_path, map_location=device)
    processed = paths["processed"] / d_cfg["processed_name"]
    char_to_idx, idx_to_char = load_vocab(processed)
    block = int(d_cfg["block_size"])
    model = CharGPT(cfg["model"], len(char_to_idx), block).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    bs = int(t_cfg["batch_size"]) * 2
    precision = t_cfg["precision"]
    train_eval = CharWindowDataset(np.load(processed / "train.npy"), block, block)
    val_eval = CharWindowDataset(np.load(processed / "val.npy"), block, block)
    tr = evaluate_loss(model, DataLoader(train_eval, batch_size=bs), device, precision)
    va = evaluate_loss(model, DataLoader(val_eval, batch_size=bs), device, precision)

    eos_id = char_to_idx[d_cfg["eos_token"]]
    decode = lambda ids: "".join(idx_to_char[i] for i in ids if i != eos_id)
    gen = torch.Generator(device=device).manual_seed(int(cfg["seed"]))
    samples, per_setting, gen_chars, gen_time = [], {}, 0, 0.0
    for temp in g_cfg["temperatures"]:
        texts = []
        n_samples = 1 if temp == 0 else int(g_cfg["samples_per_setting"])  # greedy is deterministic
        for prompt in g_cfg["prompts"]:
            prompt_ids = torch.tensor([[char_to_idx[c] for c in prompt]], device=device)
            for k in range(n_samples):
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = model.generate(prompt_ids, int(g_cfg["max_new_chars"]), float(temp), eos_id, gen)
                if device == "cuda":
                    torch.cuda.synchronize()
                gen_time += time.perf_counter() - t0
                new_ids = out[0, prompt_ids.size(1) :].tolist()
                gen_chars += len(new_ids)
                text = decode(new_ids)
                texts.append(text)
                samples.append({"temperature": temp, "decoding": "greedy" if temp == 0 else f"temperature={temp}",
                                "prompt": prompt, "sample": k, "ended_with_eos": eos_id in new_ids,
                                "generated_chars": len(new_ids), "text": prompt + text})
        per_setting[temp] = {"temperature": temp, "num_samples": len(texts), "distinct_1": distinct_n(texts, 1),
                             "distinct_2": distinct_n(texts, 2), "distinct_3": distinct_n(texts, 3),
                             "repeated_4gram_rate": repeated_4gram_rate(texts)}

    out_dir = paths["outputs"]
    with open(out_dir / "samples.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(out_dir / "samples.txt", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(f"=== {s['decoding']} | prompt: {s['prompt']!r} | sample {s['sample']} | "
                    f"ended_with_eos={s['ended_with_eos']} ===\n{s['text']}\n\n")
    with open(out_dir / "generation_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(per_setting.values())).keys()))
        w.writeheader()
        w.writerows(per_setting.values())

    history = json.loads((out_dir / "history.json").read_text())
    plot_curves(history, out_dir, run_id)
    summary = json.loads((out_dir / "train_summary.json").read_text())

    default = per_setting[g_cfg["default_temperature"]]
    row = {
        "run_id": run_id,
        "checkpoint": f"checkpoints/{run_id}/{args.checkpoint}.pt (epoch {state['epoch']}, step {state['step']})",
        "train_ce_loss": tr["loss"], "val_ce_loss": va["loss"], "perplexity": math.exp(va["loss"]),
        "bits_per_char": va["loss"] / math.log(2), "generalization_gap": va["loss"] - tr["loss"],
        "top1_next_char_acc": va["acc"], "distinct_1": default["distinct_1"], "distinct_2": default["distinct_2"],
        "distinct_3": default["distinct_3"], "repeated_4gram_rate": default["repeated_4gram_rate"],
        "max_grad_norm": summary["max_grad_norm"], "mean_grad_norm": summary["mean_grad_norm"],
        "loss_spikes": summary["loss_spikes"], "nan_count": summary["nan_count"], "param_count": summary["param_count"],
        "train_tokens_per_sec": summary["train_tokens_per_sec"], "gen_tokens_per_sec": gen_chars / gen_time,
        "peak_memory_mb": summary["peak_memory_mb"], "total_train_time_s": summary["total_train_time_s"],
        "hardware": summary["hardware"],
    }
    # smoke runs never go into the reported metrics table
    report = out_dir / "metrics_row.csv" if cfg.get("smoke") else MEMBER_DIR / "metrics_report.csv"
    rows = []
    if report.exists():
        with open(report, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get("run_id") and r["run_id"] != run_id]
    rows.append(row)
    with open(report, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    (out_dir / "eval_summary.json").write_text(json.dumps({"metrics": row, "per_temperature": list(per_setting.values()),
                                                           "train_eval_tokens": tr["tokens"], "val_eval_tokens": va["tokens"]}, indent=2))
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
