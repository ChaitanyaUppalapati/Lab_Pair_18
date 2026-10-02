"""Task 1.3 training: cross-entropy, linear warm-up + cosine decay, AdamW, grad clipping.

Usage:
    python task1_llm/member_chaitanya/src/train.py --config task1_llm/member_chaitanya/configs/full.yaml

Writes
    reproducibility/raw_logs/task1_llm/chaitanya/<run_id>.log   (raw log, append-only)
    task1_llm/member_chaitanya/checkpoints/<run_id>/{best,final}.pt
    task1_llm/member_chaitanya/outputs/<run_id>/history.json, train_summary.json
"""
import argparse
import json
import math
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from common import hardware_string, get_logger, load_config, load_vocab, run_paths, set_seed
from dataset import CharWindowDataset
from model import CharGPT, optimizer_param_groups


def lr_at(step: int, total_steps: int, cfg: dict) -> float:
    peak, floor, warmup = cfg["peak_lr"], cfg["min_lr"], cfg["warmup_steps"]
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))


def autocast_ctx(precision: str, device: str):
    if precision == "bf16" and device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if precision not in ("fp32", "bf16"):
        raise ValueError(f"training.precision must be 'fp32' or 'bf16', got {precision!r}")
    return nullcontext()


@torch.no_grad()
def evaluate_loss(model, loader, device, precision, max_batches=None) -> dict:
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast_ctx(precision, device):
            logits, loss = model(x, y)
        n = y.numel()
        total_loss += loss.item() * n
        total_correct += (logits.argmax(-1) == y).sum().item()
        total_tokens += n
    model.train()
    return {"loss": total_loss / total_tokens, "acc": total_correct / total_tokens, "tokens": total_tokens}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    d_cfg, m_cfg, t_cfg = cfg["data"], cfg["model"], cfg["training"]
    for key, value in (("data.train_stride", d_cfg["train_stride"]), ("model.activation", m_cfg["activation"]),
                       ("training.precision", t_cfg["precision"])):
        if value is None:
            raise SystemExit(f"Config value {key} is still unset in {cfg['_config_path']}")

    run_id = cfg["run_id"]
    paths = run_paths(run_id)
    log = get_logger(paths["raw_log"], f"train_{run_id}")
    set_seed(int(cfg["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processed = paths["processed"] / d_cfg["processed_name"]
    char_to_idx, _ = load_vocab(processed)
    vocab_size, block = len(char_to_idx), int(d_cfg["block_size"])
    train_ds = CharWindowDataset(np.load(processed / "train.npy"), block, int(d_cfg["train_stride"]))
    val_ds = CharWindowDataset(np.load(processed / "val.npy"), block, int(d_cfg["val_stride"]))
    # fixed train subset (same size as val) for an eval-mode train loss comparable to val loss
    sub_idx = np.random.default_rng(int(cfg["seed"])).choice(len(train_ds), size=min(len(val_ds), len(train_ds)), replace=False)
    train_eval_ds = Subset(train_ds, sub_idx.tolist())

    bs, workers = int(t_cfg["batch_size"]), int(t_cfg["num_workers"])
    g = torch.Generator().manual_seed(int(cfg["seed"]))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True, generator=g,
                              num_workers=workers, pin_memory=device == "cuda")
    val_loader = DataLoader(val_ds, batch_size=bs * 2, shuffle=False, num_workers=workers)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=bs * 2, shuffle=False, num_workers=workers)

    model = CharGPT(m_cfg, vocab_size, block).to(device)
    n_params = model.num_params()
    opt = torch.optim.AdamW(optimizer_param_groups(model, t_cfg["weight_decay"]), lr=t_cfg["peak_lr"],
                            betas=tuple(t_cfg["betas"]), fused=device == "cuda")
    epochs = int(t_cfg["epochs"])
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    precision = t_cfg["precision"]

    log.info("run_id=%s config=%s", run_id, cfg["_config_path"])
    log.info("config=%s", json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}))
    log.info("hardware=%s torch=%s", hardware_string(), torch.__version__)
    log.info("vocab_size=%d train_windows=%d val_windows=%d steps_per_epoch=%d total_steps=%d params=%d",
             vocab_size, len(train_ds), len(val_ds), steps_per_epoch, total_steps, n_params)

    history = {"steps": [], "evals": [], "epochs": []}
    recent, spikes, nan_count, grad_norms = [], 0, 0, []
    spike_factor, spike_window = float(t_cfg.get("spike_factor", 1.5)), int(t_cfg.get("spike_window", 100))
    best_val, step = float("inf"), 0
    train_compute_s, tokens_seen = 0.0, 0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    wall_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_loss_sum, epoch_batches = 0.0, 0
        for x, y in train_loader:
            t0 = time.perf_counter()
            lr = lr_at(step, total_steps, t_cfg)
            for group in opt.param_groups:
                group["lr"] = lr
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast_ctx(precision, device):
                _, loss = model(x, y)
            loss_val = loss.item()
            if not math.isfinite(loss_val):
                nan_count += 1
                log.warning("step=%d non-finite loss=%s; skipping update", step, loss_val)
                opt.zero_grad(set_to_none=True)
                step += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg["grad_clip"]).item()
            if not math.isfinite(grad_norm):
                nan_count += 1
                log.warning("step=%d non-finite grad_norm; skipping update", step)
                opt.zero_grad(set_to_none=True)
                step += 1
                continue
            opt.step()
            if device == "cuda":
                torch.cuda.synchronize()
            train_compute_s += time.perf_counter() - t0
            tokens_seen += y.numel()
            grad_norms.append(grad_norm)

            if step >= t_cfg["warmup_steps"] and len(recent) == spike_window and loss_val > spike_factor * (sum(recent) / spike_window):
                spikes += 1
                log.warning("step=%d loss spike: %.4f vs recent mean %.4f", step, loss_val, sum(recent) / spike_window)
            recent.append(loss_val)
            if len(recent) > spike_window:
                recent.pop(0)
            epoch_loss_sum += loss_val
            epoch_batches += 1

            if step % t_cfg["log_every_steps"] == 0:
                history["steps"].append({"step": step, "epoch": epoch, "loss": loss_val, "lr": lr, "grad_norm": grad_norm})
                log.info("step=%d epoch=%d loss=%.4f lr=%.3e grad_norm=%.3f train_tok_per_s=%.0f",
                         step, epoch, loss_val, lr, grad_norm, tokens_seen / max(train_compute_s, 1e-9))
            if step > 0 and step % t_cfg["eval_every_steps"] == 0:
                v = evaluate_loss(model, val_loader, device, precision, t_cfg["eval_batches"])
                history["evals"].append({"step": step, "epoch": epoch, "train_loss_running": sum(recent) / len(recent),
                                         "val_loss_partial": v["loss"]})
                log.info("step=%d eval val_loss(partial %d batches)=%.4f", step, t_cfg["eval_batches"], v["loss"])
            step += 1

        tr = evaluate_loss(model, train_eval_loader, device, precision)
        va = evaluate_loss(model, val_loader, device, precision)
        row = {"epoch": epoch, "step": step, "train_loss_running_mean": epoch_loss_sum / max(1, epoch_batches),
               "train_loss_eval": tr["loss"], "val_loss": va["loss"], "val_acc": va["acc"],
               "elapsed_s": time.perf_counter() - wall_start}
        history["epochs"].append(row)
        log.info("epoch=%d train_loss_running=%.4f train_loss_eval=%.4f val_loss=%.4f val_acc=%.4f elapsed_s=%.0f",
                 epoch, row["train_loss_running_mean"], tr["loss"], va["loss"], va["acc"], row["elapsed_s"])
        state = {"model": model.state_dict(), "config": cfg, "vocab_size": vocab_size, "epoch": epoch,
                 "step": step, "val_loss": va["loss"]}
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(state, paths["checkpoints"] / "best.pt")
            log.info("epoch=%d new best val_loss=%.4f -> checkpoints/%s/best.pt", epoch, best_val, run_id)
        (paths["outputs"] / "history.json").write_text(json.dumps(history, indent=1))

    torch.save(state, paths["checkpoints"] / "final.pt")
    total_s = time.perf_counter() - wall_start
    summary = {
        "run_id": run_id,
        "param_count": n_params,
        "vocab_size": vocab_size,
        "train_windows": len(train_ds),
        "total_steps": step,
        "epochs": epochs,
        "total_train_time_s": total_s,
        "train_compute_time_s": train_compute_s,
        "train_tokens_per_sec": tokens_seen / train_compute_s,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
        "max_grad_norm": max(grad_norms),
        "mean_grad_norm": float(np.mean(grad_norms)),
        "loss_spikes": spikes,
        "spike_rule": f"step loss > {spike_factor} x mean of previous {spike_window} step losses (after warm-up)",
        "nan_count": nan_count,
        "best_val_loss": best_val,
        "hardware": hardware_string(),
    }
    (paths["outputs"] / "train_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("done summary=%s", json.dumps(summary))


if __name__ == "__main__":
    main()
