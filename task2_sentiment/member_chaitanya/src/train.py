"""Train one Task 2 model for one seed, early-stop on validation macro-F1, restore the best epoch and write
P(positive) for val / test5k / test33k.

usage: python train.py --config configs/<model>.yaml --seed 42
outputs: checkpoints/<run_id>/best.pt, outputs/<run_id>/{history.json, train_summary.json, pred_<set>.csv}
run_id = <config run_prefix>_s<seed>
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from common import (CHECKPOINTS_DIR, EVAL_SETS, LOG_DIR, OUTPUTS_DIR, PROCESSED_DIR, get_logger, hardware_string,
                    load_config, read_json, set_seed, write_json)
from models import build_model, count_params


def load_set(name: str, field: str):
    z = np.load(PROCESSED_DIR / f"{name}.npz")
    return torch.from_numpy(z[field].astype(np.int64)), torch.from_numpy(z["labels"]).float(), z["hf_index"]


def batches(x, y, bs, shuffle, gen=None):
    idx = torch.randperm(len(x), generator=gen) if shuffle else torch.arange(len(x))
    for i in range(0, len(x), bs):
        j = idx[i:i + bs]
        yield x[j], y[j]


@torch.no_grad()
def predict(model, x, device, bs=512):
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        out.append(torch.sigmoid(model(x[i:i + bs].to(device, non_blocking=True))).float().cpu())
    return torch.cat(out).numpy()


def make_optimizer(model, t):
    """Weight decay on matrices (incl. embeddings); none on biases, LayerNorm gains or the attention context vector."""
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": t["weight_decay"]},
                              {"params": no_decay, "weight_decay": 0.0}], lr=t["lr"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = load_config(args.config)
    t = cfg["training"]
    run_id = f"{cfg['run_prefix']}_s{args.seed}"
    ckpt_dir, out_dir = CHECKPOINTS_DIR / run_id, OUTPUTS_DIR / "runs" / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(LOG_DIR / f"{run_id}.log", run_id)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False

    field = cfg["input"]  # "flat" or "hier"
    vocab_size = len(read_json(PROCESSED_DIR / "vocab.json")["itos"])
    x_tr, y_tr, _ = load_set("train", field)
    evals = {n: load_set(n, field) for n in EVAL_SETS}

    model = build_model(cfg, vocab_size).to(device)
    n_params = count_params(model)
    opt = make_optimizer(model, t)
    steps_per_epoch = (len(x_tr) + t["batch_size"] - 1) // t["batch_size"]
    total_steps = steps_per_epoch * t["max_epochs"]
    warmup = int(t.get("warmup_frac", 0.0) * total_steps)

    def lr_factor(step):
        if t.get("schedule", "constant") == "constant":
            return 1.0
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    log.info(f"run={run_id} config={cfg['_config_path']} seed={args.seed} device={device} hw={hardware_string()}")
    log.info(f"model={cfg['model']} params={n_params} vocab={vocab_size} train={len(x_tr)} input={field}{tuple(x_tr.shape[1:])}")
    log.info(f"training={t}")

    gen = torch.Generator().manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history, best_f1, best_epoch, bad, step = [], -1.0, 0, 0, 0
    train_time = 0.0
    t_start = time.time()
    for epoch in range(1, t["max_epochs"] + 1):
        model.train()
        t0 = time.time()
        tot, n = 0.0, 0
        for xb, yb in batches(x_tr, y_tr, t["batch_size"], True, gen):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = None
            if t.get("grad_clip"):
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            opt.step()
            sched.step()
            step += 1
            tot += loss.item() * len(yb)
            n += len(yb)
            if not np.isfinite(loss.item()):
                raise RuntimeError(f"non-finite loss at step {step}")
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_time += time.time() - t0
        p_val = predict(model, evals["val"][0], device)
        y_val = evals["val"][1].numpy()
        pred = (p_val >= 0.5).astype(int)
        val_loss = float(F.binary_cross_entropy(torch.tensor(p_val).clamp(1e-7, 1 - 1e-7), torch.tensor(y_val)))
        rec = {"epoch": epoch, "train_loss": tot / n, "val_loss": val_loss,
               "val_acc": accuracy_score(y_val, pred), "val_macro_f1": f1_score(y_val, pred, average="macro"),
               "lr": opt.param_groups[0]["lr"], "epoch_train_s": time.time() - t0}
        history.append(rec)
        improved = rec["val_macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch, bad = rec["val_macro_f1"], epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": cfg, "seed": args.seed,
                        "vocab_size": vocab_size}, ckpt_dir / "best.pt")
        else:
            bad += 1
        log.info(f"epoch={epoch} train_loss={rec['train_loss']:.4f} val_loss={val_loss:.4f} "
                 f"val_acc={rec['val_acc']:.4f} val_macro_f1={rec['val_macro_f1']:.4f} lr={rec['lr']:.2e} "
                 f"time_s={rec['epoch_train_s']:.1f}{' *best' if improved else ''}")
        if bad >= t["patience"]:
            log.info(f"early stop: no val macro-F1 gain for {t['patience']} epochs")
            break
    total_time = time.time() - t_start
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")

    state = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(state["model"])
    t0 = time.time()
    for name, (x, y, hf) in evals.items():
        p = predict(model, x, device)
        pd.DataFrame({"hf_index": hf, "label": y.numpy().astype(int), "prob_pos": p,
                      "pred": (p >= 0.5).astype(int)}).to_csv(out_dir / f"pred_{name}.csv", index=False)
    infer_s = time.time() - t0
    n_inf = sum(len(v[0]) for v in evals.values())
    epochs_run = len(history)
    summary = {
        "run_id": run_id, "model": cfg["model_name"], "config": cfg["_config_path"], "seed": args.seed,
        "param_count": n_params, "best_epoch": best_epoch, "epochs_run": epochs_run, "best_val_macro_f1": best_f1,
        "train_time_s": train_time, "total_time_s": total_time,
        "train_examples_per_sec": epochs_run * len(x_tr) / train_time,
        "inference_examples_per_sec": n_inf / infer_s, "peak_memory_mb": peak_mb,
        "hardware": hardware_string(), "torch": torch.__version__,
        "checkpoint": (ckpt_dir / "best.pt").relative_to(CHECKPOINTS_DIR.parent).as_posix(),
    }
    write_json(out_dir / "history.json", history)
    write_json(out_dir / "train_summary.json", summary)
    log.info(f"done summary={json.dumps(summary)}")


if __name__ == "__main__":
    main()
