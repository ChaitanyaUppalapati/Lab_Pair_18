"""Full Task 2 evaluation from saved predictions (threshold fixed at 0.5, never tuned on test).

Per model (seed-42 run) and per test set (test5k = team set, test33k = our extra set):
  accuracy; precision / recall / F1 (macro, micro, weighted); confusion matrix; ROC-AUC; PR-AUC; MCC; Brier;
  ECE (15 equal-width confidence bins); 95% bootstrap CIs (2,000 resamples) for accuracy, macro-F1 and MCC;
  McNemar (baseline vs each experimental model); macro-F1 and error rate per slice; params, time, examples/sec,
  peak memory, hardware.
Seed variance: mean +- std over seeds 42/43/44 (training variance), separate from the bootstrap (test-sample
variance). Team: Aswin's committed test5k predictions scored with the same code, McNemar vs his three models.

Writes metrics_report.csv, outputs/{seed_summary.csv, slice_metrics.csv, mcnemar.csv, team_comparison.csv,
team_mcnemar.csv, evaluation_summary.json} and plots in outputs/plots/.
"""
import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_curve, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)
from statsmodels.stats.contingency_tables import mcnemar

from common import ASWIN_DIR, MEMBER_DIR, OUTPUTS_DIR, PROCESSED_DIR, load_config, read_json, write_json

MODELS = [("baseline", "Baseline (fastText bigram)"), ("experimental_1", "Exp 1 (HAN)"),
          ("experimental_2", "Exp 2 (Transformer)")]
REF = ("t2_ref_tfidf_lr", "Reference: TF-IDF + LR (uncounted)")
TEST_SETS = ("test5k", "test33k")
MAIN_SEED = 42
N_BOOT = 2000
MIN_SLICE = load_config("task2_sentiment/member_chaitanya/configs/slices.yaml")["min_slice_size"]
ASWIN_MODELS = [("baseline", "Aswin baseline (mean-pool FFN)"), ("experimental_1", "Aswin exp 1 (BiGRU)"),
                ("experimental_2", "Aswin exp 2 (TextCNN)")]
PLOTS = OUTPUTS_DIR / "plots"


def ece_score(y, p, bins=15):
    conf = np.maximum(p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def point_metrics(y, p):
    pred = (p >= 0.5).astype(int)
    out = {"accuracy": accuracy_score(y, pred)}
    for avg in ("macro", "micro", "weighted"):
        pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average=avg, zero_division=0)
        out.update({f"precision_{avg}": pr, f"recall_{avg}": rc, f"f1_{avg}": f1})
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
                "mcc": matthews_corrcoef(y, pred), "brier": brier_score_loss(y, p), "ece": ece_score(y, p)})
    return out


def fast_binary_stats(y, pred):
    """Vectorised accuracy, macro-F1 and MCC over bootstrap rows (y, pred: [B, N])."""
    tp = ((pred == 1) & (y == 1)).sum(1).astype(float)
    tn = ((pred == 0) & (y == 0)).sum(1).astype(float)
    fp = ((pred == 1) & (y == 0)).sum(1).astype(float)
    fn = ((pred == 0) & (y == 1)).sum(1).astype(float)
    n = tp + tn + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_pos = np.nan_to_num(2 * tp / (2 * tp + fp + fn))
        f1_neg = np.nan_to_num(2 * tn / (2 * tn + fn + fp))
        mcc = np.nan_to_num((tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return (tp + tn) / n, (f1_pos + f1_neg) / 2, mcc


def bootstrap_ci(y, p, seed=0):
    rng = np.random.default_rng(seed)
    pred = (p >= 0.5).astype(int)
    res = {"accuracy": [], "f1_macro": [], "mcc": []}
    for chunk in range(0, N_BOOT, 250):
        idx = rng.integers(0, len(y), size=(min(250, N_BOOT - chunk), len(y)))
        a, f, m = fast_binary_stats(y[idx], pred[idx])
        res["accuracy"].append(a)
        res["f1_macro"].append(f)
        res["mcc"].append(m)
    return {k: tuple(np.percentile(np.concatenate(v), [2.5, 97.5])) for k, v in res.items()}


def mcnemar_test(y, pred_a, pred_b):
    a_ok, b_ok = pred_a == y, pred_b == y
    b = int((a_ok & ~b_ok).sum())   # A right, B wrong
    c = int((~a_ok & b_ok).sum())   # A wrong, B right
    table = [[int((a_ok & b_ok).sum()), b], [c, int((~a_ok & ~b_ok).sum())]]
    exact = (b + c) < 25
    r = mcnemar(table, exact=exact, correction=not exact)
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "statistic": float(r.statistic), "p_value": float(r.pvalue),
            "test": "exact binomial" if exact else "chi2 with continuity correction"}


def load_pred(run_id, name):
    return pd.read_csv(OUTPUTS_DIR / "runs" / run_id / f"pred_{name}.csv")


def slice_table(label, y, pred, slices):
    rows = []
    for col in slices.columns[1:]:
        m = slices[col].to_numpy(bool)
        n = int(m.sum())
        row = {"model": label, "slice": col, "n": n, "n_pos": int(y[m].sum()) if n else 0}
        if n >= MIN_SLICE:
            row["macro_f1"] = f1_score(y[m], pred[m], average="macro", zero_division=0)
            row["error_rate"] = float((y[m] != pred[m]).mean())
        else:
            row["macro_f1"] = row["error_rate"] = np.nan
            row["note"] = f"n < {MIN_SLICE}: not reported"
        rows.append(row)
    return rows


def plots(curves, histories):
    PLOTS.mkdir(parents=True, exist_ok=True)
    for name in TEST_SETS:
        fig, ax = plt.subplots(1, 3, figsize=(17, 5))
        for label, (y, p) in curves[name].items():
            fpr, tpr, _ = roc_curve(y, p)
            ax[0].plot(fpr, tpr, label=f"{label} (AUC {roc_auc_score(y, p):.4f})")
            pr, rc, _ = precision_recall_curve(y, p)
            ax[1].plot(rc, pr, label=f"{label} (AP {average_precision_score(y, p):.4f})")
            conf_bins = np.linspace(0, 1, 16)
            idx = np.clip(np.digitize(p, conf_bins[1:-1]), 0, 14)
            xs = [p[idx == b].mean() for b in range(15) if (idx == b).sum() > 20]
            ys = [y[idx == b].mean() for b in range(15) if (idx == b).sum() > 20]
            ax[2].plot(xs, ys, marker="o", ms=3, label=f"{label} (ECE {ece_score(y, p):.4f})")
        ax[0].plot([0, 1], [0, 1], "k:", lw=0.8)
        ax[2].plot([0, 1], [0, 1], "k:", lw=0.8)
        ax[0].set(title=f"ROC ({name})", xlabel="false positive rate", ylabel="true positive rate")
        ax[1].set(title=f"Precision-recall ({name})", xlabel="recall", ylabel="precision", ylim=(0.5, 1.01))
        ax[2].set(title=f"Reliability diagram ({name})", xlabel="mean predicted P(positive)",
                  ylabel="observed positive rate")
        for a in ax:
            a.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(PLOTS / f"roc_pr_reliability_{name}.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(1, len(curves[name]), figsize=(4 * len(curves[name]), 3.6))
        for a, (label, (y, p)) in zip(ax, curves[name].items()):
            cm = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1])
            a.imshow(cm, cmap="Blues")
            for i, j in itertools.product(range(2), range(2)):
                a.text(j, i, f"{cm[i, j]}\n{cm[i, j] / cm[i].sum():.1%}", ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
            a.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["neg", "pos"], yticklabels=["neg", "pos"],
                  xlabel="predicted", ylabel="true", title=label)
            a.title.set_fontsize(8)
        fig.tight_layout()
        fig.savefig(PLOTS / f"confusion_matrices_{name}.png", dpi=120)
        plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for k, (label, runs) in enumerate(histories.items()):
        for seed, h in runs.items():
            e = [r["epoch"] for r in h]
            ax[k].plot(e, [r["train_loss"] for r in h], color=f"C{seed - 42}", ls="--", label=f"train loss s{seed}")
            ax[k].plot(e, [r["val_loss"] for r in h], color=f"C{seed - 42}", label=f"val loss s{seed}")
        ax2 = ax[k].twinx()
        for seed, h in runs.items():
            ax2.plot([r["epoch"] for r in h], [r["val_macro_f1"] for r in h], color=f"C{seed - 42}", ls=":", marker=".")
        ax2.set_ylabel("val macro-F1 (dotted)")
        ax[k].set(title=label, xlabel="epoch", ylabel="BCE loss")
        ax[k].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOTS / "training_curves.png", dpi=120)
    plt.close(fig)


def main() -> None:
    report, seed_rows, slice_rows, mc_rows, team_rows, team_mc = [], [], [], [], [], []
    curves = {n: {} for n in TEST_SETS}
    histories, summaries = {}, {}
    slices = {n: pd.read_csv(PROCESSED_DIR / f"slices_{n}.csv") for n in TEST_SETS}
    entries = []
    for key, label in MODELS:
        cfg = load_config(f"task2_sentiment/member_chaitanya/configs/{key}.yaml")
        entries.append((key, label, cfg["run_prefix"], cfg["seeds"]))
        histories[label] = {s: read_json(OUTPUTS_DIR / "runs" / f"{cfg['run_prefix']}_s{s}" / "history.json")
                            for s in cfg["seeds"]}
    preds = {}
    for key, label, prefix, seeds in entries + [("reference", REF[1], None, [None])]:
        run_main = REF[0] if prefix is None else f"{prefix}_s{MAIN_SEED}"
        ts = read_json(OUTPUTS_DIR / "runs" / run_main / "train_summary.json")
        summaries[label] = ts
        for name in TEST_SETS:
            df = load_pred(run_main, name)
            y, p = df["label"].to_numpy(), df["prob_pos"].to_numpy()
            assert (df["hf_index"].to_numpy() == slices[name]["hf_index"].to_numpy()).all()
            preds[(label, name)] = (y, p)
            curves[name][label] = (y, p)
            m = point_metrics(y, p)
            ci = bootstrap_ci(y, p)
            row = {"model": label, "run_id": run_main, "checkpoint": ts.get("checkpoint", "n/a (scikit-learn)"),
                   "eval_set": name, **m,
                   "acc_ci_low": ci["accuracy"][0], "acc_ci_high": ci["accuracy"][1],
                   "f1_macro_ci_low": ci["f1_macro"][0], "f1_macro_ci_high": ci["f1_macro"][1],
                   "mcc_ci_low": ci["mcc"][0], "mcc_ci_high": ci["mcc"][1],
                   "param_count": ts["param_count"], "train_time_s": ts["train_time_s"],
                   "examples_per_sec": ts["train_examples_per_sec"],
                   "inference_examples_per_sec": ts.get("inference_examples_per_sec"),
                   "peak_memory_mb": ts["peak_memory_mb"], "hardware": ts["hardware"],
                   "best_epoch": ts.get("best_epoch"), "epochs_run": ts.get("epochs_run")}
            report.append(row)
            slice_rows += [{**r, "eval_set": name} for r in
                           slice_table(label, y, (p >= 0.5).astype(int), slices[name])]
        if prefix is not None:  # seed variance
            for name in TEST_SETS:
                vals = []
                for s in seeds:
                    df = load_pred(f"{prefix}_s{s}", name)
                    y, p = df["label"].to_numpy(), df["prob_pos"].to_numpy()
                    pm = point_metrics(y, p)
                    vals.append({k: pm[k] for k in ("accuracy", "f1_macro", "mcc", "roc_auc", "pr_auc", "brier", "ece")})
                v = pd.DataFrame(vals)
                seed_rows.append({"model": label, "eval_set": name, "seeds": ",".join(map(str, seeds)),
                                  **{f"{k}_mean": v[k].mean() for k in v}, **{f"{k}_std": v[k].std(ddof=1) for k in v},
                                  "best_epochs": ",".join(str(read_json(OUTPUTS_DIR / "runs" / f"{prefix}_s{s}" /
                                                                        "train_summary.json")["best_epoch"]) for s in seeds)})

    base_label = MODELS[0][1]
    for name in TEST_SETS:
        yb, pb = preds[(base_label, name)]
        for _, label in MODELS[1:] + [REF]:
            y, p = preds[(label, name)]
            r = mcnemar_test(y, (pb >= 0.5).astype(int), (p >= 0.5).astype(int))
            mc_rows.append({"eval_set": name, "model_a": base_label, "model_b": label, **r})
            for row in report:
                if row["model"] == label and row["eval_set"] == name:
                    row["mcnemar_vs_baseline_stat"], row["mcnemar_vs_baseline_p"] = r["statistic"], r["p_value"]
    for row in report:
        row.setdefault("mcnemar_vs_baseline_stat", "")
        row.setdefault("mcnemar_vs_baseline_p", "")

    # ---- team comparison on test5k (Aswin's committed predictions, same rows in the same order)
    y5 = preds[(base_label, "test5k")][0]
    team_preds = {label: preds[(label, "test5k")] for _, label in MODELS}
    for key, label in ASWIN_MODELS:
        df = pd.read_csv(ASWIN_DIR / "outputs" / "predictions" / f"{key}_test_predictions.csv")
        assert (df["true_label"].to_numpy() == y5).all(), "Aswin's rows are not aligned with test5k"
        team_preds[label] = (y5, df["prob_positive"].to_numpy())
        slice_rows += [{**r, "eval_set": "test5k"} for r in
                       slice_table(label, y5, (df["prob_positive"].to_numpy() >= 0.5).astype(int), slices["test5k"])]
    for label, (y, p) in team_preds.items():
        m = point_metrics(y, p)
        ci = bootstrap_ci(y, p)
        team_rows.append({"member": "aswin" if label.startswith("Aswin") else "chaitanya", "model": label,
                          **{k: m[k] for k in ("accuracy", "f1_macro", "mcc", "roc_auc", "pr_auc", "brier", "ece",
                                               "fp", "fn")},
                          "acc_ci": f"[{ci['accuracy'][0]:.4f}, {ci['accuracy'][1]:.4f}]",
                          "f1_macro_ci": f"[{ci['f1_macro'][0]:.4f}, {ci['f1_macro'][1]:.4f}]",
                          "mcc_ci": f"[{ci['mcc'][0]:.4f}, {ci['mcc'][1]:.4f}]"})
    for (_, mine), (_, his) in itertools.product(MODELS, ASWIN_MODELS):
        ya, pa = team_preds[mine]
        _, ph = team_preds[his]
        team_mc.append({"model_a": mine, "model_b": his,
                        **mcnemar_test(ya, (pa >= 0.5).astype(int), (ph >= 0.5).astype(int))})
    tm = pd.DataFrame(team_mc)
    tm["p_holm"] = holm(tm["p_value"].to_numpy())

    # ---- write
    cols = ["model", "run_id", "checkpoint", "eval_set", "accuracy", "acc_ci_low", "acc_ci_high", "precision_macro",
            "recall_macro", "f1_macro", "f1_macro_ci_low", "f1_macro_ci_high", "precision_micro", "recall_micro",
            "f1_micro", "precision_weighted", "recall_weighted", "f1_weighted", "tn", "fp", "fn", "tp", "roc_auc",
            "pr_auc", "mcc", "mcc_ci_low", "mcc_ci_high", "brier", "ece", "mcnemar_vs_baseline_stat",
            "mcnemar_vs_baseline_p", "param_count", "train_time_s", "examples_per_sec", "peak_memory_mb", "hardware",
            "inference_examples_per_sec", "best_epoch", "epochs_run"]
    pd.DataFrame(report)[cols].to_csv(MEMBER_DIR / "metrics_report.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(OUTPUTS_DIR / "seed_summary.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(OUTPUTS_DIR / "slice_metrics.csv", index=False)
    pd.DataFrame(mc_rows).to_csv(OUTPUTS_DIR / "mcnemar.csv", index=False)
    pd.DataFrame(team_rows).to_csv(OUTPUTS_DIR / "team_comparison.csv", index=False)
    tm.to_csv(OUTPUTS_DIR / "team_mcnemar.csv", index=False)
    plots(curves, histories)
    write_json(OUTPUTS_DIR / "evaluation_summary.json",
               {"threshold": 0.5, "bootstrap_resamples": N_BOOT, "ece_bins": 15, "main_seed": MAIN_SEED,
                "min_slice_size": MIN_SLICE, "train_summaries": summaries})
    show = pd.DataFrame(report)[["model", "eval_set", "accuracy", "acc_ci_low", "acc_ci_high", "f1_macro", "mcc",
                                 "roc_auc", "brier", "ece", "mcnemar_vs_baseline_p"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(pd.DataFrame(seed_rows)[["model", "eval_set", "accuracy_mean", "accuracy_std", "f1_macro_mean",
                                   "f1_macro_std", "best_epochs"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(pd.DataFrame(team_rows)[["model", "accuracy", "f1_macro", "mcc", "roc_auc", "brier", "ece", "acc_ci"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def holm(p):
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


if __name__ == "__main__":
    main()
