"""Uncounted reference row: TF-IDF (word uni+bigrams on our cleaned tokens) + logistic regression.

Not one of the three models (no learned embeddings); reported only as a strong, cheap floor.
C is chosen on validation macro-F1 from a small grid; threshold fixed at 0.5.
"""
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from common import EVAL_SETS, LOG_DIR, OUTPUTS_DIR, PROCESSED_DIR, get_logger, hardware_string, write_json

RUN_ID = "t2_ref_tfidf_lr"


def load(name):
    with open(PROCESSED_DIR / f"{name}_tokens.pkl", "rb") as f:
        docs = [" ".join(t for s in r for t in s) for r in pickle.load(f)]
    z = np.load(PROCESSED_DIR / f"{name}.npz")
    return docs, z["labels"], z["hf_index"]


def main() -> None:
    log = get_logger(LOG_DIR / f"{RUN_ID}.log", RUN_ID)
    out = OUTPUTS_DIR / "runs" / RUN_ID
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    x_tr, y_tr, _ = load("train")
    sets = {n: load(n) for n in EVAL_SETS}
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"\S+")
    X_tr = vec.fit_transform(x_tr)
    Xs = {n: vec.transform(d) for n, (d, _, _) in sets.items()}
    best = None
    for C in (0.25, 1.0, 4.0, 16.0, 64.0, 256.0):
        clf = LogisticRegression(C=C, max_iter=2000).fit(X_tr, y_tr)
        f1 = f1_score(sets["val"][1], clf.predict(Xs["val"]), average="macro")
        log.info(f"C={C} val_macro_f1={f1:.4f}")
        if best is None or f1 > best[0]:
            best = (f1, C, clf)
    f1, C, clf = best
    train_time = time.time() - t0
    for n, (_, y, hf) in sets.items():
        p = clf.predict_proba(Xs[n])[:, 1]
        pd.DataFrame({"hf_index": hf, "label": y, "prob_pos": p, "pred": (p >= 0.5).astype(int)}).to_csv(
            out / f"pred_{n}.csv", index=False)
    summary = {"run_id": RUN_ID, "model": "reference_tfidf_logreg", "seed": None, "C": C, "best_val_macro_f1": f1,
               "param_count": int(clf.coef_.size + 1), "n_features": int(X_tr.shape[1]),
               "train_time_s": train_time, "train_examples_per_sec": len(x_tr) / train_time,
               "peak_memory_mb": None, "hardware": hardware_string().split("; ")[-1] + " (CPU, scikit-learn)"}
    write_json(out / "train_summary.json", summary)
    log.info(f"done summary={summary}")


if __name__ == "__main__":
    main()
