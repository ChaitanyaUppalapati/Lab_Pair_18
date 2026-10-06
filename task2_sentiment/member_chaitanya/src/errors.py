"""Pull the 20 error-review candidates per model (seed-42 run, test5k = team test set, threshold 0.5):
  5 most confident false positives, 5 most confident false negatives, 5 errors nearest the threshold,
  5 errors from the model's worst slice of our slices (n >= min_slice_size; most confident errors there,
  not already picked).
Writes outputs/error_review/<model>_candidates.csv (full raw text + the cleaned tokens the model saw).
The error-type labels and the proposed fix are left for the human reviewer.
"""
import pickle

import pandas as pd
from datasets import load_dataset

from common import OUTPUTS_DIR, PROCESSED_DIR, SPLITS_DIR, load_config, read_json

SET = "test5k"
MODELS = ["baseline", "experimental_1", "experimental_2"]


def main() -> None:
    meta = read_json(SPLITS_DIR / f"{SET}_indices.json")
    texts = load_dataset("fancyzhx/yelp_polarity")[meta["hf_split"]].select(meta["indices"])["text"]
    with open(PROCESSED_DIR / f"{SET}_tokens.pkl", "rb") as f:
        cleaned = [" | ".join(" ".join(s) for s in r) for r in pickle.load(f)]
    slices = pd.read_csv(PROCESSED_DIR / f"slices_{SET}.csv")
    ours = [c for c in slices.columns if c.startswith("C: ")]
    min_n = load_config("task2_sentiment/member_chaitanya/configs/slices.yaml")["min_slice_size"]
    sm = pd.read_csv(OUTPUTS_DIR / "slice_metrics.csv")
    out_dir = OUTPUTS_DIR / "error_review"
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in MODELS:
        cfg = load_config(f"task2_sentiment/member_chaitanya/configs/{key}.yaml")
        df = pd.read_csv(OUTPUTS_DIR / "runs" / f"{cfg['run_prefix']}_s42" / f"pred_{SET}.csv")
        df["text"], df["cleaned_tokens"] = texts, cleaned
        df["slices"] = slices[ours].apply(lambda r: "; ".join(c[3:] for c in ours if r[c]), axis=1)
        err = df[df["label"] != df["pred"]].copy()
        err["margin"] = (err["prob_pos"] - 0.5).abs()
        picks = [
            ("Confident FP", err[err["pred"] == 1].nlargest(5, "prob_pos")),
            ("Confident FN", err[err["pred"] == 0].nsmallest(5, "prob_pos")),
        ]
        used = set(pd.concat([p for _, p in picks]).index)
        near = err.drop(index=list(used)).nsmallest(5, "margin")
        picks.append(("Near-threshold", near))
        used |= set(near.index)
        rows = sm[(sm["eval_set"] == SET) & (sm["model"].str.startswith(
            {"baseline": "Baseline", "experimental_1": "Exp 1", "experimental_2": "Exp 2"}[key]))
                  & sm["slice"].isin(ours) & (sm["n"] >= min_n)]
        worst = rows.nsmallest(1, "macro_f1").iloc[0]
        in_slice = err[slices.loc[err.index, worst["slice"]].to_numpy(bool)].drop(index=list(used), errors="ignore")
        picks.append((f"Slice-specific ({worst['slice'][3:]}, macro-F1 {worst['macro_f1']:.3f}, n={worst['n']})",
                      in_slice.nlargest(5, "margin")))
        out = pd.concat([p.assign(category=c) for c, p in picks])
        out = out[["category", "hf_index", "label", "pred", "prob_pos", "slices", "text", "cleaned_tokens"]]
        out.insert(0, "n", range(1, len(out) + 1))
        out.to_csv(out_dir / f"{key}_candidates.csv", index=False)
        print(key, "worst slice:", worst["slice"], f"{worst['macro_f1']:.3f}", "errors:", len(err), "picked:", len(out))


if __name__ == "__main__":
    main()
