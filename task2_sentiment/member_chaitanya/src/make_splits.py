"""Write the HF row indices of the team split (Aswin's split, reproduced from his code) plus the 33K complement.

Aswin's split (task2_sentiment/member_aswin/src/dataset.py, seed 42, scikit-learn stratified train_test_split):
  HF train (560K) -> 25K subsample -> 80/20 -> train 20K / val 5K
  HF test  (38K)  -> 5K subsample = test5k (team comparison set)
Our extra evaluation set test33k = HF test rows NOT in test5k (independent of the team test set).

The split is verified against Aswin's committed predictions (texts and labels, in order) and against his
reported vocabulary size (18,766 with his tokenizer, min_freq 3) before the index files are written.
"""
import collections
import hashlib
import sys

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from common import ASWIN_DIR, SPLITS_DIR, write_json

SEED = 42


def sha(texts) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def main() -> None:
    ds = load_dataset("fancyzhx/yelp_polarity")
    tr = pd.DataFrame(ds["train"])
    te = pd.DataFrame(ds["test"])
    tr25, _ = train_test_split(tr, train_size=25000, stratify=tr["label"], random_state=SEED)
    te5, _ = train_test_split(te, train_size=5000, stratify=te["label"], random_state=SEED)
    trs, vas = train_test_split(tr25, test_size=0.2, stratify=tr25["label"], random_state=SEED)
    te33 = te.drop(index=te5.index)

    # --- verification against Aswin's committed artefacts ---
    pred = pd.read_csv(ASWIN_DIR / "outputs" / "predictions" / "baseline_test_predictions.csv")
    ok_text = int((pred["review_text"].values == te5["text"].values).sum())
    ok_label = int((pred["true_label"].values == te5["label"].values).sum())
    sys.path.insert(0, str(ASWIN_DIR / "src"))
    from dataset import YelpTokenizer  # Aswin's tokenizer, used only for the fingerprint

    tok = YelpTokenizer(remove_stopwords=True)
    counts = collections.Counter(t for x in trs["text"] for t in tok.tokenize(x))
    vocab = 2 + min(29998, sum(1 for n in counts.values() if n >= 3))
    overlap = len(set(tr25["text"]) & set(te["text"]))
    checks = {
        "test5k_text_matches_aswin_predictions_in_order": f"{ok_text}/5000",
        "test5k_label_matches": f"{ok_label}/5000",
        "aswin_vocab_fingerprint": f"{vocab} (Aswin reports 18766)",
        "text_overlap_train25k_vs_hf_test": overlap,
    }
    print(checks)
    assert ok_text == 5000 and ok_label == 5000 and vocab == 18766 and overlap == 0, "split does not match Aswin's"

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"source": "fancyzhx/yelp_polarity", "seed": SEED, "verification": checks, "splits": {}}
    for name, hf_split, df in (("train", "train", trs), ("val", "train", vas), ("test5k", "test", te5), ("test33k", "test", te33)):
        idx = [int(i) for i in df.index]
        write_json(SPLITS_DIR / f"{name}_indices.json", {"hf_split": hf_split, "order": "as used", "indices": idx})
        meta["splits"][name] = {
            "hf_split": hf_split, "n": len(idx), "n_pos": int((df["label"] == 1).sum()),
            "text_sha256": sha(df["text"].tolist()),
        }
    write_json(SPLITS_DIR / "split_manifest.json", meta)
    print({k: v["n"] for k, v in meta["splits"].items()})


if __name__ == "__main__":
    main()
