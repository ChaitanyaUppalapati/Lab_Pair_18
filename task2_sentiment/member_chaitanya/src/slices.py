"""Tag every evaluation review with (a) our slices (configs/slices.yaml) and (b) Aswin's slices, computed with
Aswin's own code (member_aswin/src/slice_analysis.py: tag_slices) and his tokenizer/vocabulary rebuilt on the
same 20K training rows, so both members' models can be compared on identical slices.

Writes data_processed/slices_<set>.csv (one boolean column per slice, row order = split order).
"""
import re
import sys

import numpy as np
import pandas as pd
import yaml
from datasets import load_dataset

from common import ASWIN_DIR, EVAL_SETS, MEMBER_DIR, PROCESSED_DIR, SPLITS_DIR, read_json

CFG = yaml.safe_load(open(MEMBER_DIR / "configs" / "slices.yaml", encoding="utf-8"))


def our_slices(texts: pd.Series, n_clean: np.ndarray) -> pd.DataFrame:
    raw = texts.str.replace("\\n", "\n", regex=False).str.lower()
    words = raw.str.split().map(len)
    L = CFG["length_words"]
    neg = re.compile(CFG["negation_regex"])
    con = re.compile(CFG["contrast_regex"])
    rat = re.compile(CFG["rating_regex"])
    strong = set(CFG["strong_polarity_words"]["positive"]) | set(CFG["strong_polarity_words"]["negative"])
    tok = raw.map(lambda t: set(re.findall(r"[a-z]+", t)))
    return pd.DataFrame({
        "C: short (<=60 words)": words <= L["short"][1],
        "C: medium (61-180 words)": (words >= L["medium"][0]) & (words <= L["medium"][1]),
        "C: long (>180 words)": words >= L["long"][0],
        "C: truncated (>256 cleaned tokens)": n_clean > CFG["truncated_tokens"],
        "C: negation": raw.map(lambda t: bool(neg.search(t))),
        "C: contrast": raw.map(lambda t: bool(con.search(t))),
        "C: no strong polarity words": tok.map(lambda s: not (s & strong)),
        "C: explicit rating mention": raw.map(lambda t: bool(rat.search(t))),
    })


def aswin_slices(texts: pd.Series, train_texts) -> pd.DataFrame:
    sys.path.insert(0, str(ASWIN_DIR / "src"))
    from dataset import Vocabulary, YelpTokenizer
    from slice_analysis import tag_slices

    tok = YelpTokenizer(remove_stopwords=True)
    vocab = Vocabulary(max_size=30000, min_freq=3)
    vocab.build_vocab([tok.tokenize(t) for t in train_texts])
    assert len(vocab) == 18766, len(vocab)  # same vocabulary as Aswin's runs
    tags = tag_slices(pd.DataFrame({"review_text": texts.values}), vocab, tok)
    return pd.DataFrame({f"A: {k}": v.values for k, v in tags.items()})


def main() -> None:
    ds = load_dataset("fancyzhx/yelp_polarity")
    tr_meta = read_json(SPLITS_DIR / "train_indices.json")
    train_texts = ds["train"].select(tr_meta["indices"])["text"]
    for name in EVAL_SETS:
        meta = read_json(SPLITS_DIR / f"{name}_indices.json")
        texts = pd.Series(ds[meta["hf_split"]].select(meta["indices"])["text"])
        n_clean = pd.read_csv(PROCESSED_DIR / f"{name}_meta.csv")["n_clean_tokens"].to_numpy()
        df = pd.concat([our_slices(texts, n_clean), aswin_slices(texts, train_texts)], axis=1)
        df.insert(0, "hf_index", meta["indices"])
        df.to_csv(PROCESSED_DIR / f"slices_{name}.csv", index=False)
        print(name, {c: int(df[c].sum()) for c in df.columns[1:]})


if __name__ == "__main__":
    main()
