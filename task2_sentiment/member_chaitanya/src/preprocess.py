"""Preprocessing, EDA and encoding for Task 2 (Chaitanya).

Pipeline (user-specified; order matters):
  1. Replace Yelp's literal "\\n" escapes with newlines.
  2. Split the RAW text into sentences (before punctuation removal) - needed by the HAN model.
  3. Per sentence: expand contractions (incl. apostrophe-less forms like "didnt") so negations survive.
  4. Tokenise to alphabetic words and digits only (= lowercasing later + punctuation/special-char removal).
  5. POS-tag (NLTK averaged perceptron) so WordNet lemmatisation gets the right part of speech.
     Lemma choice among WordNet's morphy candidates: for an inflected tag (NNS, VBD, VBG, ...) take the first
     candidate that differs from the word ("glasses"->"glass", "dining"->"dine"); for a base-form tag keep the
     word when it is itself a lemma ("boss" stays "boss", not "bos"). Plain WordNetLemmatizer gets these wrong.
     Comparatives/superlatives (JJR/JJS/RBR/RBS) are left as-is so "best"/"worst" keep their intensity.
  6. Lowercase, remove NLTK English stopwords EXCEPT the keep-list, lemmatise.
Flat token sequence = concatenation of the sentences (identical tokens for all three models).
Vocabulary: built on the 20K train split only, min_freq 2, max 30,000 entries incl. <pad>=0 and <unk>=1.
Encodings: flat [N, 256] for baseline / Transformer, hierarchical [N, 12, 32] for the HAN.
"""
import argparse
import collections
import pickle
import re
import time
import unicodedata
from multiprocessing import Pool

import numpy as np
import pandas as pd
from datasets import load_dataset

from common import EVAL_SETS, OUTPUTS_DIR, PROCESSED_DIR, SPLITS_DIR, load_config, read_json, write_json

KEEP_STOPWORDS = {"not", "no", "nor", "never", "but", "however", "very", "too", "few", "against"}

CONTRACTIONS = [
    (r"\bwon['’]?t\b", "will not"), (r"\bcan['’]?t\b", "can not"), (r"\bshan['’]t\b", "shall not"),
    (r"\bain['’]?t\b", "is not"), (r"\bcannot\b", "can not"),
    # apostrophe-less negative forms that are common on Yelp (not "wont"/"cant" handled above)
    (r"\b(do|does|did|is|was|are|were|could|would|should|have|has|had|must|need)nt\b", r"\1 not"),
    (r"n['’]t\b", " not"),
    (r"['’]re\b", " are"), (r"['’]m\b", " am"), (r"['’]ve\b", " have"), (r"['’]ll\b", " will"),
    (r"['’]d\b", " would"), (r"['’]s\b", ""),  # 's is "is" (a stopword) or possessive -> dropped either way
]
CONTRACTIONS = [(re.compile(p, re.IGNORECASE), r) for p, r in CONTRACTIONS]
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
WORD = re.compile(r"[A-Za-z]+|\d+")

INFLECTED_TAGS = {"NNS", "NNPS", "VBD", "VBG", "VBN", "VBZ"}
# Comparatives/superlatives are NOT lemmatised: WordNet maps best->good, worst->bad, which erases intensity.
NO_LEMMA_TAGS = {"JJR", "JJS", "RBR", "RBS"}
_tagger = _morphy = _stop = None
_cache = {}


def _init_worker():
    global _tagger, _morphy, _stop
    import nltk
    from nltk.corpus import stopwords
    from nltk.corpus import wordnet as wn

    _tagger = nltk.pos_tag
    _morphy = wn._morphy
    _stop = set(stopwords.words("english")) - KEEP_STOPWORDS


def lemmatize(word: str, tag: str) -> str:
    if tag in NO_LEMMA_TAGS:
        return word
    key = (word, tag)
    if key not in _cache:
        cands = _morphy(word, _wn_pos(tag))
        if tag in INFLECTED_TAGS:
            lemma = next((c for c in cands if c != word), word)
        else:
            lemma = word if (word in cands or not cands) else cands[0]
        _cache[key] = lemma
    return _cache[key]


UNICODE_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")


def unescape(text: str) -> str:
    """Undo the dataset's literal escapes: \\n, \\r, \\t, \\" and \\uXXXX (2.4% of train reviews, e.g. "caf\\u00e9"),
    then strip accents (NFKD) so "café" -> "cafe" instead of the noise tokens "caf u 00 e 9"."""
    text = text.replace("\\n", "\n").replace("\\r", " ").replace("\\t", " ").replace('\\"', '"')
    text = UNICODE_ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), text)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def expand_contractions(s: str) -> str:
    for pat, rep in CONTRACTIONS:
        s = pat.sub(rep, s)
    return s


def _wn_pos(tag: str) -> str:
    return {"J": "a", "V": "v", "N": "n", "R": "r"}.get(tag[:1], "n")


def process_review(text) -> list:
    """Raw review -> list of sentences, each a list of cleaned lemmas (empty sentences dropped)."""
    if not isinstance(text, str):
        return []
    out = []
    for sent in SENT_SPLIT.split(unescape(text)):
        words = WORD.findall(expand_contractions(sent))
        if not words:
            continue
        toks = []
        for w, tag in _tagger(words):
            lw = w.lower()
            if lw in _stop:
                continue
            toks.append(lemmatize(lw, tag))
        if toks:
            out.append(toks)
    return out


def load_split_frames() -> dict:
    ds = load_dataset("fancyzhx/yelp_polarity")
    frames = {}
    for name in ("train",) + EVAL_SETS:
        meta = read_json(SPLITS_DIR / f"{name}_indices.json")
        src = ds[meta["hf_split"]].select(meta["indices"])
        frames[name] = pd.DataFrame({"hf_index": meta["indices"], "text": src["text"], "label": src["label"]})
    return frames


def build_vocab(train_sents, max_size: int, min_freq: int):
    counts = collections.Counter(t for review in train_sents for s in review for t in s)
    words = [w for w, n in sorted(counts.items(), key=lambda z: (-z[1], z[0])) if n >= min_freq][: max_size - 2]
    itos = ["<pad>", "<unk>"] + words
    return {w: i for i, w in enumerate(itos)}, itos, counts


def encode(reviews, stoi, max_len, max_sents, max_words):
    n = len(reviews)
    flat = np.zeros((n, max_len), dtype=np.int32)
    hier = np.zeros((n, max_sents, max_words), dtype=np.int32)
    flat_len = np.zeros(n, dtype=np.int32)
    for i, review in enumerate(reviews):
        sents = [[stoi.get(t, 1) for t in s] for s in review] or [[1]]  # empty review -> single <unk>
        ids = [t for s in sents for t in s][:max_len]
        flat[i, : len(ids)] = ids
        flat_len[i] = len(ids)
        for j, s in enumerate(sents[:max_sents]):
            s = s[:max_words]
            hier[i, j, : len(s)] = s
    return flat, flat_len, hier


def eda(frames, reviews, stoi, cfg, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = cfg["data"]
    summary = {"splits": {}}
    for name, df in frames.items():
        texts = df["text"]
        raw_words = texts.map(lambda t: len(str(t).split()))
        clean = [sum(len(s) for s in r) for r in reviews[name]]
        n_sents = [len(r) for r in reviews[name]]
        sent_lens = [len(s) for r in reviews[name] for s in r]
        toks = [t for r in reviews[name] for s in r for t in s]
        summary["splits"][name] = {
            "n": len(df),
            "class_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
            "positive_rate": round(float(df["label"].mean()), 4),
            "missing_text": int(texts.isna().sum()),
            "non_string": int((~texts.map(lambda t: isinstance(t, str))).sum()),
            "empty_or_whitespace": int(texts.map(lambda t: not str(t).strip()).sum()),
            "empty_after_cleaning": int(sum(c == 0 for c in clean)),
            "label_not_0_1": int((~df["label"].isin([0, 1])).sum()),
            "duplicate_texts_within_split": int(texts.duplicated().sum()),
            "raw_words_mean": round(float(raw_words.mean()), 1),
            "raw_words_median": float(raw_words.median()),
            "raw_words_p95": float(raw_words.quantile(0.95)),
            "raw_words_max": int(raw_words.max()),
            "raw_words_by_class_median": {str(k): float(v) for k, v in raw_words.groupby(df["label"]).median().items()},
            "clean_tokens_median": float(np.median(clean)),
            "clean_tokens_p95": float(np.percentile(clean, 95)),
            "pct_truncated_over_max_len": round(100 * float(np.mean(np.array(clean) > d["max_len"])), 2),
            "sentences_median": float(np.median(n_sents)),
            "pct_over_max_sentences": round(100 * float(np.mean(np.array(n_sents) > d["max_sentences"])), 2),
            "pct_sentences_over_max_words": round(100 * float(np.mean(np.array(sent_lens) > d["max_sentence_words"])), 2),
            "oov_token_rate": round(float(np.mean([t not in stoi for t in toks])), 4) if toks else 0.0,
        }
    tr_text = set(frames["train"]["text"])
    summary["cross_split_duplicate_texts"] = {
        n: int(frames[n]["text"].isin(tr_text).sum()) for n in EVAL_SETS
    }
    summary["handling"] = (
        "No rows dropped, so the split stays identical to Aswin's (team McNemar needs the same rows). "
        "Reviews that are empty after cleaning are encoded as a single <unk> token."
    )
    write_json(out_dir / "eda_summary.json", summary)

    df = frames["train"]
    raw_words = df["text"].map(lambda t: len(str(t).split()))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for lab, name in ((0, "negative"), (1, "positive")):
        ax[0].hist(raw_words[df["label"] == lab].clip(upper=800), bins=80, alpha=0.6, label=name)
    ax[0].set(title="Raw review length (train 20K)", xlabel="words (clipped at 800)", ylabel="reviews")
    ax[0].legend()
    clean = np.array([sum(len(s) for s in r) for r in reviews["train"]])
    ax[1].hist(np.clip(clean, 0, 600), bins=80)
    ax[1].axvline(d["max_len"], color="red", ls="--", label=f"max_len {d['max_len']}")
    ax[1].set(title="Tokens after cleaning (train)", xlabel="tokens (clipped at 600)")
    ax[1].legend()
    counts = [summary["splits"][n]["class_counts"] for n in ("train",) + EVAL_SETS]
    x = np.arange(len(counts))
    ax[2].bar(x - 0.2, [c.get("0", 0) for c in counts], 0.4, label="negative (0)")
    ax[2].bar(x + 0.2, [c.get("1", 0) for c in counts], 0.4, label="positive (1)")
    ax[2].set_xticks(x, ("train",) + EVAL_SETS)
    ax[2].set(title="Class distribution", yscale="log")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "length_distribution.png", dpi=120)
    plt.close(fig)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="task2_sentiment/member_chaitanya/configs/data.yaml")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg["data"]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    eda_dir = OUTPUTS_DIR / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    frames = load_split_frames()
    reviews = {}
    with Pool(args.workers, initializer=_init_worker) as pool:
        for name, df in frames.items():
            reviews[name] = pool.map(process_review, df["text"].tolist(), chunksize=200)
            print(f"{name}: {len(df)} reviews processed ({time.time() - t0:.0f}s)", flush=True)

    stoi, itos, counts = build_vocab(reviews["train"], d["max_vocab"], d["min_freq"])
    print(f"vocab {len(itos)} (train types {len(counts)}, with count >= {d['min_freq']}: "
          f"{sum(1 for n in counts.values() if n >= d['min_freq'])})")

    for name, df in frames.items():
        flat, flat_len, hier = encode(reviews[name], stoi, d["max_len"], d["max_sentences"], d["max_sentence_words"])
        np.savez_compressed(PROCESSED_DIR / f"{name}.npz", flat=flat, flat_len=flat_len, hier=hier,
                            labels=df["label"].to_numpy(np.int64), hf_index=df["hf_index"].to_numpy(np.int64))
        with open(PROCESSED_DIR / f"{name}_tokens.pkl", "wb") as f:
            pickle.dump(reviews[name], f)
        df[["hf_index", "label"]].assign(n_clean_tokens=[sum(len(s) for s in r) for r in reviews[name]]).to_csv(
            PROCESSED_DIR / f"{name}_meta.csv", index=False)

    write_json(PROCESSED_DIR / "vocab.json", {"itos": itos})
    summary = eda(frames, reviews, stoi, cfg, eda_dir)
    summary["vocab"] = {"size": len(itos), "min_freq": d["min_freq"], "max_vocab": d["max_vocab"],
                        "train_types": len(counts)}
    summary["examples"] = [
        {"raw": frames["train"]["text"].iloc[i][:400], "cleaned": [" ".join(s) for s in reviews["train"][i]][:6]}
        for i in range(3)
    ]
    summary["preprocessing_seconds"] = round(time.time() - t0, 1)
    write_json(eda_dir / "eda_summary.json", summary)
    write_json(PROCESSED_DIR / "preprocessing_config.json", {**d, "keep_stopwords": sorted(KEEP_STOPWORDS),
                                                             "vocab_size": len(itos)})
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
