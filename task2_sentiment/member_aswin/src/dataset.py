"""Dataset loading, exploratory data analysis (EDA), text preprocessing,
vocabulary construction, and PyTorch DataLoader pipeline for Yelp Polarity.
"""

import re
import json
import logging
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import nltk
from nltk.corpus import stopwords


# Explicit negation words preserved during stopword removal
NEGATION_WORDS = {
    "not", "no", "never", "neither", "hardly", "scarcely", "barely", "nor",
    "none", "cannot", "n't", "without", "against", "wasn't", "weren't",
    "isn't", "aren't", "won't", "wouldn't", "couldn't", "shouldn't",
    "hasn't", "haven't", "hadn't", "doesn't", "don't", "didn't", "aint",
    "ain't", "cant", "can't", "dont", "wont"
}


class YelpTokenizer:
    """Tokenizer with lowercasing, regex tokenization, negation-preserving stopwords."""

    def __init__(self, remove_stopwords: bool = True):
        self.remove_stopwords = remove_stopwords
        try:
            standard_stopwords = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            standard_stopwords = set(stopwords.words("english"))

        # Preserve all negation terms
        self.stopwords = standard_stopwords - NEGATION_WORDS
        self.token_pattern = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[^\w\s]")

    def tokenize(self, text: str) -> List[str]:
        """Normalize, lowercase, and tokenize text while handling punctuation."""
        if not isinstance(text, str):
            return []
        # Normalization
        text = text.lower()
        # Unescape escaped newlines/tabs
        text = text.replace("\\n", " ").replace("\\t", " ")
        text = re.sub(r"\s+", " ", text).strip()

        tokens = self.token_pattern.findall(text)
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        return tokens


class Vocabulary:
    """Vocabulary mapping tokens to integer IDs, fitted exclusively on training data."""

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, max_size: int = 30000, min_freq: int = 3):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {self.PAD_TOKEN: self.PAD_IDX, self.UNK_TOKEN: self.UNK_IDX}
        self.idx2word: Dict[int, str] = {self.PAD_IDX: self.PAD_TOKEN, self.UNK_IDX: self.UNK_TOKEN}
        self.word_counts: Counter = Counter()

    def build_vocab(self, tokenized_texts: List[List[str]]) -> None:
        """Count tokens and assign IDs to the top frequent tokens meeting min_freq."""
        for tokens in tokenized_texts:
            self.word_counts.update(tokens)

        # Filter by min_freq and sort by frequency
        valid_words = [
            word for word, count in self.word_counts.most_common()
            if count >= self.min_freq and word not in self.word2idx
        ]

        # Truncate to max_size (reserving 2 slots for PAD and UNK)
        max_words = self.max_size - 2
        for word in valid_words[:max_words]:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def __len__(self) -> int:
        return len(self.word2idx)

    def encode(self, tokens: List[str]) -> List[int]:
        """Convert a sequence of tokens to token IDs."""
        return [self.word2idx.get(t, self.UNK_IDX) for t in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        """Convert a sequence of token IDs back to tokens."""
        return [self.idx2word.get(i, self.UNK_TOKEN) for i in indices]

    def save(self, filepath: str) -> None:
        """Save vocabulary mappings and frequency stats to JSON."""
        data = {
            "max_size": self.max_size,
            "min_freq": self.min_freq,
            "vocab_size": len(self.word2idx),
            "word2idx": self.word2idx,
            "word_counts": dict(self.word_counts.most_common(1000)),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "Vocabulary":
        """Load vocabulary from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(max_size=data["max_size"], min_freq=data["min_freq"])
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(v): k for k, v in data["word2idx"].items()}
        vocab.word_counts = Counter(data.get("word_counts", {}))
        return vocab


class YelpTextDataset(Dataset):
    """PyTorch Dataset for padded numericalized Yelp reviews."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: YelpTokenizer,
        vocab: Vocabulary,
        max_seq_len: int = 256,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        label = self.labels[idx]

        tokens = self.tokenizer.tokenize(text)
        token_ids = self.vocab.encode(tokens)
        seq_len = min(len(token_ids), self.max_seq_len)

        # Padding / Truncation
        if len(token_ids) < self.max_seq_len:
            padded_ids = token_ids + [self.vocab.PAD_IDX] * (self.max_seq_len - len(token_ids))
        else:
            padded_ids = token_ids[:self.max_seq_len]

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "length": torch.tensor(seq_len if seq_len > 0 else 1, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float),
            "raw_text": text,
        }


def perform_eda(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Perform and document exploratory data analysis on the Yelp Polarity dataset."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Total counts
    n_train = len(train_df)
    n_test = len(test_df)

    # 2. Missing values / empty strings / duplicates
    train_missing = int(train_df["text"].isna().sum())
    train_empty = int((train_df["text"].str.strip() == "").sum())
    train_dups = int(train_df.duplicated(subset=["text"]).sum())

    test_missing = int(test_df["text"].isna().sum())
    test_empty = int((test_df["text"].str.strip() == "").sum())
    test_dups = int(test_df.duplicated(subset=["text"]).sum())

    # 3. Class balance
    train_class_counts = train_df["label"].value_counts().to_dict()
    test_class_counts = test_df["label"].value_counts().to_dict()

    # 4. Length distributions
    train_word_lens = train_df["text"].apply(lambda t: len(str(t).split())).values
    train_char_lens = train_df["text"].apply(lambda t: len(str(t))).values

    eda_stats = {
        "dataset": "fancyzhx/yelp_polarity",
        "train_size": n_train,
        "test_size": n_test,
        "train_missing_values": train_missing,
        "train_empty_strings": train_empty,
        "train_duplicates": train_dups,
        "test_missing_values": test_missing,
        "test_empty_strings": test_empty,
        "test_duplicates": test_dups,
        "train_class_balance": {str(k): int(v) for k, v in train_class_counts.items()},
        "test_class_balance": {str(k): int(v) for k, v in test_class_counts.items()},
        "train_word_length": {
            "mean": float(np.mean(train_word_lens)),
            "std": float(np.std(train_word_lens)),
            "median": float(np.median(train_word_lens)),
            "min": int(np.min(train_word_lens)),
            "max": int(np.max(train_word_lens)),
            "p90": float(np.percentile(train_word_lens, 90)),
            "p95": float(np.percentile(train_word_lens, 95)),
            "p99": float(np.percentile(train_word_lens, 99)),
        },
        "train_char_length": {
            "mean": float(np.mean(train_char_lens)),
            "std": float(np.std(train_char_lens)),
            "median": float(np.median(train_char_lens)),
            "p95": float(np.percentile(train_char_lens, 95)),
        },
    }

    # Plot 1: Class distribution
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    classes = [0, 1]
    class_labels = ["Negative (0)", "Positive (1)"]
    axes[0].bar(class_labels, [train_class_counts.get(c, 0) for c in classes], color=["#e74c3c", "#2ecc71"])
    axes[0].set_title(f"Train Split Class Distribution (N={n_train})")
    axes[0].set_ylabel("Count")

    axes[1].bar(class_labels, [test_class_counts.get(c, 0) for c in classes], color=["#e74c3c", "#2ecc71"])
    axes[1].set_title(f"Test Split Class Distribution (N={n_test})")
    axes[1].set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(plots_dir / "class_distribution.png", dpi=300)
    plt.close(fig)

    # Plot 2: Review length distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(train_word_lens, bins=50, range=(0, 500), color="#3498db", edgecolor="black", alpha=0.7)
    axes[0].axvline(np.median(train_word_lens), color="red", linestyle="--", label=f"Median: {int(np.median(train_word_lens))}")
    axes[0].axvline(256, color="green", linestyle=":", label="max_seq_len (256)")
    axes[0].set_title("Word Length Distribution")
    axes[0].set_xlabel("Number of Words")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    axes[1].hist(train_char_lens, bins=50, range=(0, 2500), color="#9b59b6", edgecolor="black", alpha=0.7)
    axes[1].axvline(np.median(train_char_lens), color="red", linestyle="--", label=f"Median: {int(np.median(train_char_lens))}")
    axes[1].set_title("Character Length Distribution")
    axes[1].set_xlabel("Number of Characters")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(plots_dir / "review_length_distribution.png", dpi=300)
    plt.close(fig)

    logger.info(f"EDA Completed: Train={n_train}, Test={n_test}, Duplicates={train_dups}")
    logger.info(f"Word length mean={eda_stats['train_word_length']['mean']:.1f}, 95th pct={eda_stats['train_word_length']['p95']:.1f}")

    return eda_stats


def prepare_yelp_dataloaders(
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Tuple[DataLoader, DataLoader, DataLoader, Vocabulary, Dict[str, Any]]:
    """Load Yelp Polarity, perform EDA, build vocab on train only, and return DataLoaders."""
    data_cfg = config["data"]
    train_cfg = config["training"]
    seed = config.get("seed", 42)

    logger.info("Loading Yelp Polarity dataset from Hugging Face...")
    ds = load_dataset(data_cfg["dataset_name"])

    train_df = pd.DataFrame(ds["train"])
    test_df = pd.DataFrame(ds["test"])

    # Optional subsampling
    if data_cfg.get("subsample_train") is not None and data_cfg["subsample_train"] < len(train_df):
        logger.info(f"Subsampling training set to {data_cfg['subsample_train']} examples (stratified, seed={seed})...")
        train_df, _ = train_test_split(
            train_df,
            train_size=data_cfg["subsample_train"],
            stratify=train_df["label"],
            random_state=seed,
        )

    if data_cfg.get("subsample_test") is not None and data_cfg["subsample_test"] < len(test_df):
        logger.info(f"Subsampling test set to {data_cfg['subsample_test']} examples (stratified, seed={seed})...")
        test_df, _ = train_test_split(
            test_df,
            train_size=data_cfg["subsample_test"],
            stratify=test_df["label"],
            random_state=seed,
        )

    # Perform EDA and save plots
    output_dir = Path(config["paths"]["output_dir"])
    eda_stats = perform_eda(train_df, test_df, output_dir, logger)

    # Create stratified train/val split (e.g. 80% train, 20% val)
    val_ratio = data_cfg.get("val_split_ratio", 0.2)
    train_split_df, val_split_df = train_test_split(
        train_df,
        test_size=val_ratio,
        stratify=train_df["label"],
        random_state=seed,
    )

    logger.info(f"Split sizes: Train={len(train_split_df)}, Val={len(val_split_df)}, Test={len(test_df)}")

    # Initialize tokenizer and fit vocabulary ONLY on train split
    tokenizer = YelpTokenizer(remove_stopwords=True)
    vocab = Vocabulary(
        max_size=data_cfg.get("max_vocab_size", 30000),
        min_freq=data_cfg.get("min_freq", 3),
    )

    logger.info("Tokenizing training set and building vocabulary (from scratch on train split only)...")
    train_tokens = [tokenizer.tokenize(t) for t in train_split_df["text"].tolist()]
    vocab.build_vocab(train_tokens)
    logger.info(f"Vocabulary successfully built! Total vocab size: {len(vocab)} tokens.")

    # Save vocabulary and preprocessing config to data_processed/
    proc_dir = Path(data_cfg.get("processed_dir", "task2_sentiment/member_aswin/data_processed"))
    proc_dir.mkdir(parents=True, exist_ok=True)
    vocab.save(str(proc_dir / "vocab.json"))

    prep_config = {
        "tokenizer": "YelpTokenizer (regex tokenization + negation preservation)",
        "negation_words_preserved": sorted(list(NEGATION_WORDS)),
        "max_vocab_size": data_cfg.get("max_vocab_size", 30000),
        "min_freq": data_cfg.get("min_freq", 3),
        "actual_vocab_size": len(vocab),
        "max_seq_len": data_cfg.get("max_seq_len", 256),
        "train_size": len(train_split_df),
        "val_size": len(val_split_df),
        "test_size": len(test_df),
        "random_seed": seed,
    }
    with open(proc_dir / "preprocessing_config.json", "w", encoding="utf-8") as f:
        json.dump(prep_config, f, indent=2)

    with open(proc_dir / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(eda_stats, f, indent=2)

    # Create PyTorch datasets
    max_len = data_cfg.get("max_seq_len", 256)
    train_dataset = YelpTextDataset(
        train_split_df["text"].tolist(),
        train_split_df["label"].tolist(),
        tokenizer,
        vocab,
        max_seq_len=max_len,
    )
    val_dataset = YelpTextDataset(
        val_split_df["text"].tolist(),
        val_split_df["label"].tolist(),
        tokenizer,
        vocab,
        max_seq_len=max_len,
    )
    test_dataset = YelpTextDataset(
        test_df["text"].tolist(),
        test_df["label"].tolist(),
        tokenizer,
        vocab,
        max_seq_len=max_len,
    )

    batch_size = train_cfg.get("batch_size", 64)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, vocab, eda_stats
