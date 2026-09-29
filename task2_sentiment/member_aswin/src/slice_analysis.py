"""Data slicing and robustness analysis across linguistic and structural slices.
Slices:
1. Short reviews (< 50 words)
2. Medium reviews (50 - 150 words)
3. Long reviews (> 150 words)
4. Reviews containing negation
5. Reviews containing strong sentiment terms
6. Mixed-sentiment reviews (co-occurring positive and negative cues)
7. High unknown-token proportion reviews (> 5% UNK)
"""

import re
import argparse
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from dataset import YelpTokenizer, Vocabulary, NEGATION_WORDS

STRONG_POS = {"amazing", "exceptional", "perfection", "outstanding", "incredible", "superb", "delightful", "phenomenal"}
STRONG_NEG = {"terrible", "horrible", "disgusting", "worst", "awful", "dreadful", "atrocious", "abysmal"}
GENERAL_POS = {"good", "great", "delicious", "nice", "love", "loved", "friendly", "clean", "excellent", "enjoyed"}
GENERAL_NEG = {"bad", "slow", "expensive", "rude", "cold", "dirty", "poor", "hate", "hated", "overpriced"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run Slice Robustness Analysis")
    parser.add_argument("--preds-dir", type=str, default="task2_sentiment/member_aswin/outputs/predictions")
    parser.add_argument("--vocab-path", type=str, default="task2_sentiment/member_aswin/data_processed/vocab.json")
    parser.add_argument("--output-csv", type=str, default="task2_sentiment/member_aswin/outputs/slice_analysis.csv")
    parser.add_argument("--output-plot", type=str, default="task2_sentiment/member_aswin/outputs/plots/slice_robustness_comparison.png")
    return parser.parse_args()


def tag_slices(df: pd.DataFrame, vocab: Vocabulary, tokenizer: YelpTokenizer) -> Dict[str, pd.Series]:
    """Tag boolean mask for each slice."""
    texts = df["review_text"].astype(str)
    word_counts = texts.apply(lambda t: len(t.split()))

    # 1. Length slices
    slice_short = word_counts < 50
    slice_medium = (word_counts >= 50) & (word_counts <= 150)
    slice_long = word_counts > 150

    # 2. Negation slice
    def has_negation(text: str) -> bool:
        tokens = set(re.findall(r"\b\w+(?:'\w+)?\b", text.lower()))
        return bool(tokens & NEGATION_WORDS)

    slice_negation = texts.apply(has_negation)

    # 3. Strong sentiment slice
    strong_terms = STRONG_POS | STRONG_NEG
    def has_strong_sentiment(text: str) -> bool:
        tokens = set(re.findall(r"\b\w+\b", text.lower()))
        return bool(tokens & strong_terms)

    slice_strong = texts.apply(has_strong_sentiment)

    # 4. Mixed sentiment slice
    def is_mixed_sentiment(text: str) -> bool:
        tokens = set(re.findall(r"\b\w+\b", text.lower()))
        has_pos = bool(tokens & GENERAL_POS)
        has_neg = bool(tokens & GENERAL_NEG) or bool(tokens & NEGATION_WORDS)
        return has_pos and has_neg

    slice_mixed = texts.apply(is_mixed_sentiment)

    # 5. High UNK proportion slice (> 5% UNK)
    def is_high_unk(text: str) -> bool:
        toks = tokenizer.tokenize(text)
        if not toks:
            return False
        unk_count = sum(1 for t in toks if t not in vocab.word2idx)
        return (unk_count / len(toks)) >= 0.05

    slice_high_unk = texts.apply(is_high_unk)

    return {
        "Short Reviews (<50 words)": slice_short,
        "Medium Reviews (50-150 words)": slice_medium,
        "Long Reviews (>150 words)": slice_long,
        "Negation-Bearing": slice_negation,
        "Strong Sentiment Cues": slice_strong,
        "Mixed Sentiment": slice_mixed,
        "High Unknown Token (>5% UNK)": slice_high_unk,
    }


def main():
    args = parse_args()
    preds_dir = Path(args.preds_dir)

    pred_files = [
        f for f in sorted(preds_dir.glob("*_test_predictions.csv"))
        if not f.name.startswith("smoke_")
    ]
    if not pred_files:
        raise FileNotFoundError(f"No prediction CSV files found in {preds_dir}")

    # Load vocab and tokenizer
    vocab = Vocabulary.load(args.vocab_path)
    tokenizer = YelpTokenizer(remove_stopwords=True)

    # Load reference df for slice tagging
    first_df = pd.read_csv(pred_files[0])
    slice_masks = tag_slices(first_df, vocab, tokenizer)

    results = []
    models_evaluated = []

    for pred_file in sorted(pred_files):
        model_name = pred_file.name.replace("_test_predictions.csv", "")
        models_evaluated.append(model_name)
        df = pd.read_csv(pred_file)

        y_true = df["true_label"].values
        y_pred = df["predicted_label"].values

        for slice_name, mask in slice_masks.items():
            n_examples = int(mask.sum())
            if n_examples == 0:
                continue

            slice_true = y_true[mask]
            slice_pred = y_pred[mask]

            error_rate = float(np.mean(slice_true != slice_pred))
            f1_macro = float(f1_score(slice_true, slice_pred, average="macro", zero_division=0))

            results.append({
                "model": model_name,
                "slice": slice_name,
                "n_examples": n_examples,
                "macro_f1": round(f1_macro, 4),
                "error_rate": round(error_rate, 4),
            })

    results_df = pd.DataFrame(results)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_csv, index=False)
    print(f"Saved slice robustness metrics to {out_csv}")
    print(results_df.to_string(index=False))

    # Plotting comparison
    if len(models_evaluated) > 0:
        plot_path = Path(args.output_plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        pivot_f1 = results_df.pivot(index="slice", columns="model", values="macro_f1")
        fig, ax = plt.subplots(figsize=(12, 6))
        pivot_f1.plot(kind="bar", ax=ax, width=0.8, colormap="tab10")
        ax.set_title("Slice Robustness Analysis: Macro-F1 across Data Slices", fontsize=14)
        ax.set_ylabel("Macro-F1 Score")
        ax.set_xlabel("Data Slice")
        ax.set_ylim(0.5, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f"Saved slice comparison plot to {plot_path}")


if __name__ == "__main__":
    main()
