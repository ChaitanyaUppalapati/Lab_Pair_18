# Task 2 — Yelp Polarity Sentiment Classification (Aswin)

## 1. Dataset Description & Exploratory Data Analysis (EDA)
* **Dataset**: Official Yelp Review Polarity benchmark (`fancyzhx/yelp_polarity`).
* **Task Objective**: Binary text sentiment classification (0 = Negative, 1 = Positive).
* **Data Splits**:
  * Official Raw Train Set: 560,000 examples (balanced 50/50 negative/positive).
  * Official Raw Test Set: 38,000 examples (balanced 50/50 negative/positive).
  * Evaluated Stratified Training Split: 20,000 reviews (seed=42).
  * Stratified Validation Split: 5,000 reviews (seed=42).
  * Evaluated Untouched Test Split: 5,000 reviews (seed=42).
* **Data Quality Checks**:
  * Missing Values: 0
  * Empty Strings: 0
  * Duplicate Reviews: 0
  * Class Balance: Exact 50.0% Negative / 50.0% Positive across all splits.
* **Length Distributions**:
  * Word Length: Median = 88.0 words, Mean = 133.3 words, 90th percentile = 296 words, 95th percentile = 377 words.
  * Maximum Sequence Length: Set to 256 tokens to cover >90% of reviews without truncation loss.

---

## 2. Text Preprocessing & Vocabulary Decisions
* **Academic Constraint**: All embeddings learned strictly from scratch. No pretrained embeddings (Word2Vec, GloVe) or language models (BERT, RoBERTa, GPT) were utilized.
* **Normalization**: Full lowercasing, whitespace collapse, newline/tab unescaping.
* **Tokenization**: Regex-based boundary extraction (`[a-z]+(?:'[a-z]+)?|[0-9]+|[^\w\s]`).
* **Negation-Preserving Stopword Policy**: Standard NLTK English stopwords were pruned, but **all 31 negation tokens were strictly preserved**:
  `{"not", "no", "never", "neither", "hardly", "scarcely", "barely", "nor", "none", "cannot", "n't", "without", "against", "wasn't", "weren't", "isn't", "aren't", "won't", "wouldn't", "couldn't", "shouldn't", "hasn't", "haven't", "hadn't", "doesn't", "don't", "didn't", "cant", "dont", "wont"}`.
* **Vocabulary Construction**:
  * Fitted strictly on the training split only (zero data leakage).
  * Parameters: `max_vocab_size=30,000`, `min_freq=3`.
  * Special tokens: `<PAD>` (index 0), `<UNK>` (index 1).
  * Final vocabulary size: 18,766 tokens.

---

## 3. Model Lineup & Hardware Disclosure

| Characteristic | Baseline (Model 1) | Experimental 1 (Model 2) | Experimental 2 (Model 3) |
|---|---|---|---|
| **Architecture** | Masked Mean-Pooling FFN | Bidirectional GRU (BiGRU) | Multi-Scale 1D TextCNN |
| **Layers** | Embedding $\to$ Masked Mean Pooling $\to$ Dense(128) $\to$ ReLU $\to$ Dropout(0.3) $\to$ Logit | Embedding $\to$ BiGRU(128) $\to$ Dual Pooling (Max+Avg) $\to$ Dense(128) $\to$ Dropout(0.4) $\to$ Logit | Embedding $\to$ 3 Parallel Conv1D (k=3,4,5, 100 filters) $\to$ MaxPool1D $\to$ Dense(128) $\to$ Dropout(0.4) $\to$ Logit |
| **Embedding Dimension** | 128 (Learned from scratch) | 128 (Learned from scratch) | 128 (Learned from scratch) |
| **Hidden / Feature Dim** | 128 | 256 (bidirectional) $\to$ 512 pooled | 300 (100 $\times$ 3 kernels) |
| **Trainable Parameters** | 2,418,689 | 2,665,985 | 2,594,605 |
| **Optimizer** | AdamW (`lr=1e-3, wd=1e-4`) | AdamW (`lr=5e-4, wd=1e-4`) | AdamW (`lr=1e-3, wd=1e-4`) |
| **Loss Function** | Binary Cross-Entropy with Logits | Binary Cross-Entropy with Logits | Binary Cross-Entropy with Logits |
| **Hardware Used** | Apple M4 (MPS / 16 GB Unified Memory) | Apple M4 (MPS / 16 GB Unified Memory) | Apple M4 (MPS / 16 GB Unified Memory) |
| **Throughput** | 3,040.29 examples/sec | 31.18 examples/sec | 2,157.80 examples/sec |
| **Training Time** | 32.89 seconds | 3,206.96 seconds (~53 min) | 46.34 seconds |

---

## 4. Evaluation Metrics & Statistical Testing

All metrics were computed on the untouched 5,000-example test split. 95% confidence intervals were generated via 1,000 bootstrap resamples.

| Metric | Baseline (Mean Pooling) | Experimental 1 (BiGRU) | Experimental 2 (TextCNN) |
|---|---|---|---|
| **Accuracy** | 0.9162 [0.9084, 0.9242] | 0.9194 [0.9116, 0.9272] | **0.9252** [0.9176, 0.9324] |
| **Macro-Precision** | 0.9162 | 0.9197 | **0.9253** |
| **Macro-Recall** | 0.9162 | 0.9194 | **0.9252** |
| **Macro-F1** | 0.9162 [0.9084, 0.9242] | 0.9194 [0.9116, 0.9272] | **0.9252** [0.9176, 0.9324] |
| **Weighted F1** | 0.9162 | 0.9194 | **0.9252** |
| **Confusion Matrix** | TN: 2295, FP: 205, FN: 214, TP: 2286 | TN: 2334, FP: 166, FN: 237, TP: 2263 | TN: 2290, FP: 210, FN: 164, TP: 2336 |
| **ROC-AUC** | 0.9737 | 0.9756 | **0.9800** |
| **PR-AUC** | 0.9741 | 0.9766 | **0.9804** |
| **MCC** | 0.8324 [0.8168, 0.8484] | 0.8391 [0.8236, 0.8549] | **0.8505** [0.8353, 0.8650] |
| **Brier Score** (lower is better) | 0.0605 | 0.0610 | **0.0556** |
| **ECE** (lower is better) | **0.0112** | 0.0190 | 0.0177 |
| **McNemar vs. Baseline $\chi^2$** | — | 0.9783 | **5.2752** |
| **McNemar vs. Baseline $p$-value** | — | 0.3226 (Not Sig.) | **0.0216** (Significant, $p < 0.05$) |

### Statistical Testing Findings:
* **Baseline vs. Experimental 1 (BiGRU)**: $\chi^2 = 0.9783, p = 0.3226$. The 0.32% accuracy improvement over the baseline is **not statistically significant** at $\alpha = 0.05$.
* **Baseline vs. Experimental 2 (TextCNN)**: $\chi^2 = 5.2752, p = 0.0216$. The 0.90% accuracy improvement ($91.62\% \to 92.52\%$) is **statistically significant** ($p < 0.05$), confirming that local n-gram convolutional patterns provide genuine predictive edge over mean pooling.

---

## 5. Slice Robustness Analysis

| Slice Definition | Examples | Baseline Macro-F1 (Err Rate) | BiGRU Macro-F1 (Err Rate) | TextCNN Macro-F1 (Err Rate) |
|---|---|---|---|---|
| **Short Reviews (<50 words)** | 1,199 | 0.9142 (8.26%) | 0.9144 (8.26%) | **0.9149** (8.17%) |
| **Medium Reviews (50-150 words)** | 2,255 | 0.9188 (8.12%) | 0.9224 (7.76%) | **0.9361** (6.39%) |
| **Long Reviews (>150 words)** | 1,546 | 0.9084 (8.86%) | **0.9134** (8.34%) | 0.9126 (8.54%) |
| **Negation-Bearing Reviews** | 3,765 | 0.9128 (8.45%) | 0.9156 (8.13%) | **0.9246** (7.33%) |
| **Strong Sentiment Cues** | 1,133 | 0.9545 (4.41%) | 0.9480 (5.03%) | **0.9592** (3.97%) |
| **Mixed Sentiment (Contrasts)** | 2,338 | 0.9066 (9.32%) | 0.9085 (9.11%) | **0.9221** (7.78%) |
| **High Unknown Token (>5% UNK)** | 509 | 0.8880 (10.81%) | 0.8792 (11.59%) | **0.8997** (9.63%) |

**Slice Insights**:
1. TextCNN achieved highest Macro-F1 across 6 out of 7 slices, performing exceptionally well on reviews with **Strong Sentiment Cues** (0.9592 F1) and **Medium Reviews** (0.9361 F1).
2. BiGRU demonstrated slightly superior retention on **Long Reviews** (0.9134 F1 vs 0.9126 TextCNN), consistent with recurrent memory handling long-span context.
3. All models suffer degraded performance on **High Unknown Token** reviews (~88-90% F1), highlighting the vulnerability of vocabulary-bound tokenizers to OOV words.

---

## 6. Error Analysis Summary (Best Model: TextCNN)
See complete breakdown of all 20 grounded cases in `failure_analysis.md`.
* **Confident False Positives (5 cases)**: Reviews utilizing sarcastic positive words ("Oh thank you for the wonderful 45-minute wait") or describing polite staff during an otherwise awful experience.
* **Confident False Negatives (5 cases)**: Understated positive reviews or reviews featuring self-deprecating humor.
* **Near-Threshold Errors (5 cases)**: 3-star reviews forced into binary labels containing evenly split positive food remarks and negative ambiance complaints.
* **Slice-Specific Errors (5 cases)**: Multi-clause contrastive sentences where the final clause determines the outcome, but convolution max-pooling attends to an earlier clause.

---

## 7. Strengths, Weaknesses, Limitations & Future Work
* **Strengths**:
  * High test accuracy (>92.5%) achieved with randomly initialized embeddings without any pretrained models.
  * Rigorous paired McNemar testing confirms statistically significant superiority of TextCNN over Baseline.
  * Multi-scale convolutions run 70x faster than recurrent BiGRU while achieving superior test metrics.
* **Weaknesses**:
  * Inability to detect sarcasm when surface vocabulary is effusively positive.
  * OOV vulnerability when reviews contain slang, typos, or foreign food terminology.
* **Limitations**:
  * Truncation at 256 tokens leaves out information in the remaining 5% of very long reviews.
  * Binary classification discards nuanced neutral/mixed ratings.
* **Future Work**:
  * Subword BPE tokenization (e.g. Byte-level BPE) to eliminate UNK rate.
  * Self-attention pooling layer to dynamically weigh clauses.
  * Auxiliary sarcasm classification head.

---

## 8. Artifact and Evidence Index
* **Metrics Table**: `metrics_report.csv`
* **Error Analysis**: `failure_analysis.md` and `outputs/error_analysis/error_analysis.csv`
* **Checkpoints**:
  * Baseline: `checkpoints/baseline/best_model.pt`
  * Experimental 1: `checkpoints/experimental_1/best_model.pt`
  * Experimental 2: `checkpoints/experimental_2/best_model.pt`
* **Plots**: `outputs/plots/`, `outputs/confusion_matrices/`, `outputs/calibration/`
* **Raw Logs**: `reproducibility/raw_logs/task2_sentiment/aswin/`
* **Manifests**: `reproducibility/manifests/task2_sentiment/aswin/`
* **Notebook**: `src/task2_sentiment_aswin.ipynb` (executable on RTX 5090, Apple Silicon, or CPU)
