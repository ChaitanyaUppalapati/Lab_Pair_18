# Task 2 — Yelp Polarity Sentiment (Chaitanya)

> Design decisions (data budget, split, preprocessing, embeddings, the three architectures, training settings,
> slices) are the member's own; they were implemented by Claude. Implementation choices Claude made where the
> spec was silent, and the deviations it found necessary, are listed in **Implementation notes** at the end.
> The "Comparative analysis" interpretation and the error-type labels are left for the member to write.

## Data analysis & preprocessing

**Split (shared with Aswin).** Aswin's split is reproduced from his code (`member_aswin/src/dataset.py`, seed 42,
stratified scikit-learn `train_test_split`) and verified before use: all 5,000 test reviews match his committed
predictions text-for-text and label-for-label in the same order, and his tokenizer rebuilt on our train rows gives
exactly his reported vocabulary (18,766). The HF row indices are committed in `splits/` (send to Aswin to confirm).

| Set | Source | Size | Use |
|---|---|---|---|
| train | HF train (560K) → 25K stratified → 80% | 20,000 | training |
| val | same 25K → 20% | 5,000 | early stopping (macro-F1) |
| test5k | HF test (38K) → 5K stratified | 5,000 | **team comparison**, McNemar vs Aswin, error review |
| test33k | HF test rows **not** in test5k | 33,000 | tighter CIs; confirms slice findings |

The 25K training pool shares no review with the 38K HF test split; test33k is disjoint from test5k, so the two
test sets are independent. Training on 20K (not 560K) keeps the data budget equal to Aswin's.

- **Review length** (`outputs/eda/length_distribution.png`): raw words mean 133, median 96, 95th percentile 376,
  max 982 (train). Negative reviews are longer (median 112 words) than positive ones (83).
- **Class balance:** exactly 50/50 in every set (train 10,000/10,000; val, test5k 2,500/2,500; test33k 16,500/16,500).
- **Missing / malformed:** no missing, empty or non-string texts; labels all 0/1; no duplicate texts within or
  across sets. 2 train reviews are empty after cleaning (encoded as one `<unk>`). No rows dropped (the split must
  stay identical to Aswin's). The dataset stores literal escapes (`\n`, `\"`, `\uXXXX`, `\r`); 472 train reviews
  (2.4%) contain `\uXXXX` escapes, decoded and accent-stripped ("café" → "cafe").
- **Preprocessing** (`src/preprocess.py`, in order): split the raw text into sentences (before punctuation is
  removed) → expand contractions, including apostrophe-less forms ("didn't", "didnt" → "did not") → keep only
  alphabetic words and digits (removes punctuation and special characters) → POS-tag → lowercase → remove NLTK
  English stopwords **except** not, no, nor, never, but, however, very, too, few, against → WordNet lemmatisation
  (POS-aware). Example: "I didn't like it. The fries weren't great, but the service was very friendly!" →
  `not like | fry not great but service very friendly`.
- **Tokenizer / vocabulary / length:** whole words; vocabulary built on the 20K train split only, min frequency 2
  (words seen once → `<unk>`), cap 30,000 → **19,646** entries incl. `<pad>`, `<unk>`. Max 256 tokens (2.4% of
  train reviews are truncated; median 52 cleaned tokens). The HAN view is 32 sentences × 32 tokens (2.3% of reviews
  have more than 32 sentences, 0.3% of sentences more than 32 tokens). Out-of-vocabulary rate: 1.9% of test tokens.
- **Embeddings:** learned from scratch, dimension 128, random N(0, 0.1²) init, trained end to end; identical
  vocabulary, dimension, init and embedding dropout (0.3) in all three models, so the architecture is the only
  variable. No pretrained embeddings or language models anywhere.

## Models

| | Baseline | Experimental 1 | Experimental 2 |
|---|---|---|---|
| Architecture | fastText-style: mean of unigram + hashed-bigram embeddings → linear output (no hidden layer) | Hierarchical attention network: word BiLSTM (64/dir) + additive attention → sentence vectors → sentence BiLSTM (64/dir) + attention → document vector → linear | Hand-written Transformer encoder: 2 layers, 4 heads, width 128, FFN 512, learned positions, pre-LayerNorm, masked mean pooling → linear |
| Input | 256 tokens + their consecutive bigrams (2^17 hash buckets) | 32 sentences × 32 tokens (sentences split on raw text) | 256 tokens |
| Embedding choice | shared setup (128-d, scratch); plus a separate 128-d hashed-bigram table | shared setup | shared setup |
| Key hyperparameters | AdamW lr 3e-3 constant, wd 0.01, batch 64, ≤20 epochs, patience 3, emb. dropout 0.3, no clipping | AdamW lr 1e-3 constant, wd 0.01, batch 64, ≤15 epochs, patience 3, dropout 0.3, clip 1.0 | AdamW lr 5e-4, 5% warm-up then linear decay, wd 0.01, batch 64, ≤20 epochs, patience 3, attention dropout 0.1, other 0.2, clip 1.0 |
| Parameter count | 19,292,033 (16.8M of them in the bigram table) | 2,746,753 | 2,944,385 |
| Best epoch (seeds 42/43/44) | 2 / 2 / 2 | 2 / 1 / 2 | 3 / 3 / 3 |
| Hardware (exact) | GPU: NVIDIA GeForce RTX 4090; CPU: AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD | same | same |
| Distinct from Aswin because | his baseline has no n-grams and an MLP head | hierarchy, LSTM cells, attention pooling vs his flat BiGRU with max+avg pooling | global self-attention; no recurrence or convolution |
| Hypothesis (member's) | How much do local word pairs ("not good") buy without any sequence model? | Does modelling sentence structure help mixed reviews ("food great, service awful")? HAN was introduced on Yelp data. | Can attention trained from scratch compete at 20K examples? (expected to underperform) |
| Justification | *(member)* | *(member)* | *(member)* |

Common: early stopping on validation macro-F1 with the best epoch restored; threshold fixed at 0.5; seeds 42, 43,
44 (seed 42 is the run used for the bootstrap, McNemar and the error review). Binary cross-entropy on one logit.
Reference row (not one of the three; no learned embeddings): TF-IDF word uni+bigrams on the same cleaned tokens +
logistic regression, C = 16 chosen on validation macro-F1 from {0.25 … 256}.

## Metrics

Full table: `metrics_report.csv` (both test sets, every listed metric). Seed-42 runs; 95% bootstrap CIs, 2,000
resamples.

| Model | Set | Accuracy [95% CI] | Macro-F1 [95% CI] | MCC [95% CI] | ROC-AUC | PR-AUC | Brier | ECE | McNemar vs baseline |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (fastText bigram) | test5k | 0.9294 [0.9218, 0.9362] | 0.9294 [0.9218, 0.9362] | 0.8588 [0.8436, 0.8724] | 0.9786 | 0.9792 | 0.0543 | 0.0137 | — |
| Exp 1 (HAN) | test5k | 0.9286 [0.9218, 0.9354] | 0.9286 [0.9217, 0.9354] | 0.8577 [0.8440, 0.8713] | 0.9810 | 0.9819 | 0.0537 | 0.0131 | χ² 0.03, p = 0.86 |
| Exp 2 (Transformer) | test5k | 0.9128 [0.9050, 0.9206] | 0.9128 [0.9049, 0.9206] | 0.8258 [0.8100, 0.8413] | 0.9728 | 0.9750 | 0.0668 | 0.0337 | χ² 23.6, p = 1.2e-6 |
| *Ref: TF-IDF + LR* | test5k | 0.9350 [0.9278, 0.9416] | 0.9350 [0.9278, 0.9415] | 0.8700 [0.8557, 0.8831] | 0.9840 | 0.9848 | 0.0489 | 0.0212 | χ² 4.4, p = 0.035 |
| Baseline (fastText bigram) | test33k | 0.9264 [0.9236, 0.9292] | 0.9264 [0.9236, 0.9292] | 0.8529 [0.8472, 0.8586] | 0.9765 | 0.9761 | 0.0557 | 0.0144 | — |
| Exp 1 (HAN) | test33k | 0.9222 [0.9192, 0.9250] | 0.9221 [0.9191, 0.9250] | 0.8447 [0.8387, 0.8504] | 0.9773 | 0.9778 | 0.0587 | 0.0157 | χ² 9.8, p = 0.0017 |
| Exp 2 (Transformer) | test33k | 0.9108 [0.9077, 0.9139] | 0.9108 [0.9077, 0.9139] | 0.8219 [0.8156, 0.8281] | 0.9707 | 0.9714 | 0.0683 | 0.0338 | χ² 137.5, p = 9e-32 |
| *Ref: TF-IDF + LR* | test33k | 0.9320 [0.9294, 0.9347] | 0.9320 [0.9294, 0.9347] | 0.8640 [0.8587, 0.8695] | 0.9818 | 0.9824 | 0.0512 | 0.0184 | χ² 31.6, p = 1.9e-8 |

Precision, recall and F1 (macro / micro / weighted) are all within 0.001 of accuracy because the classes are
exactly balanced (see the CSV). Confusion matrices (test5k, TN / FP / FN / TP): baseline 2334 / 166 / 187 / 2313;
HAN 2281 / 219 / 138 / 2362; Transformer 2258 / 242 / 194 / 2306. Plots: `outputs/plots/`
(`confusion_matrices_<set>.png`, `roc_pr_reliability_<set>.png` = ROC, PR and reliability diagrams,
`training_curves.png`).

**Training variance (3 seeds, mean ± std)** — separate from the bootstrap, which is test-sample variance
(`outputs/seed_summary.csv`):

| Model | test5k accuracy | test33k accuracy | test33k MCC | test33k ECE |
|---|---|---|---|---|
| Baseline | 0.9295 ± 0.0001 | 0.9269 ± 0.0004 | 0.8538 ± 0.0009 | 0.0142 ± 0.0006 |
| HAN | 0.9251 ± 0.0050 | 0.9198 ± 0.0025 | 0.8400 ± 0.0049 | 0.0133 ± 0.0053 |
| Transformer | 0.9151 ± 0.0039 | 0.9106 ± 0.0003 | 0.8217 ± 0.0005 | 0.0338 ± 0.0079 |

**Efficiency** (seed 42; wall-clock on 20K examples is dominated by per-epoch overhead, so examples/sec is the
fairer number):

| Model | Params | Train time (s) | Train examples/s | Inference examples/s | Peak GPU memory (MB) |
|---|---|---|---|---|---|
| Baseline | 19.29M | 6.1 | 16,463 | 275K | 698 |
| HAN | 2.75M | 49.0 | 2,040 | 40K | 387 |
| Transformer | 2.94M | 16.5 | 7,264 | 23K | 1,535 |
| Ref: TF-IDF + LR | 0.17M (features) | 10.7 (CPU, incl. C search) | 1,875 | — | — |

## Slices used for robustness

Definitions are in `configs/slices.yaml` (shared with Aswin), computed on the raw text. Slices with fewer than 150
reviews are reported with their size but no metric. Aswin's own slices are also computed, with his code and his
vocabulary, so both members' models are compared on identical rows.

| Slice | Definition | n (test5k) | n (test33k) |
|---|---|---|---|
| Short / medium / long | ≤ 60, 61–180, > 180 raw words | 1512 / 2328 / 1160 | 10058 / 15078 / 7864 |
| Truncated | > 256 tokens after cleaning | 104 (not reported) | 745 |
| Negation | not, no, never, or an n't contraction | 3688 | 24223 |
| Contrast | but, however, although, though, except | 2919 | 19414 |
| No strong polarity words | none of 40 fixed strong words (amazing, worst, terrible, love, horrible, …) | 1918 | 12785 |
| Explicit rating mention | "5 stars", "one star", "zero stars", "4/5", "3 out of 5", … | 470 | 2962 |
| Aswin's slices | short < 50, medium 50–150, long > 150 words; negation-bearing; strong sentiment cues; mixed sentiment; > 5% unknown tokens | 1199 / 2255 / 1546 / 3765 / 1133 / 2338 / 509 | — |

Macro-F1 per slice (seed 42; error rate = 1 − accuracy, in `outputs/slice_metrics.csv`):

| Slice | test5k: Baseline | HAN | Transformer | test33k: Baseline | HAN | Transformer |
|---|---|---|---|---|---|---|
| short (≤ 60) | 0.925 | 0.929 | 0.907 | 0.924 | 0.921 | 0.904 |
| medium (61–180) | 0.931 | 0.930 | 0.918 | 0.927 | 0.922 | 0.912 |
| long (> 180) | 0.926 | 0.920 | 0.903 | 0.920 | 0.916 | 0.906 |
| truncated (> 256 tokens) | — | — | — | 0.900 | 0.878 | 0.880 |
| negation | 0.928 | 0.924 | 0.906 | 0.921 | 0.917 | 0.902 |
| contrast | 0.922 | 0.922 | 0.906 | 0.917 | 0.912 | 0.900 |
| no strong polarity words | **0.893** | **0.890** | **0.874** | **0.895** | **0.890** | **0.872** |
| explicit rating mention | 0.915 | 0.922 | 0.896 | 0.919 | 0.899 | 0.899 |
| A: strong sentiment cues | 0.958 | 0.956 | 0.947 | 0.957 | 0.957 | 0.950 |
| A: mixed sentiment | 0.923 | 0.921 | 0.898 | 0.918 | 0.911 | 0.898 |
| A: > 5% unknown tokens | 0.901 | 0.904 | 0.879 | 0.884 | 0.875 | 0.861 |

"No strong polarity words" is the worst slice of ours for every model on both test sets.

## Comparative analysis

**Team comparison on test5k** (Aswin's committed predictions scored with the same code; `outputs/team_comparison.csv`,
McNemar in `outputs/team_mcnemar.csv`, Holm-corrected over the 9 pairs):

| Model | Accuracy [95% CI] | Macro-F1 | MCC | ROC-AUC | Brier | ECE |
|---|---|---|---|---|---|---|
| Chaitanya baseline (fastText bigram) | 0.9294 [0.9218, 0.9362] | 0.9294 | 0.8588 | 0.9786 | 0.0543 | 0.0137 |
| Chaitanya exp 1 (HAN) | 0.9286 [0.9218, 0.9354] | 0.9286 | 0.8577 | 0.9810 | 0.0537 | 0.0131 |
| Chaitanya exp 2 (Transformer) | 0.9128 [0.9050, 0.9206] | 0.9128 | 0.8258 | 0.9728 | 0.0668 | 0.0337 |
| Aswin baseline (mean-pool FFN) | 0.9162 [0.9082, 0.9234] | 0.9162 | 0.8324 | 0.9737 | 0.0605 | 0.0118 |
| Aswin exp 1 (BiGRU) | 0.9194 [0.9118, 0.9266] | 0.9194 | 0.8391 | 0.9756 | 0.0610 | 0.0175 |
| Aswin exp 2 (TextCNN) | 0.9252 [0.9178, 0.9320] | 0.9252 | 0.8505 | 0.9800 | 0.0556 | 0.0159 |

Significant pairwise differences after Holm correction (p_holm < 0.05): my baseline beats Aswin's baseline
(p_holm 0.0005) and BiGRU (0.027); my HAN beats Aswin's baseline (0.0075); Aswin's TextCNN beats my Transformer
(0.018). All other pairs are not significant.

**Known confounds in the team comparison** (pipelines differ; architecture is not the only variable between members):
- Aswin trains on the same 20K rows, but with a different pipeline: no lemmatisation; contractions kept as single
  tokens; a larger negation keep-list (≈33 forms) but "but / however / very / too / few" removed as stopwords;
  min frequency 3 (vocabulary 18,766 vs 19,646).
- Different training settings (Aswin: ≤5 epochs, lr 1e-3, weight decay 1e-4, ReduceLROnPlateau) and a single
  seed, so his rows have no training-variance estimate.

To write (member):
- Own models:
- Versus teammates:
- Strengths / weaknesses / limitations:
- Future work: *(planned points: data-scaling run 25K → 100K → 560K on the best model; BPE tokenisation;
  word2vec pre-trained on the unlabeled remainder of the 560K — each changes the data budget or adds a variable,
  so they are outside this comparison)*

## Implementation notes (choices made by Claude where the spec was silent, and deviations)

1. **HAN input 32 × 32 instead of 12 × 32.** The plan's rationale was to match the 256-token budget, but 12
   sentences kept only 77.9% of train tokens vs 97.2% for the flat 256 limit (24% of reviews cut); 32 × 32 keeps
   97.1%, so all models see the same text.
2. **POS-aware lemmatisation with two guards.** Plain WordNet lemmatisation produced "boss" → "bos" and
   "dining" → "din"; candidates are chosen by POS tag (inflected tags take the first different lemma, base tags
   keep the word if it is itself a lemma). Comparatives/superlatives are not lemmatised: WordNet maps
   best → good and worst → bad, erasing intensity. The NLTK averaged-perceptron POS tagger is a pretrained
   tagger (not embeddings or a language model).
3. **Escape decoding** (`\uXXXX`, `\r`) was added after the first full run showed "fiancé" → "fianc u 00 e 9".
   All runs were redone; the first pass's logs are in `reproducibility/raw_logs/task2_sentiment/chaitanya/superseded_unicode_escape_bug/`.
4. Unspecified sizes: 2^17 bigram hash buckets (deterministic hash (a·1,000,003 + b) mod 2^17); HAN LSTM 64 per
   direction, attention dimension 128; Transformer FFN 512 (4× width), GELU; embedding init N(0, 0.1²).
5. Embedding dropout 0.3 in all three models (embedding layer as a controlled constant); the Transformer's
   internal dropout is 0.1 (attention) / 0.2 (residual, FFN, before the output), as specified.
6. Weight decay applies to weight matrices including embeddings; none on biases, LayerNorm gains or the attention
   context vectors. Baseline and HAN learning rates are constant (no schedule was specified for them).
7. ECE: 15 equal-width confidence bins on max(p, 1 − p). McNemar: χ² with continuity correction (exact binomial
   when fewer than 25 discordant pairs).

## Evidence
- Configs: `configs/` (`data.yaml`, `baseline.yaml`, `experimental_1.yaml`, `experimental_2.yaml`, `slices.yaml`)
- Split indices: `splits/` (+ `split_manifest.json` with the verification results and text hashes)
- One command reproduces everything: `python task2_sentiment/member_chaitanya/src/run_pipeline.py`
- Raw logs: `reproducibility/raw_logs/task2_sentiment/chaitanya/`
- Manifests (environment, packages, checkpoint sha256): `reproducibility/manifests/task2_sentiment/chaitanya/`
- Checkpoint IDs: `t2_baseline_s{42,43,44}`, `t2_exp1_han_s{42,43,44}`, `t2_exp2_transformer_s{42,43,44}`
  (`checkpoints/<id>/best.pt`, gitignored; sha256 in the manifests)
- Predictions: `outputs/runs/<id>/pred_{val,test5k,test33k}.csv`; EDA: `outputs/eda/`
