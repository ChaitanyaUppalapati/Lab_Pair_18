# Architecture Decisions and Hyperparameter Reasoning

## Overview & Constraints
* **Task**: Binary sentiment classification on the Yelp Polarity dataset (0 = Negative, 1 = Positive).
* **Constraints**:
  * No pretrained embeddings (Word2Vec, GloVe, FastText) are allowed.
  * No pretrained language models (BERT, RoBERTa, DistilBERT, GPT) are allowed.
  * All token representations must be initialized from scratch (uniform/normal random initialization) and trained solely on the training split.
  * Three models required: 1 Baseline and 2 Experimental models differing in inductive biases and sequence representations.

---

## Model Architectures

### 1. Baseline Model: Global Mean-Pooling Feed-Forward Network
* **Layer Composition**:
  1. `nn.Embedding(num_embeddings=vocab_size, embedding_dim=128, padding_idx=0)`
  2. Sequence Masked Mean Pooling: Calculates the element-wise average across the token embeddings for each review, ignoring `<PAD>` tokens.
  3. `nn.Linear(128, 128)`
  4. `nn.ReLU()`
  5. `nn.Dropout(p=0.3)`
  6. `nn.Linear(128, 1)` (Binary Logit output)
* **Design Rationale**:
  * A continuous bag-of-words (CBOW) / Deep Averaging Network style baseline represents the simplest reasonable neural approach.
  * It tests how much sentiment information can be extracted purely from word frequencies and learned token semantics, without any regard for token order, syntax, or compositionality.
  * It is fast to train and serves as the lower-bound benchmark for sequence-aware models.

### 2. Experimental Model 1: Bidirectional Gated Recurrent Unit (BiGRU)
* **Layer Composition**:
  1. `nn.Embedding(num_embeddings=vocab_size, embedding_dim=128, padding_idx=0)`
  2. `nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)`
  3. Dual Pooling: Concatenation of Global Max Pooling and Global Average Pooling across GRU time steps ($2 \times 256 = 512$ dim).
  4. `nn.Linear(512, 128)`
  5. `nn.ReLU()`
  6. `nn.Dropout(p=0.4)`
  7. `nn.Linear(128, 1)` (Binary Logit output)
* **Design Rationale**:
  * Unlike the baseline, a recurrent architecture processes tokens sequentially and maintains hidden states across the sequence.
  * The bidirectional nature allows each token to be conditioned on both prior and subsequent context (critical for resolving negation and modifiers, e.g. "not particularly tasty").
  * Dual pooling (max + avg) captures both the peak activation along any hidden dimension and the overall average semantic trajectory.

### 3. Experimental Model 2: Multi-Scale 1D TextCNN
* **Layer Composition**:
  1. `nn.Embedding(num_embeddings=vocab_size, embedding_dim=128, padding_idx=0)`
  2. Parallel 1D Convolutional Blocks:
     * Filter Size 3: `nn.Conv1d(in_channels=128, out_channels=100, kernel_size=3)`
     * Filter Size 4: `nn.Conv1d(in_channels=128, out_channels=100, kernel_size=4)`
     * Filter Size 5: `nn.Conv1d(in_channels=128, out_channels=100, kernel_size=5)`
  3. Activation: `nn.ReLU()` for each convolution branch.
  4. 1D Global Max-over-Time Pooling: Extracts the maximum feature map response for each kernel across the sequence length ($3 \times 100 = 300$ dim).
  5. `nn.Linear(300, 128)`
  6. `nn.ReLU()`
  7. `nn.Dropout(p=0.4)`
  8. `nn.Linear(128, 1)` (Binary Logit output)
* **Design Rationale**:
  * Convolutional networks apply localized n-gram feature extractors (3-grams, 4-grams, 5-grams) irrespective of their absolute position in the text.
  * Max-over-time pooling detects whether a specific sentiment phrase (e.g. "service was terrible", "absolutely loved the food") appeared anywhere in the review.
  * This tests whether local phrase-level patterns outperform sequential recurrence (BiGRU) and unordered token pooling (Baseline).

---

## Hyperparameter Decisions & Reasoning

| Hyperparameter | Value | Reasoning |
|---|---|---|
| `vocab_size` | 30,000 | Balances vocabulary coverage (capturing rare descriptive words) against memory footprint and parameter count of the randomly initialized embedding table. |
| `min_freq` | 3 | Tokens appearing fewer than 3 times in the training set are treated as `<UNK>` to prevent overfitting to typos and idiosyncratic tokens. |
| `max_seq_len` | 256 | Empirical review length distribution shows >90% of Yelp reviews fall within 256 tokens. Truncating at 256 balances computational efficiency and memory usage while preserving essential context. |
| `embedding_dim` | 128 | Since embeddings must be learned from scratch, 128 dimensions provide sufficient capacity to represent semantic clusters without requiring hundreds of epochs to converge. |
| `batch_size` | 64 | Provides stable gradient estimates and fits comfortably within Apple Silicon unified memory / GPU cache. |
| `learning_rate` | 1e-3 (Baseline/CNN), 5e-4 (BiGRU) | BiGRU utilizes a slightly lower learning rate to maintain recurrent gradient stability. |
| `optimizer` | AdamW (`weight_decay=1e-4`) | Decoupled weight decay prevents overfitting of the randomly initialized embedding layer and linear projection layers. |
| `dropout` | 0.3 - 0.4 | Regularization to prevent co-adaptation of features, especially important when learning embeddings from scratch. |
| `loss_fn` | Binary Cross-Entropy with Logits (`nn.BCEWithLogitsLoss`) | Numerically stable sigmoid + cross-entropy computation with log-sum-exp trick. |

*Note: Claims regarding which model achieves superior empirical accuracy, calibration, or slice robustness will only be made following rigorous training, validation, and testing.*
