# Team Model Registry — Task 2 Yelp Polarity Sentiment Classification

## Teammate Model Verification Status
As of **September 28, 2026**:
* **Member Chaitanya (`member_chaitanya`)**: Registered below (October 6, 2026); no architecture duplicates Aswin's.
* **Coordination Protocol**: Member Aswin's selected architectures are registered below. When Chaitanya registers their models, they must verify against this registry to avoid architectural duplication as mandated by DATA266 Lab 1 Section 2 ("Two members cannot submit near-identical models").

---

## Member Model Lineup

### Member: Aswin (`member_aswin`)
* **Baseline (Model 1)**:
  * **Architecture**: Global Mean Pooling Feed-Forward Network (`Trainable Embedding -> Mean Pooling -> Dense(128) -> ReLU -> Dropout(0.3) -> Dense(1) -> Sigmoid / Logits`).
  * **Rationale**: Provides a linear/bag-of-embeddings baseline that ignores token order, establishing the baseline performance of learned token representations without sequential context.
* **Experimental 1 (Model 2)**:
  * **Architecture**: Bidirectional Gated Recurrent Unit (`Trainable Embedding -> BiGRU (hidden_dim=128, 1 layer, bidirectional=True) -> Global Max + Average Pooling Concatenation -> Dense(128) -> Dropout(0.4) -> Dense(1)`).
  * **Rationale**: Tests whether modeling bidirectional word order, long-range dependencies, and sequential context improves sentiment classification over unordered token pooling.
* **Experimental 2 (Model 3)**:
  * **Architecture**: Multi-Scale 1D TextCNN (`Trainable Embedding -> Parallel Conv1D [filter sizes 3, 4, 5, 100 filters each] -> ReLU -> 1D Max-over-Time Pooling -> Concatenation(300) -> Dense(128) -> ReLU -> Dropout(0.4) -> Dense(1)`).
  * **Rationale**: Tests whether extracting local n-gram feature detectors (trigrams, 4-grams, 5-grams) via convolutional filters captures salient phrase-level sentiment patterns (e.g., negation phrases like "not worth the wait") more effectively than recurrent models.

### Member: Chaitanya (`member_chaitanya`)
* **Baseline (Model 1)**:
  * **Architecture**: fastText-style (`Trainable Embedding (unigrams) + Trainable hashed-bigram Embedding (2^17 buckets) -> masked mean of all unigram and bigram vectors -> Dropout(0.3 on embeddings) -> Dense(1)`), no hidden layer.
  * **Distinct from Aswin's baseline**: uses local word pairs (bigrams) and a purely linear head (his: unigram mean pooling + MLP).
* **Experimental 1 (Model 2)**:
  * **Architecture**: Hierarchical Attention Network (`Trainable Embedding -> word-level BiLSTM (64/dir) + additive attention -> sentence vectors -> sentence-level BiLSTM (64/dir) + additive attention -> document vector -> Dense(1)`), input 32 sentences x 32 tokens split on the raw text.
  * **Distinct from Aswin's BiGRU**: two-level hierarchy, LSTM cells and attention pooling (his: flat BiGRU with max + average pooling).
* **Experimental 2 (Model 3)**:
  * **Architecture**: hand-written Transformer encoder (`Trainable Embedding + learned positions -> 2 x pre-LN blocks (4-head self-attention, FFN 512) -> final LayerNorm -> masked mean pooling -> Dense(1)`), width 128, no prebuilt attention modules.
  * **Distinct from Aswin's TextCNN**: global self-attention; no recurrence or convolution.
* **Shared setup**: Aswin's exact 20K / 5K / 5K split (HF row indices in `member_chaitanya/splits/`, verified against his predictions), embeddings learned from scratch (128-d), vocabulary min frequency 2. Details and results: `member_chaitanya/results.md`.
* **Status**: Registered and trained (October 6, 2026).
