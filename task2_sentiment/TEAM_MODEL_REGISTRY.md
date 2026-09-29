# Team Model Registry — Task 2 Yelp Polarity Sentiment Classification

## Teammate Model Verification Status
As of **September 28, 2026**:
* **Member Chaitanya (`member_chaitanya`)**: Model architectures, hyperparameters, and configurations are currently **unspecified** in the repository. The template files (`results.md`, `configs/`, `src/`) contain empty placeholder tables without architecture definitions.
* **Status**: **Uncertain / Pending Teammate Implementation**.
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
* **Baseline (Model 1)**: [Pending teammate registration - currently unverified]
* **Experimental 1 (Model 2)**: [Pending teammate registration - currently unverified]
* **Experimental 2 (Model 3)**: [Pending teammate registration - currently unverified]
