# Task 1 - GPT from scratch (Aswin)

## Status

The implementation and local smoke-test pathway are provided. Full TinyStories training must be run on Aswin's RTX 5090 machine before adding final numbers or claims here. Only rows explicitly marked `full` in `metrics_report.csv` may be used in the team report.

## Dataset and character tokenization

The source is `roneneldan/TinyStories`. Preprocessing rejects missing, non-string, empty, and null-byte-only records and records those counts. A seeded permutation (`26617`) selects disjoint sets of 100,000 training stories and 10,000 validation stories. Each selected story creates one example of 161 IDs: `<BOS>`, literal characters, `<EOS>`, then either a seeded crop or `<PAD>` padding. Input is the first 160 IDs and target is the same sequence shifted one character forward.

The vocabulary contains four documented special IDs followed by sorted literal characters: `<PAD>` masks padded targets from loss, `<UNK>` handles unseen prompt characters, `<BOS>` marks the beginning, and `<EOS>` permits stopping. `char_to_idx`, `idx_to_char`, frequencies, split sizes, seed, and context length are saved in `data_processed/full_run01/metadata.json`.

## Architecture

The model is a pre-layer-normalized decoder-only Transformer with 5 blocks, 240-dimensional embeddings, 6 heads (40 dimensions per head), 960-dimensional GELU feed-forward layers, dropout 0.12, and context length 160. Token and position embeddings are learned. Every block has manual linear Q/K/V projections, manual head reshaping, scaled dot-product scores, an explicit lower-triangular Boolean mask, softmax, concatenation, output projection, two residual paths, and two layer normalizations. The final normalized states are projected to character logits.

Chaitanya's Task 1 architecture and hyperparameters were blank when this configuration was chosen (repository inspected on 2026-09-28), so a direct difference cannot yet be claimed. Before full training, compare `configs/full.yaml` against Chaitanya's completed config; if it matches, coordinate and change Aswin's configuration before training.

## Hyperparameters and rationale

| Setting | Value | Rationale |
|---|---:|---|
| Blocks | 5 | More depth than the four-block starting example while remaining lab-scale |
| Embedding / heads | 240 / 6 | Head dimension 40; intentionally distinct candidate design |
| Feed-forward | 960 | Standard 4x expansion |
| Context | 160 | Captures short story dependencies with moderate attention cost |
| Dropout | 0.12 | Mild regularization for a small decoder |
| Batch size | 48 | Conservative starting point for GPU memory |
| AdamW LR / decay | 2.5e-4 / 0.1 | Stable decoder training defaults |
| Epochs | 12 | Exceeds the required minimum of 10 |
| Schedule | 5% linear warm-up, cosine decay to 10% LR | Avoids abrupt early updates and decays smoothly |
| Gradient clipping | 1.0 | Limits unstable steps; unclipped norm is recorded |

## Training and evaluation procedure

Cross-entropy ignores `<PAD>`. Every optimization step records loss, learning rate, and the pre-clipping gradient norm in the raw log. A loss spike is a step more than three standard deviations above the previous 20 finite losses. Non-finite losses are counted and skipped. Each epoch computes held-out loss and top-1 accuracy; best and final checkpoints are saved. Runtime, non-padding tokens/second, parameter count, and peak process/GPU memory are recorded. Validation perplexity is `exp(validation CE)`, bits per character is `validation CE / ln(2)`, and the generalization gap is validation CE minus training CE.

Generation includes greedy decoding and temperatures 0.7, 1.0, and 1.2 for three fixed prompts. Each JSONL record stores the prompt, mode/temperature, checkpoint, output, token count, elapsed time, and speed. Distinct-1/2/3 are unique character n-grams divided by total character n-grams across continuations; repeated 4-gram rate is excess repeated 4-gram occurrences divided by all 4-gram occurrences.

## Results, samples, and hardware

Populate this section only after completing the full run. The authoritative values will be the `full_run01` row in `metrics_report.csv`, the plot in `outputs/plots/full_run01_loss_curves.png`, generated samples in `outputs/generated_text/full_run01_samples.*`, and the captured reproducibility manifest. Hardware must come from the run log/manifest rather than memory or estimation.

## Strengths, weaknesses, limitations, and future work

The implementation is auditable because attention and masking are explicit, and all requested performance/stability measurements share one run ID. Character tokenization has no word-level out-of-vocabulary problem, but produces long dependencies and slower semantic learning than subword tokenization. One fixed crop per story leaves some text unused. Future controlled experiments could compare context lengths, tied input/output embeddings, or multiple windows per story without changing the evaluation split.
