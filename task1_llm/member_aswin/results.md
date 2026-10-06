# Task 1 - GPT from scratch (Aswin)

## Status

`full_run01` is complete. It trained for 12 epochs on Google Colab Pro with an NVIDIA A100-SXM4-40GB and produced a best/final checkpoint, raw log, training history, loss curve, metrics, and 12 generated samples. The `full_run01` row in `metrics_report.csv` is the authoritative result; the earlier `smoke_run01` row is pipeline validation only and must not be reported as model performance.

## Dataset and character tokenization

The source is `roneneldan/TinyStories`. Of 2,119,719 raw records inspected, preprocessing found 0 missing records, 230 empty records, and 0 malformed records. A seeded permutation (`26617`) selected disjoint sets of exactly 100,000 training stories and 10,000 validation stories, containing 98,609,553 selected story characters. Each selected story creates one example of 161 IDs: `<BOS>`, literal characters, `<EOS>`, then either a seeded crop or `<PAD>` padding. Input is the first 160 IDs and target is the same sequence shifted one character forward. The executed notebook verified the tensor shapes `(100000, 161)` and `(10000, 161)` and confirmed the shift invariant.

The learned vocabulary has 111 entries: four documented special IDs followed by 107 sorted literal characters. `<PAD>` masks padded targets from loss, `<UNK>` handles unseen prompt characters, `<BOS>` marks the beginning, and `<EOS>` permits stopping. `char_to_idx`, `idx_to_char`, frequencies, split sizes, seed, and context length are saved in `data_processed/full_run01/metadata.json` when preprocessing is rerun and are embedded in both preserved checkpoints.

## Architecture

The model is a pre-layer-normalized decoder-only Transformer with 5 blocks, 240-dimensional embeddings, 6 heads (40 dimensions per head), 960-dimensional GELU feed-forward layers, dropout 0.12, and context length 160. Token and position embeddings are learned. Every block has manual linear Q/K/V projections, manual head reshaping, scaled dot-product scores, an explicit lower-triangular Boolean mask, softmax, concatenation, output projection, two residual paths, and two layer normalizations. The final normalized states are projected to character logits.

This is materially different from Chaitanya's submitted model: Aswin used 5 blocks, width 240, 6 heads, context 160, and 3,558,960 parameters; Chaitanya used 6 blocks, width 384, 6 heads, context 256, and 10,811,136 parameters. Chaitanya also used cleaned sliding windows, whereas Aswin used one seeded crop/padded sequence per selected story. Consequently, the two members' metric values are not a controlled architecture-only comparison.

## Hyperparameters and rationale

| Setting | Value | Rationale |
|---|---:|---|
| Blocks | 5 | More depth than the four-block starting example while remaining lab-scale |
| Embedding / heads | 240 / 6 | Head dimension 40; intentionally distinct candidate design |
| Feed-forward | 960 | Standard 4x expansion |
| Context | 160 | Captures short story dependencies with moderate attention cost |
| Dropout | 0.12 | Mild regularization for a small decoder |
| Batch size | 48 | Stable full-run batch size with ample A100 memory headroom |
| AdamW LR / decay | 2.5e-4 / 0.1 | Stable decoder training defaults |
| Epochs | 12 | Exceeds the required minimum of 10 |
| Schedule | 5% linear warm-up, cosine decay to 10% LR | Avoids abrupt early updates and decays smoothly |
| Gradient clipping | 1.0 | Limits unstable steps; unclipped norm is recorded |

## Training and evaluation procedure

Cross-entropy ignores `<PAD>`. Every optimization step records loss, learning rate, and the pre-clipping gradient norm in the raw log. A loss spike is a step more than three standard deviations above the previous 20 finite losses. Non-finite losses are counted and skipped. Each epoch computes held-out loss and top-1 accuracy; best and final checkpoints are saved. Runtime, non-padding tokens/second, parameter count, and peak process/GPU memory are recorded. Validation perplexity is `exp(validation CE)`, bits per character is `validation CE / ln(2)`, and the generalization gap is validation CE minus training CE.

Generation includes greedy decoding and temperatures 0.7, 1.0, and 1.2 for three fixed prompts. Each JSONL record stores the prompt, mode/temperature, checkpoint, output, token count, elapsed time, and speed. Distinct-1/2/3 are unique character n-grams divided by total character n-grams across continuations; repeated 4-gram rate is excess repeated 4-gram occurrences divided by all 4-gram occurrences.

## Results, samples, and hardware

### Learning behavior

Training and validation loss decreased in every epoch. Validation accuracy increased from 63.4923% at epoch 1 to 76.0059% at epoch 12, so the final epoch was also the best checkpoint.

| Epoch | Train CE | Validation CE | Validation accuracy |
|---:|---:|---:|---:|
| 1 | 1.94440 | 1.17773 | 0.63492 |
| 2 | 1.13335 | 0.97699 | 0.69409 |
| 3 | 0.99810 | 0.90299 | 0.71571 |
| 4 | 0.93609 | 0.86261 | 0.72846 |
| 5 | 0.89726 | 0.83630 | 0.73674 |
| 6 | 0.86935 | 0.81192 | 0.74323 |
| 7 | 0.84740 | 0.79645 | 0.74856 |
| 8 | 0.83001 | 0.78401 | 0.75202 |
| 9 | 0.81648 | 0.77426 | 0.75515 |
| 10 | 0.80576 | 0.76629 | 0.75770 |
| 11 | 0.79804 | 0.76141 | 0.75922 |
| 12 | **0.79321** | **0.75822** | **0.76006** |

### Final metrics

| Metric | `full_run01` value |
|---|---:|
| Training CE loss | 0.793211 |
| Validation CE loss | 0.758225 |
| Perplexity | 2.134483 |
| Bits per character | 1.093887 |
| Generalization gap (validation - train CE) | -0.034986 |
| Top-1 next-character accuracy | 0.760059 |
| Distinct-1 / Distinct-2 / Distinct-3 | 0.016686 / 0.117963 / 0.322015 |
| Repeated 4-gram rate | 0.511463 |
| Maximum / mean pre-clipping gradient norm | 6.870716 / 0.720087 |
| Loss spikes / non-finite losses | 125 / 0 |
| Parameters | 3,558,960 |
| Training throughput | 233,568.39 non-padding tokens/s |
| Generation throughput | 241.11 tokens/s |
| Peak GPU memory | 1,017.89 MB |
| Total training time | 822.02 s (13 min 42.02 s) |

The slightly negative reported generalization gap is not evidence that validation data leaked into training. Training CE was measured while dropout was active and over all training updates, whereas validation CE was measured with dropout disabled at the end of the epoch; the two values are therefore not perfectly like-for-like. The 125 detected loss spikes use the deliberately sensitive rule defined above. There were no NaNs, and gradient clipping at 1.0 kept optimization stable; recorded gradient norms are the values before clipping.

### Representative generation

Greedy output was grammatical at first but tended to collapse into repetitions. Temperature 0.7 was usually more coherent, while 1.2 increased variety at the cost of grammar and semantic consistency.

| Prompt / decoding | Representative output excerpt |
|---|---|
| `Once upon a time`, temperature 0.7 | “Once upon a time, there was a little girl named Lily. She loved to play outside in the park…” |
| `Lily found a little`, greedy | “Lily found a little bird in the grass. She was so happy…” followed by repeated friend/play clauses |
| `The dragon was afraid`, temperature 0.7 | “The dragon was afraid of himself and said, \"I know, my name is Ben. It is too high.\"” |

All 12 complete samples and per-sample generation times are in `outputs/generated_text/full_run01_samples.jsonl`. The aggregate repeated 4-gram rate of 0.5115 quantitatively agrees with the visible repetition. `failure_analysis.md` documents exactly three real failures from that file.

### Hardware and artifacts

The run used PyTorch 2.11.0+cu130 with CUDA runtime 13.0 on an NVIDIA A100-SXM4-40GB in Google Colab Pro. The best checkpoint is epoch 12 (`best_model.pt`, SHA-256 `faac36643b139b32497399b3931351017224a8787f6f1f3717ce9967d21d35e7`); the separately saved final checkpoint is also epoch 12 (`final_model.pt`, SHA-256 `80d39990a1b118a36574749c14ab244a617ff2799f2b16716e9f5873f50f4ee0`). Exact Python, OS, and full package-freeze details were not printed by the run and are explicitly marked as not recorded in the evidence manifest rather than reconstructed after the fact.

## Strengths, weaknesses, limitations, and future work

The implementation is auditable because attention and causal masking are explicit, and all requested performance/stability measurements share one immutable run ID. Optimization was stable: both losses improved monotonically, no non-finite loss occurred, and the 3.56M-parameter model trained quickly with only about 1.0 GB peak GPU memory.

The main weakness is generation quality rather than next-character prediction. Character tokenization creates long dependencies, the 160-character context is short for story-level consistency, and one fixed crop per story leaves most characters unused. Greedy decoding enters repetitive loops, reflected by the 0.5115 repeated 4-gram rate. The selected raw character vocabulary also retained mojibake sequences such as `â€œ`, which appear in a temperature-1.2 sample.

The highest-priority future experiment is to normalize Unicode and reject mojibake before vocabulary construction, then train on multiple non-overlapping/sliding windows per story using the same fixed validation split. Controlled follow-ups could compare context 256, tied input/output embeddings, a larger model, and repetition-aware decoding. Only one factor should change per run so improvements can be attributed correctly.
