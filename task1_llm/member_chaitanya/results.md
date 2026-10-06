# Task 1 — GPT From Scratch (Chaitanya)

> Values below come from `configs/full.yaml` and the run artifacts.

## Data preprocessing
- Source: TinyStories `train` split, 2,119,719 stories.
- Cleaning: dropped 230 empty stories and 130,139 stories containing characters outside printable ASCII + newline + `“ ”` (mis-decoded text such as `â€™`); then removed 299,039 exact duplicates → 1,690,311 unique stories.
- Split (story level, seed 1337): 100,000 train / 10,000 validation stories; no story appears in both.
- Tokenisation: character level; `char_to_idx` / `idx_to_char` built from the training stories plus `<EOS>` (end of story) and `<UNK>`. Vocabulary size: 85. Unknown characters in validation: 0.
- Encoding: each split is one stream `story <EOS> story <EOS> …` (train 89,654,873 chars, val 9,003,962 chars).
- Sequences: block size 256; each window is 257 chars, input = first 256, target = last 256 (shifted by one). Training windows use stride 128 (700,427 windows); validation uses non-overlapping windows (35,171).
- Details: `data_processed/tinystories_s1337_clean/meta.json` (regenerable, not committed).

| Choice | Value | Why |
|---|---|---|
| Interpretation of "100K" | 100K stories | The assignment fixes the budget at 100K/10K, and I read it as stories because the story is the natural unit to shuffle and split; splitting windows would let overlapping windows from one story land on both sides. 100K stories already give 89.7M training characters (about 8 per parameter per epoch); all 1.69M clean stories would have meant about 17× the compute (≈ 40 h for 10 epochs instead of 2.3 h). |
| Drop garbled stories | yes (keep `“ ”`) | 130,139 stories (6%) contained mis-decoded text such as `â€™`; training on them would spend vocabulary on mojibake and teach the model to produce it, and with 1.69M clean stories left, dropping them costs nothing. Curly quotes are legitimate punctuation in the GPT-4 stories and cost only two vocabulary entries; removing every story containing them would have thrown away good data. Result: vocabulary 85, zero unknown characters in validation. |
| Deduplication | exact match | 299,039 stories (14%) were exact duplicates; a duplicate split across train and validation inflates the validation score, so removing them protects the evaluation. Exact matching is cheap and deterministic. I did not remove near-duplicates (MinHash), a limitation since TinyStories is formulaic; the 0.025 generalisation gap suggests leakage is not driving the numbers. |
| Block size | 256 | An average story is about 900 characters, so 256 covers roughly 30% of one: enough for sentence-level grammar and a few sentences of context. Attention cost grows with the square of the context, and 256 let 10 epochs finish in 2.3 h on the 4090. The cost is long-range coherence, which shows up in my failure cases. |
| Train stride | 128 | With stride 128 every character is a target twice: once early in a window with little context and once late with full context; non-overlapping windows would only ever show a character with one fixed amount of context. It doubles the windows (700,427 vs ≈ 350K) and so the cost per epoch; a smaller stride would add highly correlated windows for more compute. Validation uses non-overlapping windows so each character is scored once. |
| `<EOS>` separator | yes | Without a separator the model treats the start of the next story as a continuation of the previous one. With `<EOS>` it learns where stories end, generation can stop cleanly, and windows that cross a boundary teach it to reset its context. A dedicated token rather than a blank line, because newlines also occur inside stories. |

## Architecture
Pre-LN decoder-only Transformer written from scratch in `src/model.py` (own attention, causal mask and LayerNorm; no prebuilt Transformer/attention modules).

| Setting | Value | Why |
|---|---|---|
| n_layers | 6 | 10.8M parameters against 89.7M training characters per epoch is a moderate ratio. Six layers give enough depth for context tracking, which Eldan & Li (2023) found matters more than width for coherence. The results suggest I could have gone bigger: the gap is only 0.025 and validation loss was still falling at epoch 10. |
| n_heads | 6 (head dim 64) | Six heads of dimension 64 (the GPT-2 head size) split d_model = 384 evenly; part of the same size budget as n_layers. |
| d_model | 384 | Width chosen together with depth for the ≈ 10.8M-parameter budget (see n_layers); the same shape as nanoGPT's character-level reference model. |
| d_ff | 1536 | The standard 4 × d_model feed-forward expansion. |
| FFN activation | GELU | GELU is the GPT-2 standard: smooth, with no dead units as with ReLU. At this scale the difference is small, and I did not ablate it. |
| dropout | 0.1 (embeddings, attention weights, residual branches) | Karpathy uses 0.2 for Tiny Shakespeare (1.1M characters); my data is about 80× larger, so I halved it. In hindsight even 0.1 may be more than needed: train and validation loss stayed within 0.025 and validation never turned upward. |
| LayerNorm placement | Pre-LN + final LayerNorm before LM head | Pre-LN keeps gradients well scaled at initialisation (Xiong et al., 2020), so training is stable without depending on warm-up; Post-LN is known to need careful warm-up. The final LayerNorm is needed because in Pre-LN the residual stream itself is never normalised and its scale grows with depth before it reaches the LM head. |
| Positional embedding | learned, 256 positions | The context is fixed at 256 and generation always crops to the last 256 characters, so the main advantage of sinusoidal encodings, extrapolating to longer sequences, is never used. Learned positions are the GPT-2 choice and cost only 98K parameters (under 1%). |
| Weight tying | no | Tying mainly saves parameters when the vocabulary is large; with 85 characters it would save 32,640 parameters (0.3%), so I kept input and output embeddings separate and let the head learn its own projection. Not ablated. |
| Parameter count | 10,811,136 | Result of the shape above (see n_layers for the sizing argument). |

## Training
| Setting | Value | Why |
|---|---|---|
| Loss | next-character cross-entropy | The objective required by the assignment (1.3.1); perplexity and bits per character are derived directly from it. |
| Optimizer | AdamW, betas (0.9, 0.99) | With β₂ = 0.99 the second-moment estimate averages over roughly the last 100 steps instead of 1,000, so Adam adapts faster; nanoGPT uses this for character-level models because each step contains relatively few tokens (16K here). With 0 loss spikes it caused no instability. |
| Weight decay | 0.1 on 2-D weights only | 0.1 is the GPT-2 / nanoGPT standard. Decaying biases and LayerNorm gains pulls them toward zero, which interferes with normalisation without adding useful regularisation, so decay applies only to the 2-D matrices, embeddings included. |
| Peak LR | 1e-3 | Small models tolerate a higher learning rate, and nanoGPT uses 1e-3 for this exact shape. Pre-LN, warm-up and clipping made it safe, and the run had 0 spikes. I did not sweep it. |
| Warm-up | 1,000 steps, linear from 0 | About 0.9% of the 109,440 steps: a short ramp that lets Adam's moment estimates settle before full-size steps. The largest logged gradient norm (9.63) is at the very first step and the mean afterwards was 0.20, so warm-up handled the only unstable phase. Because stride 128 doubled the step count this is a smaller fraction than I first planned (≈ 2%), and it was still enough. |
| Schedule | cosine decay to 1e-4 at the last step | A floor of 10% of the peak is the nanoGPT / Chinchilla convention. Decaying to zero makes the last stretch of training nearly useless, and my validation loss was still improving in epoch 10, so those final steps were productive. Decaying to zero sometimes gives a slightly lower final loss; I did not test it. |
| Batch size | 64 sequences × 256 chars | 16,384 characters per step matches the reference configuration. It used only 4.6 GB, so memory was not the limit; with a fixed number of epochs a larger batch would mean fewer optimiser steps, and throughput was already good at 235K tokens/s. |
| Epochs | 10 (109,440 steps) | Ten is the assignment minimum. The model was still underfitting rather than overfitting: validation loss fell every epoch (by 0.003 in the last one) and the gap is only 0.025, so more epochs would likely have helped. With stride 128 every character is a target twice per epoch, so 10 epochs is closer to 20 passes over the data. |
| Grad clipping | 1.0 (global norm) | Insurance. Of the logged norms (every 50 steps), the only large one, 9.63, is at the first step, and the mean of 0.20 means clipping was almost never active after warm-up. It cost nothing; 0 spikes, 0 NaNs. |
| Precision | fp32 | Simplicity, and no numerical issues in hand-written attention (softmax over a −∞ mask and the loss are the usual trouble spots in fp16). The 4090 had plenty of headroom (4.6 GB peak) and 2.3 h was acceptable; bf16 would probably have roughly halved the time, a trade I knowingly gave up. |
| Init | N(0, 0.02) for linear/embedding weights, zero biases | The GPT-2 initialisation. It keeps the initial logits small, so the starting loss is close to ln(85) ≈ 4.44, which I used as a sanity check (logged first-step loss: 4.40). Biases start at zero. |

Loss curves: `outputs/full_run01/loss_curves.png`; gradient norm and LR: `outputs/full_run01/stability.png`.

## Generation
- Default decoding: temperature 0.8. Also greedy and temperature 1.0 for the failure analysis.
- Prompts: "Once upon a time", "Lily found a little", "The dog was sad because"; up to 400 new characters, stopping at `<EOS>`.
- Samples: `outputs/full_run01/samples.txt`

## Metrics
See `metrics_report.csv` (decoding metrics at temperature 0.8) and `outputs/full_run01/generation_metrics.csv` (per temperature). Metric definitions are in the docstring of `src/evaluate.py`.

Summary (final checkpoint, epoch 10):

| Metric | Value |
|---|---|
| Train CE loss (eval mode) | 0.4842 |
| Validation CE loss | 0.5097 |
| Perplexity | 1.665 |
| Bits per character | 0.735 |
| Generalization gap (val − train) | 0.0255 |
| Top-1 next-char accuracy (val) | 83.58% |
| Distinct-1 / 2 / 3 (T = 0.8) | 0.344 / 0.778 / 0.929 |
| Repeated 4-gram rate (T = 0.8) | 0.0195 |
| Grad norm, mean / max (pre-clip) | 0.201 / 9.63 (max is the first step) |
| Loss spikes / NaNs | 0 / 0 |
| Parameters | 10,811,136 |
| Training tokens/sec | 235,385 |
| Generation tokens/sec (batch 1) | 180 |
| Peak GPU memory | 4,598 MB |
| Total training time | 8,458 s (2 h 21 min, incl. per-epoch evaluation) |

Per-epoch losses:

| Epoch | Train (eval mode) | Val | Val acc |
|---|---|---|---|
| 1 | 0.6217 | 0.6271 | 79.98% |
| 2 | 0.5878 | 0.5953 | 80.97% |
| 3 | 0.5689 | 0.5777 | 81.50% |
| 4 | 0.5545 | 0.5651 | 81.88% |
| 5 | 0.5418 | 0.5544 | 82.22% |
| 6 | 0.5268 | 0.5412 | 82.59% |
| 7 | 0.5141 | 0.5317 | 82.92% |
| 8 | 0.4999 | 0.5201 | 83.23% |
| 9 | 0.4895 | 0.5129 | 83.46% |
| 10 | 0.4840 | 0.5097 | 83.58% |

## Hardware disclosure
- GPU: NVIDIA GeForce RTX 4090 (24 GB); CPU: AMD Ryzen 9 7950X; PyTorch 2.11.0+cu128.
- Peak memory: 4,598 MB (GPU, allocated)
- Total training time: 8,458 s

## Comparison with teammates
Aswin's full run is not in the repository yet (only a smoke test), so I compare the designs and state what I expect. His numbers will be added only if they arrive in time **and** are evaluated on the same validation set and vocabulary; otherwise bits per character are not comparable.

| | Mine | Aswin |
|---|---|---|
| Model | 6 layers, d_model 384 (10.8M parameters) | 5 layers, d_model 240 (≈ 3.5M parameters) |
| Context | 256 | 160 |
| Text seen per epoch | sliding windows, stride 128: ≈ 179M target characters | one crop per story: ≈ 16M characters |
| Peak LR / epochs | 1e-3 / 10 | 2.5e-4 / 12 |

Expectations: his 160-character context covers only about 18% of a story, so I expect weaker consistency of names and plot. His smaller model should give a clearly higher validation loss. One crop per story means about 10× less text seen per epoch, so I expect a higher loss and possibly earlier overfitting, and his lower learning rate should slow convergence further.

## Evidence
- Config: `configs/full.yaml`
- Raw log: `reproducibility/raw_logs/task1_llm/chaitanya/full_run01.log`
- Manifest: `reproducibility/manifests/task1_llm/chaitanya/full_run01.json`
- Checkpoint ID: `checkpoints/full_run01/final.pt` (sha256 in the manifest)
- Notebook with outputs: `src/task1_chaitanya.ipynb`
