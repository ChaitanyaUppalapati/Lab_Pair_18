# Task 1 — GPT From Scratch (Chaitanya)

> Values below come from `configs/full.yaml` and the run artifacts. The **Why** column is mine to write.

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
| Interpretation of "100K" | 100K stories | |
| Drop garbled stories | yes (keep `“ ”`) | |
| Deduplication | exact match | |
| Block size | 256 | |
| Train stride | 128 | |
| `<EOS>` separator | yes | |

## Architecture
Pre-LN decoder-only Transformer written from scratch in `src/model.py` (own attention, causal mask and LayerNorm; no prebuilt Transformer/attention modules).

| Setting | Value | Why |
|---|---|---|
| n_layers | 6 | |
| n_heads | 6 (head dim 64) | |
| d_model | 384 | |
| d_ff | 1536 | |
| FFN activation | GELU | |
| dropout | 0.1 (embeddings, attention weights, residual branches) | |
| LayerNorm placement | Pre-LN + final LayerNorm before LM head | |
| Positional embedding | learned, 256 positions | |
| Weight tying | no | |
| Parameter count | 10,811,136 | |

## Training
| Setting | Value | Why |
|---|---|---|
| Loss | next-character cross-entropy | |
| Optimizer | AdamW, betas (0.9, 0.99) | |
| Weight decay | 0.1 on 2-D weights only | |
| Peak LR | 1e-3 | |
| Warm-up | 1,000 steps, linear from 0 | |
| Schedule | cosine decay to 1e-4 at the last step | |
| Batch size | 64 sequences × 256 chars | |
| Epochs | 10 (109,440 steps) | |
| Grad clipping | 1.0 (global norm) | |
| Precision | fp32 | |
| Init | N(0, 0.02) for linear/embedding weights, zero biases | |

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

## Evidence
- Config: `configs/full.yaml`
- Raw log: `reproducibility/raw_logs/task1_llm/chaitanya/full_run01.log`
- Manifest: `reproducibility/manifests/task1_llm/chaitanya/full_run01.json`
- Checkpoint ID: `checkpoints/full_run01/final.pt` (sha256 in the manifest)
- Notebook with outputs: `src/task1_chaitanya.ipynb`
