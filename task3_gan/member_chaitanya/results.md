# Task 3 — CycleGAN Monet ↔ Photo (Chaitanya)

> Values below come from `configs/full.yaml` and the run artifacts. The **Why** column and the analysis sections are mine to write.

## Data
- Domain naming follows the Kaggle competition: **A = Monet** (300 images, `monet_jpg`), **B = photo** (7,038 images, `photo_jpg`). All images are 256×256 JPG.
- One epoch = one pass over the 7,038 photos (shuffled), each paired with a Monet painting drawn uniformly at random.
- Training augmentation (both domains, independently): resize to 286×286 (bicubic), random crop 256, random horizontal flip. No colour jitter, rotation or vertical flip.
- Evaluation / inference: images used at 256×256 with no augmentation.

| Choice | Value | Why |
|---|---|---|
| Epoch definition | one pass over photos | |
| Augmentation | resize 286 → crop 256, h-flip, both domains | |
| No colour jitter / rotation / v-flip | — | |

## Architecture
| Component | Design | Why |
|---|---|---|
| Generator G_A2B: Monet → photo | ResNet: c7s1-64, d128, d256, 9 × R256, u128, u64, c7s1-3 + tanh | |
| Generator G_B2A: photo → Monet | same as G_A2B | |
| Upsampling | nearest-neighbour 2× + 3×3 conv | |
| Padding | reflection | |
| Discriminator D_A (Monet), D_B (photo) | 70×70 PatchGAN: C64-C128-C256-C512 (4×4), 4×4 conv → 1 channel (30×30 output map) | |
| Normalization | InstanceNorm (none on first D layer); LeakyReLU 0.2 in D | |
| Initialisation | N(0, 0.02) conv weights, zero bias | |
| Parameter count | G: 11,378,179 each; D: 2,764,737 each; total 28,285,832 | |

## Training
| Setting | Value | Why |
|---|---|---|
| Adversarial loss type | LSGAN (MSE) | |
| D loss scale | × 0.5 | |
| λ_cycle | 10 | |
| λ_identity | 5 (0.5 × λ_cycle) | |
| Optimizer / LR / betas | Adam, 2e-4 for G and D, (0.5, 0.999) | |
| LR schedule | constant for epochs 1–20, then linear decay to 0 at the last iteration of epoch 40 | |
| Batch size, epochs | 1, 40 (281,520 iterations) | |
| Image buffer | 50 per discriminator | |
| Precision | fp32 (cuDNN autotuning on) | |
| Gradient clipping | none | |
| DiffAugment on D_A | off (optional switch in config) | |

Loss curves: `outputs/full_run01/loss_curves.png`. Per-epoch translation grids: `outputs/full_run01/grids/epoch_001.png` … `epoch_040.png` (rows: Monet | Monet→photo | reconstruction | photo | photo→Monet | reconstruction).

### Per-epoch losses (epoch means, from the raw log)
| Epoch | G total | cycle A | cycle B | identity A | identity B | D_A (Monet) | D_B (photo) | D_A(real) / D_A(fake) |
|---|---|---|---|---|---|---|---|---|
| 1 | 7.958 | 0.228 | 0.244 | 0.221 | 0.216 | 0.193 | 0.221 | 0.72 / 0.28 |
| 5 | 5.967 | 0.157 | 0.172 | 0.157 | 0.151 | 0.132 | 0.146 | 0.77 / 0.23 |
| 10 | 5.386 | 0.130 | 0.147 | 0.133 | 0.132 | 0.067 | 0.134 | 0.89 / 0.11 |
| 13 | 5.166 | 0.121 | 0.137 | 0.121 | 0.122 | 0.050 | 0.126 | 0.92 / 0.08 |
| 20 | 4.881 | 0.107 | 0.124 | 0.105 | 0.110 | 0.038 | 0.095 | 0.94 / 0.06 |
| 25 | 4.601 | 0.097 | 0.113 | 0.090 | 0.100 | 0.031 | 0.086 | 0.95 / 0.05 |
| 30 | 4.362 | 0.090 | 0.108 | 0.082 | 0.093 | 0.047 | 0.078 | 0.91 / 0.09 |
| 35 | 4.222 | 0.084 | 0.099 | 0.074 | 0.087 | 0.040 | 0.072 | 0.92 / 0.08 |
| 40 | 4.119 | 0.079 | 0.092 | 0.068 | 0.082 | 0.032 | 0.063 | 0.93 / 0.07 |

Every epoch is in `reproducibility/raw_logs/task3_gan/chaitanya/full_run01.log` and `outputs/full_run01/history.json`.

## Cycle-consistency verification

## Training stability / convergence
Facts from the run (analysis to be written):
- NaN count: 0. Max gradient norm (pre-step, no clipping): G 1,561.7, D 482.9.
- D_A (Monet) loss fell below 0.05 at epoch 13 (overfitting warning logged for epochs 13–16), bottomed at 0.031 (epoch 25), rose to 0.048 (epoch 29) during LR decay, ended at 0.032.
- Visual observations from the grids: dark circular blob artifacts at epoch 1 (mostly gone by epoch 10); green colour cast in photo→Monet at epoch 10, gone at 13, teal cast back at 20, gone by 40; vertical streak texture in skies around epochs 13–20; orange "fire" hallucination in the olive-tree Monet→photo sample at epoch 20.

## Visual quality assessment

## Evaluation metrics
All metrics: `full_metrics_report.csv` (definitions in the docstring of `evaluate_local.py`). Feature-based metrics use Inception-v3 (torchmetrics) for evaluation only.

| Metric | A2B (Monet → photo) | B2A (photo → Monet) |
|---|---|---|
| Inputs translated | 300 | 7,038 |
| FID | 87.49 | 91.46 |
| KID (mean ± std) | 0.0244 ± 0.0026 | 0.0193 ± 0.0030 |
| Precision / recall (k = 5) | 0.683 / 0.430 | 0.407 / 0.730 |
| Density / coverage | 0.725 / 0.760 | 0.263 / 0.587 |
| Cycle-reconstruction L1 ([0,1] pixels) | 0.0458 | 0.0500 |
| LPIPS input vs translation | 0.360 | 0.417 |
| LPIPS input vs reconstruction | 0.429 | 0.318 |
| Content cosine similarity (input vs translation) | 0.782 | 0.740 |
| Generator inference images/sec | 216 | 300 |

Training: 26,494 s (7 h 22 min), 10.63 it/s (21.3 images/sec, one image per domain per iteration). Peak GPU memory allocated: 19,154 MB (steady state ≈ 3.2 GB; the peak most likely comes from cuDNN autotuning trying memory-hungry algorithms).

## Human audit (30 fixed samples, 2 raters, blinded)
- Rubric:
- Scores:
- Cohen's kappa / % agreement:

## Kaggle
Official protocol: `src/official_eval.py`, a port of the instructor's `Part3_Evaluation_Script.ipynb` (first 300 images per folder, both directions, FID and MiFID averaged over A2B and B2A).

| Date (2026-10-02) | FID | MiFID | Public score | Note |
|---|---|---|---|---|
| 15:46 UTC | 83.9280 | 0.4026 | −42.1653 | **Not official**: computed with `src/kaggle_score.py` (photo→Monet only, all 7,038 images, vs `real_stats.npz`) before the evaluation script was available. Instructor asked to remove it. |
| 17:35 UTC | 107.2547 | 0.4148 | −53.8347 | **Correction**, official script on the same epoch-40 model (B2A: FID 104.887 / MiFID 0.4066; A2B: FID 109.623 / MiFID 0.4230). |

- Current `submission.csv` = the official values (row 2). Details: `outputs/official_eval.json`.
- Leaderboard rank: pending removal of the non-official entry (Kaggle shows a team's best public score).
- Official score of every saved snapshot (epochs 5–40) improved monotonically: −61.75, −58.35, −56.65, −56.27, −55.53, −54.64, −54.50, −53.87; the final model is the best.

## Run 2 — DiffAugment on both discriminators (submitted model)
Config `configs/full_run02.yaml`: identical to run 1 except `diffaugment_monet_d: true` and `diffaugment_photo_d: true` (translation + cutout applied to real and generated images before D_A and D_B). Same seed, schedule, 40 epochs. The training loop was also changed to keep per-iteration statistics on the GPU (identical results, verified on the smoke config); the first attempt with the old loop was aborted after ~13 min and its log kept as `full_run02_attempt1_aborted.log`.

| Choice | Value | Why |
|---|---|---|
| DiffAugment | on, both discriminators | |
| Model selection | best official score among 5-epoch snapshots | |

Official score of every snapshot (`outputs/full_run02/snapshot_scores.csv`; run 1 in `outputs/full_run01/snapshot_scores_official.json`):

| Epoch | Run 1 score | Run 2 FID B2A | Run 2 FID A2B | Run 2 FID | Run 2 MiFID | Run 2 score |
|---|---|---|---|---|---|---|
| 5 | −61.75 | 125.53 | 123.06 | 124.30 | 0.4135 | −62.36 |
| 10 | −58.35 | 108.87 | 119.47 | 114.17 | 0.4129 | −57.29 |
| 15 | −56.65 | 103.74 | 126.87 | 115.31 | 0.4123 | −57.86 |
| 20 | −56.27 | 105.99 | 113.02 | 109.51 | 0.4136 | −54.96 |
| 25 | −55.53 | 101.16 | 109.83 | 105.50 | 0.4119 | −52.95 |
| 30 | −54.64 | 101.33 | 109.28 | 105.31 | 0.4088 | −52.86 |
| **35** | −54.50 | 99.42 | 105.65 | 102.53 | 0.4082 | **−51.47** |
| 40 | −53.87 | 100.15 | 105.89 | 103.02 | 0.4048 | −51.72 |

Training facts: 28,102 s (7 h 48 min), 10.02 it/s, NaN count 0, max grad norm G 1,627.3 / D 381.5. D_A (Monet) loss stayed between 0.078 and 0.214 for the whole run (run 1: fell to 0.031); final epoch cycle losses 0.071 / 0.083 (run 1: 0.079 / 0.092).

Local metrics of the submitted epoch-35 generators (`full_metrics_report.csv`, run_id `full_run02`):

| Metric | A2B (Monet → photo) | B2A (photo → Monet) |
|---|---|---|
| FID (torchmetrics, all images) | 86.65 | 84.12 |
| KID mean | 0.0236 | 0.0146 |
| Precision / recall | 0.730 / 0.427 | 0.453 / 0.727 |
| Density / coverage | 0.892 / 0.863 | 0.377 / 0.733 |
| Cycle L1 | 0.0427 | 0.0481 |
| LPIPS input vs translation | 0.360 | 0.409 |
| Content cosine similarity | 0.780 | 0.735 |

Kaggle (official script on these outputs): FID 102.5307, MiFID 0.4082, public score −51.4694 (2026-10-03). Full leaderboard has 35 teams: this score ranks 25th once the non-official −42.1653 entry is removed (top 20 needs > −50.29, top 10 > −47.69).

## Hardware disclosure
- GPU: NVIDIA GeForce RTX 4090 (24 GB); CPU: AMD Ryzen 9 7950X; PyTorch 2.11.0+cu128.
- Training time: 26,494 s. Peak memory: 19,154 MB allocated (see note above).

## Shortcomings

## Evidence
- Config: `configs/full.yaml`
- Raw log: `reproducibility/raw_logs/task3_gan/chaitanya/full_run01.log`
- Manifest: `reproducibility/manifests/task3_gan/chaitanya/full_run01.json`
- Metrics: `full_metrics_report.csv`, `submission.csv`, `outputs/kaggle_score.json`
- Checkpoints: `checkpoints/full_run01/G_A2B.pt`, `G_B2A.pt`, `D_A.pt`, `D_B.pt` (sha256 of G_B2A in the manifest)
- Predictions: `outputs/pred_A2B/` (300 Monet → photo), `outputs/pred_B2A/` (7,038 photo → Monet)
