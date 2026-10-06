# Task 3 — CycleGAN Monet ↔ Photo (Chaitanya)

> Values below come from `configs/full.yaml` and the run artifacts.

## Data
- Domain naming follows the Kaggle competition: **A = Monet** (300 images, `monet_jpg`), **B = photo** (7,038 images, `photo_jpg`). All images are 256×256 JPG.
- One epoch = one pass over the 7,038 photos (shuffled), each paired with a Monet painting drawn uniformly at random.
- Training augmentation (both domains, independently): resize to 286×286 (bicubic), random crop 256, random horizontal flip. No colour jitter, rotation or vertical flip.
- Evaluation / inference: images used at 256×256 with no augmentation.

| Choice | Value | Why |
|---|---|---|
| Epoch definition | one pass over photos | Matches the official implementation's max(len(A), len(B)) and guarantees every photo is used once per epoch, with a random Monet each step, so "epoch" is a unit teammates can compare. The consequence is that each painting is reused about 23 times per epoch, which is why I watched the Monet discriminator for overfitting. |
| Augmentation | resize 286 → crop 256, h-flip, both domains | The paper's default. Random crops add translation jitter, effectively extra data for only 300 paintings, and horizontal flips do not change what a landscape is. Applied to both domains so neither discriminator sees systematically different image statistics. (Run 9 later showed the 286 resize also creates a small mismatch between training and inference scale.) |
| No colour jitter / rotation / v-flip | — | Colour is part of the Monet style: jitter would blur the palette the generator is trying to learn and fight the identity loss. Rotations and vertical flips create unnatural compositions (sky at the bottom) that the discriminator would accept as real. |

## Architecture
| Component | Design | Why |
|---|---|---|
| Generator G_A2B: Monet → photo | ResNet: c7s1-64, d128, d256, 9 × R256, u128, u64, c7s1-3 + tanh | ResNet-9 is the paper's generator for 256×256: it transforms features at 64×64 through residual blocks and keeps the global structure. A U-Net's skip connections pass high-resolution input detail straight to the output, which makes it easy to copy photo textures and under-stylise, and texture is exactly what should change here. |
| Generator G_B2A: photo → Monet | same as G_A2B | Symmetric task, same reasoning as G_A2B. |
| Upsampling | nearest-neighbour 2× + 3×3 conv | Transposed convolutions with stride 2 produce checkerboard artifacts from uneven kernel overlap (Odena et al., 2016), especially in flat sky and water. The human audit scores artifacts, so avoiding them was worth deviating from the paper, at the same compute cost. |
| Padding | reflection | Zero padding makes border pixels see artificial black, which creates dark frames and edge artifacts; reflection keeps the statistics continuous at the borders. The paper's choice. |
| Discriminator D_A (Monet), D_B (photo) | 70×70 PatchGAN: C64-C128-C256-C512 (4×4), 4×4 conv → 1 channel (30×30 output map) | Style is mostly local texture, which a 70×70 patch can judge. The PatchGAN outputs a 30×30 grid of decisions per image, so each painting supplies many training signals, which matters with only 300, and it is small (2.76M parameters); a full-image discriminator would memorise 300 compositions faster. The catch, which I learned later, is that FID depends on global colour and lighting, which a patch discriminator does not enforce. |
| Normalization | InstanceNorm (none on first D layer); LeakyReLU 0.2 in D | InstanceNorm normalises each image's own contrast and colour statistics, the standard for style transfer; BatchNorm at batch size 1 is unstable. The first discriminator layer has no normalisation (pix2pix / DCGAN convention) so it sees raw colour and intensity, exactly what separates the two domains. |
| Initialisation | N(0, 0.02) conv weights, zero bias | The paper / DCGAN default: small initial weights keep early outputs near neutral, so neither network starts with an extreme advantage. |
| Parameter count | G: 11,378,179 each; D: 2,764,737 each; total 28,285,832 | Follows from the paper architecture above. |

## Training
| Setting | Value | Why |
|---|---|---|
| Adversarial loss type | LSGAN (MSE) | LSGAN penalises samples by their distance from the decision boundary, so the generator keeps getting useful gradients even when the discriminator is confident, where BCE saturates. The paper found it more stable with better quality; I had 0 NaNs. |
| D loss scale | × 0.5 | Slows the discriminator relative to the generator, as in the paper: telling real from fake is the easier task, especially with only 300 paintings. |
| λ_cycle | 10 | The paper value. It weights cycle consistency well above the adversarial term, so content is preserved and outputs do not drift into hallucinated content. Later runs showed this metric does not reward content preservation and λ_cycle 2 scored better, but for a paper-faithful first run 10 was right. |
| λ_identity | 5 (0.5 × λ_cycle) | For Monet ↔ photo the paper adds an identity loss at 0.5 × λ_cycle to stop the generators shifting colours unnecessarily (e.g. turning daytime scenes into sunset tints). I wanted palette preservation in the first run. |
| Optimizer / LR / betas | Adam, 2e-4 for G and D, (0.5, 0.999) | The paper's settings. β₁ = 0.5 lowers momentum, which helps when each network's objective keeps changing as the other learns (the standard DCGAN-era recommendation). |
| LR schedule | constant for epochs 1–20, then linear decay to 0 at the last iteration of epoch 40 | The paper's 100 + 100 schedule halved to fit compute: the constant phase finds the solution and the linear decay lets both networks settle. Run 1's snapshot scores improved monotonically through the decay, which supports it. |
| Batch size, epochs | 1, 40 (281,520 iterations) | Batch 1 is paper-faithful and suits per-image InstanceNorm. 40 epochs = 281,520 iterations, close to the paper's horse ↔ zebra step count, and took 7.4 h on the 4090. |
| Image buffer | 50 per discriminator | Shows the discriminator a history of past fakes, not just the latest, which damps the back-and-forth where each network chases the other's newest move (Shrivastava et al., 2017). 50 is the paper default. |
| Precision | fp32 (cuDNN autotuning on) | Simplicity, and no fp16 issues in the squared LSGAN terms; 10.6 it/s was acceptable. The 19 GB peak came from cuDNN autotuning; steady state was about 3.2 GB. |
| Gradient clipping | none | The paper does not clip, and Adam scales each parameter's step by its own gradient history, so a large raw norm does not turn into a huge update. The curves show norms were large mostly at the start, with 0 NaNs, and picking a clip threshold without evidence could shift the generator/discriminator balance. The max norms (G 1,562 / D 483) are reported as the stability evidence. |
| DiffAugment on D_A | off (optional switch in config) | Run 1 is the paper-faithful baseline; I kept DiffAugment as a switch to turn on if the Monet discriminator overfit the 300 paintings, which it did (run 2). |

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
- Reconstruction error of the final model: cycle L1 ≈ 0.05 (photo → Monet → photo, 0.052) and 0.045 (Monet → photo →
  Monet), against ≈ 0.2 between unrelated images (live check in `src/task3_chaitanya.ipynb`). F(G(x)) is about four
  times closer to x than chance in both directions, and the cycle loss fell steadily throughout training, so the
  constraint is implemented and optimised correctly.
- What it does not prove is that the translation preserves content: failure case 2 has a normal cycle error (0.052)
  but the lowest content similarity of all 7,038 photos (0.357). Together these suggest the generator can hide
  information in the translation that the inverse generator decodes, a known CycleGAN behaviour (Chu et al., 2017).
- Reconstructions are slightly softer and less saturated than the inputs.

## Training stability / convergence
Facts from run 1:
- NaN count: 0. Max gradient norm (pre-step, no clipping): G 1,561.7, D 482.9.
- D_A (Monet) loss fell below 0.05 at epoch 13 (overfitting warning logged for epochs 13–16), bottomed at 0.031 (epoch 25), rose to 0.048 (epoch 29) during LR decay, ended at 0.032.
- Visual observations from the grids: dark circular blob artifacts at epoch 1 (mostly gone by epoch 10); green colour cast in photo→Monet at epoch 10, gone at 13, teal cast back at 20, gone by 40; vertical streak texture in skies around epochs 13–20; orange "fire" hallucination in the olive-tree Monet→photo sample at epoch 20.

Analysis: the cycle and identity losses converged smoothly and monotonically, but the adversarial part did not reach an
equilibrium.
- *Discriminators winning:* both discriminator losses kept falling while the generators' adversarial losses rose for the
  whole run.
- *A brief rebalance:* the one exception is around epochs 25–29, where the Monet discriminator's loss rose to 0.047 as
  the photo → Monet adversarial loss dipped (0.862 → 0.797).
- *Spikes:* small spikes appear throughout, but none diverge and there were no NaNs.
- *Visual oscillation:* colour casts appeared at epoch 10, cleared at 13 and returned at 20, so the generator–discriminator
  game was oscillating even when the epoch-mean losses looked smooth.

So the run was stable but not converged in the adversarial sense; the learning-rate decay is what made it settle.

## Visual quality assessment
From the fixed-sample grids (`outputs/full_run01/grids/epoch_040.png`, final model `outputs/full_run18/grids/epoch_123.png`):
- *Run 1, photo → Monet:* the brushwork and warm palette are convincing and structure is preserved (hills, figures,
  river). Skies are over-textured: the storm sky turns into pink-orange streaks and the sun gets a purple halo.
- *Run 1, Monet → photo:* mostly smoothing; brushwork is removed but the result looks like a blurred painting, not a
  photograph.
- *Final, Monet → photo:* much more photographic, with higher contrast and clean blue skies, but it invents lighting
  (the misty river becomes an orange sunset), which follows from dropping the identity loss; textures are still airbrushed.
- *Final, photo → Monet:* the palette is closer to real Monet (muted blue-green). Vertical streak artifacts remain in
  the skies, the sun washes out, and the "NT" watermark survives.
- *Both:* reconstructions are good, if slightly soft.

**Corner artifact.** Almost every output has a small dark blob in the top-left corner. I measured how many of the 16
generated tiles in each run's last grid have a top-left 5×5 corner more than 40 grey levels darker than the area beside it:

| Run (last grid) | Tiles with the blob |
|---|---|
| Run 1, epoch 40 | 1 of 16 |
| Run 2 (first run with DiffAugment), epoch 40 | 6 of 16 |
| Run 2b, epoch 80 | 11 of 16 |
| Runs 5–18 | 11–15 of 16 (15 of 16 in runs 7, 12, 16, 18) |

So it arrived with DiffAugment. My implementation shifts images by up to 32 pixels (12.5%) with fill, so the original
corner is often shifted out or replaced, and the discriminator rarely sees the true corner and cannot penalise what the
generator puts there. Why the generator wants a dark spot at all is a hypothesis, not a measured fact: possibly the
InstanceNorm "droplet" effect described for StyleGAN2 (a strong localised spike lets the generator manipulate the
normalisation statistics), with the top-left corner favoured because stride-2 convolutions align their sampling grid
to it. Two tests would settle it: inpaint the corners and re-score to measure the cost, and train with
reflection-padded translation to see whether it disappears.

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
- **Samples:** 30 fixed inputs chosen with a fixed seed (266) from the sorted file lists: 20 photo → Monet (the Kaggle
  direction) and 10 Monet → photo, translated by the submitted generators (`checkpoints/full_run18/epoch_123_ema/`).
  Items were shuffled and shown under anonymous IDs (S01–S30) with no file names, run names or scores; the answer key
  (`outputs/human_audit/key.csv`) was not opened until both raters had submitted. Tool: `src/human_audit.py`.
- **Raters:** rater 1 Chaitanya, rater 2 Aswin, each rating alone.
- **Rubric (integers 1–5):** *style*: does the output convincingly look like the target domain (5 = indistinguishable,
  1 = not at all); *content*: is the input's scene and layout preserved (5 = fully, 1 = unrecognisable);
  *artifacts*: 5 = none visible, 1 = severe (blotches, checkerboard, colour blow-outs, smears).

| Criterion | Mean (both raters) | Rater 1 / rater 2 mean | Photo → Monet / Monet → photo | Cohen's κ (unweighted / quadratic) | Exact agreement | Within 1 point |
|---|---|---|---|---|---|---|
| Style | 4.28 | 4.50 / 4.07 | 4.42 / 4.00 | −0.16 / −0.15 | 27% | 83% |
| Content | 4.28 | 4.80 / 3.77 | 4.33 / 4.20 | −0.06 / −0.01 | 23% | 60% |
| Artifacts | 3.85 | 3.87 / 3.83 | 3.67 / 4.20 | 0.28 / 0.23 | 50% | 83% |

Score distributions (counts of 2 / 3 / 4 / 5; nobody used 1): style rater 1 0/2/11/17, rater 2 2/5/12/11; content
rater 1 0/2/2/26, rater 2 3/9/10/8; artifacts rater 1 2/5/18/5, rater 2 3/8/10/9. Ratings:
`outputs/human_audit/rater_1.csv`, `rater_2.csv`; results: `outputs/human_audit/audit_results.json`.

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
| DiffAugment | on, both discriminators | A Monet-discriminator loss below 0.05 at epoch 13 meant it had memorised the 300 paintings, so its feedback stopped being informative. DiffAugment applies the same differentiable augmentation to real and fake images, so the discriminator cannot memorise and the generator is not taught to produce augmented-looking images. Both discriminators, to keep the setup symmetric. Result: D_A stayed between 0.078 and 0.214 and the score improved from −53.87 to −51.47. It also appears to have caused the corner blob (see Visual quality). |
| Model selection | best official score among 5-epoch snapshots | GAN loss values do not track output quality, so the official score, the number the leaderboard uses, is the only reliable selection signal, and scoring every 5 epochs was cheap. Caveats: selecting on the same images I am scored on biases the reported score upward, and the epoch 35 vs 40 difference (0.25) is within snapshot noise. |

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

## Final submitted model (as of 2026-10-04 19:50)

> From 2026-10-03 the model design was handed to Claude at the user's direction; every change after run 4, its reason and its result are in `DESIGN_LOG.md` (entries marked **[Claude]**).

The submission uses the two generators of one CycleGAN lineage (run 2 → 2b → 6 → 7 → … → 16 → 18), each taken from the training state that was best for its own direction (each direction's FID depends only on its own generator):

| Generator | Source | How it was trained |
|---|---|---|
| G_A2B (Monet → photo) | weight average of run-7 EMA snapshots 88 and 90 | run 6 (λ_cyc 5, λ_id 0.5) → run 7 (λ_cyc 2, λ_id 0), generator EMA 0.9999 |
| G_B2A (photo → Monet) | run-18 EMA snapshot 123 | runs 10 → 12 → 16 → 18: 2-scale discriminators, evaluation-scale training, λ_cyc 2 / λ_id 0, then fine-tuned with G_A2B frozen (the frozen generator acts as the fixed inverse in the cycle loss); run 18 = 6 more epochs at LR 1.5e-5 → 0 |

Checkpoint: `checkpoints/full_run18/epoch_123_ema/` (G_A2B is the frozen copy of `full_blends/epoch_b1_r7ema88-90/G_A2B.pt`).

| Metric | A2B (Monet → photo) | B2A (photo → Monet) |
|---|---|---|
| FID (torchmetrics, all images) | 81.89 | 75.50 |
| KID mean | 0.0211 | 0.0108 |
| Precision / recall | 0.737 / 0.400 | 0.597 / 0.690 |
| Density / coverage | 0.797 / 0.863 | 0.611 / 0.840 |
| Cycle L1 | 0.0445 | 0.0523 |
| LPIPS input vs translation | 0.411 | 0.441 |
| Content cosine similarity | 0.739 | 0.721 |

Note: on all 7,038 photos the run-16 EMA 116 generator has a slightly lower FID (75.03 vs 75.50); the official 300-image score, which Kaggle uses, favours run-18 EMA 123 by 0.08 FID. The difference is within snapshot-to-snapshot noise.

Official script (first 300 images per folder): FID_A2B 97.81, FID_B2A 94.21 → submission **FID 96.0061 / MiFID 0.3998**, Kaggle public score **−48.2029** (previous: run-16 EMA 116, −48.2244), rank ≈ 16 of 44 once the non-official −42.1653 entry is removed.

Score progression of the Kaggle submissions (official script): −53.83 (run 1) → −51.47 (run 2, DiffAugment) → −50.16 (run 2b) → −49.73 (weight average) → −49.49 (run 6, lower λ + EMA) → −48.93 / −48.69 (run 7, λ_cyc 2 / λ_id 0) → −48.55 → −48.40 → −48.27 → −48.22 → **−48.20** (per-direction generators, photo → Monet specialisation; run 18 EMA 123).

Reference points measured with the official script: untranslated images −63.5; two disjoint sets of 300 real photos −39 to −40.

## Run 2b — run 2 continued to 80 epochs
Config `configs/full_run02b.yaml`: resumes full_run02's epoch-40 state (generators, discriminators, Adam state, history; the 50-image pools start empty) and trains epochs 41–80 with a warm restart at LR 1e-4 decaying linearly to 0 at epoch 80. Everything else as run 2.

| Choice | Value | Why |
|---|---|---|
| Continue instead of retraining | resume run 2 at epoch 40 | The score was still improving at epoch 40. Continuing reused 7.8 h of training and tested the idea for about half the cost of a fresh 80-epoch run. Trade-off: it is not identical to a clean 80-epoch schedule (the Adam state carried over, but the image buffers restarted empty). |
| Restart LR | 1e-4 (half of the original 2e-4), linear decay to 0 | Restarting at the full 2e-4 could have pushed the weights out of the good solution they had reached; half the rate is large enough to keep improving and small enough not to undo progress (the warm-restart idea). It gained about 1.3 points (−51.47 → −50.16). |

| Epoch | FID B2A | FID A2B | FID | MiFID | Official score |
|---|---|---|---|---|---|
| 45 | 99.10 | 103.92 | 101.51 | 0.4057 | −50.96 |
| 50 | 99.37 | 103.74 | 101.55 | 0.4067 | −50.98 |
| 55 | 99.20 | 102.16 | 100.68 | 0.4054 | −50.54 |
| 60 | 99.58 | 102.95 | 101.27 | 0.4056 | −50.84 |
| 65 | 99.87 | 100.81 | 100.34 | 0.4063 | −50.37 |
| **70** | **99.00** | 100.83 | **99.91** | **0.4033** | **−50.16** |
| 75 | 99.51 | 100.43 | 99.97 | 0.403 | −50.19 |
| 80 | 100.18 | 100.44 | 100.31 | 0.403 | −50.36 |

Training facts: epochs 41–80 took ≈ 8.3 h at 9.4–9.5 it/s (`train_summary.json` reports 58,039 s cumulative including run 2's 40 epochs). NaN count 0; max grad norm G 1,824.5 / D 381.5 (cumulative). End of epoch 80: cycle 0.069 / 0.075, identity 0.046 / 0.068, D_A 0.058, D_B 0.114.

Local metrics of the epoch-70 generators (the `full_run02b` row of `full_metrics_report.csv` was later replaced by the weight-averaged model, epochs 55–80, which scored −49.73):

| Metric | A2B (Monet → photo) | B2A (photo → Monet) |
|---|---|---|
| FID (torchmetrics, all images) | 84.40 | 82.20 |
| KID mean ± std | 0.0228 ± 0.0024 | 0.0141 ± 0.0020 |
| Precision / recall | 0.730 / 0.443 | 0.517 / 0.700 |
| Density / coverage | 0.745 / 0.810 | 0.489 / 0.803 |
| Cycle L1 | 0.0408 | 0.0448 |
| LPIPS input vs translation | 0.354 | 0.408 |
| Content cosine similarity | 0.776 | 0.741 |
| Generator images/sec | 217 | 302 |

Kaggle: FID 99.9142, MiFID 0.4033 → public score −50.1587 (2026-10-03). The leaderboard now has 40 teams; this ranks 22nd once the non-official −42.1653 entry is removed (top 20 > −49.73, top 10 > −46.99).

Reference point measured with the official script: two disjoint sets of 300 **real** photos give FID 77.2–80.5 and MiFID 0.425–0.436 (score ≈ −39 to −40), i.e. the practical ceiling of this metric at N = 300.

## Runs 3–4 — self-attention and spectral normalisation
The colour casts and sky streaks looked like global inconsistencies, which a convolutional generator with local
receptive fields struggles to fix, so I tried SAGAN self-attention for long-range interactions.
- *Run 3* (attention in G and D, spectral norm in the discriminators only): the discriminators dominated from epoch 1,
  the generators got little useful gradient and produced blob artifacts (−83.28 at epoch 10); stopped at epoch 23.
- *Run 4* (spectral norm in both): rebalanced, but it trailed run 2 at the same epoch (−61.82 vs −57.29 at epoch 10).

At this data scale the extra capacity did not pay for its cost. Since I stopped run 4 at epoch 13, the result is
inconclusive rather than negative. The global-consistency problem was later handled more cheaply with a second,
coarser discriminator scale (run 10).

## Comparison with teammates
Official script, same protocol: my submitted model −48.20 (FID 96.01 / MiFID 0.400) vs Aswin's canonical V2 run −52.76
(FID 105.11 / MiFID 0.414; per direction: photo → Monet 100.67, Monet → photo 109.56). Both use a ResNet-9 generator
with nearest-neighbour resize-convolution and a 70×70 PatchGAN. Aswin's V2: spectral-normalised single-scale
discriminators, λ_cycle 10, λ_identity 2.5 decaying to 0 over the first half, Adam G 2e-4 / D 1e-4, AMP, gradient
clipping, EMA 0.999 and no DiffAugment; 150 epochs with decay from epoch 75, with epoch 130 selected by validation FID.
No single factor explains a 4.6-point gap and I cannot separate them without ablations. My candidates:
- *Selection:* my final model went through EMA, weight averaging and per-direction selection on the official score;
  his checkpoint was selected by a validation FID on held-out images.
- *Content constraints:* my later runs cut λ_cycle to 2 and λ_identity to 0. The metric rewards only matching the Monet
  distribution, so looser constraints let outputs move further toward it, while he kept λ_cycle at 10.
- *No DiffAugment:* his V2 trains its discriminators on un-augmented images. DiffAugment was my biggest single gain
  (−53.87 → −51.47, run 1 → run 2): with only 300 paintings the Monet discriminator memorises them, as mine did in run 1
  (loss below 0.05 by epoch 13), and its feedback to the generator stops being informative.

Runs 5 onward were designed with Claude (marked in `DESIGN_LOG.md`). My own runs 2–2b reached −51.47 / −50.16, which
already beat −52.76, so most of the gap predates those changes. Part of my score also carries the optimistic bias of
selecting on the evaluated images.

## Hardware disclosure
- GPU: NVIDIA GeForce RTX 4090 (24 GB); CPU: AMD Ryzen 9 7950X; PyTorch 2.11.0+cu128.
- Training time: 26,494 s. Peak memory: 19,154 MB allocated (see note above).

## Shortcomings
In order of importance:
1. **Only 300 Monet paintings.** This is the root cause of most problems: discriminator overfitting, the DiffAugment
   side effect (corner blob), MiFID's memorisation penalty, and the fact that the reference set is the training set.
2. **FID noise on 300 images.** The last few gains (≈ 0.02) are below the snapshot-to-snapshot noise, and repeated
   selection on the evaluated images inflates the score, so the final leaderboard improvements are not real.
3. **Monet → photo stuck at FID 97.8.** It is now the limiting direction, and three specialisation attempts (runs 13,
   15, 17) made it worse.
4. **Per-direction generators.** The submitted pair comes from different training states, so it is not one jointly
   trained CycleGAN. The deployed pair's cycle error is still low (0.045 / 0.052), but this needs disclosing.
5. **The −42.17 non-official entry.** Procedural: it is disclosed, a corrected score was submitted, and it matters only
   because Kaggle displays a team's best score.

## Evidence
- Config: `configs/full.yaml`
- Raw log: `reproducibility/raw_logs/task3_gan/chaitanya/full_run01.log`
- Manifest: `reproducibility/manifests/task3_gan/chaitanya/full_run01.json`
- Metrics: `full_metrics_report.csv`, `submission.csv`, `outputs/kaggle_score.json`
- Checkpoints: `checkpoints/full_run01/G_A2B.pt`, `G_B2A.pt`, `D_A.pt`, `D_B.pt` (sha256 of G_B2A in the manifest)
- Predictions: `outputs/pred_A2B/` (300 Monet → photo), `outputs/pred_B2A/` (7,038 photo → Monet)
