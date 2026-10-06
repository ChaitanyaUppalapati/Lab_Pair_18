# Task 3 - CycleGAN results (Aswin)

## Status

The V2 GPU run completed all 150 epochs on an NVIDIA GeForce RTX 4090. The executed notebook is `src/task3_cyclegan_aswin_v2.ipynb`; its configuration and outputs match `checkpoints/aswin_cyclegan_v2/`, the training log/history, generated images, and `full_metrics_report.csv`. Validation selected `best_model.pt` at epoch 130. The supplied two-direction evaluation script has been rerun on those generated outputs. The two-rater human audit and actual Kaggle leaderboard submission remain pending.

## Independent architecture

- Generator A→B and B→A: 9-block ResNet, 64 base filters, two strided downsampling stages, instance normalization, reflection padding, and nearest-neighbor resize-convolution upsampling.
- Discriminator A and B: single-scale spectral-normalized 70×70 PatchGAN discriminators.
- Objective: least-squares adversarial loss, cycle L1 weight 10, and identity L1 weight 2.5, with the identity weight decaying to zero over the first half of training.
- Stability: 50-image replay buffers, AMP, gradient clipping, exponential-moving-average generators, and linear learning-rate decay.

Chaitanya's architecture and hyperparameters were blank when this design was created. Recheck both completed configurations before the full run and coordinate a change if they become materially identical.

## Data and preprocessing

- Real Monet images: 300.
- Real photo images: 7,038.
- Generated Monet-to-photo evaluation images: 300.
- Generated photo-to-Monet competition images: 7,038.
- Evaluation preprocessing: resize to 299, center crop to 299, ImageNet normalization, and Inception-v3 features, exactly as implemented by the supplied evaluation notebook.

## Training behavior and convergence

Training completed in 9,567.69 seconds (2 h 39 min 27.69 s) at 16.92 images/s, with 3,988.75 MB peak GPU memory. Validation B2A FID reached its best value of 189.550 at epoch 130; epoch 150 ended at 192.001, so the epoch-130 checkpoint was correctly retained. Final epoch losses were generator 3.087, discriminator A 0.127, discriminator B 0.178, and cycle 0.210. No NaN loss occurred. The run recorded 70 non-finite-gradient steps, which the training loop safely skipped rather than applying corrupted updates.

## Cycle-consistency verification

Measured cycle-reconstruction L1 is 0.110149 for Monet→Photo→Monet and 0.123067 for Photo→Monet→Photo. The fixed qualitative evidence is in `outputs/plots/aswin_cyclegan_v2_qualitative_grid.png`.

## Quantitative metrics

The supplied `Part3_Evaluation_Script.ipynb` evaluated the first 300 sorted images in every set:

| Direction | FID | MiFID |
|---|---:|---:|
| Photo → Monet (B2A) | 100.668 | 0.4054 |
| Monet → Photo (A2B) | 109.561 | 0.4220 |
| Submission average | 105.1144417304836 | 0.4136924761280517 |

Evidence is preserved in `src/Part3_Evaluation_Script_evaluated.ipynb`, and the exact submission row is in `submission.csv`. These values are distinct from the additional measured training and generation metrics preserved in `full_metrics_report.csv`.

## Visual quality assessment

Describe recurring style-transfer strengths, content preservation, color changes, texture artifacts, and failure patterns from fixed, non-cherry-picked outputs.

## Human audit

Report the 30-sample, two-rater style/content/artifact means and inter-rater agreement only after both raters independently complete `human_audit_30.csv`.

## Kaggle evidence

Record submission date, checkpoint ID/hash, public score, private score, and leaderboard rank from the actual team submission.

## Limitations and future work

Discuss findings supported by the completed run. Possible controlled follow-ups include discriminator-scale ablation, loss-weight tuning, longer schedules, and higher-resolution fine-tuning.
