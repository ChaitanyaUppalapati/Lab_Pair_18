# Task 3 - CycleGAN results (Aswin)

## Status

The V2 GPU run completed all 150 epochs on an NVIDIA GeForce RTX 4090. The executed notebook is `src/task3_cyclegan_aswin_v2.ipynb`; its configuration and outputs match `checkpoints/aswin_cyclegan_v2/`, the training log/history, generated images, and `full_metrics_report.csv`. Validation selected `best_model.pt` at epoch 130. The supplied two-direction evaluation script has been rerun on those generated outputs. The team's Kaggle submission uses Chaitanya's model (team rank 15th); V2 was scored with the official script but not submitted. The V2 human audit is being rated in the team's blinded round-2 sheet.

## Independent architecture

- Generator A→B and B→A: 9-block ResNet, 64 base filters, two strided downsampling stages, instance normalization, reflection padding, and nearest-neighbor resize-convolution upsampling.
- Discriminator A and B: single-scale 70×70 PatchGAN discriminators with instance normalization (no spectral normalization).
- Objective: least-squares adversarial loss, cycle L1 weight 10, and identity L1 weight 2.5, with the identity weight decaying to zero over the first half of training.
- Stability: 50-image replay buffers, AMP, gradient clipping (10), exponential-moving-average generators (decay 0.999), and linear learning-rate decay from epoch 75.
- Augmentation: random horizontal flip only. The discriminators see un-augmented images (no DiffAugment).

The design was fixed before Chaitanya's configuration was known. The completed models differ: Chaitanya's selected model uses lambda_cycle 2, lambda_identity 0, DiffAugment, and a two-scale photo-to-Monet discriminator.

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

Observations come from the fixed qualitative grid (`outputs/plots/aswin_cyclegan_v2_qualitative_grid.png`) and the 30 fixed audit inputs (`outputs/audit/team_audit/`), not from selected favourable examples.

- **Photo → Monet:** scene layout is preserved closely (rocks, waterfalls, mountains, and buildings stay in place), and the outputs gain visible brushstroke texture. The strong cycle weight keeps the translation conservative, so many outputs look like textured photos rather than paintings.
- **Colour shift:** the generator pushes most scenes toward a cool blue-green palette. A red forest turns blue, and warm sunset coasts lose their orange tones.
- **Skies and dark scenes:** smooth skies become streaky or speckled. A night sky becomes grainy and brighter than the input, which is the clearest out-of-distribution failure in the grid.
- **Monet → photo:** outputs keep much of the painted texture and mostly change contrast and colour (darker water, deeper blues). Realistic photographic detail is rarely produced, which matches the higher A2B FID (109.6 vs 100.7).
- **Watermarks and text** in source photos are carried through rather than removed.
- **Cycle reconstructions** keep structure but are blurrier than the inputs and sometimes shift colour (a haystack reconstruction turns yellower).

## Human audit

The 30 fixed audit outputs from `best_model.pt` (epoch 130) are in `outputs/audit/team_audit/` (20 photo → Monet, 10 Monet → photo). They are mixed with Chaitanya's submitted model in the team's blinded 60-item round-2 sheet (`task3_gan/member_chaitanya/outputs/human_audit/`). Style, content, and artifact means and Cohen's kappa will be recorded in `human_audit_30.csv` once both raters finish.

## Kaggle evidence

- Team submission: Chaitanya's model (run 18, EMA epoch 123), public score −48.2029, **team rank 15th**.
- V2 (`best_model.pt`, epoch 130) official-script score: −(105.1144 + 0.4137)/2 = **−52.7641**. V2 was not submitted, so it has no leaderboard entry of its own.
- Private leaderboard score: available after the competition closes.

## Limitations and future work

- **No discriminator augmentation.** With only 300 Monet paintings, an un-augmented discriminator can memorise the training set. DiffAugment was the largest single gain in Chaitanya's runs (−53.87 → −51.47).
- **Strong content constraint.** lambda_cycle 10 keeps outputs close to the inputs, which helps content preservation but limits how far outputs move toward the Monet distribution. Chaitanya's lambda_cycle 2 / identity 0 runs scored better on FID.
- **Selection signal.** The best checkpoint was chosen by a validation FID on 120 images, which is noisy (189.6 at epoch 130 vs 192.0 at epoch 150).
- **Non-finite gradients.** 70 steps were skipped under AMP. No NaN loss occurred, but the cause was not investigated.

Next experiments, one factor at a time: add DiffAugment to both discriminators; sweep lambda_cycle over {10, 5, 2}; then test a two-scale discriminator with the same schedule.

## AI use

Code for this task was written with AI coding assistants (OpenAI Codex and Claude) from Aswin's design decisions. Aswin chose the architecture and hyperparameters, ran the training, and reviewed the outputs.
