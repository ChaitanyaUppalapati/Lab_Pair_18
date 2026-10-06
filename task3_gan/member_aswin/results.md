# Task 3 - CycleGAN results (Aswin)

## Status

Implementation and a GPU run through epoch 90 are preserved in the v3 notebooks and `checkpoints/aswin_cyclegan_v3_improved/`. The supplied two-direction evaluation script has been completed using `epoch_090.pt`. The two-rater human audit and actual Kaggle leaderboard submission remain pending.

## Independent architecture

- Generator A→B and B→A: 9-block ResNet, 64 base filters, two strided downsampling stages, instance normalization, reflection padding, and nearest-neighbor resize-convolution upsampling.
- Discriminator A and B: two-scale spectral-normalized 70×70 PatchGAN discriminators.
- Objective: least-squares adversarial loss, cycle L1 weight 10, and identity L1 weight 5.
- Stability: 50-image replay buffers, AMP, gradient clipping, exponential-moving-average generators, and linear learning-rate decay.

Chaitanya's architecture and hyperparameters were blank when this design was created. Recheck both completed configurations before the full run and coordinate a change if they become materially identical.

## Data and preprocessing

- Real Monet images: 300.
- Real photo images: 7,038.
- Generated Monet-to-photo evaluation images: 300.
- Generated photo-to-Monet competition images: 7,038.
- Evaluation preprocessing: resize to 299, center crop to 299, ImageNet normalization, and Inception-v3 features, exactly as implemented by the supplied evaluation notebook.

## Training behavior and convergence

Add evidence-backed observations from `outputs/plots/<run_id>_training_curves.png` and the raw JSONL log. Discuss adversarial balance, cycle/identity trends, gradient norms, instability, and non-finite counts.

## Cycle-consistency verification

Report both-direction cycle L1 values and reference the qualitative grid showing inputs, translations, and reconstructions.

## Quantitative metrics

The supplied `Part3_Evaluation_Script.ipynb` evaluated the first 300 sorted images in every set:

| Direction | FID | MiFID |
|---|---:|---:|
| Photo → Monet (B2A) | 103.126 | 0.4077 |
| Monet → Photo (A2B) | 119.646 | 0.4348 |
| Submission average | 111.38606856321958 | 0.4212227378974539 |

Evidence is preserved in `src/Part3_Evaluation_Script_evaluated.ipynb`, and the exact submission row is in `submission.csv`. These values are distinct from the additional metrics requested in `full_metrics_report.csv`, which should only be filled from measured evidence.

## Visual quality assessment

Describe recurring style-transfer strengths, content preservation, color changes, texture artifacts, and failure patterns from fixed, non-cherry-picked outputs.

## Human audit

Report the 30-sample, two-rater style/content/artifact means and inter-rater agreement only after both raters independently complete `human_audit_30.csv`.

## Kaggle evidence

Record submission date, checkpoint ID/hash, public score, private score, and leaderboard rank from the actual team submission.

## Limitations and future work

Discuss findings supported by the completed run. Possible controlled follow-ups include discriminator-scale ablation, loss-weight tuning, longer schedules, and higher-resolution fine-tuning.
