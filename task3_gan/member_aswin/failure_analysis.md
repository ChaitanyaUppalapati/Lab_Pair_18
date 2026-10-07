# Task 3 — Issues and fixes (Aswin)

Model: V2 CycleGAN, `checkpoints/aswin_cyclegan_v2/best_model.pt` (epoch 130, EMA generators). Every issue below is
visible in the raw log `logs/aswin_cyclegan_v2.jsonl`, the notebook `src/task3_cyclegan_aswin_v2.ipynb`, or the commit
history.

| # | Issue | What it was | How it was handled | Result |
|---|---|---|---|---|
| 1 | Non-finite gradients under mixed precision | With AMP, some fp16 gradients overflowed to inf/NaN. The log column `nonfinite_gradient_steps` shows 7 such steps in epoch 1, while the gradient scaler was still settling, then about one every few epochs, 70 in total over 81,000 steps. | Before each optimizer step the loop checks that every gradient is finite. If not, it skips that step, clears the gradients, and calls `scaler.update()` so the GradScaler lowers its scale. Losses are checked separately (`nan_count`). | `nan_count` stayed 0 for all 150 epochs and training completed. Only 0.09% of steps were skipped. |
| 2 | Large generator gradients early in training | Mean generator gradient norm was 65.6 in epoch 1 and fell to about 32 by epoch 150 (`grad_g`). Unbounded updates this large can destabilise the adversarial game at the start. | Gradient-norm clipping at 10 for both generators and discriminators (`grad_clip` in `config.json`). | Losses decreased smoothly with no divergence. See `outputs/plots/aswin_cyclegan_v2_training_curves.png`. |
| 3 | Quality got worse after epoch 130 | Validation photo → Monet FID fell from 277.0 (epoch 5) to 189.6 (epoch 130), then drifted up to 190.8–192.8 over the last 20 epochs as the learning rate decayed to 0 (`val_b2a_fid`). Using the last checkpoint would have been slightly worse. | Validation FID on held-out images every 5 epochs, saving `best_model.pt` at the minimum instead of taking `final_model.pt`. | Epoch 130 was used for all evaluation and the Kaggle submission. The 2–3 point gap is close to the noise of a 120-image FID, so the gain is small. |
| 4 | Missing Monet → photo images for the official evaluation | The training notebook generated the 7,038 photo → Monet competition images, but the supplied Part 3 evaluation script also needs Monet → photo outputs. | Wrote `src/generate_eval_images.py`, which loads `EMA_A2B` from `best_model.pt` and translates all 300 paintings with the same preprocessing as training. | Both directions were scored (FID 109.56 A2B, 100.67 B2A) in `src/Part3_Evaluation_Script_evaluated.ipynb`. |
| 5 | Unclear which checkpoint produced the prediction folders | `pred_A2B/` and `pred_B2A/` were written at the end of training, so their file dates did not prove whether they came from epoch 130 or epoch 150. | Regenerated the 30 audit images with epoch 130 and, as a control, with epoch 150, and compared each with the saved files. | Mean pixel difference was about 2/255 for epoch 130 versus about 10/255 for epoch 150, which confirms the folders come from `best_model.pt`. |
| 6 | Several notebook versions | V1, V2, and V3 notebooks were in `src/` together (commit `c64887f53`), which made it easy to report numbers from the wrong run. | V2, the completed 150-epoch run, was made canonical and V1/V3 were removed from `src/` (commit `5a0e106f3`). | Every metric, plot, checkpoint, and submission now traces to one run. |
| 7 | Human-audit sample did not fit the team protocol | The notebook's audit cell picked 30 random photo → Monet outputs (`outputs/audit/sample_*.jpg`). They covered only one direction, used different inputs from Chaitanya's, and could not be blinded against another model. | Used the team's 30 fixed inputs (20 B2A, 10 A2B) instead (`outputs/audit/team_audit/`). They were mixed with Chaitanya's outputs for the same inputs in one blinded 60-item sheet rated by both members. | Paired, blinded comparison: V2 style 4.70, content 4.65, artifacts 4.68, and rated cleaner on artifacts than the submitted model (p = 0.002). Per-item scores are in `human_audit_30.csv`. |
| 8 | Documentation did not match the code | `results.md` described spectral-normalised discriminators, but the V2 code uses plain InstanceNorm PatchGANs with no spectral norm and no DiffAugment. | Corrected `results.md` and the team report against the notebook code. | The reported architecture now matches the trained model. The missing DiffAugment is listed as a likely reason for V2's higher FID. |

## Issues that remain

- **Conservative translation.** lambda_cycle 10 keeps outputs close to their inputs, so many photo → Monet outputs look
  like textured photos and Monet → photo outputs stay painterly. This was not tuned; a lambda_cycle sweep is the next
  experiment.
- **Colour bias and out-of-distribution scenes.** Outputs drift toward a cool blue-green palette, and dark or night
  photos become grainy and brighter, because the 300 Monet paintings contain almost no night scenes.
- **No discriminator augmentation.** With 300 paintings the Monet discriminator can memorise its training data;
  DiffAugment was the largest single gain in Chaitanya's runs and should be added first.
