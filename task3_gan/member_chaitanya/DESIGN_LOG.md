# Task 3 — Design log (Chaitanya)

Chronological record of every model / training change after the initial design, why it was made, who decided it, and what it did to the **official score** (instructor's `Part3_Evaluation_Script`, ported in `src/official_eval.py`; score = −(FID + MiFID)/2, higher is better).

**Authorship note.** Runs 1–4 were designed by Chaitanya (Claude implemented them). On 2026-10-03 the user lifted the restriction on AI design decisions and asked Claude to take over model design to maximise the leaderboard score, documenting each change and its reason. Entries marked **[Claude]** below are Claude's decisions under that instruction.

| # | Run | Change vs previous best | Decided by | Reason | Best official score |
|---|---|---|---|---|---|
| 1 | full_run01 | Initial design: ResNet-9 G (nearest-neighbour upsampling), 70×70 PatchGAN D, InstanceNorm, LSGAN, λ_cyc 10, λ_id 5, 40 ep | Chaitanya | Paper baseline | −53.87 (epoch 40) |
| 2 | full_run02 | DiffAugment (translation + cutout) before both discriminators | Chaitanya | Monet discriminator overfit the 300 paintings in run 1 (loss → 0.05 at epoch 13) | −51.47 (epoch 35) |
| 3 | full_run02b | Resume run 2 at epoch 40, warm restart LR 1e-4 → 0 over epochs 41–80 | Chaitanya | Score was still improving at epoch 40 | −50.16 (epoch 70) |
| 4 | full_run03 | + SAGAN self-attention in G and D, spectral norm in D | Chaitanya | Global colour / lighting consistency | −83.28 (epoch 10); stopped at epoch 23 — discriminators dominated from epoch 1, blob / colour-spot artifacts |
| 5 | full_run04 | + self-attention in G and D, spectral norm in G and D | Chaitanya | Same, with SN in G to rebalance | −61.82 at epoch 10 (run 2: −57.29); stable but slower; stopped at epoch 13 to free the GPU |
| 6 | full_run02b weight average | Average generator weights of run 2b snapshots 55, 60, 65, 70, 75, 80 (`src/average_checkpoints.py`) | **[Claude]** | Late run-2b snapshots differ by noise (−50.2 … −50.8) around a plateau; averaging nearby checkpoints of one run moves to the centre of that region. Windows tried: 70–80 −49.98, 65–80 −49.93, **55–80 −49.73**, 45–80 −50.14 (too wide: pulls in early warm-restart weights) | **−49.73** (Kaggle −49.7306, submitted) |
| 8 | full_run06 | Resume run 2b at epoch 80; **λ_cycle 10 → 5, λ_identity 5 → 0.5**; warm restart 1e-4 → 0 over epochs 81–90; EMA 0.9999; snapshots every 2 epochs; trained in parallel with run 5 | **[Claude]** | Reference points measured with the official script: untranslated images score −63.5, two sets of real photos −39 to −40, our best −49.7, 1st place −29.2. The metric scores only the match to the target distribution (content is never rewarded; MiFID pairs images by index), and the reference Monet set is the training set. The strong content constraints (λ_cyc 10, λ_id 5) keep outputs close to their inputs, so loosening them should let the generators move further toward the target distribution while remaining a CycleGAN. | running |
| 7 | full_run05 | Resume run 2b at epoch 80; second warm restart at 5e-5 → 0 over epochs 81–120; **generator EMA** (decay 0.9999, half-life ≈ 1 epoch) saved as extra snapshots; weight averages of the last 4 / 6 raw and EMA snapshots also scored | **[Claude]** | The run-2b restart gave a steady gain; a smaller second restart continues that without destabilising. EMA smooths the high-frequency weight noise that weight averaging only removes at 5-epoch granularity, and is standard in GAN training for better FID. | running |

## Rules kept throughout
- Every submitted image is the direct output of our own trained generators (no pretrained or foundation image models, no editing, no hand-picking). Weight averaging / EMA only combine our own generator weights.
- Model selection uses the official script on the fixed evaluation images, never a subset chosen per image.
- A submission is made only when its recomputed official score beats the best previous submission.
