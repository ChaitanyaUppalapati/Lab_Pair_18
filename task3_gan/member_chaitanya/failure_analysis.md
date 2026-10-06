# Task 3 — Failure Cases (Chaitanya)

Model: the submitted generators `checkpoints/full_run18/epoch_123_ema/` (G_A2B = frozen copy of the run-7 EMA 88+90
average; G_B2A = run-18 EMA 123; Kaggle −48.2029). Each image is a strip: **input | translation G(x) |
reconstruction F(G(x))**.

The three cases are **suggested** by `src/failure_cases.py`, which scores every image (all 7,038 photos and
300 Monet paintings) and ranks the extremes: largest cycle error, lowest content similarity, smallest change.
Typical values for comparison (photo → Monet medians): cycle L1 0.050, LPIPS change 0.429, content cos 0.729.
Four candidates per category and direction are in `outputs/failure_cases/candidates.md`; swap any of them in.
**To fill in: Failure and Observation** (for example artifacts, content loss, colour shift, under-stylisation,
or cycle not satisfied).

| # | Input · Translation · Reconstruction (+ scores) | Failure (e.g. artifacts, mode collapse, content loss) | Observation |
|---|---|---|---|
| 1 | ![](outputs/failure_cases/B2A_high_cycle_error/1_d0a1eed9dd.png) `d0a1eed9dd.jpg`, photo → Monet; largest cycle-reconstruction error of 7,038 photos. Cycle L1 0.1926, LPIPS change 0.515, LPIPS rec. 0.342, content cos 0.718 | | |
| 2 | ![](outputs/failure_cases/B2A_low_content/1_3a7a0992dd.png) `3a7a0992dd.jpg`, photo → Monet; lowest input-vs-translation content similarity of 7,038 photos. Cycle L1 0.0516, LPIPS change 0.521, LPIPS rec. 0.320, content cos 0.357 | | |
| 3 | ![](outputs/failure_cases/A2B_high_cycle_error/1_4f7e01f097.png) `4f7e01f097.jpg`, Monet → photo; largest cycle-reconstruction error of 300 Monet paintings. Cycle L1 0.0859, LPIPS change 0.575, LPIPS rec. 0.777, content cos 0.747 | | |
