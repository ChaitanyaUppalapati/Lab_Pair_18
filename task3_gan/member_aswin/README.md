# Task 3 - CycleGAN Monet and Photo (Aswin)

The completed V1 and V2 experiments, including their immutable outputs, are preserved in `src/task3_cyclegan_aswin.ipynb` and `src/task3_cyclegan_aswin_v2.ipynb`. The recommended next experiment is `src/task3_cyclegan_aswin_v3.ipynb`. V3 retains V2's successful generators and adds BF16 mixed precision, differentiable discriminator augmentation, lazy R1 regularization, adaptive Monet-discriminator updates, an identity-loss floor, and composite FID-plus-cycle checkpoint selection.

## Recommended run environment

- Google Colab with an NVIDIA GPU, or Jupyter on the RTX 5090 machine.
- Python 3.11 or 3.12.
- CUDA-enabled PyTorch installed for the selected GPU.
- Competition data arranged as `task3_gan/data/monet_jpg/` and `task3_gan/data/photo_jpg/`, or `/content/data/monet_jpg/` and `/content/data/photo_jpg/` in Colab.

## Run instructions

1. Open `src/task3_cyclegan_aswin_v3.ipynb` from the repository root. Keep the completed V1 and V2 notebooks unchanged as evidence.
2. Select a GPU runtime.
3. Run the dependency cell.
4. Review `CFG`, `DATA_ROOT`, `WORK_ROOT`, and the Kaggle competition slug.
5. Run all cells sequentially through training, generation, and evaluation.
6. Preserve the raw JSONL log and checkpoints without editing them.
7. Give the generated 30-image audit package to two independent raters, fill `human_audit_30.csv`, and rerun the audit analysis cell.
8. Verify the competition's required archive layout, then submit `images.zip` without editing or selecting individual generated images.
9. Record the real Kaggle scores/rank and complete `results.md` using the notebook evidence.

## Completed evaluation artifacts

The supplied `Part3_Evaluation_Script.ipynb` was run against the restored v3-improved epoch-90 outputs with 300 images per set, as required by its `N_EVAL` setting.

- Executed notebook: `src/Part3_Evaluation_Script_evaluated.ipynb`
- Reusable A2B generation utility: `src/generate_eval_images.py`
- Kaggle/lab metric CSV: `submission.csv`
- Complete Photo-to-Monet competition archive: `images.zip` (7,038 JPEG files)
- Real data: `../data/monet_jpg/` (300) and `../data/photo_jpg/` (7,038)
- Generated evaluation data: `outputs/pred_A2B/` (300) and `outputs/pred_B2A/` (7,038)

Large checkpoints, generated JPEGs, datasets, and ZIP archives are retained locally and excluded from normal Git. The notebooks, source code, metric CSV, and written evidence remain eligible for version control.

The default full run uses 100 epochs, nine generator residual blocks, two discriminator scales, AMP, replay buffers, and EMA generators. Set `CFG.smoke = True` only to verify execution; smoke outputs are not assignment results.

No architecture can guarantee the best leaderboard result. Treat the provided configuration as a strong, stable starting point and only report results produced by an actual run.
