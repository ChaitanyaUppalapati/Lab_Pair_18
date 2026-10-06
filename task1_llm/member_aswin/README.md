# Task 1 - Character GPT from scratch (Aswin)

This folder implements Task 1 using TinyStories and a manually implemented decoder-only Transformer. No pretrained model/tokenizer, `nn.Transformer`, `nn.MultiheadAttention`, or prebuilt attention operation is used.

## Completed full run

`full_run01` was completed for 12 epochs on Google Colab Pro using an NVIDIA A100-SXM4-40GB. The output-bearing notebook is `src/task1_colab_all_in_one.ipynb`; it records the setup checks, data shapes, training log, metrics, loss plot, and generated samples. The authoritative numerical row is the `full_run01` row in `metrics_report.csv`.

| Artifact | Repository path |
|---|---|
| Best and final checkpoints | `checkpoints/full_run01/` |
| Frozen run config and epoch history | `checkpoints/full_run01/config.yaml`, `checkpoints/full_run01/training_history.json` |
| Metrics | `metrics_report.csv` |
| Loss curve | `outputs/plots/full_run01_loss_curves.png` |
| Generated samples | `outputs/generated_text/full_run01_samples.txt` and `.jsonl` |
| Failure analysis | `failure_analysis.md` |
| Raw training log | `reproducibility/raw_logs/task1_llm/aswin/full_run01.log` |
| Environment/evidence manifest | `reproducibility/manifests/task1_llm/aswin/full_run01.json` |

The full run achieved validation CE 0.758225, perplexity 2.13448, 1.09389 bits/character, and 76.0059% next-character accuracy. See `results.md` for the complete interpretation and limitations.

## Google Colab (recommended)

Keep the entire repository folder in Google Drive or upload it to Colab session storage, preserving the `task1_llm/member_aswin/` directory structure. Do not upload only the notebook or only the `src` folder, because the notebook also needs the configuration and supporting scripts. Opening the notebook from Drive does not itself expose the other Drive files to the runtime; the setup cell therefore mounts Drive automatically and asks you to authorize access.

1. In Colab, select **Runtime > Change runtime type > GPU**.
2. Open `src/task1_colab_all_in_one.ipynb` from the uploaded folder.
3. Select **Runtime > Run all**.
4. Before the Colab session ends, use the final cell's link to download `task1_<run_id>_results.zip`. It contains the checkpoints, metrics, plot, generated samples, configuration, history, and raw log. Colab's `/content` storage is temporary. A completed run is already preserved in this repository; rerunning creates new evidence and should use a new `run_id`.

The all-in-one notebook installs only missing Python packages, searches both `/content` and mounted Google Drive even when the top-level folder has been renamed, verifies CUDA and all supporting files, and runs preprocessing, training, evaluation, and generation in the same runtime. The three stage-specific notebooks below are also Colab-compatible, but the all-in-one notebook avoids losing state when moving between separate Colab sessions.

## RTX 5090 setup

Start in the repository root. Create an environment with Python 3.11 or 3.12. Install a current NVIDIA driver and a CUDA-enabled PyTorch build compatible with the RTX 5090 by following the official PyTorch installation selector; then install the repository dependencies.

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
# Install the CUDA-enabled torch build selected at https://pytorch.org/get-started/locally/
pip install -r requirements.txt
python -m ipykernel install --user --name data266-lab1 --display-name "DATA266 Lab 1"
jupyter lab
```

Choose the `DATA266 Lab 1` kernel and open these notebooks from the repository root, in order:

1. `src/preprocessing.ipynb`
2. `src/model_training.ipynb`
3. `src/text_generation.ipynb`

The first notebook downloads TinyStories and creates exactly 100,000 training examples and 10,000 validation examples. The second trains for 12 epochs and saves best/final checkpoints, an append-only raw log, history, a loss plot, and metrics. The third evaluates and generates greedy plus temperature 0.7/1.0/1.2 samples.

## Equivalent terminal commands

```bash
python task1_llm/member_aswin/src/preprocess.py --config task1_llm/member_aswin/configs/full.yaml
python task1_llm/member_aswin/src/train.py --config task1_llm/member_aswin/configs/full.yaml
python task1_llm/member_aswin/src/evaluate.py --config task1_llm/member_aswin/configs/full.yaml
python task1_llm/member_aswin/src/generate.py --config task1_llm/member_aswin/configs/full.yaml
```

One-command local validation uses synthetic data and does not replace the full TinyStories run:

```bash
python task1_llm/member_aswin/src/smoke_test.py
```

For a future run, first copy the config and assign a new `run_id`; after that run, capture its environment and checkpoint hash before leaving the runtime:

```bash
python reproducibility/capture_manifest.py --member aswin --task task1_llm --run-id <new_run_id> --config <new_config_path> --checkpoint task1_llm/member_aswin/checkpoints/<new_run_id>/best_model.pt
```

Do not run this command retroactively for `full_run01`: that run's evidence manifest is already preserved and explicitly identifies environment fields that were not captured in Colab.

Raw training evidence is written to `reproducibility/raw_logs/task1_llm/aswin/`. Do not edit it after training. Processed tensors are excluded from Git because they are reproducible. Back up checkpoints and outputs before leaving the GPU machine; use Git LFS or the team's approved external checkpoint storage if a checkpoint is too large for normal Git.

Treat each `run_id` as immutable. If a run is interrupted or you change any setting, change `run_id` in a copied config before restarting so the new log never appends to an earlier experiment's evidence.
