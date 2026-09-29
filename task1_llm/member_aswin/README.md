# Task 1 - Character GPT from scratch (Aswin)

This folder implements Task 1 using TinyStories and a manually implemented decoder-only Transformer. No pretrained model/tokenizer, `nn.Transformer`, `nn.MultiheadAttention`, or prebuilt attention operation is used.

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

After the full run, capture its environment and checkpoint hash:

```bash
python reproducibility/capture_manifest.py --member aswin --task task1_llm --run-id full_run01 --config task1_llm/member_aswin/configs/full.yaml --checkpoint task1_llm/member_aswin/checkpoints/full_run01/best_model.pt
```

Raw training evidence is written to `reproducibility/raw_logs/task1_llm/aswin/`. Do not edit it after training. Processed tensors are excluded from Git because they are reproducible. Back up checkpoints and outputs before leaving the GPU machine; use Git LFS or the team's approved external checkpoint storage if a checkpoint is too large for normal Git.

Treat each `run_id` as immutable. If a run is interrupted or you change any setting, change `run_id` in a copied config before restarting so the new log never appends to an earlier experiment's evidence.
