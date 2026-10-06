# DATA266 Lab 1 — Team 18

LLM pretraining (TinyStories) · Yelp Polarity sentiment classification · CycleGAN Monet style transfer.

Repo: `git@github.com:ChaitanyaUppalapati/Lab_Pair_18.git`

## Team ownership

| Member | Task 1 folder | Task 2 folder | Task 3 folder |
|---|---|---|---|
| Chaitanya | `task1_llm/member_chaitanya/` | `task2_sentiment/member_chaitanya/` | `task3_gan/member_chaitanya/` |
| Aswin | `task1_llm/member_aswin/` | `task2_sentiment/member_aswin/` | `task3_gan/member_aswin/` |

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

*Note for NVIDIA RTX GPUs (e.g. RTX 5090): Install PyTorch with CUDA support first: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`.*

Datasets are **not** committed (size). Download instructions: see `task*/data/README.md`.

## Smoke test (one command)

Run a rapid verification (< 1 minute) of the data pipeline, model forward pass, checkpointing, and evaluation:
```bash
python task2_sentiment/member_aswin/src/train.py --config task2_sentiment/member_aswin/configs/smoke.yaml
```

Task 1 (Chaitanya) — preprocess, causal-mask/shift checks, train, evaluate on a tiny config (a few minutes; downloads TinyStories on first run):
```bash
python task1_llm/member_chaitanya/src/smoke_test.py
```

Task 1 (Aswin) — the equivalent synthetic pipeline check:
```bash
python task1_llm/member_aswin/src/smoke_test.py
```

## Reproducing a member's run

1. Pick the config from `task*/member_*/configs/`.
2. Run that member's entry point with `--config <path>` (all paths are relative to the repo root; no hard-coded personal paths).
3. Capture the environment manifest:
   ```bash
   python reproducibility/capture_manifest.py --member chaitanya --task task1_llm --run-id <run_id>
   ```
4. Raw logs go to `reproducibility/raw_logs/<task>/<member>/<run_id>.log` and are never edited after the run.

## Where results live

| What | Where |
|---|---|
| Code / notebooks (with outputs) | `task*/member_*/src/` |
| Configs | `task*/member_*/configs/` |
| Checkpoints | `task*/member_*/checkpoints/` (large files: see note below) |
| Samples, plots, confusion matrices | `task*/member_*/outputs/` |
| All metrics for a task | `task*/member_*/metrics_report.csv` (Task 3: `full_metrics_report.csv` + `submission.csv`) |
| Failure / error analysis | `task*/member_*/failure_analysis.md` |
| Design justification | `task*/member_*/results.md` |
| Environment manifests | `reproducibility/manifests/` |
| Raw training logs | `reproducibility/raw_logs/` |
| Final report | `report/DATA266_Lab1_Report_Team_18.pdf` |

Checkpoints over 100 MB: use Git LFS or link to external storage in the member's `results.md` with the checkpoint ID.

## References

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
- Eldan & Li, *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*, 2023.
- Zhu et al., *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*, ICCV 2017.
