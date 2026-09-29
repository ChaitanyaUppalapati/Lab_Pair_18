# Task 2 — Yelp Polarity Sentiment Classification (Aswin)

This directory contains the complete, reproducible implementation of **Task 2: Yelp Polarity Sentiment Classification** for DATA266 Lab 1.

All models learn embeddings strictly **from scratch** (no pretrained embeddings, no pretrained language models).

---

## Hardware Support & Portability (RTX 5090 / CUDA / Apple Silicon / CPU)

The codebase and notebook **automatically detect and utilize your hardware**:
* **NVIDIA GPU (e.g. RTX 5090 / 4090 / 3090)**: Automatically activates PyTorch CUDA acceleration (`device = torch.device("cuda")`).
* **Apple Silicon (M1/M2/M3/M4)**: Automatically activates Metal Performance Shaders (`device = torch.device("mps")`).
* **CPU**: Automatic fallback if no GPU is detected.

---

## 1. Environment Setup

### Option A: Using Conda / Mamba (Recommended for NVIDIA RTX 5090)
```bash
# Create dedicated environment with Python 3.12
conda create -n data266 python=3.12 -y
conda activate data266

# Install PyTorch with CUDA 12.4+ for RTX 5090 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install required dependencies
pip install datasets pandas scikit-learn statsmodels nltk pyyaml matplotlib tqdm psutil jupyter nbformat ipykernel
```

### Option B: Using standard Python venv
```bash
python3 -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
```

---

## 2. Running via Jupyter Notebook (`src/task2_sentiment_aswin.ipynb`)

A fully self-contained, end-to-end interactive notebook is available at:
```text
task2_sentiment/member_aswin/src/task2_sentiment_aswin.ipynb
```
To run it on your RTX 5090 laptop:
1. Launch Jupyter Lab or VS Code:
   ```bash
   jupyter lab
   ```
2. Open `task2_sentiment/member_aswin/src/task2_sentiment_aswin.ipynb`.
3. Select your Python kernel (`data266` or `.venv`).
4. Run all cells: The notebook will automatically detect the RTX 5090, load Yelp Polarity, perform EDA, train/evaluate all 3 models, generate confusion matrices and ROC/PR plots, run McNemar statistical tests, slice robustness analysis, and display the 20-error review.

---

## 3. One-Command Smoke Test (< 1 Minute)

To verify the entire pipeline (data loading, preprocessing, model forward pass, checkpointing, and evaluation) in seconds on any CPU or GPU:
```bash
python task2_sentiment/member_aswin/src/train.py --config task2_sentiment/member_aswin/configs/smoke.yaml
python task2_sentiment/member_aswin/src/evaluate.py --config task2_sentiment/member_aswin/configs/smoke.yaml
```

---

## 4. Reproducing CLI Training & Evaluation

All execution is 100% config-driven. No hard-coded personal paths.

### A. Train the Three Models
```bash
# 1. Baseline: Masked Mean-Pooling Feed-Forward Network
python task2_sentiment/member_aswin/src/train.py --config task2_sentiment/member_aswin/configs/baseline.yaml

# 2. Experimental 1: Bidirectional GRU (BiGRU)
python task2_sentiment/member_aswin/src/train.py --config task2_sentiment/member_aswin/configs/experimental_1.yaml

# 3. Experimental 2: Multi-Scale 1D TextCNN
python task2_sentiment/member_aswin/src/train.py --config task2_sentiment/member_aswin/configs/experimental_2.yaml
```

### B. Evaluate Models on Test Set
```bash
python task2_sentiment/member_aswin/src/evaluate.py --config task2_sentiment/member_aswin/configs/baseline.yaml
python task2_sentiment/member_aswin/src/evaluate.py --config task2_sentiment/member_aswin/configs/experimental_1.yaml
python task2_sentiment/member_aswin/src/evaluate.py --config task2_sentiment/member_aswin/configs/experimental_2.yaml
```

### C. Run McNemar Statistical Significance Tests
```bash
python task2_sentiment/member_aswin/src/statistical_tests.py
```

### D. Run Slice Robustness Analysis
```bash
python task2_sentiment/member_aswin/src/slice_analysis.py
```

### E. Run Manual Error Analysis (20 Grounded Test Errors)
```bash
python task2_sentiment/member_aswin/src/error_analysis.py \
  --preds-file task2_sentiment/member_aswin/outputs/predictions/experimental_2_test_predictions.csv \
  --model-name "Experimental 2 (TextCNN)"
```

---

## 5. Output Directory Structure

```text
task2_sentiment/member_aswin/
├── src/
│   ├── dataset.py                # EDA, Tokenizer with negation preservation, DataLoader
│   ├── models.py                 # Baseline, BiGRU, TextCNN definitions
│   ├── train.py                  # Training loop, checkpointing, loss logging
│   ├── evaluate.py               # Test evaluation, metrics, bootstrap CI
│   ├── statistical_tests.py      # Paired McNemar tests with continuity correction
│   ├── slice_analysis.py         # Robustness across 7 linguistic & structural slices
│   ├── error_analysis.py         # 20-error categorization with testable fixes
│   ├── utils.py                  # Seeding, metrics, hardware tracking, ECE
│   └── task2_sentiment_aswin.ipynb # Self-contained end-to-end interactive notebook
├── configs/
│   ├── baseline.yaml             # Baseline configuration
│   ├── experimental_1.yaml       # BiGRU configuration
│   ├── experimental_2.yaml       # TextCNN configuration
│   └── smoke.yaml                # Rapid smoke test configuration
├── data_processed/
│   ├── vocab.json                # Fitted vocabulary (from train split only)
│   ├── preprocessing_config.json # Preprocessing hyperparameters & negation list
│   └── eda_summary.json          # Dataset EDA summary statistics
├── checkpoints/
│   ├── baseline/best_model.pt
│   ├── experimental_1/best_model.pt
│   └── experimental_2/best_model.pt
├── outputs/
│   ├── plots/                    # ROC/PR curves, training loss curves, slice comparison
│   ├── confusion_matrices/       # Heatmap confusion matrices
│   ├── predictions/              # Per-example test predictions with probabilities
│   ├── calibration/              # Reliability diagrams & confidence histograms
│   ├── statistical_tests/        # McNemar contingency tables and p-values
│   ├── error_analysis/           # Structured CSV of 20 categorized test errors
│   └── slice_analysis.csv        # Macro-F1 and error rates across data slices
├── metrics_report.csv            # Official unified metrics table matching team schema
├── failure_analysis.md           # 20-case error review with testable hypotheses
├── results.md                    # Synthesis report tying together architecture, metrics, viva prep
├── hardware_manifest.json        # Machine specs and runtime hardware disclosure
└── README.md                     # This documentation
```
