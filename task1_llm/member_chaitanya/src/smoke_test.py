"""One-command smoke test: preprocess -> correctness checks -> train -> evaluate on a tiny config.

    python task1_llm/member_chaitanya/src/smoke_test.py

Checks: input/target windows are shifted by exactly one character, and the
causal mask stops a position from seeing any later position.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from common import load_config, run_paths
from dataset import CharWindowDataset
from model import CharGPT

SRC = Path(__file__).resolve().parent
CONFIG = "task1_llm/member_chaitanya/configs/smoke.yaml"


def run(script: str) -> None:
    subprocess.run([sys.executable, str(SRC / script), "--config", CONFIG], check=True)


def main() -> None:
    run("preprocess.py")
    cfg = load_config(CONFIG)
    processed = run_paths(cfg["run_id"])["processed"] / cfg["data"]["processed_name"]
    ids = np.load(processed / "train.npy")
    block = cfg["data"]["block_size"]
    ds = CharWindowDataset(ids, block, cfg["data"]["train_stride"])
    x, y = ds[3]
    assert torch.equal(x[1:], y[:-1]), "target is not the input shifted by one character"

    torch.manual_seed(0)
    vocab_size = int(ids.max()) + 1
    model = CharGPT(cfg["model"], vocab_size, block).eval()
    seq = x.unsqueeze(0)
    pivot = 50
    altered = seq.clone()
    altered[:, pivot + 1 :] = (altered[:, pivot + 1 :] + 1) % vocab_size
    with torch.no_grad():
        a, _ = model(seq)
        b, _ = model(altered)
    assert torch.allclose(a[:, : pivot + 1], b[:, : pivot + 1], atol=1e-5), "causal mask leaks future tokens"
    assert not torch.allclose(a[:, pivot + 1 :], b[:, pivot + 1 :]), "altered positions should change the output"

    run("train.py")
    run("evaluate.py")
    print("SMOKE TEST PASSED: window shift, causal mask, training, checkpointing, evaluation, generation")


if __name__ == "__main__":
    main()
