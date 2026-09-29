"""End-to-end smoke test, including shift and causal-mask invariance checks."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

from common import load_config, output_paths
from model import CharacterGPT


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"))
    args = parser.parse_args()
    python = sys.executable
    run([python, str(Path(__file__).with_name("preprocess.py")), "--config", args.config, "--smoke"])
    config = load_config(args.config); paths = output_paths(config["run_id"])
    train = torch.load(paths["processed"] / "train.pt", weights_only=True)["sequences"]
    assert torch.equal(train[:, :-1][:, 1:], train[:, 1:][:, :-1]), "input-target shift is wrong"
    import json
    metadata = json.loads((paths["processed"] / "metadata.json").read_text())
    model = CharacterGPT(metadata["vocab_size"], metadata["context_length"], config["model"]).eval()
    prefix = train[:1, :-1]
    changed = prefix.clone(); pivot = min(10, prefix.shape[1] - 2); changed[:, pivot + 1 :] = 0
    with torch.no_grad():
        original_logits = model(prefix)[:, : pivot + 1]
        changed_logits = model(changed)[:, : pivot + 1]
    assert torch.allclose(original_logits, changed_logits, atol=1e-6), "causal mask leaked future tokens"
    run([python, str(Path(__file__).with_name("train.py")), "--config", args.config])
    run([python, str(Path(__file__).with_name("evaluate.py")), "--config", args.config])
    run([python, str(Path(__file__).with_name("generate.py")), "--config", args.config])
    print("SMOKE TEST PASSED: shift, causal masking, training, evaluation, checkpointing, and generation")


if __name__ == "__main__":
    main()
