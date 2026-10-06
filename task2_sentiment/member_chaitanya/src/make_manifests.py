"""Write one reproducibility manifest per Task 2 run (environment, hardware, config, checkpoint sha256, log)."""
import hashlib
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata

import torch

from common import CHECKPOINTS_DIR, LOG_DIR, MANIFEST_DIR, MEMBER, OUTPUTS_DIR, REPO_ROOT, TASK, read_json, write_json


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())
    hw = {"cpu": platform.processor(), "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
          "cuda": torch.version.cuda,
          "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    for run_dir in sorted((OUTPUTS_DIR / "runs").iterdir()):
        s = read_json(run_dir / "train_summary.json")
        ckpt = CHECKPOINTS_DIR / run_dir.name / "best.pt"
        write_json(MANIFEST_DIR / f"{run_dir.name}.json", {
            "member": MEMBER, "task": TASK, "run_id": run_dir.name,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0], "os": platform.platform(), "hardware": hw,
            "trained_on": s["hardware"], "config": s.get("config", "src/tfidf_reference.py"), "seed": s.get("seed"),
            "split_indices": "task2_sentiment/member_chaitanya/splits/",
            "checkpoint": ckpt.relative_to(REPO_ROOT).as_posix() if ckpt.exists() else None,
            "checkpoint_sha256": sha256(ckpt) if ckpt.exists() else None,
            "checkpoint_note": "checkpoints are gitignored; regenerate with src/train.py --config <config> --seed <seed>",
            "raw_log": (LOG_DIR / f"{run_dir.name}.log").relative_to(REPO_ROOT).as_posix(),
            "packages": packages,
        })
        print("manifest", run_dir.name)


if __name__ == "__main__":
    main()
