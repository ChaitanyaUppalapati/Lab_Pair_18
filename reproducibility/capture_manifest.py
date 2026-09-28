"""Write an environment manifest for one training run.

Usage:
    python reproducibility/capture_manifest.py --member chaitanya --task task1_llm --run-id run01 \
        --checkpoint task1_llm/member_chaitanya/checkpoints/best.pt --config task1_llm/member_chaitanya/configs/config.yaml
"""
import argparse
import datetime
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hardware_info() -> dict:
    info = {"cpu": platform.processor() or platform.machine()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        info["torch"] = None
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--member", required=True)
    p.add_argument("--task", required=True, choices=["task1_llm", "task2_sentiment", "task3_gan"])
    p.add_argument("--run-id", required=True)
    p.add_argument("--checkpoint", help="repo-relative path of the checkpoint this run produced")
    p.add_argument("--config", help="repo-relative path of the config used")
    p.add_argument("--notes", default="")
    args = p.parse_args()

    packages = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
    ).stdout.splitlines()

    manifest = {
        "member": args.member,
        "task": args.task,
        "run_id": args.run_id,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "hardware": hardware_info(),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256(REPO_ROOT / args.checkpoint) if args.checkpoint else None,
        "raw_log": f"reproducibility/raw_logs/{args.task}/{args.member}/{args.run_id}.log",
        "notes": args.notes,
        "packages": packages,
    }

    out_dir = REPO_ROOT / "reproducibility" / "manifests" / args.task / args.member
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.run_id}.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
