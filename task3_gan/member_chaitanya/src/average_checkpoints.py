"""Average generator weights over several snapshots of one run ("checkpoint averaging" / weight averaging).

Nearby checkpoints of the same training run lie in the same low-loss region; their weight average often
generalises better than any single snapshot and smooths out snapshot-to-snapshot noise. Only weights of our
own trained generators are combined; nothing pretrained is involved.

Usage:
    python task3_gan/member_chaitanya/src/average_checkpoints.py --config task3_gan/member_chaitanya/configs/full_run02b.yaml \
        --snapshots epoch_065 epoch_070 epoch_075 epoch_080 --name epoch_avg065-080
Writes checkpoints/<run_id>/<name>/G_A2B.pt and G_B2A.pt (scored by score_snapshots.py like any snapshot).
"""
import argparse

import torch

from common import load_config, run_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--snapshots", nargs="+", required=True)
    parser.add_argument("--name", required=True, help="output folder name; start with 'epoch_' so score_snapshots picks it up")
    args = parser.parse_args()
    paths = run_paths(load_config(args.config)["run_id"])
    out = paths["checkpoints"] / args.name
    out.mkdir(exist_ok=True)
    for key in ("G_A2B", "G_B2A"):
        states = [torch.load(paths["checkpoints"] / s / f"{key}.pt", map_location="cpu") for s in args.snapshots]
        avg = {}
        for k in states[0]:
            t = torch.stack([s[k].float() for s in states]).mean(0)
            avg[k] = t.to(states[0][k].dtype)
        torch.save(avg, out / f"{key}.pt")
    print(f"wrote {out} from {', '.join(args.snapshots)}")


if __name__ == "__main__":
    main()
