"""Average generator weights across snapshots of *different runs that share a starting point* ("model soup").

full_run05 and full_run06 both resume from full_run02b's epoch-80 state, so their generators stay in the same
loss basin and can be averaged like snapshots of one run. Only our own generator weights are combined.

Usage:
    python task3_gan/member_chaitanya/src/soup.py --out-config task3_gan/member_chaitanya/configs/full_run06.yaml \
        --name epoch_soupA --sources full_run02b/epoch_avg055-080 full_run06/epoch_084_ema
Writes checkpoints/<out run>/<name>/G_A2B.pt and G_B2A.pt.
"""
import argparse

import torch

from common import MEMBER_DIR, load_config, run_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-config", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--sources", nargs="+", required=True, help="<run_id>/<snapshot> folders under checkpoints/")
    args = parser.parse_args()
    out = run_paths(load_config(args.out_config)["run_id"])["checkpoints"] / args.name
    out.mkdir(exist_ok=True)
    for key in ("G_A2B", "G_B2A"):
        states = [torch.load(MEMBER_DIR / "checkpoints" / s / f"{key}.pt", map_location="cpu") for s in args.sources]
        torch.save({k: torch.stack([s[k].float() for s in states]).mean(0).to(states[0][k].dtype) for k in states[0]},
                   out / f"{key}.pt")
    (out / "sources.txt").write_text("\n".join(args.sources) + "\n")
    print(f"wrote {out} from {', '.join(args.sources)}")


if __name__ == "__main__":
    main()
