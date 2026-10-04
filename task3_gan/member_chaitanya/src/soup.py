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
    parser.add_argument("--sources", nargs="+", help="<run_id>/<snapshot> folders under checkpoints/ (both generators)")
    parser.add_argument("--sources-a2b", nargs="+", help="override: sources for G_A2B (Monet -> photo) only")
    parser.add_argument("--sources-b2a", nargs="+", help="override: sources for G_B2A (photo -> Monet) only")
    args = parser.parse_args()
    out = run_paths(load_config(args.out_config)["run_id"])["checkpoints"] / args.name
    out.mkdir(exist_ok=True)
    # the two generators are independent at inference (pred_A2B uses only G_A2B, pred_B2A only G_B2A),
    # so each can be averaged over its own set of snapshots
    per_key = {"G_A2B": args.sources_a2b or args.sources, "G_B2A": args.sources_b2a or args.sources}
    for key, sources in per_key.items():
        states = [torch.load(MEMBER_DIR / "checkpoints" / s / f"{key}.pt", map_location="cpu") for s in sources]
        torch.save({k: torch.stack([s[k].float() for s in states]).mean(0).to(states[0][k].dtype) for k in states[0]},
                   out / f"{key}.pt")
    (out / "sources.txt").write_text("".join(f"{k}: {' '.join(v)}\n" for k, v in per_key.items()))
    print(f"wrote {out}: " + "; ".join(f"{k} <- {', '.join(v)}" for k, v in per_key.items()))


if __name__ == "__main__":
    main()
