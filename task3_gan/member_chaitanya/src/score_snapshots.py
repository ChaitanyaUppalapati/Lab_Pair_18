"""Score every saved generator snapshot of a run with the official evaluation script.

For each checkpoints/<run_id>/epoch_XXX/ (and the final G_*.pt), translate exactly the images the
official script reads (first 300 sorted photos -> pred_B2A, first 300 sorted Monet -> pred_A2B),
then compute the official both-direction FID / MiFID. Results: outputs/<run_id>/snapshot_scores.csv.

Usage:
    python task3_gan/member_chaitanya/src/score_snapshots.py --config task3_gan/member_chaitanya/configs/full_run02.yaml
"""
import argparse
import csv
import shutil
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from common import REPO_ROOT, load_config, repo_path, run_paths
from data import FolderDataset, eval_transform, list_images
from models import make_generator
from official_eval import N_EVAL, evaluate


def translate(gen, files, out_dir: Path, size: int, device: str) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    loader = DataLoader(FolderDataset(None, eval_transform(size), files), batch_size=16)
    with torch.no_grad():
        for x, names in loader:
            y = ((gen(x.to(device)).clamp(-1, 1) * 0.5 + 0.5) * 255).round().to(torch.uint8)
            for img, n in zip(y.permute(0, 2, 3, 1).cpu().numpy(), names):
                Image.fromarray(img).save(out_dir / (Path(n).stem + ".jpg"), quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    m, d = cfg["model"], cfg["data"]
    paths = run_paths(cfg["run_id"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    monets = list_images(repo_path(d["monet_dir"]))[:N_EVAL]
    photos = list_images(repo_path(d["photo_dir"]))[:N_EVAL]
    snaps = [(p.name, p) for p in sorted(paths["checkpoints"].glob("epoch_*"))]
    if (paths["checkpoints"] / "G_A2B.pt").exists():
        snaps.append(("final", paths["checkpoints"]))
    tmp = paths["outputs"] / "snapshot_eval"
    rows = []
    for name, folder in snaps:
        for key, files, sub in (("G_A2B", monets, "pred_A2B"), ("G_B2A", photos, "pred_B2A")):
            g = make_generator(m).to(device).eval()
            g.load_state_dict(torch.load(folder / f"{key}.pt", map_location=device))
            translate(g, files, tmp / sub, d["crop_size"], device)
        r = evaluate((tmp / "pred_A2B").relative_to(REPO_ROOT).as_posix(), (tmp / "pred_B2A").relative_to(REPO_ROOT).as_posix())
        row = {"snapshot": name, "FID_B2A": r["FID_B2A"], "MiFID_B2A": r["MiFID_B2A"], "FID_A2B": r["FID_A2B"],
               "MiFID_A2B": r["MiFID_A2B"], "FID": r["FID"], "MiFID": r["MiFID"], "score": -(r["FID"] + r["MiFID"]) / 2}
        rows.append(row)
        print(f"{name:10s} FID_B2A={row['FID_B2A']:7.2f} FID_A2B={row['FID_A2B']:7.2f} FID={row['FID']:7.2f} "
              f"MiFID={row['MiFID']:.4f} score={row['score']:8.3f}", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    with open(paths["outputs"] / "snapshot_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    best = max(rows, key=lambda r: r["score"])
    print(f"best snapshot: {best['snapshot']} score={best['score']:.4f}")


if __name__ == "__main__":
    main()
