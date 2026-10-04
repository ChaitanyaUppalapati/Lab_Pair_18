"""Write pred_A2B (all Monet -> photo) and pred_B2A (all photos -> Monet) from one generator checkpoint folder.

Usage:
    python task3_gan/member_chaitanya/src/predict.py --config task3_gan/member_chaitanya/configs/full_run06.yaml \
        --checkpoint-dir task3_gan/member_chaitanya/checkpoints/full_run06/epoch_soupB
Outputs go to task3_gan/member_chaitanya/outputs/pred_A2B and pred_B2A (256x256 RGB JPEG, quality 95).
"""
import argparse
import shutil
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from common import MEMBER_DIR, load_config, repo_path
from data import FolderDataset, eval_transform
from models import make_generator


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    d, m = cfg["data"], cfg["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = repo_path(args.checkpoint_dir)
    for key, src, sub in (("G_A2B", d["monet_dir"], "pred_A2B"), ("G_B2A", d["photo_dir"], "pred_B2A")):
        g = make_generator(m).to(device).eval()
        g.load_state_dict(torch.load(ckpt / f"{key}.pt", map_location=device))
        out = MEMBER_DIR / "outputs" / sub
        for f in out.glob("*.jpg"):
            f.unlink()
        out.mkdir(parents=True, exist_ok=True)
        loader = DataLoader(FolderDataset(repo_path(src), eval_transform(d["crop_size"])), batch_size=16, num_workers=2)
        n = 0
        for x, names in loader:
            y = ((g(x.to(device)).clamp(-1, 1) * 0.5 + 0.5) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for img, name in zip(y, names):
                Image.fromarray(img).save(out / (Path(name).stem + ".jpg"), quality=95)
                n += 1
        print(f"{sub}: {n} images from {ckpt.name}/{key}.pt")
    (MEMBER_DIR / "outputs" / "pred_source.txt").write_text(f"{args.checkpoint_dir}\n")


if __name__ == "__main__":
    main()
