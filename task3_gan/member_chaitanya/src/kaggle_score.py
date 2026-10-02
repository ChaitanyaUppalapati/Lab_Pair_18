"""Score generated photo->Monet JPGs the way the Kaggle competition describes.

Feature extractor (verified to reproduce task3_gan/data/real_stats.npz exactly on the 300
Monet images): torchvision Inception-v3 (IMAGENET1K_V1) with fc = Identity -> 2048-d,
input resized to 299x299 and ImageNet-normalised.

FID   = ||mu_r - mu_g||^2 + Tr(S_r + S_g - 2 (S_r S_g)^(1/2)), with mu_r / S_r from real_stats.npz.
MiFID = mean cosine distance between generated and real features after subsampling to equal
        set sizes. The competition text admits two readings, so both are reported until the
        provided evaluation script settles it:
          mifid_paired : random equal-size subsets, d_i = 1 - cos(g_i, r_i)
          mifid_nearest: d_i = min_j (1 - cos(g_i, r_j))
score = (FID + MiFID) / 2

Usage:
    python task3_gan/member_chaitanya/src/kaggle_score.py --images task3_gan/member_chaitanya/outputs/pred_B2A
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from scipy import linalg

from common import REPO_ROOT

REAL_STATS = REPO_ROOT / "task3_gan" / "data" / "real_stats.npz"


def inception() -> torch.nn.Module:
    m = torchvision.models.inception_v3(weights="IMAGENET1K_V1")
    m.fc = torch.nn.Identity()
    return m.eval()


@torch.no_grad()
def features(files: list[Path], device: str, batch: int = 50) -> np.ndarray:
    tf = T.Compose([T.Resize((299, 299)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    net, out = inception().to(device), []
    for i in range(0, len(files), batch):
        x = torch.stack([tf(Image.open(f).convert("RGB")) for f in files[i : i + batch]]).to(device)
        out.append(net(x).cpu())
    return torch.cat(out).numpy().astype(np.float64)


def fid(mu1, s1, mu2, s2) -> float:
    covmean = linalg.sqrtm(s1 @ s2)
    if not np.isfinite(covmean).all():
        eps = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + eps) @ (s2 + eps))
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def cosine_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return 1.0 - a @ b.T


def score(image_dir: Path, seed: int = 1337) -> dict:
    files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"})
    bad = [f.name for f in files if Image.open(f).size != (256, 256) or Image.open(f).mode != "RGB"]
    if bad:
        raise ValueError(f"{len(bad)} images are not 256x256 RGB, e.g. {bad[:3]}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = features(files, device)
    ref = np.load(REAL_STATS)
    r, mu_r, s_r = ref["feats_real"].astype(np.float64), ref["mu_real"].astype(np.float64), ref["sigma_real"]
    f = fid(g.mean(0), np.cov(g, rowvar=False), mu_r, s_r)
    rng = np.random.default_rng(seed)
    n = min(len(g), len(r))
    gi, ri = rng.choice(len(g), n, replace=False), rng.choice(len(r), n, replace=False)
    mifid_paired = float(np.mean(np.diag(cosine_dist(g[gi], r[ri]))))
    mifid_nearest = float(cosine_dist(g, r).min(1).mean())
    return {"images": str(image_dir.relative_to(REPO_ROOT).as_posix()), "n_images": len(files), "fid": f,
            "mifid_paired": mifid_paired, "mifid_nearest": mifid_nearest,
            "score_paired": (f + mifid_paired) / 2, "score_nearest": (f + mifid_nearest) / 2}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="repo-relative folder of generated photo->Monet JPGs")
    parser.add_argument("--out", help="optional repo-relative JSON output path")
    args = parser.parse_args()
    result = score(REPO_ROOT / args.images)
    print(json.dumps(result, indent=2))
    if args.out:
        (REPO_ROOT / args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
