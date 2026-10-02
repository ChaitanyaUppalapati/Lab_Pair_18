"""Instructor-provided Part 3 evaluation script (Part3_Evaluation_Script.ipynb), ported to a CLI.

The computation is kept identical to the notebook:
  - first N_EVAL = 300 images (sorted file names) per folder
  - torchvision Inception-v3 IMAGENET1K_V1, fc = Identity; Resize(299), CenterCrop(299), ImageNet norm
  - FID from the real / generated activations; MiFID = mean scipy cosine distance, paired by index
  - submission FID = (FID_A2B + FID_B2A) / 2, submission MiFID = (MiFID_A2B + MiFID_B2A) / 2
Only change: newer SciPy removed sqrtm(..., disp=False), so the call falls back to sqrtm(...) when
needed (same matrix square root).

Usage:
    python task3_gan/member_chaitanya/src/official_eval.py \
        --pred-a2b task3_gan/member_chaitanya/outputs/pred_A2B \
        --pred-b2a task3_gan/member_chaitanya/outputs/pred_B2A [--write-submission]
"""
import argparse
import glob
import json
import os

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from scipy.spatial.distance import cosine

from common import MEMBER_DIR, REPO_ROOT

N_EVAL = 300
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INCEPTION_TF = T.Compose([
    T.Resize(299),
    T.CenterCrop(299),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def list_images(folder):
    exts = (".jpg", ".jpeg", ".png")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(list(set(paths)))


def take_n(paths, n):
    return paths if n is None else paths[: min(n, len(paths))]


def get_inception_model():
    inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    inception.fc = nn.Identity()
    return inception.to(device).eval()


def load_batch(paths):
    return torch.stack([INCEPTION_TF(Image.open(p).convert("RGB")) for p in paths], dim=0)


@torch.no_grad()
def get_activations(model, image_paths, batch_size=32):
    feats = []
    for i in range(0, len(image_paths), batch_size):
        x = load_batch(image_paths[i : i + batch_size]).to(device)
        feats.append(model(x).detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


def _sqrtm(m):
    try:
        covmean, _ = scipy.linalg.sqrtm(m, disp=False)
    except TypeError:  # SciPy >= 1.16 dropped `disp`
        covmean = scipy.linalg.sqrtm(m)
    return covmean


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    covmean = _sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = _sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


def calculate_fid_mifid(model, real_paths, gen_paths, batch_size=32, subsample_to_match=True):
    real_paths, gen_paths = sorted(real_paths), sorted(gen_paths)
    if subsample_to_match:
        n = min(len(real_paths), len(gen_paths))
        real_paths, gen_paths = real_paths[:n], gen_paths[:n]
    real_act = get_activations(model, real_paths, batch_size=batch_size)
    gen_act = get_activations(model, gen_paths, batch_size=batch_size)
    mu_r, sig_r = real_act.mean(axis=0), np.cov(real_act, rowvar=False)
    mu_g, sig_g = gen_act.mean(axis=0), np.cov(gen_act, rowvar=False)
    fid = frechet_distance(mu_r, sig_r, mu_g, sig_g)
    m = min(len(real_act), len(gen_act))
    mifid = float(np.mean([cosine(real_act[i], gen_act[i]) for i in range(m)]))
    return fid, mifid


def evaluate(pred_a2b: str, pred_b2a: str, real_monet: str = "task3_gan/data/monet_jpg",
             real_photo: str = "task3_gan/data/photo_jpg") -> dict:
    model = get_inception_model()
    folders = {k: str(REPO_ROOT / v) for k, v in
               (("real_monet", real_monet), ("real_photo", real_photo), ("a2b", pred_a2b), ("b2a", pred_b2a))}
    for d in folders.values():
        assert os.path.isdir(d), f"Missing folder: {d}"
    imgs = {k: take_n(list_images(v), N_EVAL) for k, v in folders.items()}
    fid_b2a, mifid_b2a = calculate_fid_mifid(model, imgs["real_monet"], imgs["b2a"], BATCH_SIZE)
    fid_a2b, mifid_a2b = calculate_fid_mifid(model, imgs["real_photo"], imgs["a2b"], BATCH_SIZE)
    return {"FID_B2A": fid_b2a, "MiFID_B2A": mifid_b2a, "FID_A2B": fid_a2b, "MiFID_A2B": mifid_a2b,
            "FID": (fid_a2b + fid_b2a) / 2, "MiFID": (mifid_a2b + mifid_b2a) / 2,
            "counts": {k: len(v) for k, v in imgs.items()}, "pred_a2b": pred_a2b, "pred_b2a": pred_b2a}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-a2b", required=True)
    parser.add_argument("--pred-b2a", required=True)
    parser.add_argument("--write-submission", action="store_true",
                        help="write task3_gan/member_chaitanya/submission.csv and outputs/official_eval.json")
    args = parser.parse_args()
    r = evaluate(args.pred_a2b, args.pred_b2a)
    print(f"[Photo->Monet] FID={r['FID_B2A']:.3f}  MiFID={r['MiFID_B2A']:.4f}")
    print(f"[Monet->Photo] FID={r['FID_A2B']:.3f}  MiFID={r['MiFID_A2B']:.4f}")
    print(f"submission: FID={r['FID']}  MiFID={r['MiFID']}")
    if args.write_submission:
        import pandas as pd

        pd.DataFrame([{"ID": 1, "FID": float(r["FID"]), "MiFID": float(r["MiFID"])}]).to_csv(
            MEMBER_DIR / "submission.csv", index=False)
        (MEMBER_DIR / "outputs" / "official_eval.json").write_text(json.dumps(r, indent=2))
        print((MEMBER_DIR / "submission.csv").read_text())


if __name__ == "__main__":
    main()
