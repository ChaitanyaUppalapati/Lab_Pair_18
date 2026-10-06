"""Per-image scores of the submitted generators and ranked failure candidates (input | translation | reconstruction).

For every image, both directions (Kaggle naming: A = Monet, B = photo):
  cycle_l1        mean |F(G(x)) - x| in [0,1] pixel units   (high -> cycle constraint not met / content lost)
  lpips_change    LPIPS(x, G(x))                              (very low -> barely translated / under-stylised)
  lpips_rec       LPIPS(x, F(G(x)))
  content_cos     cosine of Inception features, x vs G(x)    (low -> content not preserved)
Candidates per category are only a ranking; the failure type and the observation are the member's judgement.

usage: python task3_gan/member_chaitanya/src/failure_cases.py --config task3_gan/member_chaitanya/configs/full_run18.yaml --snapshot epoch_123_ema
writes outputs/failure_cases/{per_image_scores_<dir>.csv, <category>/<rank>_<name>.png, candidates.md}
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MEMBER_DIR, load_config, repo_path, run_paths, set_seed  # noqa: E402
from data import FolderDataset, eval_transform  # noqa: E402
from models import make_generator  # noqa: E402

OUT = MEMBER_DIR / "outputs" / "failure_cases"
TOP = 4
CATEGORIES = {  # name: (column, ascending, description)
    "high_cycle_error": ("cycle_l1", False, "largest cycle-reconstruction error"),
    "low_content": ("content_cos", True, "lowest input-vs-translation content similarity"),
    "barely_translated": ("lpips_change", True, "smallest perceptual change (possible under-translation)"),
}


def to_u8(x):
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def strip(x, y, r, labels):
    tiles = [Image.fromarray(t.permute(1, 2, 0).cpu().numpy()) for t in (x, y, r)]
    w, h = tiles[0].size
    canvas = Image.new("RGB", (3 * w + 20, h + 22), "white")
    d = ImageDraw.Draw(canvas)
    for i, (t, lab) in enumerate(zip(tiles, labels)):
        canvas.paste(t, (i * (w + 10), 22))
        d.text((i * (w + 10) + 4, 4), lab, fill="black")
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--snapshot", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = run_paths(cfg["run_id"])["checkpoints"] / args.snapshot
    gens = {}
    for k in ("G_A2B", "G_B2A"):
        g = make_generator(cfg["model"]).to(device).eval()
        g.load_state_dict(torch.load(ckpt / f"{k}.pt", map_location=device))
        gens[k] = g

    import lpips
    from torchmetrics.image.fid import NoTrainInceptionV3

    inception = NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048"]).to(device).eval()
    lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    tf = eval_transform(cfg["data"]["crop_size"])
    OUT.mkdir(parents=True, exist_ok=True)
    md = ["# Task 3 — failure candidates (auto-ranked; the member picks 3 and labels them)", "",
          f"Generators: `{ckpt.relative_to(MEMBER_DIR).as_posix()}` (the submitted model). Each strip is "
          "input | translation G(x) | reconstruction F(G(x)). Per-image scores for every image are in "
          "`per_image_scores_<direction>.csv`. The ranking is automatic; the failure type is a human judgement.", ""]
    for tag, (fwd, back, src, labels) in {
        "B2A": ("G_B2A", "G_A2B", cfg["data"]["photo_dir"], ("photo (input)", "-> Monet", "-> photo (rec.)")),
        "A2B": ("G_A2B", "G_B2A", cfg["data"]["monet_dir"], ("Monet (input)", "-> photo", "-> Monet (rec.)")),
    }.items():
        ds = FolderDataset(repo_path(src), tf)
        rows, cache = [], {}
        with torch.no_grad():
            for x, names in DataLoader(ds, batch_size=32, num_workers=2):
                x = x.to(device)
                y = gens[fwd](x)
                r = gens[back](y)
                cos = F.cosine_similarity(inception(to_u8(x)), inception(to_u8(y)), dim=1)
                for i, n in enumerate(names):
                    rows.append({"image": n, "cycle_l1": (r[i] - x[i]).abs().mean().item() / 2,
                                 "lpips_change": lp(x[i:i + 1], y[i:i + 1]).item(),
                                 "lpips_rec": lp(x[i:i + 1], r[i:i + 1]).item(), "content_cos": cos[i].item()})
        df = pd.DataFrame(rows)
        df.to_csv(OUT / f"per_image_scores_{tag}.csv", index=False)
        md += [f"## {tag} ({labels[0].split()[0]} -> {labels[1].split()[-1]}), {len(df)} images", "",
               "| Score | median | 5th pct | 95th pct |", "|---|---|---|---|"]
        md += [f"| {c} | {df[c].median():.4f} | {df[c].quantile(.05):.4f} | {df[c].quantile(.95):.4f} |"
               for c in ("cycle_l1", "lpips_change", "lpips_rec", "content_cos")]
        md.append("")
        for cat, (col, asc, desc) in CATEGORIES.items():
            picks = df.sort_values(col, ascending=asc).head(TOP)
            (OUT / f"{tag}_{cat}").mkdir(exist_ok=True)
            md += [f"### {tag}: {desc}", ""]
            for rank, row in enumerate(picks.itertuples(), 1):
                i = ds.files.index(next(f for f in ds.files if f.name == row.image))
                x = ds[i][0].unsqueeze(0).to(device)
                with torch.no_grad():
                    y = gens[fwd](x)
                    r = gens[back](y)
                rel = f"{tag}_{cat}/{rank}_{Path(row.image).stem}.png"
                strip(to_u8(x[0]), to_u8(y[0]), to_u8(r[0]), labels).save(OUT / rel)
                md += [f"**{rank}. `{row.image}`** — cycle L1 {row.cycle_l1:.4f}, LPIPS change {row.lpips_change:.3f}, "
                       f"LPIPS rec. {row.lpips_rec:.3f}, content cos {row.content_cos:.3f}", "", f"![]({rel})", ""]
        print(tag, len(df), "images scored")
    (OUT / "candidates.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
