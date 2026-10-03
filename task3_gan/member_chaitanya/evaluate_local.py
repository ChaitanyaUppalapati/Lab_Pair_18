"""Task 3 local evaluation -> full_metrics_report.csv (everything except the Kaggle score).

Usage:
    python task3_gan/member_chaitanya/evaluate_local.py --config task3_gan/member_chaitanya/configs/full.yaml

Directions (Kaggle naming, A = Monet, B = photo): A2B = Monet -> photo (generated vs real photos),
B2A = photo -> Monet (generated vs real Monet; the direction Kaggle scores).
All translations are written to outputs/pred_A2B/ and outputs/pred_B2A/.

Metric definitions (pretrained networks are used for evaluation only, never to make images)
- FID / KID: torchmetrics, Inception-v3 pool features (2048-d); KID uses subsets of
  min(kid_subset_size, n_real, n_fake) images.
- generative precision / recall (Kynkaanniemi et al., 2019) and density / coverage
  (Naeem et al., 2020): k-NN (k = 5) manifolds in Inception feature space, using
  equal-size seeded random subsets of real and generated features.
- cycle L1: mean |F(G(x)) - x| in [0, 1] pixel units over all inputs of that direction.
- LPIPS: LPIPS-AlexNet distance between each input and its translation (how much the
  image changed perceptually), mean over inputs. lpips_reconstruction: input vs F(G(x)).
- content cosine similarity: cosine similarity of Inception features of input vs translation.
- images/sec: generator inference throughput (batch of eval_batch_size, fp32).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from common import MEMBER_DIR, hardware_string, load_config, repo_path, run_paths, set_seed  # noqa: E402
from data import FolderDataset, eval_transform  # noqa: E402
from models import count_params, make_generator  # noqa: E402

CSV_COLUMNS = ["run_id", "checkpoint", "direction", "fid", "kid_mean", "kid_std", "gen_precision", "gen_recall",
               "density", "coverage", "cycle_l1", "lpips", "content_cosine_sim", "final_g_loss", "final_d_loss",
               "cycle_loss", "identity_loss", "max_grad_norm", "nan_count", "human_style", "human_content",
               "human_artifacts", "inter_rater_kappa", "pct_agreement", "param_count", "train_time_s",
               "images_per_sec", "peak_memory_mb", "kaggle_public", "kaggle_private", "kaggle_rank", "hardware"]
KEEP_MANUAL = ["human_style", "human_content", "human_artifacts", "inter_rater_kappa", "pct_agreement",
               "kaggle_public", "kaggle_private", "kaggle_rank"]


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1, 1) * 0.5 + 0.5) * 255).round().to(torch.uint8)


def knn_radii(feats: torch.Tensor, k: int) -> torch.Tensor:
    d = torch.cdist(feats, feats)
    return d.kthvalue(k + 1, dim=1).values  # +1 skips the zero self-distance


def prdc(real: torch.Tensor, fake: torch.Tensor, k: int) -> dict:
    r_rad, f_rad = knn_radii(real, k), knn_radii(fake, k)
    d_fr = torch.cdist(fake, real)  # (n_fake, n_real)
    precision = (d_fr <= r_rad.unsqueeze(0)).any(1).float().mean().item()
    recall = (d_fr.t() <= f_rad.unsqueeze(0)).any(1).float().mean().item()
    density = ((d_fr <= r_rad.unsqueeze(0)).float().sum(1) / k).mean().item()
    coverage = (d_fr.min(0).values <= r_rad).float().mean().item()
    return {"gen_precision": precision, "gen_recall": recall, "density": density, "coverage": coverage}


@torch.no_grad()
def translate_and_score(g_fwd, g_back, loader, out_dir, inception, lpips_fn, device, fake_metrics):
    out_dir.mkdir(parents=True, exist_ok=True)

    in_feats, out_feats, cyc, lp, lp_rec = [], [], [], [], []
    gen_time, n_imgs = 0.0, 0
    for x, names in loader:
        x = x.to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = g_fwd(x)
        if device == "cuda":
            torch.cuda.synchronize()
        gen_time += time.perf_counter() - t0
        n_imgs += x.size(0)
        rec = g_back(y)
        cyc.append((rec - x).abs().mul(0.5).flatten(1).mean(1).cpu())  # [0,1] pixel units
        lp.append(lpips_fn(x, y).flatten().cpu())
        lp_rec.append(lpips_fn(x, rec).flatten().cpu())
        y_u8 = to_uint8(y)
        for metric in fake_metrics:
            metric.update(y_u8, real=False)
        in_feats.append(inception(to_uint8(x)).cpu())
        out_feats.append(inception(y_u8).cpu())
        for img, n in zip(y_u8.permute(0, 2, 3, 1).cpu().numpy(), names):
            Image.fromarray(img).save(out_dir / (Path(n).stem + ".jpg"), quality=95)
    return {"in_feats": torch.cat(in_feats), "out_feats": torch.cat(out_feats), "cycle_l1": torch.cat(cyc).mean().item(),
            "lpips": torch.cat(lp).mean().item(), "lpips_reconstruction": torch.cat(lp_rec).mean().item(),
            "images_per_sec": n_imgs / gen_time, "n": n_imgs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--kid-subset-size", type=int, default=100)
    parser.add_argument("--k", type=int, default=5, help="k for precision/recall/density/coverage")
    parser.add_argument("--snapshot", default=None,
                        help="evaluate checkpoints/<run_id>/<snapshot>/ (e.g. epoch_035) instead of the final generators")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_id, d_cfg, m_cfg = cfg["run_id"], cfg["data"], cfg["model"]
    paths = run_paths(run_id)
    set_seed(int(cfg["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = paths["checkpoints"] / args.snapshot if args.snapshot else paths["checkpoints"]
    ckpt_rel = ckpt_dir.relative_to(MEMBER_DIR).as_posix()

    import lpips
    from torchmetrics.image.fid import FrechetInceptionDistance, NoTrainInceptionV3
    from torchmetrics.image.kid import KernelInceptionDistance

    gens = {}
    for k in ("G_A2B", "G_B2A"):
        g = make_generator(m_cfg).to(device).eval()
        g.load_state_dict(torch.load(ckpt_dir / f"{k}.pt", map_location=device))
        gens[k] = g
    inception = NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048"]).to(device).eval()
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    etf = eval_transform(d_cfg["crop_size"])
    photo_dir, monet_dir = repo_path(d_cfg["photo_dir"]), repo_path(d_cfg["monet_dir"])
    loaders = {name: DataLoader(FolderDataset(folder, etf), batch_size=args.eval_batch_size, num_workers=2)
               for name, folder in (("photo", photo_dir), ("monet", monet_dir))}
    pred_root = paths["outputs"] if cfg.get("smoke") else MEMBER_DIR / "outputs"

    summary = json.loads((paths["outputs"] / "train_summary.json").read_text())
    last = summary["final_epoch"]
    rng = torch.Generator().manual_seed(int(cfg["seed"]))
    rows = []
    for direction, (fwd, back, src, real_loader) in {  # Kaggle naming: A = Monet, B = photo
        "A2B (Monet->photo)": ("G_A2B", "G_B2A", "monet", "photo"),
        "B2A (photo->Monet)": ("G_B2A", "G_A2B", "photo", "monet"),
    }.items():
        tag = direction.split()[0]
        n_real, n_src = len(loaders[real_loader].dataset), len(loaders[src].dataset)
        fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
        kid = KernelInceptionDistance(feature=2048, subset_size=min(args.kid_subset_size, n_real, n_src),
                                      normalize=False).to(device)
        # translations are scored from memory (uint8), then also saved as JPEG
        res = translate_and_score(gens[fwd], gens[back], loaders[src], pred_root / f"pred_{tag}",
                                  inception, lpips_fn, device, (fid, kid))
        real_feats = []
        with torch.no_grad():
            for x, _ in loaders[real_loader]:
                u = to_uint8(x.to(device))
                fid.update(u, real=True)
                kid.update(u, real=True)
                real_feats.append(inception(u).cpu())
        real_feats = torch.cat(real_feats)
        kid_mean, kid_std = kid.compute()
        n = min(len(real_feats), len(res["out_feats"]))
        ri = torch.randperm(len(real_feats), generator=rng)[:n]
        fi = torch.randperm(len(res["out_feats"]), generator=rng)[:n]
        pr = prdc(real_feats[ri].double(), res["out_feats"][fi].double(), args.k)
        cos = F.cosine_similarity(res["in_feats"], res["out_feats"], dim=1).mean().item()
        rows.append({
            "run_id": run_id, "checkpoint": f"{ckpt_rel}/{fwd}.pt", "direction": direction,
            "fid": fid.compute().item(), "kid_mean": kid_mean.item(), "kid_std": kid_std.item(), **pr,
            "cycle_l1": res["cycle_l1"], "lpips": res["lpips"], "content_cosine_sim": cos,
            "final_g_loss": last["loss_G"], "final_d_loss": last["loss_D_B"] if tag == "A2B" else last["loss_D_A"],
            "cycle_loss": last["cycle_A"] if tag == "A2B" else last["cycle_B"],
            "identity_loss": last["identity_B"] if tag == "A2B" else last["identity_A"],
            "max_grad_norm": f"G={summary['max_grad_norm_G']:.3f}; D={summary['max_grad_norm_D']:.3f}",
            "nan_count": summary["nan_count"], "param_count": count_params(gens[fwd]),
            "train_time_s": summary["train_time_s"], "images_per_sec": res["images_per_sec"],
            "peak_memory_mb": summary["peak_memory_mb"], "hardware": hardware_string(),
            "_extra": {"lpips_reconstruction": res["lpips_reconstruction"], "n_translated": res["n"],
                       "n_real_target": n_real, "prdc_subset": n, "k": args.k},
        })

    report = paths["outputs"] / "metrics_rows.csv" if cfg.get("smoke") else MEMBER_DIR / "full_metrics_report.csv"
    old = {}
    if report.exists():
        with open(report, newline="", encoding="utf-8") as f:
            old = {(r["run_id"], r["direction"]): r for r in csv.DictReader(f) if r.get("run_id")}
    for r in rows:  # keep manually entered human-audit / Kaggle values for this run
        prev = old.pop((r["run_id"], r["direction"]), {})
        for k in KEEP_MANUAL:
            r[k] = prev.get(k, "")
    extra = {r["direction"]: r.pop("_extra") for r in rows}
    with open(report, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(list(old.values()) + rows)
    (paths["outputs"] / "eval_summary.json").write_text(json.dumps({"rows": rows, "extra": extra}, indent=2))
    plot_losses(paths["outputs"])
    print(json.dumps(rows, indent=2))


def plot_losses(out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = json.loads((out_dir / "history.json").read_text())["intervals"]
    if not h:
        return
    it = [r["iter"] for r in h]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, keys, title in (
        (axes[0, 0], ["adv_A2B", "adv_B2A"], "generator adversarial (LSGAN)"),
        (axes[0, 1], ["loss_D_A", "loss_D_B"], "discriminator loss (x0.5)"),
        (axes[1, 0], ["cycle_A", "cycle_B", "identity_A", "identity_B"], "cycle / identity L1"),
        (axes[1, 1], ["grad_norm_G", "grad_norm_D"], "gradient norm (log)"),
    ):
        for k in keys:
            ax.plot(it, [r[k] for r in h], lw=0.8, label=k)
        ax.set(title=title, xlabel="iteration")
        ax.legend()
    axes[1, 1].set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
