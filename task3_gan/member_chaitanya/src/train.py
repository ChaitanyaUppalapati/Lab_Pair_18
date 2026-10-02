"""Task 3.1 CycleGAN training (photo <-> Monet).

Naming: A = photo, B = Monet. G_A2B: photo -> Monet, G_B2A: Monet -> photo.
D_A judges photos, D_B judges Monet paintings.

Generator objective (one step for both generators):
    L_G = LSGAN(D_B(G_A2B(a)), 1) + LSGAN(D_A(G_B2A(b)), 1)
        + lambda_cycle    * (|G_B2A(G_A2B(a)) - a|_1 + |G_A2B(G_B2A(b)) - b|_1)
        + lambda_identity * (|G_A2B(b) - b|_1 + |G_B2A(a) - a|_1)
Discriminator objective (each D, fakes drawn from its 50-image history pool):
    L_D = d_loss_scale * (MSE(D(real), 1) + MSE(D(fake), 0))
LR: constant for constant_lr_epochs, then linear decay to 0 at the last iteration.

Usage:
    python task3_gan/member_chaitanya/src/train.py --config task3_gan/member_chaitanya/configs/full.yaml [--resume]
"""
import argparse
import itertools
import json
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from common import get_logger, hardware_string, load_config, repo_path, run_paths, set_seed
from data import FolderDataset, UnpairedDataset, eval_transform, list_images, train_transform
from models import ImagePool, PatchDiscriminator, ResnetGenerator, count_params, diff_augment, init_weights


def build_models(cfg: dict, device: str) -> dict:
    m = cfg["model"]
    if m["upsampling"] != "nearest_conv" or m["norm"] != "instance":
        raise ValueError("this implementation supports upsampling=nearest_conv and norm=instance")
    nets = {
        "G_A2B": ResnetGenerator(m["ngf"], m["n_res_blocks"], m["n_downsampling"]),
        "G_B2A": ResnetGenerator(m["ngf"], m["n_res_blocks"], m["n_downsampling"]),
        "D_A": PatchDiscriminator(m["ndf"]),
        "D_B": PatchDiscriminator(m["ndf"]),
    }
    for net in nets.values():
        net.apply(lambda mod: init_weights(mod, m["init_std"]))
        net.to(device)
    return nets


def grad_norm(params) -> float:
    norms = [p.grad.detach().norm(2) for p in params if p.grad is not None]
    return torch.norm(torch.stack(norms), 2).item() if norms else 0.0


def lsgan(pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
    return F.mse_loss(pred, torch.ones_like(pred) if target_is_real else torch.zeros_like(pred))


@torch.no_grad()
def save_grid(nets: dict, fixed_a: torch.Tensor, fixed_b: torch.Tensor, path) -> None:
    """Rows: photo | photo->Monet | reconstruction ; Monet | Monet->photo | reconstruction."""
    for n in nets.values():
        n.eval()
    fake_b = nets["G_A2B"](fixed_a)
    rec_a = nets["G_B2A"](fake_b)
    fake_a = nets["G_B2A"](fixed_b)
    rec_b = nets["G_A2B"](fake_a)
    grid = torch.cat([fixed_a, fake_b, rec_a, fixed_b, fake_a, rec_b], 0)
    save_image(grid * 0.5 + 0.5, path, nrow=fixed_a.size(0))
    for n in nets.values():
        n.train()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true", help="continue from checkpoints/<run_id>/latest.pt")
    args = parser.parse_args()
    cfg = load_config(args.config)
    d_cfg, l_cfg, t_cfg = cfg["data"], cfg["loss"], cfg["training"]
    if t_cfg["precision"] != "fp32" or l_cfg["adversarial"] != "lsgan":
        raise ValueError("this implementation supports precision=fp32 and adversarial=lsgan")
    run_id = cfg["run_id"]
    paths = run_paths(run_id)
    log = get_logger(paths["raw_log"], f"train_{run_id}")
    set_seed(int(cfg["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    photo_dir, monet_dir = repo_path(d_cfg["photo_dir"]), repo_path(d_cfg["monet_dir"])
    ds = UnpairedDataset(photo_dir, monet_dir,
                         train_transform(d_cfg["load_size"], d_cfg["crop_size"], d_cfg["hflip"]),
                         train_transform(d_cfg["load_size"], d_cfg["crop_size"], d_cfg["hflip"]))
    loader = DataLoader(ds, batch_size=t_cfg["batch_size"], shuffle=True, drop_last=True,
                        num_workers=d_cfg["num_workers"], pin_memory=device == "cuda",
                        persistent_workers=d_cfg["num_workers"] > 0,
                        generator=torch.Generator().manual_seed(int(cfg["seed"])))
    n_fixed = int(t_cfg["num_fixed_samples"])
    etf = eval_transform(d_cfg["crop_size"])
    fixed_a = torch.stack([FolderDataset(photo_dir, etf)[i][0] for i in range(n_fixed)]).to(device)
    fixed_b = torch.stack([FolderDataset(monet_dir, etf)[i][0] for i in range(n_fixed)]).to(device)

    nets = build_models(cfg, device)
    params = {k: count_params(v) for k, v in nets.items()}
    g_params = list(itertools.chain(nets["G_A2B"].parameters(), nets["G_B2A"].parameters()))
    d_params = list(itertools.chain(nets["D_A"].parameters(), nets["D_B"].parameters()))
    opt_g = torch.optim.Adam(g_params, lr=t_cfg["lr_g"], betas=tuple(t_cfg["betas"]))
    opt_d = torch.optim.Adam(d_params, lr=t_cfg["lr_d"], betas=tuple(t_cfg["betas"]))
    pool_a, pool_b = ImagePool(t_cfg["pool_size"]), ImagePool(t_cfg["pool_size"])

    epochs, const_epochs = int(t_cfg["epochs"]), int(t_cfg["constant_lr_epochs"])
    iters_per_epoch = len(loader)
    total_iters, const_iters = epochs * iters_per_epoch, const_epochs * iters_per_epoch
    lam_c, lam_i, d_scale = l_cfg["lambda_cycle"], l_cfg["lambda_identity"], l_cfg["d_loss_scale"]
    use_diffaug = bool(t_cfg["diffaugment_monet_d"])

    start_epoch, it = 1, 0
    history = {"intervals": [], "epochs": []}
    nan_count, max_gn = 0, {"G": 0.0, "D": 0.0}
    train_time_prev = 0.0
    latest = paths["checkpoints"] / "latest.pt"
    if args.resume and latest.exists():
        state = torch.load(latest, map_location=device)
        for k, net in nets.items():
            net.load_state_dict(state[k])
        opt_g.load_state_dict(state["opt_g"])
        opt_d.load_state_dict(state["opt_d"])
        start_epoch, it = state["epoch"] + 1, state["iter"]
        history, nan_count, max_gn = state["history"], state["nan_count"], state["max_grad_norm"]
        train_time_prev = state["train_time_s"]
        log.info("resumed from %s at epoch %d iter %d", latest, state["epoch"], it)

    log.info("run_id=%s config=%s", run_id, cfg["_config_path"])
    log.info("config=%s", json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}))
    log.info("hardware=%s torch=%s", hardware_string(), torch.__version__)
    log.info("photos=%d monets=%d iters_per_epoch=%d total_iters=%d params=%s", len(ds.photos), len(ds.monets),
             iters_per_epoch, total_iters, json.dumps(params))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    run_start = time.perf_counter()

    for epoch in range(start_epoch, epochs + 1):
        sums, count, epoch_start = {}, 0, time.perf_counter()
        for real_a, real_b in loader:
            lr_scale = 1.0 if it < const_iters else max(0.0, 1.0 - (it - const_iters) / max(1, total_iters - const_iters))
            for g in opt_g.param_groups:
                g["lr"] = t_cfg["lr_g"] * lr_scale
            for g in opt_d.param_groups:
                g["lr"] = t_cfg["lr_d"] * lr_scale
            real_a, real_b = real_a.to(device, non_blocking=True), real_b.to(device, non_blocking=True)
            d_b_in = diff_augment if use_diffaug else (lambda x: x)

            # ---- generators ----
            for p in d_params:
                p.requires_grad_(False)
            fake_b = nets["G_A2B"](real_a)
            fake_a = nets["G_B2A"](real_b)
            rec_a = nets["G_B2A"](fake_b)
            rec_b = nets["G_A2B"](fake_a)
            idt_b = nets["G_A2B"](real_b)
            idt_a = nets["G_B2A"](real_a)
            adv_a2b = lsgan(nets["D_B"](d_b_in(fake_b)), True)
            adv_b2a = lsgan(nets["D_A"](fake_a), True)
            cyc_a, cyc_b = F.l1_loss(rec_a, real_a), F.l1_loss(rec_b, real_b)
            idt_l_a, idt_l_b = F.l1_loss(idt_a, real_a), F.l1_loss(idt_b, real_b)
            loss_g = adv_a2b + adv_b2a + lam_c * (cyc_a + cyc_b) + lam_i * (idt_l_a + idt_l_b)
            if not math.isfinite(loss_g.item()):
                nan_count += 1
                log.warning("iter=%d non-finite generator loss; skipping", it)
                it += 1
                continue
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            gn_g = grad_norm(g_params)
            opt_g.step()

            # ---- discriminators ----
            for p in d_params:
                p.requires_grad_(True)
            pooled_a, pooled_b = pool_a.query(fake_a), pool_b.query(fake_b)
            pred_real_a, pred_fake_a = nets["D_A"](real_a), nets["D_A"](pooled_a)
            pred_real_b, pred_fake_b = nets["D_B"](d_b_in(real_b)), nets["D_B"](d_b_in(pooled_b))
            loss_d_a = d_scale * (lsgan(pred_real_a, True) + lsgan(pred_fake_a, False))
            loss_d_b = d_scale * (lsgan(pred_real_b, True) + lsgan(pred_fake_b, False))
            loss_d = loss_d_a + loss_d_b
            if not math.isfinite(loss_d.item()):
                nan_count += 1
                log.warning("iter=%d non-finite discriminator loss; skipping", it)
                it += 1
                continue
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            gn_d = grad_norm(d_params)
            opt_d.step()

            max_gn["G"], max_gn["D"] = max(max_gn["G"], gn_g), max(max_gn["D"], gn_d)
            vals = {"loss_G": loss_g.item(), "adv_A2B": adv_a2b.item(), "adv_B2A": adv_b2a.item(),
                    "cycle_A": cyc_a.item(), "cycle_B": cyc_b.item(), "identity_A": idt_l_a.item(),
                    "identity_B": idt_l_b.item(), "loss_D_A": loss_d_a.item(), "loss_D_B": loss_d_b.item(),
                    "D_B_real_mean": pred_real_b.mean().item(), "D_B_fake_mean": pred_fake_b.mean().item(),
                    "grad_norm_G": gn_g, "grad_norm_D": gn_d}
            for k, v in vals.items():
                sums[k] = sums.get(k, 0.0) + v
            count += 1
            it += 1
            if it % t_cfg["log_every_iters"] == 0:
                row = {"iter": it, "epoch": epoch, "lr": t_cfg["lr_g"] * lr_scale,
                       **{k: v / count for k, v in sums.items()}}
                history["intervals"].append(row)
                elapsed = time.perf_counter() - epoch_start
                log.info("iter=%d epoch=%d lr=%.2e G=%.3f advA2B=%.3f advB2A=%.3f cycA=%.3f cycB=%.3f idtA=%.3f idtB=%.3f "
                         "D_A=%.3f D_B=%.3f D_B(real)=%.2f D_B(fake)=%.2f gnG=%.2f gnD=%.2f it_per_s=%.2f",
                         it, epoch, row["lr"], row["loss_G"], row["adv_A2B"], row["adv_B2A"], row["cycle_A"],
                         row["cycle_B"], row["identity_A"], row["identity_B"], row["loss_D_A"], row["loss_D_B"],
                         row["D_B_real_mean"], row["D_B_fake_mean"], row["grad_norm_G"], row["grad_norm_D"],
                         count / elapsed)

        epoch_s = time.perf_counter() - epoch_start
        ep = {"epoch": epoch, "iters": count, "epoch_time_s": epoch_s, "iters_per_s": count / epoch_s,
              **{k: v / max(1, count) for k, v in sums.items()}}
        history["epochs"].append(ep)
        log.info("epoch=%d done time_s=%.0f it_per_s=%.2f G=%.3f cycA=%.3f cycB=%.3f idtA=%.3f idtB=%.3f D_A=%.3f D_B=%.3f "
                 "D_B(real)=%.2f D_B(fake)=%.2f nan_count=%d", epoch, epoch_s, ep["iters_per_s"], ep["loss_G"],
                 ep["cycle_A"], ep["cycle_B"], ep["identity_A"], ep["identity_B"], ep["loss_D_A"], ep["loss_D_B"],
                 ep["D_B_real_mean"], ep["D_B_fake_mean"], nan_count)
        if ep["loss_D_B"] < 0.05:
            log.warning("epoch=%d D_B (Monet) loss %.4f is near 0: possible discriminator overfitting", epoch, ep["loss_D_B"])
        if epoch % t_cfg["grid_every_epochs"] == 0 or epoch == epochs:
            (paths["outputs"] / "grids").mkdir(exist_ok=True)
            save_grid(nets, fixed_a, fixed_b, paths["outputs"] / "grids" / f"epoch_{epoch:03d}.png")
        train_time = train_time_prev + (time.perf_counter() - run_start)
        torch.save({**{k: n.state_dict() for k, n in nets.items()}, "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(), "epoch": epoch, "iter": it, "history": history,
                    "nan_count": nan_count, "max_grad_norm": max_gn, "train_time_s": train_time, "config": cfg}, latest)
        if epoch % t_cfg["checkpoint_every_epochs"] == 0 or epoch == epochs:
            snap = paths["checkpoints"] / f"epoch_{epoch:03d}"
            snap.mkdir(exist_ok=True)
            for k in ("G_A2B", "G_B2A"):
                torch.save(nets[k].state_dict(), snap / f"{k}.pt")
        (paths["outputs"] / "history.json").write_text(json.dumps(history, indent=1))

    for k, n in nets.items():
        torch.save(n.state_dict(), paths["checkpoints"] / f"{k}.pt")
    total_s = train_time_prev + (time.perf_counter() - run_start)
    summary = {"run_id": run_id, "params": params, "param_count_total": sum(params.values()),
               "param_count_generators": params["G_A2B"] + params["G_B2A"], "photos": len(ds.photos),
               "monets": len(ds.monets), "iters": it, "train_time_s": total_s,
               "train_iters_per_sec": it / total_s,
               "train_images_per_sec": it * t_cfg["batch_size"] * 2 / total_s,
               "peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
               "max_grad_norm_G": max_gn["G"], "max_grad_norm_D": max_gn["D"], "nan_count": nan_count,
               "final_epoch": history["epochs"][-1], "hardware": hardware_string()}
    (paths["outputs"] / "train_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("done summary=%s", json.dumps(summary))


if __name__ == "__main__":
    main()
