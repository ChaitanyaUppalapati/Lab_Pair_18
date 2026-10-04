"""Task 3.1 CycleGAN training (photo <-> Monet).

Naming follows the Kaggle competition: A = Monet, B = photo.
G_A2B: Monet -> photo, G_B2A: photo -> Monet (the direction Kaggle scores).
D_A judges Monet paintings, D_B judges photos.

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
from models import ImagePool, MultiScaleDiscriminator, count_params, diff_augment, make_discriminator, make_generator


def build_models(cfg: dict, device: str) -> dict:
    m = cfg["model"]
    if m["upsampling"] != "nearest_conv" or m["norm"] != "instance":
        raise ValueError("this implementation supports upsampling=nearest_conv and norm=instance")
    # attention / spectral norm are optional config switches (default off = the original architecture)
    nets = {"G_A2B": make_generator(m), "G_B2A": make_generator(m),
            "D_A": make_discriminator(m), "D_B": make_discriminator(m)}
    for net in nets.values():
        net.to(device)
    return nets


def grad_norm(params) -> torch.Tensor:
    """Global L2 norm of all gradients, kept on the device (no host sync)."""
    grads = [p.grad.detach() for p in params if p.grad is not None]
    return torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))


# per-iteration statistics, accumulated on the device and read back only when logging
STAT_KEYS = ["loss_G", "adv_A2B", "adv_B2A", "cycle_A", "cycle_B", "identity_A", "identity_B", "loss_D_A",
             "loss_D_B", "D_A_real_mean", "D_A_fake_mean", "grad_norm_G", "grad_norm_D"]


def lsgan(pred, target_is_real: bool) -> torch.Tensor:
    """LSGAN loss; for a multi-scale discriminator (list of patch maps) the per-scale losses are averaged."""
    if isinstance(pred, (list, tuple)):
        return sum(lsgan(p, target_is_real) for p in pred) / len(pred)
    return F.mse_loss(pred, torch.ones_like(pred) if target_is_real else torch.zeros_like(pred))


def first_scale(pred):
    return pred[0] if isinstance(pred, (list, tuple)) else pred


@torch.no_grad()
def save_grid(nets: dict, fixed_a: torch.Tensor, fixed_b: torch.Tensor, path) -> None:
    """Rows: Monet | Monet->photo | reconstruction ; photo | photo->Monet | reconstruction."""
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
    # faster conv algorithm selection; same model and maths, not bitwise reproducible
    torch.backends.cudnn.benchmark = bool(t_cfg.get("cudnn_benchmark", False))

    monet_dir, photo_dir = repo_path(d_cfg["monet_dir"]), repo_path(d_cfg["photo_dir"])
    ds = UnpairedDataset(monet_dir, photo_dir,
                         train_transform(d_cfg["load_size"], d_cfg["crop_size"], d_cfg["hflip"]),
                         train_transform(d_cfg["load_size"], d_cfg["crop_size"], d_cfg["hflip"]))
    loader = DataLoader(ds, batch_size=t_cfg["batch_size"], shuffle=True, drop_last=True,
                        num_workers=d_cfg["num_workers"], pin_memory=device == "cuda",
                        persistent_workers=d_cfg["num_workers"] > 0,
                        generator=torch.Generator().manual_seed(int(cfg["seed"])))
    n_fixed = int(t_cfg["num_fixed_samples"])
    etf = eval_transform(d_cfg["crop_size"])
    fixed_a = torch.stack([FolderDataset(monet_dir, etf)[i][0] for i in range(n_fixed)]).to(device)
    fixed_b = torch.stack([FolderDataset(photo_dir, etf)[i][0] for i in range(n_fixed)]).to(device)

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
    use_diffaug_a = bool(t_cfg["diffaugment_monet_d"])
    use_diffaug_b = bool(t_cfg.get("diffaugment_photo_d", False))

    start_epoch, it = 1, 0
    history = {"intervals": [], "epochs": []}
    nan_count, max_gn = 0, {"G": 0.0, "D": 0.0}
    train_time_prev = 0.0
    latest = paths["checkpoints"] / "latest.pt"
    if args.resume and latest.exists():
        state = torch.load(latest, map_location=device)
        upgraded_d = False
        for k, net in nets.items():
            saved = state[k]
            if k.startswith("D_") and isinstance(net, MultiScaleDiscriminator) and not any(n.startswith("ds.") for n in saved):
                # upgrading a single-scale discriminator: its weights become scale 0, coarser scales start fresh
                net.ds[0].load_state_dict(saved)
                upgraded_d = True
            else:
                net.load_state_dict(saved)
        opt_g.load_state_dict(state["opt_g"])
        if upgraded_d:
            log.info("discriminators upgraded to %d scales: scale 0 loaded, other scales and D optimiser state re-initialised",
                     len(nets["D_A"].ds))
        else:
            opt_d.load_state_dict(state["opt_d"])
        start_epoch, it = state["epoch"] + 1, state["iter"]
        history, nan_count, max_gn = state["history"], state["nan_count"], state["max_grad_norm"]
        train_time_prev = state["train_time_s"]
        log.info("resumed from %s at epoch %d iter %d", latest, state["epoch"], it)

    # optional generator EMA: an exponential moving average of the generator weights, used only for
    # snapshots / inference (training itself is unchanged). Starts from the current (possibly resumed) weights.
    ema_decay = float(t_cfg.get("ema_decay", 0.0))
    ema = None
    if ema_decay > 0:
        ema = {k: {n: p.detach().clone() for n, p in nets[k].state_dict().items()} for k in ("G_A2B", "G_B2A")}
        if args.resume and latest.exists() and state.get("ema"):
            ema = {k: {n: t.to(device) for n, t in v.items()} for k, v in state["ema"].items()}
            log.info("resumed EMA weights from %s", latest)
        log.info("generator EMA enabled, decay=%s", ema_decay)

    def update_ema() -> None:
        for k in ("G_A2B", "G_B2A"):
            cur = nets[k].state_dict()
            names = [n for n, t in ema[k].items() if t.is_floating_point()]
            torch._foreach_lerp_([ema[k][n] for n in names], [cur[n].detach() for n in names], 1.0 - ema_decay)

    log.info("run_id=%s config=%s", run_id, cfg["_config_path"])
    log.info("config=%s", json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}))
    log.info("hardware=%s torch=%s", hardware_string(), torch.__version__)
    log.info("photos=%d monets=%d iters_per_epoch=%d total_iters=%d params=%s", len(ds.photos), len(ds.monets),
             iters_per_epoch, total_iters, json.dumps(params))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    run_start = time.perf_counter()
    max_gn_t = torch.tensor([max_gn["G"], max_gn["D"]], device=device)

    for epoch in range(start_epoch, epochs + 1):
        sums_t, count, epoch_start = None, 0, time.perf_counter()
        for real_a, real_b in loader:
            lr_scale = 1.0 if it < const_iters else max(0.0, 1.0 - (it - const_iters) / max(1, total_iters - const_iters))
            for g in opt_g.param_groups:
                g["lr"] = t_cfg["lr_g"] * lr_scale
            for g in opt_d.param_groups:
                g["lr"] = t_cfg["lr_d"] * lr_scale
            real_a, real_b = real_a.to(device, non_blocking=True), real_b.to(device, non_blocking=True)
            # optional DiffAugment (same random transform family for real and fake) before each discriminator
            d_a_in = diff_augment if use_diffaug_a else (lambda x: x)
            d_b_in = diff_augment if use_diffaug_b else (lambda x: x)

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
            adv_b2a = lsgan(nets["D_A"](d_a_in(fake_a)), True)
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
            if ema is not None:
                update_ema()

            # ---- discriminators ----
            for p in d_params:
                p.requires_grad_(True)
            pooled_a, pooled_b = pool_a.query(fake_a), pool_b.query(fake_b)
            pred_real_a, pred_fake_a = nets["D_A"](d_a_in(real_a)), nets["D_A"](d_a_in(pooled_a))
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

            stats = torch.stack([loss_g, adv_a2b, adv_b2a, cyc_a, cyc_b, idt_l_a, idt_l_b, loss_d_a, loss_d_b,
                                 first_scale(pred_real_a).mean(), first_scale(pred_fake_a).mean(), gn_g, gn_d]).detach().float()
            sums_t = stats if sums_t is None else sums_t + stats
            max_gn_t = torch.maximum(max_gn_t, stats[-2:])
            count += 1
            it += 1
            if it % t_cfg["log_every_iters"] == 0:
                sums = dict(zip(STAT_KEYS, sums_t.tolist()))
                row = {"iter": it, "epoch": epoch, "lr": t_cfg["lr_g"] * lr_scale,
                       **{k: v / count for k, v in sums.items()}}
                history["intervals"].append(row)
                elapsed = time.perf_counter() - epoch_start
                log.info("iter=%d epoch=%d lr=%.2e G=%.3f advA2B=%.3f advB2A=%.3f cycA=%.3f cycB=%.3f idtA=%.3f idtB=%.3f "
                         "D_A=%.3f D_B=%.3f D_A(real)=%.2f D_A(fake)=%.2f gnG=%.2f gnD=%.2f it_per_s=%.2f",
                         it, epoch, row["lr"], row["loss_G"], row["adv_A2B"], row["adv_B2A"], row["cycle_A"],
                         row["cycle_B"], row["identity_A"], row["identity_B"], row["loss_D_A"], row["loss_D_B"],
                         row["D_A_real_mean"], row["D_A_fake_mean"], row["grad_norm_G"], row["grad_norm_D"],
                         count / elapsed)

        epoch_s = time.perf_counter() - epoch_start
        sums = dict(zip(STAT_KEYS, sums_t.tolist())) if sums_t is not None else {k: float("nan") for k in STAT_KEYS}
        max_gn = dict(zip(("G", "D"), max_gn_t.tolist()))
        ep = {"epoch": epoch, "iters": count, "epoch_time_s": epoch_s, "iters_per_s": count / epoch_s,
              **{k: v / max(1, count) for k, v in sums.items()}}
        history["epochs"].append(ep)
        log.info("epoch=%d done time_s=%.0f it_per_s=%.2f G=%.3f cycA=%.3f cycB=%.3f idtA=%.3f idtB=%.3f D_A=%.3f D_B=%.3f "
                 "D_A(real)=%.2f D_A(fake)=%.2f nan_count=%d", epoch, epoch_s, ep["iters_per_s"], ep["loss_G"],
                 ep["cycle_A"], ep["cycle_B"], ep["identity_A"], ep["identity_B"], ep["loss_D_A"], ep["loss_D_B"],
                 ep["D_A_real_mean"], ep["D_A_fake_mean"], nan_count)
        if ep["loss_D_A"] < 0.05:
            log.warning("epoch=%d D_A (Monet) loss %.4f is near 0: possible discriminator overfitting", epoch, ep["loss_D_A"])
        if epoch % t_cfg["grid_every_epochs"] == 0 or epoch == epochs:
            (paths["outputs"] / "grids").mkdir(exist_ok=True)
            save_grid(nets, fixed_a, fixed_b, paths["outputs"] / "grids" / f"epoch_{epoch:03d}.png")
        train_time = train_time_prev + (time.perf_counter() - run_start)
        torch.save({**{k: n.state_dict() for k, n in nets.items()}, "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(), "epoch": epoch, "iter": it, "history": history,
                    "nan_count": nan_count, "max_grad_norm": max_gn, "train_time_s": train_time, "config": cfg,
                    "ema": ema}, latest)
        if epoch % t_cfg["checkpoint_every_epochs"] == 0 or epoch == epochs:
            snap = paths["checkpoints"] / f"epoch_{epoch:03d}"
            snap.mkdir(exist_ok=True)
            for k in ("G_A2B", "G_B2A"):
                torch.save(nets[k].state_dict(), snap / f"{k}.pt")
            if ema is not None:  # EMA generators saved as their own snapshot, scored like any other
                snap_ema = paths["checkpoints"] / f"epoch_{epoch:03d}_ema"
                snap_ema.mkdir(exist_ok=True)
                for k in ("G_A2B", "G_B2A"):
                    torch.save(ema[k], snap_ema / f"{k}.pt")
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
