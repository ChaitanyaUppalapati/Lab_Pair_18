"""Blinded human audit of 30 fixed samples (style, content, artifacts), 2 raters, inter-rater agreement.

build:  picks 30 FIXED inputs (seeded, from the sorted file lists: 20 photo -> Monet, the Kaggle direction, and
        10 Monet -> photo), translates them with the submitted generators and writes, in outputs/human_audit/:
          sheet.html        what the raters open: input + translation per item, items shuffled, anonymous IDs
          images/           S01_in.jpg / S01_out.jpg ... (no file names, run names or scores)
          rater_template.csv  copy to rater_1.csv and rater_2.csv; each rater fills it in alone
          key.csv           item -> source image, direction, model.  DO NOT OPEN BEFORE RATING.
          inputs.txt        the 30 input file names, so a teammate's model can be audited on the same inputs:
                            pass --other NAME=DIR (DIR holds that model's translations named like the inputs)
                            and its items are mixed into the same blinded sheet.
score:  reads rater_1.csv and rater_2.csv, writes audit_results.json and the human_* columns of
        full_metrics_report.csv (rows of the submitted model).

Scale (1-5, integers): style = how convincingly the output belongs to the target domain (Monet painting /
real photo); content = how well the input's scene and layout are preserved; artifacts = 5 means no visible
artifacts, 1 means severe (blotches, checkerboard, colour blow-outs).
Agreement: Cohen's kappa (unweighted and quadratic-weighted, the latter suits an ordinal scale), % exact
agreement and % within 1 point, per criterion.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MEMBER_DIR, load_config, repo_path, run_paths  # noqa: E402
from data import eval_transform, list_images, load_rgb  # noqa: E402
from models import make_generator  # noqa: E402

OUT = MEMBER_DIR / "outputs" / "human_audit"
CRITERIA = ("style", "content", "artifacts")
SEED = 266
N_B2A, N_A2B = 20, 10


def build(args) -> None:
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = run_paths(cfg["run_id"])["checkpoints"] / args.snapshot
    tf = eval_transform(cfg["data"]["crop_size"])
    rng = random.Random(SEED)
    photos = sorted(list_images(repo_path(cfg["data"]["photo_dir"])))
    monets = sorted(list_images(repo_path(cfg["data"]["monet_dir"])))
    chosen = [("B2A", p) for p in rng.sample(photos, N_B2A)] + [("A2B", m) for m in rng.sample(monets, N_A2B)]

    gens = {}
    for k in ("G_A2B", "G_B2A"):
        g = make_generator(cfg["model"]).to(device).eval()
        g.load_state_dict(torch.load(ckpt / f"{k}.pt", map_location=device))
        gens[k] = g
    models = {"chaitanya_submitted": None}
    for spec in args.other or []:
        name, folder = spec.split("=", 1)
        models[name] = Path(folder)

    items = []
    for direction, path in chosen:
        for model, folder in models.items():
            if folder is None:
                with torch.no_grad():
                    y = gens[f"G_{direction}"](tf(load_rgb(path)).unsqueeze(0).to(device))[0]
                out = Image.fromarray(((y.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy())
            else:
                out = load_rgb(next(folder.glob(path.stem + ".*")))
            items.append({"direction": direction, "source": path.name, "model": model, "out": out, "in": path})
    rng.shuffle(items)

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    size = cfg["data"]["crop_size"]
    rows, html = [], []
    for i, it in enumerate(items, 1):
        sid = f"S{i:02d}"
        load_rgb(it["in"]).resize((size, size), Image.BICUBIC).save(OUT / "images" / f"{sid}_in.jpg", quality=95)
        it["out"].resize((size, size), Image.BICUBIC).save(OUT / "images" / f"{sid}_out.jpg", quality=95)
        target = "Monet painting" if it["direction"] == "B2A" else "real photo"
        rows.append({"item": sid, "direction": it["direction"], "source_image": it["source"], "model": it["model"]})
        html.append(f"<div class='it'><h3>{sid} &mdash; should look like a <b>{target}</b></h3>"
                    f"<img src='images/{sid}_in.jpg'><img src='images/{sid}_out.jpg'>"
                    f"<p>left: input &nbsp; right: output</p></div>")
    with open(OUT / "key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "rater_template.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", *CRITERIA, "notes"])
        w.writerows([[r["item"], "", "", "", ""] for r in rows])
    (OUT / "inputs.txt").write_text("\n".join(f"{d}\t{p.name}" for d, p in chosen) + "\n", encoding="utf-8")
    (OUT / "sheet.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Task 3 human audit</title><style>"
        "body{font-family:sans-serif;max-width:900px;margin:auto;padding:16px}"
        ".it{border-top:1px solid #ccc;padding:8px 0}img{width:49%;margin-right:1%}</style>"
        "<h1>Task 3 — blinded human audit</h1>"
        "<p>Rate every item on its own, without discussing with the other rater, in your copy of "
        "<code>rater_template.csv</code> (save as rater_1.csv / rater_2.csv). Integers 1&ndash;5:</p><ul>"
        "<li><b>style</b>: does the output convincingly look like the target domain? 5 = indistinguishable, 1 = not at all</li>"
        "<li><b>content</b>: is the input's scene/layout preserved? 5 = fully, 1 = unrecognisable</li>"
        "<li><b>artifacts</b>: 5 = none visible, 1 = severe (blotches, checkerboard, colour blow-outs, smears)</li></ul>"
        "<p>Do not open key.csv until both raters are done.</p>" + "".join(html), encoding="utf-8")
    print(f"{len(items)} items ({', '.join(models)}) -> {OUT}")


def kappa(a, b, weights=None):
    from sklearn.metrics import cohen_kappa_score

    if len(set(a) | set(b)) < 2:
        return float("nan")
    return float(cohen_kappa_score(a, b, labels=[1, 2, 3, 4, 5], weights=weights))


def score(args) -> None:
    key = {r["item"]: r for r in csv.DictReader(open(OUT / "key.csv", encoding="utf-8"))}
    raters = []
    for name in ("rater_1.csv", "rater_2.csv"):
        rows = {r["item"]: r for r in csv.DictReader(open(OUT / name, encoding="utf-8"))}
        for item, r in rows.items():
            for c in CRITERIA:
                v = int(str(r[c]).strip())
                assert 1 <= v <= 5, f"{name} {item} {c}={v}"
        raters.append(rows)
    assert set(raters[0]) == set(raters[1]) == set(key), "both raters must rate every item"
    results = {}
    for model in sorted({k["model"] for k in key.values()}):
        items = sorted(i for i, k in key.items() if k["model"] == model)
        res = {"n_items": len(items)}
        for c in CRITERIA:
            a = [int(raters[0][i][c]) for i in items]
            b = [int(raters[1][i][c]) for i in items]
            res[c] = {"mean": float(np.mean(a + b)), "rater_1_mean": float(np.mean(a)), "rater_2_mean": float(np.mean(b)),
                      "kappa": kappa(a, b), "kappa_quadratic": kappa(a, b, "quadratic"),
                      "pct_exact_agreement": float(np.mean(np.array(a) == np.array(b)) * 100),
                      "pct_within_1": float(np.mean(np.abs(np.array(a) - np.array(b)) <= 1) * 100)}
            for d in ("B2A", "A2B"):
                sub = [i for i in items if key[i]["direction"] == d]
                if sub:
                    res[c][f"mean_{d}"] = float(np.mean([int(raters[r][i][c]) for r in (0, 1) for i in sub]))
        results[model] = res
    (OUT / "audit_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    mine = results["chaitanya_submitted"]
    report = MEMBER_DIR / "full_metrics_report.csv"
    rows = list(csv.DictReader(open(report, encoding="utf-8")))
    fields = list(rows[0])
    for r in rows:
        if r["run_id"] == args.report_run:
            d = "B2A" if r["direction"].startswith("B2A") else "A2B"
            r["human_style"] = f"{mine['style'][f'mean_{d}']:.2f}"
            r["human_content"] = f"{mine['content'][f'mean_{d}']:.2f}"
            r["human_artifacts"] = f"{mine['artifacts'][f'mean_{d}']:.2f}"
            r["inter_rater_kappa"] = "; ".join(f"{c} {mine[c]['kappa_quadratic']:.2f}" for c in CRITERIA) + " (quadratic-weighted, all 30 items)"
            r["pct_agreement"] = "; ".join(f"{c} {mine[c]['pct_exact_agreement']:.0f}%" for c in CRITERIA) + " (exact)"
    with open(report, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(results, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--config", default="task3_gan/member_chaitanya/configs/full_run18.yaml")
    b.add_argument("--snapshot", default="epoch_123_ema")
    b.add_argument("--other", action="append", help="NAME=DIR with that model's translations of inputs.txt")
    s = sub.add_parser("score")
    s.add_argument("--report-run", default="full_run18", help="rows of full_metrics_report.csv to fill")
    args = ap.parse_args()
    build(args) if args.cmd == "build" else score(args)


if __name__ == "__main__":
    main()
