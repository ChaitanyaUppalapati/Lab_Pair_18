"""Train the three models for every seed in their configs, sequentially (one GPU). Skips finished runs."""
import subprocess
import sys

from common import MEMBER_DIR, OUTPUTS_DIR, load_config

CONFIGS = ["baseline", "experimental_1", "experimental_2"]

for name in CONFIGS:
    rel = f"task2_sentiment/member_chaitanya/configs/{name}.yaml"
    cfg = load_config(rel)
    for seed in cfg["seeds"]:
        if (OUTPUTS_DIR / "runs" / f"{cfg['run_prefix']}_s{seed}" / "train_summary.json").exists():
            print(f"skip {name} seed {seed} (done)", flush=True)
            continue
        print(f"train {name} seed {seed}", flush=True)
        subprocess.run([sys.executable, str(MEMBER_DIR / "src" / "train.py"), "--config", rel, "--seed", str(seed)],
                       check=True, stdout=subprocess.DEVNULL)
print("all training done", flush=True)
