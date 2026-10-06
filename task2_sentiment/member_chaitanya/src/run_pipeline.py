"""End-to-end Task 2 pipeline: splits -> preprocessing/EDA -> slices -> 9 training runs -> TF-IDF reference ->
evaluation -> error-review candidates. Each step is a separate process; stops on the first failure."""
import subprocess
import sys

from common import LOG_DIR, MEMBER_DIR

STEPS = ["make_splits.py", "preprocess.py", "slices.py", "run_all_training.py", "tfidf_reference.py",
         "evaluate.py", "errors.py"]

LOG_DIR.mkdir(parents=True, exist_ok=True)
for step in STEPS:
    print(f"== {step}", flush=True)
    with open(LOG_DIR / f"pipeline_{step.replace('.py', '')}.log", "a", encoding="utf-8") as f:
        subprocess.run([sys.executable, str(MEMBER_DIR / "src" / step)], check=True, stdout=f, stderr=subprocess.STDOUT)
print("pipeline done", flush=True)
