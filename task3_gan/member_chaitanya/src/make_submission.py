"""Write submission.csv for the Kaggle class competition using the provided evaluation script.

Format: columns ID, FID, MiFID; one result row; numeric ID. Values come from official_eval.py
(the instructor's Part3_Evaluation_Script: both directions, first 300 images per folder, averaged).

Usage:
    python task3_gan/member_chaitanya/src/make_submission.py \
        --pred-a2b task3_gan/member_chaitanya/outputs/pred_A2B --pred-b2a task3_gan/member_chaitanya/outputs/pred_B2A
"""
import argparse
import json

import pandas as pd

from common import MEMBER_DIR
from official_eval import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-a2b", default="task3_gan/member_chaitanya/outputs/pred_A2B")
    parser.add_argument("--pred-b2a", default="task3_gan/member_chaitanya/outputs/pred_B2A")
    parser.add_argument("--id", type=int, default=1)
    args = parser.parse_args()
    r = evaluate(args.pred_a2b, args.pred_b2a)
    pd.DataFrame([{"ID": args.id, "FID": float(r["FID"]), "MiFID": float(r["MiFID"])}]).to_csv(
        MEMBER_DIR / "submission.csv", index=False)
    (MEMBER_DIR / "outputs" / "official_eval.json").write_text(json.dumps(r, indent=2))
    print((MEMBER_DIR / "submission.csv").read_text())
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
