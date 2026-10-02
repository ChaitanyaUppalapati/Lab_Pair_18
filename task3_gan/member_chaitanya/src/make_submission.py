"""Write submission.csv for the Kaggle class competition from generated photo->Monet images.

Format (instructor announcement): columns ID, FID, MiFID; one result row; numeric ID.
FID / MiFID come from kaggle_score.py (MiFID = index-paired reading, which matches the
magnitude of the instructor's example row).

Usage:
    python task3_gan/member_chaitanya/src/make_submission.py --images task3_gan/member_chaitanya/outputs/pred_B2A
"""
import argparse
import csv
import json

from common import MEMBER_DIR, REPO_ROOT
from kaggle_score import score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="repo-relative folder of generated photo->Monet JPGs")
    parser.add_argument("--id", type=int, default=1)
    args = parser.parse_args()
    result = score(REPO_ROOT / args.images)
    out = MEMBER_DIR / "submission.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "FID", "MiFID"])
        w.writerow([args.id, f"{result['fid']:.4f}", f"{result['mifid_paired']:.4f}"])
    (MEMBER_DIR / "outputs" / "kaggle_score.json").write_text(json.dumps(result, indent=2))
    print(out.read_text())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
