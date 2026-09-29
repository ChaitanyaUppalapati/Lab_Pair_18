"""Paired McNemar test execution between Baseline and Experimental models.
Reads the exact paired test prediction files, builds 2x2 contingency tables,
runs continuity-corrected chi-square McNemar tests, and updates metrics_report.csv.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np

from utils import run_mcnemar_test


def parse_args():
    parser = argparse.ArgumentParser(description="Run McNemar Tests for Task 2")
    parser.add_argument("--preds-dir", type=str, default="task2_sentiment/member_aswin/outputs/predictions")
    parser.add_argument("--metrics-report", type=str, default="task2_sentiment/member_aswin/metrics_report.csv")
    parser.add_argument("--output-json", type=str, default="task2_sentiment/member_aswin/outputs/statistical_tests/mcnemar_tests.json")
    return parser.parse_args()


def main():
    args = parse_args()
    preds_dir = Path(args.preds_dir)
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # File paths for models
    base_file = preds_dir / "baseline_test_predictions.csv"
    if not base_file.exists():
        # Fallback to smoke baseline if running smoke test
        base_file = preds_dir / "smoke_baseline_test_predictions.csv"
        if not base_file.exists():
            raise FileNotFoundError(f"Baseline test predictions not found in {preds_dir}")

    exp1_file = preds_dir / "experimental_1_test_predictions.csv"
    exp2_file = preds_dir / "experimental_2_test_predictions.csv"

    base_df = pd.read_csv(base_file)
    y_true = base_df["true_label"].values
    y_pred_base = base_df["predicted_label"].values

    test_results = {}

    # Model comparisons
    comparisons = [
        ("experimental_1", exp1_file),
        ("experimental_2", exp2_file),
    ]

    metrics_df = pd.read_csv(args.metrics_report) if Path(args.metrics_report).exists() else None

    for model_name, pred_file in comparisons:
        if not pred_file.exists():
            print(f"Skipping {model_name}: {pred_file} not found.")
            continue

        model_df = pd.read_csv(pred_file)
        if len(model_df) != len(base_df):
            raise ValueError(f"Length mismatch: {model_name} has {len(model_df)} predictions, baseline has {len(base_df)}.")

        y_pred_model = model_df["predicted_label"].values
        stat, p_val, contingency_table, is_sig = run_mcnemar_test(y_true, y_pred_base, y_pred_model)

        interpretation = (
            f"The difference in classification accuracy between Baseline and {model_name} is "
            f"{'statistically significant (p < 0.05)' if is_sig else 'NOT statistically significant (p >= 0.05)'} "
            f"under the McNemar test with continuity correction (chi2={stat:.4f}, p={p_val:.4e})."
        )

        test_results[model_name] = {
            "comparison": f"baseline_vs_{model_name}",
            "sample_size": len(y_true),
            "contingency_table": {
                "both_correct_n00": contingency_table[0][0],
                "base_correct_model_wrong_n01": contingency_table[0][1],
                "base_wrong_model_correct_n10": contingency_table[1][0],
                "both_wrong_n11": contingency_table[1][1],
            },
            "mcnemar_statistic": stat,
            "p_value": p_val,
            "is_significant_at_alpha_0_05": is_sig,
            "interpretation": interpretation,
        }

        print("=" * 60)
        print(f"McNemar Test: Baseline vs {model_name}")
        print(f"Contingency Table: {contingency_table}")
        print(f"Statistic: {stat:.4f}, p-value: {p_val:.4e}, Significant: {is_sig}")
        print(interpretation)

        # Update metrics_report.csv
        if metrics_df is not None:
            mask = metrics_df["model"] == model_name
            if mask.any():
                metrics_df.loc[mask, "mcnemar_vs_baseline_stat"] = f"{stat:.4f}"
                metrics_df.loc[mask, "mcnemar_vs_baseline_p"] = f"{p_val:.4e}"

    # Save results
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)
    print(f"Saved McNemar test results to {out_json}")

    if metrics_df is not None:
        metrics_df.to_csv(args.metrics_report, index=False)
        print(f"Updated McNemar columns in {args.metrics_report}")


if __name__ == "__main__":
    main()
