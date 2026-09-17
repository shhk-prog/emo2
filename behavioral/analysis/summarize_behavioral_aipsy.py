#!/usr/bin/env python3
"""
behavioral/analysis/summarize_behavioral_aipsy.py

Comprehensive Summary and Analysis for AIPsy-Affect 4-Split Behavioral Evaluation.
Focuses on the 4 Behavioral Dimensions:
  1. Sensitivity: Clinical vs. Neutral paired response (Cohen's d_z, paired t-test)
  2. Dose-Response: Monotonic trend across Neutral -> Moderate -> Clinical
  3. Specificity: Complex Neutral control vs. Clinical
  4. Reader-Self Coupling: Correlation between Delta(Reader) and Delta(Self)
"""

import argparse
import glob
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_rel

from affective_empathy_eval.statistics import (
    apply_benjamini_hochberg,
    compute_bootstrap_ci,
    compute_d_z,
)

EMOTIONS = [
    ("grief", "V-"),
    ("terror", "V- A+"),
    ("rage", "V- A+"),
    ("loathing", "V-"),
    ("ecstasy", "V+ A+"),
    ("admiration", "V+"),
    ("amazement", "A+"),
    ("vigilance", "A+"),
]


def safe_corr(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_c, y_c = x[mask], y[mask]
    if len(x_c) < 3 or np.std(x_c) == 0 or np.std(y_c) == 0:
        return np.nan, np.nan, np.nan, np.nan
    r, p = pearsonr(x_c, y_c)
    rho, rho_p = spearmanr(x_c, y_c)
    return float(r), float(p), float(rho), float(rho_p)


def analyze_model_aipsy(csv_path):
    df = pd.read_csv(csv_path)
    model_name = Path(csv_path).stem.replace("_aipsy_4split", "")

    # Group by base_id or split
    # Splits: neutral, moderate, clinical, complex_neutral
    splits = df["split"].unique()

    # RQ1: Sensitivity (Clinical vs Neutral)
    rq1_rows = []
    if "clinical" in splits and "neutral" in splits:
        clin_df = df[df["split"] == "clinical"].sort_values("id")
        neut_df = df[df["split"] == "neutral"].sort_values("id")

        # Match pairs if possible by base text or emotion
        common_len = min(len(clin_df), len(neut_df))
        if common_len > 0:
            for dim in ["v", "a", "d"]:
                for task in ["w", "r", "s"]:
                    col = f"{task}_e{dim}"
                    if col in clin_df.columns and col in neut_df.columns:
                        c_vals = clin_df[col].values[:common_len]
                        n_vals = neut_df[col].values[:common_len]
                        diff = c_vals - n_vals
                        dz = compute_d_z(diff)
                        t_stat, p_val = ttest_rel(c_vals, n_vals)
                        rq1_rows.append(
                            {
                                "model": model_name,
                                "task": task,
                                "dimension": dim.upper(),
                                "mean_diff": float(np.mean(diff)),
                                "d_z": float(dz),
                                "t_stat": float(t_stat),
                                "p_value": float(p_val),
                            }
                        )

    # RQ4: Reader-Self Coupling (Task 2 vs Task 3)
    rq4_rows = []
    for dim in ["v", "a", "d"]:
        r_col = f"r_e{dim}"
        s_col = f"s_e{dim}"
        if r_col in df.columns and s_col in df.columns:
            r, p, rho, rho_p = safe_corr(df[r_col].values, df[s_col].values)
            ci_low, ci_high = compute_bootstrap_ci(
                df[r_col].values, df[s_col].values, stat_fn="pearson"
            )
            rq4_rows.append(
                {
                    "model": model_name,
                    "dimension": dim.upper(),
                    "r_RS": r,
                    "p_RS": p,
                    "rho_RS": rho,
                    "ci_95_low": ci_low,
                    "ci_95_high": ci_high,
                }
            )

    return pd.DataFrame(rq1_rows), pd.DataFrame(rq4_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize AIPsy-Affect 4-Split Behavioral Results"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="behavioral/results/aipsy_4split",
        help="Directory containing *_aipsy_4split.csv files",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="behavioral/results/aipsy_4split_summary",
        help="Output directory for reports",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "*_aipsy_4split.csv")))
    if not files:
        v1_fallback = "v1/results/aipsy_4split_eval"
        if os.path.exists(v1_fallback):
            files = sorted(
                glob.glob(os.path.join(v1_fallback, "*_aipsy_4split.csv"))
            )

    if not files:
        print(f"No result CSVs found in {args.input_dir} or fallback.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    all_rq1 = []
    all_rq4 = []

    for f in files:
        df_rq1, df_rq4 = analyze_model_aipsy(f)
        all_rq1.append(df_rq1)
        all_rq4.append(df_rq4)

    res_rq1 = pd.concat(all_rq1, ignore_index=True) if all_rq1 else pd.DataFrame()
    res_rq4 = pd.concat(all_rq4, ignore_index=True) if all_rq4 else pd.DataFrame()

    if not res_rq1.empty:
        # Apply Benjamini-Hochberg FDR
        res_rq1["p_fdr"] = apply_benjamini_hochberg(res_rq1["p_value"].values)
        rq1_path = os.path.join(args.out_dir, "behavioral_aipsy_sensitivity_rq1.csv")
        res_rq1.to_csv(rq1_path, index=False)
        print(f"Saved RQ1 Sensitivity to {rq1_path}")

    if not res_rq4.empty:
        rq4_path = os.path.join(args.out_dir, "behavioral_aipsy_coupling_rq4.csv")
        res_rq4.to_csv(rq4_path, index=False)
        print(f"Saved RQ4 Coupling to {rq4_path}")


if __name__ == "__main__":
    main()
