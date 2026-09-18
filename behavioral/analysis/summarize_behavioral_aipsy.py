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
    splits = set(df["split"].dropna().unique())

    # 1. RQ1: Sensitivity (Clinical vs Neutral matched by pair_id)
    rq1_rows = []
    if "clinical" in splits and "neutral" in splits:
        clin_df = df[df["split"] == "clinical"]
        neut_df = df[df["split"] == "neutral"]
        
        # Merge on pair_id if available, fallback to index
        if "pair_id" in df.columns and clin_df["pair_id"].notna().any() and neut_df["pair_id"].notna().any():
            merged_cn = pd.merge(clin_df, neut_df, on="pair_id", suffixes=("_clin", "_neut"))
        else:
            common_len = min(len(clin_df), len(neut_df))
            merged_cn = pd.concat([
                clin_df.iloc[:common_len].reset_index(drop=True).add_suffix("_clin"),
                neut_df.iloc[:common_len].reset_index(drop=True).add_suffix("_neut")
            ], axis=1)

        if len(merged_cn) > 0:
            for dim in ["v", "a", "d"]:
                for task in ["w", "r", "s"]:
                    c_col = f"{task}_e{dim}_clin"
                    n_col = f"{task}_e{dim}_neut"
                    if c_col in merged_cn.columns and n_col in merged_cn.columns:
                        c_vals = merged_cn[c_col].values
                        n_vals = merged_cn[n_col].values
                        diff = c_vals - n_vals
                        dz = compute_d_z(diff)
                        t_stat, p_val = ttest_rel(c_vals, n_vals)
                        rq1_rows.append(
                            {
                                "model": model_name,
                                "task": task,
                                "dimension": dim.upper(),
                                "n_pairs": len(diff),
                                "mean_diff": float(np.mean(diff)),
                                "d_z": float(dz),
                                "t_stat": float(t_stat),
                                "p_value": float(p_val),
                            }
                        )

    # 2. RQ2: Dose-Response (Neutral -> Moderate -> Clinical)
    rq2_rows = []
    if {"neutral", "moderate", "clinical"}.issubset(splits):
        # Match triplets by triplet_id if available, otherwise domain/emotion
        mod_df = df[df["split"] == "moderate"]
        clin_df = df[df["split"] == "clinical"]
        neut_df = df[df["split"] == "neutral"]

        trip_merged = None
        if "triplet_id" in df.columns and df["triplet_id"].notna().any():
            m_cn = pd.merge(neut_df, mod_df, on="triplet_id", suffixes=("_neut", "_mod"))
            trip_merged = pd.merge(m_cn, clin_df, on="triplet_id")
            trip_merged = trip_merged.rename(columns={c: f"{c}_clin" for c in clin_df.columns if c != "triplet_id"})

        if trip_merged is not None and len(trip_merged) > 0:
            for dim in ["v", "a", "d"]:
                for task in ["w", "r", "s"]:
                    n_col = f"{task}_e{dim}_neut"
                    m_col = f"{task}_e{dim}_mod"
                    c_col = f"{task}_e{dim}_clin"
                    if n_col in trip_merged.columns and m_col in trip_merged.columns and c_col in trip_merged.columns:
                        n_arr = trip_merged[n_col].values
                        m_arr = trip_merged[m_col].values
                        c_arr = trip_merged[c_col].values
                        # Monotonic dose trend: Spearman rank corr with dose level [0, 1, 2]
                        all_doses = np.repeat([0, 1, 2], len(n_arr))
                        all_vals = np.concatenate([n_arr, m_arr, c_arr])
                        rho, rho_p = spearmanr(all_doses, all_vals)
                        rq2_rows.append(
                            {
                                "model": model_name,
                                "task": task,
                                "dimension": dim.upper(),
                                "n_triplets": len(trip_merged),
                                "mean_neutral": float(np.mean(n_arr)),
                                "mean_moderate": float(np.mean(m_arr)),
                                "mean_clinical": float(np.mean(c_arr)),
                                "dose_spearman_rho": float(rho),
                                "dose_p_value": float(rho_p),
                            }
                        )

    # 3. RQ3: Specificity (Clinical vs Complex Neutral)
    rq3_rows = []
    if "clinical" in splits and "complex_neutral" in splits:
        clin_df = df[df["split"] == "clinical"]
        cneu_df = df[df["split"] == "complex_neutral"]
        for dim in ["v", "a", "d"]:
            for task in ["w", "r", "s"]:
                c_col = f"{task}_e{dim}"
                if c_col in clin_df.columns and c_col in cneu_df.columns:
                    c_vals = clin_df[c_col].dropna().values
                    cn_vals = cneu_df[c_col].dropna().values
                    if len(c_vals) > 2 and len(cn_vals) > 2:
                        from scipy.stats import ttest_ind
                        t_stat, p_val = ttest_ind(c_vals, cn_vals, equal_var=False)
                        pooled_std = np.sqrt((np.var(c_vals, ddof=1) + np.var(cn_vals, ddof=1)) / 2.0)
                        d_val = float((np.mean(c_vals) - np.mean(cn_vals)) / (pooled_std + 1e-12))
                        rq3_rows.append(
                            {
                                "model": model_name,
                                "task": task,
                                "dimension": dim.upper(),
                                "n_clinical": len(c_vals),
                                "n_complex_neutral": len(cn_vals),
                                "mean_diff": float(np.mean(c_vals) - np.mean(cn_vals)),
                                "cohen_d": d_val,
                                "t_stat": float(t_stat),
                                "p_value": float(p_val),
                            }
                        )

    # 4. RQ4: Reader-Self Coupling (Task 2 vs Task 3)
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

    return pd.DataFrame(rq1_rows), pd.DataFrame(rq2_rows), pd.DataFrame(rq3_rows), pd.DataFrame(rq4_rows)


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
    all_rq2 = []
    all_rq3 = []
    all_rq4 = []

    for f in files:
        df_rq1, df_rq2, df_rq3, df_rq4 = analyze_model_aipsy(f)
        all_rq1.append(df_rq1)
        all_rq2.append(df_rq2)
        all_rq3.append(df_rq3)
        all_rq4.append(df_rq4)

    res_rq1 = pd.concat(all_rq1, ignore_index=True) if all_rq1 else pd.DataFrame()
    res_rq2 = pd.concat(all_rq2, ignore_index=True) if all_rq2 else pd.DataFrame()
    res_rq3 = pd.concat(all_rq3, ignore_index=True) if all_rq3 else pd.DataFrame()
    res_rq4 = pd.concat(all_rq4, ignore_index=True) if all_rq4 else pd.DataFrame()

    if not res_rq1.empty:
        # Apply Benjamini-Hochberg FDR
        res_rq1["p_fdr"] = apply_benjamini_hochberg(res_rq1["p_value"].values)
        rq1_path = os.path.join(args.out_dir, "behavioral_aipsy_sensitivity_rq1.csv")
        res_rq1.to_csv(rq1_path, index=False)
        print(f"Saved RQ1 Sensitivity to {rq1_path}")

    if not res_rq2.empty:
        res_rq2["p_fdr"] = apply_benjamini_hochberg(res_rq2["dose_p_value"].values)
        rq2_path = os.path.join(args.out_dir, "behavioral_aipsy_dose_response_rq2.csv")
        res_rq2.to_csv(rq2_path, index=False)
        print(f"Saved RQ2 Dose-Response to {rq2_path}")

    if not res_rq3.empty:
        res_rq3["p_fdr"] = apply_benjamini_hochberg(res_rq3["p_value"].values)
        rq3_path = os.path.join(args.out_dir, "behavioral_aipsy_specificity_rq3.csv")
        res_rq3.to_csv(rq3_path, index=False)
        print(f"Saved RQ3 Specificity to {rq3_path}")

    if not res_rq4.empty:
        rq4_path = os.path.join(args.out_dir, "behavioral_aipsy_coupling_rq4.csv")
        res_rq4.to_csv(rq4_path, index=False)
        print(f"Saved RQ4 Coupling to {rq4_path}")


if __name__ == "__main__":
    main()
