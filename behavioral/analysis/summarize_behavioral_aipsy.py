#!/usr/bin/env python3
"""
behavioral/analysis/summarize_behavioral_aipsy.py

Comprehensive Summary and Analysis for AIPsy-Affect 4-Split Behavioral Evaluation.
Focuses on the 4 Behavioral Dimensions:
  1. Sensitivity: Clinical vs. Neutral paired response (paired mean diff, Cohen's d_z, 95% Bootstrap CI, paired t-test)
  2. Dose-Response: Repeated-measures within-triplet linear slope across Neutral -> Moderate -> Clinical
  3. Specificity: Complex Neutral control vs. Clinical
  4. Reader-Self Coupling: Primary metric is correlation of matched pair changes:
       corr(Delta_reader, Delta_self) with Delta = affective - neutral.
       Raw correlation is preserved as secondary/supplementary analysis.
"""

import argparse
import glob
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_rel, ttest_1samp

from affective_empathy_eval.statistics import (
    apply_benjamini_hochberg,
    compute_bootstrap_ci,
    compute_correlation_bootstrap_ci,
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


def resolve_alignment(model_name: str) -> str:
    """model_name から instruct または base を判定"""
    lower = model_name.lower()
    if any(k in lower for k in ("instruct", "chat", "it")):
        return "instruct"
    return "base"


def safe_corr(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_c, y_c = x[mask], y[mask]
    if len(x_c) < 3 or np.std(x_c) == 0 or np.std(y_c) == 0:
        return np.nan, np.nan, np.nan, np.nan
    r, p = pearsonr(x_c, y_c)
    rho, rho_p = spearmanr(x_c, y_c)
    return float(r), float(p), float(rho), float(rho_p)


def analyze_model_aipsy(csv_path: str):
    df = pd.read_csv(csv_path)
    model_name = Path(csv_path).stem.replace("_aipsy_4split", "")
    alignment = resolve_alignment(model_name)
    splits = set(df["split"].dropna().unique())

    # 1. RQ1: Sensitivity (Clinical vs Neutral matched by pair_id)
    rq1_rows = []
    merged_cn = None
    if "clinical" in splits and "neutral" in splits:
        clin_df = df[df["split"] == "clinical"]
        neut_df = df[df["split"] == "neutral"]

        if "pair_id" in df.columns and clin_df["pair_id"].notna().any() and neut_df["pair_id"].notna().any():
            merged_cn = pd.merge(clin_df, neut_df, on="pair_id", suffixes=("_clin", "_neut"))
        else:
            common_len = min(len(clin_df), len(neut_df))
            merged_cn = pd.concat([
                clin_df.iloc[:common_len].reset_index(drop=True).add_suffix("_clin"),
                neut_df.iloc[:common_len].reset_index(drop=True).add_suffix("_neut")
            ], axis=1)

        if merged_cn is not None and len(merged_cn) > 0:
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

                        # Bootstrap 95% CI for paired mean diff and Cohen's dz
                        _, diff_ci_low, diff_ci_high = compute_bootstrap_ci(diff, statistic_fn=np.mean)
                        _, dz_ci_low, dz_ci_high = compute_bootstrap_ci(diff, statistic_fn=compute_d_z)

                        rq1_rows.append(
                            {
                                "model": model_name,
                                "alignment": alignment,
                                "task": task,
                                "dimension": dim.upper(),
                                "n_pairs": len(diff),
                                "mean_diff": float(np.mean(diff)),
                                "mean_diff_ci_low": float(diff_ci_low),
                                "mean_diff_ci_high": float(diff_ci_high),
                                "d_z": float(dz),
                                "dz_ci_low": float(dz_ci_low),
                                "dz_ci_high": float(dz_ci_high),
                                "t_stat": float(t_stat),
                                "p_value": float(p_val),
                            }
                        )

    # 2. RQ2: Dose-Response (Repeated-measures within-triplet linear slope)
    rq2_rows = []
    if {"neutral", "moderate", "clinical"}.issubset(splits):
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

                        # Primary: triplet-level linear contrast slope b_i = (c_i - n_i) / 2.0
                        # Dose levels are [0, 1, 2], centered x = [-1, 0, 1], denominator sum(x^2)=2
                        slopes = (c_arr - n_arr) / 2.0
                        mean_slope = float(np.mean(slopes))
                        t_stat, p_val = ttest_1samp(slopes, popmean=0.0)

                        # Triplet-level bootstrap CI for slope
                        _, slope_low, slope_high = compute_bootstrap_ci(slopes, statistic_fn=np.mean)

                        # Secondary: across-triplet concatenated rank correlation (for legacy reference)
                        all_doses = np.repeat([0, 1, 2], len(n_arr))
                        all_vals = np.concatenate([n_arr, m_arr, c_arr])
                        sec_rho, sec_p = spearmanr(all_doses, all_vals)

                        rq2_rows.append(
                            {
                                "model": model_name,
                                "alignment": alignment,
                                "task": task,
                                "dimension": dim.upper(),
                                "n_triplets": len(trip_merged),
                                "mean_slope": mean_slope,
                                "slope_ci_low": float(slope_low),
                                "slope_ci_high": float(slope_high),
                                "t_stat": float(t_stat),
                                "p_value": float(p_val),
                                "mean_neutral": float(np.mean(n_arr)),
                                "mean_moderate": float(np.mean(m_arr)),
                                "mean_clinical": float(np.mean(c_arr)),
                                "secondary_spearman_rho": float(sec_rho),
                                "secondary_spearman_p": float(sec_p),
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
                                "alignment": alignment,
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

    # 4. RQ4: Reader-Self Coupling (Task r vs Task s)
    # Primary: Matched pair change coupling: corr(Delta_reader, Delta_self)
    # Secondary: Raw correlation corr(reader_raw, self_raw)
    rq4_rows = []
    for dim in ["v", "a", "d"]:
        dim_u = dim.upper()
        # a. Primary: Delta coupling from matched pairs (Clinical - Neutral)
        if merged_cn is not None and len(merged_cn) > 0:
            r_clin = merged_cn[f"r_e{dim}_clin"].values
            r_neut = merged_cn[f"r_e{dim}_neut"].values
            s_clin = merged_cn[f"s_e{dim}_clin"].values
            s_neut = merged_cn[f"s_e{dim}_neut"].values

            delta_reader = r_clin - r_neut
            delta_self = s_clin - s_neut

            r, p, rho, rho_p = safe_corr(delta_reader, delta_self)
            _, ci_low, ci_high = compute_correlation_bootstrap_ci(delta_reader, delta_self, method="pearson")

            rq4_rows.append(
                {
                    "model": model_name,
                    "alignment": alignment,
                    "dimension": dim_u,
                    "n_pairs": len(delta_reader),
                    "correlation_type": "delta_coupling",  # PRIMARY METRIC
                    "r": r,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": p,
                    "spearman_rho": rho,
                    "spearman_p": rho_p,
                }
            )

        # b. Secondary: Raw cross-stimulus correlation
        r_col = f"r_e{dim}"
        s_col = f"s_e{dim}"
        if r_col in df.columns and s_col in df.columns:
            r_raw, p_raw, rho_raw, rho_p_raw = safe_corr(df[r_col].values, df[s_col].values)
            _, raw_low, raw_high = compute_correlation_bootstrap_ci(df[r_col].values, df[s_col].values, method="pearson")

            rq4_rows.append(
                {
                    "model": model_name,
                    "alignment": alignment,
                    "dimension": dim_u,
                    "n_pairs": len(df),
                    "correlation_type": "raw_coupling",  # SECONDARY METRIC
                    "r": r_raw,
                    "ci_low": raw_low,
                    "ci_high": raw_high,
                    "p_value": p_raw,
                    "spearman_rho": rho_raw,
                    "spearman_p": rho_p_raw,
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
        default="behavioral/results/derived/aipsy_4split_summary",
        help="Output directory for reports",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "*_aipsy_4split.csv")))
    if not files:
        # Fallback paths
        fallbacks = [
            "behavioral/results/raw/aipsy_4split",
            "results/raw/behavioral/aipsy_4split",
            "v1/results/aipsy_4split_eval",
        ]
        for fb in fallbacks:
            if os.path.exists(fb):
                files = sorted(glob.glob(os.path.join(fb, "*_aipsy_4split.csv")))
                if files:
                    break

    if not files:
        print(f"No result CSVs found in {args.input_dir} or fallbacks.")
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
        # 1 family for Sensitivity tests across models/tasks/dims
        res_rq1["p_fdr"] = apply_benjamini_hochberg(res_rq1["p_value"].values)
        rq1_path = os.path.join(args.out_dir, "behavioral_aipsy_sensitivity_rq1.csv")
        res_rq1.to_csv(rq1_path, index=False)
        print(f"Saved RQ1 Sensitivity to {rq1_path}")

    if not res_rq2.empty:
        # 1 family for Repeated-measures Dose-Response tests
        res_rq2["p_fdr"] = apply_benjamini_hochberg(res_rq2["p_value"].values)
        rq2_path = os.path.join(args.out_dir, "behavioral_aipsy_dose_response_rq2.csv")
        res_rq2.to_csv(rq2_path, index=False)
        print(f"Saved RQ2 Dose-Response to {rq2_path}")

    if not res_rq3.empty:
        # 1 family for Specificity controls
        res_rq3["p_fdr"] = apply_benjamini_hochberg(res_rq3["p_value"].values)
        rq3_path = os.path.join(args.out_dir, "behavioral_aipsy_specificity_rq3.csv")
        res_rq3.to_csv(rq3_path, index=False)
        print(f"Saved RQ3 Specificity to {rq3_path}")

    if not res_rq4.empty:
        # Primary delta_coupling tests form 1 family for FDR correction
        delta_mask = res_rq4["correlation_type"] == "delta_coupling"
        res_rq4["p_fdr"] = np.nan
        if delta_mask.any():
            res_rq4.loc[delta_mask, "p_fdr"] = apply_benjamini_hochberg(res_rq4.loc[delta_mask, "p_value"].values)
        rq4_path = os.path.join(args.out_dir, "behavioral_aipsy_coupling_rq4.csv")
        res_rq4.to_csv(rq4_path, index=False)
        print(f"Saved RQ4 Coupling to {rq4_path}")


if __name__ == "__main__":
    main()
