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


from affective_empathy_eval.affect_directions import AIPSY_EXPECTED_DIRECTION as EXPECTED_DIRECTION


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
    stem = Path(csv_path).stem
    if stem.startswith("behavioral_aipsy_") and stem.endswith("_4split"):
        model_name = stem[len("behavioral_aipsy_") : -len("_4split")]
    else:
        model_name = stem.replace("_aipsy_4split", "")
    alignment = resolve_alignment(model_name)
    splits = set(df["split"].dropna().unique())

    # 1. RQ1: Sensitivity (Clinical vs Neutral matched by pair_id, direction-aligned)
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
                dim_u = dim.upper()
                for task in ["w", "r", "s"]:
                    c_col = f"{task}_e{dim}_clin"
                    n_col = f"{task}_e{dim}_neut"
                    if c_col in merged_cn.columns and n_col in merged_cn.columns:
                        c_vals = merged_cn[c_col].values
                        n_vals = merged_cn[n_col].values
                        raw_diff = c_vals - n_vals

                        # Direction-aligned difference to prevent positive/negative emotion cancellation
                        aligned_diffs = []
                        for _, row_cn in merged_cn.iterrows():
                            emo = str(row_cn.get("emotion_clin", row_cn.get("emotion", ""))).lower().strip()
                            sign = EXPECTED_DIRECTION.get(emo, {}).get(dim_u, None)
                            if sign is not None:
                                aligned_diffs.append(sign * (row_cn[c_col] - row_cn[n_col]))

                        aligned_arr = np.array(aligned_diffs) if aligned_diffs else np.array([])
                        n_defined = len(aligned_arr)

                        if n_defined > 2:
                            align_mean = float(np.mean(aligned_arr))
                            align_dz = compute_d_z(aligned_arr)
                            t_stat_a, p_val_a = ttest_1samp(aligned_arr, popmean=0.0)
                            _, align_ci_low, align_ci_high = compute_bootstrap_ci(aligned_arr, statistic_fn=np.mean)
                            _, align_dz_low, align_dz_high = compute_bootstrap_ci(aligned_arr, statistic_fn=compute_d_z)
                        else:
                            align_mean = np.nan
                            align_dz = np.nan
                            t_stat_a = np.nan
                            p_val_a = np.nan
                            align_ci_low, align_ci_high = np.nan, np.nan
                            align_dz_low, align_dz_high = np.nan, np.nan

                        raw_dz = compute_d_z(raw_diff)
                        t_stat_raw, p_val_raw = ttest_rel(c_vals, n_vals)
                        _, raw_ci_low, raw_ci_high = compute_bootstrap_ci(raw_diff, statistic_fn=np.mean)
                        _, raw_dz_low, raw_dz_high = compute_bootstrap_ci(raw_diff, statistic_fn=compute_d_z)


                        rq1_rows.append(
                            {
                                "model": model_name,
                                "alignment": alignment,
                                "task": task,
                                "dimension": dim_u,
                                "n_pairs": len(raw_diff),
                                "n_direction_defined": n_defined,
                                "direction_aligned_mean_diff": align_mean,  # PRIMARY METRIC
                                "aligned_ci_low": float(align_ci_low),
                                "aligned_ci_high": float(align_ci_high),
                                "aligned_d_z": float(align_dz),
                                "aligned_dz_ci_low": float(align_dz_low),
                                "aligned_dz_ci_high": float(align_dz_high),
                                "aligned_t_stat": float(t_stat_a),
                                "aligned_p_value": float(p_val_a),
                                "raw_mean_diff": float(np.mean(raw_diff)),
                                "raw_ci_low": float(raw_ci_low),
                                "raw_ci_high": float(raw_ci_high),
                                "raw_d_z": float(raw_dz),
                                "raw_t_stat": float(t_stat_raw),
                                "raw_p_value": float(p_val_raw),
                                # Backward compatible aliases
                                "mean_diff": float(np.mean(raw_diff)),
                                "mean_diff_ci_low": float(raw_ci_low),
                                "mean_diff_ci_high": float(raw_ci_high),
                                "d_z": float(raw_dz),
                                "dz_ci_low": float(raw_dz_low),
                                "dz_ci_high": float(raw_dz_high),
                                "p_value": float(p_val_raw),
                            }
                        )

    # 2. RQ2: Dose-Response (Step-1 Moderate vs Neutral, Step-2 Clinical vs Moderate, Monotonicity)
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
                dim_u = dim.upper()
                for task in ["w", "r", "s"]:
                    n_col = f"{task}_e{dim}_neut"
                    m_col = f"{task}_e{dim}_mod"
                    c_col = f"{task}_e{dim}_clin"
                    if n_col in trip_merged.columns and m_col in trip_merged.columns and c_col in trip_merged.columns:
                        n_arr = trip_merged[n_col].values
                        m_arr = trip_merged[m_col].values
                        c_arr = trip_merged[c_col].values

                        # Step-1 (Moderate - Neutral) and Step-2 (Clinical - Moderate)
                        aligned_s1 = []
                        aligned_s2 = []
                        midpoint_devs = []

                        for _, row_tr in trip_merged.iterrows():
                            emo = str(row_tr.get("emotion_clin", row_tr.get("emotion", ""))).lower().strip()
                            sign = EXPECTED_DIRECTION.get(emo, {}).get(dim_u, None)
                            s1 = row_tr[m_col] - row_tr[n_col]
                            s2 = row_tr[c_col] - row_tr[m_col]
                            midpoint_devs.append(row_tr[m_col] - (row_tr[n_col] + row_tr[c_col]) / 2.0)
                            if sign is not None:
                                aligned_s1.append(sign * s1)
                                aligned_s2.append(sign * s2)

                        aligned_s1_arr = np.array(aligned_s1) if aligned_s1 else np.array([])
                        aligned_s2_arr = np.array(aligned_s2) if aligned_s2 else np.array([])

                        if len(aligned_s1_arr) > 2:
                            mean_step_1 = float(np.mean(aligned_s1_arr))
                            mean_step_2 = float(np.mean(aligned_s2_arr))
                            monotonic = (aligned_s1_arr > 0) & (aligned_s2_arr > 0)
                            monotonicity_rate = float(np.mean(monotonic))
                            _, s1_low, s1_high = compute_bootstrap_ci(aligned_s1_arr, statistic_fn=np.mean)
                            _, s2_low, s2_high = compute_bootstrap_ci(aligned_s2_arr, statistic_fn=np.mean)
                            _, p_s1 = ttest_1samp(aligned_s1_arr, popmean=0.0, alternative="greater")
                            _, p_s2 = ttest_1samp(aligned_s2_arr, popmean=0.0, alternative="greater")
                            p_monotonic_iut = float(max(p_s1, p_s2))
                        else:
                            mean_step_1, mean_step_2, monotonicity_rate = np.nan, np.nan, np.nan
                            s1_low, s1_high = np.nan, np.nan
                            s2_low, s2_high = np.nan, np.nan
                            p_s1, p_s2 = np.nan, np.nan
                            p_monotonic_iut = np.nan

                        midpoint_deviation = float(np.mean(midpoint_devs)) if midpoint_devs else 0.0

                        # Secondary: raw slope (clinical - neutral) / 2.0
                        sec_slopes = (c_arr - n_arr) / 2.0
                        sec_mean_slope = float(np.mean(sec_slopes))
                        sec_t, sec_p = ttest_1samp(sec_slopes, popmean=0.0)
                        _, sec_low, sec_high = compute_bootstrap_ci(sec_slopes, statistic_fn=np.mean)

                        rq2_rows.append(
                            {
                                "model": model_name,
                                "alignment": alignment,
                                "task": task,
                                "dimension": dim_u,
                                "n_triplets": len(trip_merged),
                                "n_direction_defined": len(aligned_s1_arr),
                                "mean_step_1_aligned": mean_step_1,  # PRIMARY: Moderate - Neutral
                                "step_1_ci_low": float(s1_low),
                                "step_1_ci_high": float(s1_high),
                                "step_1_p_value": float(p_s1),
                                "mean_step_2_aligned": mean_step_2,  # PRIMARY: Clinical - Moderate
                                "step_2_ci_low": float(s2_low),
                                "step_2_ci_high": float(s2_high),
                                "step_2_p_value": float(p_s2),
                                "dose_response_p_value": float(p_monotonic_iut),  # PRIMARY IUT p-value
                                "monotonicity_rate": monotonicity_rate,  # Effect description
                                "midpoint_deviation": midpoint_deviation,
                                "secondary_mean_slope": sec_mean_slope,
                                "secondary_slope_p": float(sec_p),
                                "mean_slope": sec_mean_slope,  # Backward compatible alias
                                "slope_ci_low": float(sec_low),  # Valid bootstrap CI for secondary slope
                                "slope_ci_high": float(sec_high),
                                "p_value": float(p_monotonic_iut),  # Primary monotonic p-value for compatibility
                                "mean_neutral": float(np.mean(n_arr)),
                                "mean_moderate": float(np.mean(m_arr)),
                                "mean_clinical": float(np.mean(c_arr)),
                            }
                        )


    # 3. RQ3: Specificity (Affective Displacement vs Complex Neutral)
    rq3_rows = []
    if "clinical" in splits and "complex_neutral" in splits:
        clin_df = df[df["split"] == "clinical"]
        cneu_df = df[df["split"] == "complex_neutral"]
        neut_df = df[df["split"] == "neutral"] if "neutral" in splits else cneu_df

        for dim in ["v", "a", "d"]:
            dim_u = dim.upper()
            for task in ["w", "r", "s"]:
                c_col = f"{task}_e{dim}"
                if c_col in clin_df.columns and c_col in cneu_df.columns:
                    c_vals = clin_df[c_col].dropna().values
                    cn_vals = cneu_df[c_col].dropna().values
                    if c_col in neut_df.columns and len(neut_df[c_col].dropna()) > 0:
                        neut_baseline = float(np.mean(neut_df[c_col].dropna().values))
                    else:
                        neut_baseline = np.nan

                    if len(c_vals) > 2 and len(cn_vals) > 2 and not np.isnan(neut_baseline):
                        # Primary: Affective Displacement D(x) = |E(x) - mu_neutral|
                        # Prevents valence cancellation between positive and negative emotions
                        disp_c = np.abs(c_vals - neut_baseline)
                        disp_cn = np.abs(cn_vals - neut_baseline)

                        from scipy.stats import ttest_ind
                        t_stat_disp, p_val_disp = ttest_ind(disp_c, disp_cn, equal_var=False)
                        pooled_std_disp = np.sqrt((np.var(disp_c, ddof=1) + np.var(disp_cn, ddof=1)) / 2.0)
                        d_disp = float((np.mean(disp_c) - np.mean(disp_cn)) / (pooled_std_disp + 1e-12))

                        # Secondary: raw signed mean diff
                        t_stat_raw, p_val_raw = ttest_ind(c_vals, cn_vals, equal_var=False)

                        rq3_rows.append(
                            {
                                "model": model_name,
                                "alignment": alignment,
                                "task": task,
                                "dimension": dim_u,
                                "n_clinical": len(c_vals),
                                "n_complex_neutral": len(cn_vals),
                                "mean_clinical_displacement": float(np.mean(disp_c)),  # PRIMARY METRIC
                                "mean_complex_neutral_displacement": float(np.mean(disp_cn)),
                                "displacement_diff": float(np.mean(disp_c) - np.mean(disp_cn)),
                                "displacement_cohen_d": d_disp,
                                "displacement_t_stat": float(t_stat_disp),
                                "displacement_p_value": float(p_val_disp),
                                "secondary_raw_mean_diff": float(np.mean(c_vals) - np.mean(cn_vals)),
                                "secondary_raw_p_value": float(p_val_raw),
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
        default="behavioral/results/raw/aipsy_4split",
        help="Directory containing *_aipsy_4split.csv files",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="behavioral/results/derived/aipsy_4split_summary",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Summarize only dry_run results",
    )
    args = parser.parse_args()

    if args.dry_run:
        args.input_dir = str(Path(args.input_dir) / "dry_run")
        args.out_dir = str(Path(args.out_dir) / "dry_run")
        files = sorted(glob.glob(os.path.join(args.input_dir, "behavioral_aipsy_*_4split.csv")))
    else:
        files = sorted(glob.glob(os.path.join(args.input_dir, "behavioral_aipsy_*_4split.csv")))
        if not files:
            # Fallback paths
            fallbacks = [
                "behavioral/results/raw/aipsy_4split",
                "results/raw/behavioral/aipsy_4split",
                "v1/results/aipsy_4split_eval",
            ]
            for fb in fallbacks:
                if os.path.exists(fb):
                    files = sorted(glob.glob(os.path.join(fb, "behavioral_aipsy_*_4split.csv")))
                    if files:
                        break

    if not files:
        raise FileNotFoundError(
            f"No AIPsy behavioral result CSVs found in {args.input_dir}"
        )

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
        # Primary inferential family: task in ['r', 's'] & dimension in ['V', 'A']
        primary_mask = (
            res_rq1["task"].isin(["r", "s"])
            & res_rq1["dimension"].isin(["V", "A"])
            & (res_rq1["n_direction_defined"] > 0)
            & res_rq1["aligned_p_value"].notna()
        )
        res_rq1["primary_p_value"] = res_rq1["aligned_p_value"]
        res_rq1["p_fdr"] = np.nan
        if primary_mask.any():
            res_rq1.loc[primary_mask, "p_fdr"] = apply_benjamini_hochberg(
                res_rq1.loc[primary_mask, "primary_p_value"].values
            )
        rq1_path = os.path.join(args.out_dir, "behavioral_aipsy_sensitivity_rq1.csv")
        res_rq1.to_csv(rq1_path, index=False)
        print(f"Saved RQ1 Sensitivity to {rq1_path}")

    if not res_rq2.empty:
        # Primary inferential family: task in ['r', 's'] & dimension in ['V', 'A'] with valid IUT p-value
        primary_mask = (
            res_rq2["task"].isin(["r", "s"])
            & res_rq2["dimension"].isin(["V", "A"])
            & res_rq2["dose_response_p_value"].notna()
        )
        res_rq2["primary_p_value"] = res_rq2["dose_response_p_value"]
        res_rq2["p_fdr"] = np.nan
        if primary_mask.any():
            res_rq2.loc[primary_mask, "p_fdr"] = apply_benjamini_hochberg(
                res_rq2.loc[primary_mask, "primary_p_value"].values
            )
        rq2_path = os.path.join(args.out_dir, "behavioral_aipsy_dose_response_rq2.csv")
        res_rq2.to_csv(rq2_path, index=False)
        print(f"Saved RQ2 Dose-Response to {rq2_path}")

    if not res_rq3.empty:
        # Primary inferential family: task in ['r', 's'] & dimension in ['V', 'A'] with valid displacement p-value
        primary_mask = (
            res_rq3["task"].isin(["r", "s"])
            & res_rq3["dimension"].isin(["V", "A"])
            & res_rq3["displacement_p_value"].notna()
        )
        res_rq3["primary_p_value"] = res_rq3["displacement_p_value"]
        res_rq3["p_fdr"] = np.nan
        if primary_mask.any():
            res_rq3.loc[primary_mask, "p_fdr"] = apply_benjamini_hochberg(
                res_rq3.loc[primary_mask, "primary_p_value"].values
            )
        rq3_path = os.path.join(args.out_dir, "behavioral_aipsy_specificity_rq3.csv")
        res_rq3.to_csv(rq3_path, index=False)
        print(f"Saved RQ3 Specificity to {rq3_path}")

    if not res_rq4.empty:
        # Primary delta_coupling tests form 1 family for FDR correction (V / A dimensions)
        delta_mask = (
            (res_rq4["correlation_type"] == "delta_coupling")
            & res_rq4["dimension"].isin(["V", "A"])
            & res_rq4["p_value"].notna()
        )
        res_rq4["p_fdr"] = np.nan
        if delta_mask.any():
            res_rq4.loc[delta_mask, "p_fdr"] = apply_benjamini_hochberg(
                res_rq4.loc[delta_mask, "p_value"].values
            )
        rq4_path = os.path.join(args.out_dir, "behavioral_aipsy_coupling_rq4.csv")
        res_rq4.to_csv(rq4_path, index=False)
        print(f"Saved RQ4 Coupling to {rq4_path}")



if __name__ == "__main__":
    main()
