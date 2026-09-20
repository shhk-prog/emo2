#!/usr/bin/env python3
"""
behavioral/analysis/summarize_behavioral_emobank.py

Comprehensive Summary for EmoBank 3-Way VAD Behavioral Evaluation.
Organized around the 4 Core Behavioral Metrics:
  1. Human Grounding: Correlation (Pearson r & Spearman rho) between Model outputs and Human Ground Truth
     - Writer Estimation: W <-> Human Writer VAD
     - Reader Prediction: R <-> Human Reader VAD
     - Self-Report:       S <-> Human Reader VAD
  2. Sensitivity: Dispersion, Cohen's d_z, and Neutral Rate P(5,5,5) across conditions
  3. Dose-Response & Calibration: MAE, RMSE against human scales
  4. Reader-Self Coupling: Model-internal cognitive coupling Corr(R, S)
"""

import argparse
import glob
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

TASKS = [
    ("writer", "Writer-State Estimation (W)", "human_writer"),
    ("reader", "Reader-Response Prediction (R)", "human_reader"),
    ("self", "Self-Report (S)", "human_reader"),
]

DIMS = [("v", "Valence"), ("a", "Arousal"), ("d", "Dominance")]


def format_p(p):
    if np.isnan(p):
        return "N/A"
    if p < 0.001:
        return "<.001"
    return f"{p:.3f}"


def safe_corr(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_c, y_c = x[mask], y[mask]
    if len(x_c) < 3:
        return (np.nan, np.nan), (np.nan, np.nan)
    pr, pp = pearsonr(x_c, y_c)
    sr, sp = spearmanr(x_c, y_c)
    return (float(pr), float(pp)), (float(sr), float(sp))


TASK_PREFIX = {
    "writer": "w",
    "reader": "r",
    "self": "s",
}


def analyze_file(csv_path):
    df = pd.read_csv(csv_path)
    model_name = Path(csv_path).stem.replace("_3way_vad", "")
    n_samples = len(df)

    rows = []
    # 1. Task x Dimension Evaluation
    for t_key, t_label, gt_prefix in TASKS:
        col_prefix = TASK_PREFIX.get(t_key, t_key)
        for d_key, d_label in DIMS:
            ev_col = f"{col_prefix}_e{d_key}" if f"{col_prefix}_e{d_key}" in df.columns else f"{t_key}_e{d_key}"
            gv_col = f"{col_prefix}_g{d_key}" if f"{col_prefix}_g{d_key}" in df.columns else f"{t_key}_g{d_key}"
            gt_col = f"{gt_prefix}_{d_key}"

            if (
                ev_col not in df.columns
                or gv_col not in df.columns
                or gt_col not in df.columns
            ):
                continue

            (pr_cont, pp_cont), (sr_cont, sp_cont) = safe_corr(
                df[ev_col].values, df[gt_col].values
            )
            (pr_grd, pp_grd), (sr_grd, sp_grd) = safe_corr(
                df[gv_col].values, df[gt_col].values
            )

            mask = ~np.isnan(df[ev_col].values) & ~np.isnan(df[gt_col].values)
            if np.sum(mask) >= 1:
                mae_cont = float(mean_absolute_error(df[gt_col].values[mask], df[ev_col].values[mask]))
                rmse_cont = float(root_mean_squared_error(df[gt_col].values[mask], df[ev_col].values[mask]))
            else:
                mae_cont = np.nan
                rmse_cont = np.nan

            rows.append(
                {
                    "model": model_name,
                    "task": t_key,
                    "task_label": t_label,
                    "dimension": d_key.upper(),
                    "dim_label": d_label,
                    "n": n_samples,
                    "r_continuous": pr_cont,
                    "p_continuous": pp_cont,
                    "rho_continuous": sr_cont,
                    "r_greedy": pr_grd,
                    "p_greedy": pp_grd,
                    "mae": mae_cont,
                    "rmse": rmse_cont,
                    "pred_mean": float(np.nanmean(df[ev_col].values)),
                    "pred_std": float(np.nanstd(df[ev_col].values)),
                    "human_mean": float(np.nanmean(df[gt_col].values)),
                    "human_std": float(np.nanstd(df[gt_col].values)),
                }
            )

    # 2. Reader-Self Coupling
    coupling_rows = []
    for d_key, d_label in DIMS:
        r_col = f"r_e{d_key}"
        s_col = f"s_e{d_key}"
        if r_col in df.columns and s_col in df.columns:
            (pr, pp), (sr, sp) = safe_corr(df[r_col].values, df[s_col].values)
            diff = df[s_col].values - df[r_col].values
            coupling_rows.append(
                {
                    "model": model_name,
                    "dimension": d_key.upper(),
                    "dim_label": d_label,
                    "r_RS": pr,
                    "p_RS": pp,
                    "rho_RS": sr,
                    "delta_mean": float(np.nanmean(diff)),
                    "delta_std": float(np.nanstd(diff)),
                }
            )

    # 3. Neutral Rate P(5,5,5)
    p555_info = {}
    for t_key in ["w", "r", "s"]:
        p_col = f"{t_key}_p555"
        if p_col in df.columns:
            p555_info[f"{t_key}_p555_mean"] = float(df[p_col].mean() * 100.0)
        gv_col, ga_col, gd_col = (
            f"{t_key}_gv",
            f"{t_key}_ga",
            f"{t_key}_gd",
        )
        if (
            gv_col in df.columns
            and ga_col in df.columns
            and gd_col in df.columns
        ):
            exact_neutral = (
                (df[gv_col] == 5) & (df[ga_col] == 5) & (df[gd_col] == 5)
            ).mean() * 100.0
            p555_info[f"{t_key}_exact_neutral_pct"] = float(exact_neutral)

    return pd.DataFrame(rows), pd.DataFrame(coupling_rows), p555_info


def main():
    parser = argparse.ArgumentParser(
        description="Summarize EmoBank 3-Way VAD Behavioral Evaluation."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="behavioral/results/raw/emobank_3way",
        help="Directory containing *_3way_vad.csv files",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="behavioral/results/derived/emobank_3way_summary",
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
        files = sorted(glob.glob(os.path.join(args.input_dir, "*_3way_vad.csv")))
    else:
        files = sorted(glob.glob(os.path.join(args.input_dir, "*_3way_vad.csv")))
        if not files:
            # Fallbacks: legacy path and v1 results
            fallbacks = [
                "behavioral/results/emobank_3way",
                "v1/results/emobank_3way_vad_test1k",
            ]
            for fb in fallbacks:
                if os.path.exists(fb):
                    files = sorted(glob.glob(os.path.join(fb, "*_3way_vad.csv")))
                    if files:
                        break

    if not files:
        print(f"No result CSVs found in {args.input_dir} or fallback.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    all_metrics = []
    all_couplings = []
    neutral_summaries = []

    for f in files:
        df_m, df_c, p555 = analyze_file(f)
        all_metrics.append(df_m)
        all_couplings.append(df_c)
        p555["model"] = Path(f).stem.replace("_3way_vad", "")
        neutral_summaries.append(p555)

    res_metrics = pd.concat(all_metrics, ignore_index=True)
    res_couplings = pd.concat(all_couplings, ignore_index=True)
    res_neutrals = pd.DataFrame(neutral_summaries)

    metrics_csv = os.path.join(args.out_dir, "behavioral_emobank_metrics.csv")
    couplings_csv = os.path.join(args.out_dir, "behavioral_emobank_coupling.csv")
    neutrals_csv = os.path.join(
        args.out_dir, "behavioral_emobank_neutral_rates.csv"
    )

    res_metrics.to_csv(metrics_csv, index=False)
    res_couplings.to_csv(couplings_csv, index=False)
    res_neutrals.to_csv(neutrals_csv, index=False)

    print(
        f"Saved behavioral EmoBank summaries to {args.out_dir}:\n"
        f"  - {metrics_csv}\n"
        f"  - {couplings_csv}\n"
        f"  - {neutrals_csv}"
    )


if __name__ == "__main__":
    main()
