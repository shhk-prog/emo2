#!/usr/bin/env python3
"""
scripts/repair_e3_causal_map.py

Re-aggregates e3_causal_map.csv from intact e3_causal_map_pair_level.csv
using robust NaN-filtering mean calculations.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

def repair_e3_map(pair_csv_path: str, out_map_path: str):
    if not os.path.exists(pair_csv_path):
        print(f"Error: {pair_csv_path} does not exist.")
        sys.exit(1)

    df_pairs = pd.read_csv(pair_csv_path)
    print(f"Loaded {len(df_pairs)} rows from {pair_csv_path}")

    layers = sorted(df_pairs["layer"].unique())
    num_layers = len(layers)
    records = []

    for l in layers:
        sub = df_pairs[df_pairs["layer"] == l].reset_index(drop=True)
        rel_d = float(sub["relative_depth"].iloc[0]) if "relative_depth" in sub.columns else float(l / max(1, num_layers - 1))

        valid_sv = sub["shift_V_self"].dropna().tolist()
        valid_sa = sub["shift_A_self"].dropna().tolist()
        valid_rv = sub["shift_V_reader"].dropna().tolist()
        valid_ra = sub["shift_A_reader"].dropna().tolist()

        mean_sv = float(np.mean(valid_sv)) if len(valid_sv) > 0 else 0.0
        mean_sa = float(np.mean(valid_sa)) if len(valid_sa) > 0 else 0.0
        mean_rv = float(np.mean(valid_rv)) if len(valid_rv) > 0 else 0.0
        mean_ra = float(np.mean(valid_ra)) if len(valid_ra) > 0 else 0.0

        mag_s = float(np.sqrt(mean_sv**2 + mean_sa**2))
        mag_r = float(np.sqrt(mean_rv**2 + mean_ra**2))
        denom_mean = mag_s * mag_r
        cos_sim_of_means = (
            float((mean_sv * mean_rv + mean_sa * mean_ra) / denom_mean)
            if denom_mean > 1e-6
            else 0.0
        )

        valid_cos = sub["pairwise_cos"].dropna().tolist() if "pairwise_cos" in sub.columns else []
        mean_pairwise_cos = float(np.mean(valid_cos)) if len(valid_cos) > 0 else 0.0

        disc_sub = sub[sub["eval_split"] == "discovery"]
        conf_sub = sub[sub["eval_split"] == "confirmation"]

        disc_vals_s = [
            np.sqrt(v**2 + a**2)
            for v, a in zip(disc_sub["shift_V_self"], disc_sub["shift_A_self"])
            if not (np.isnan(v) or np.isnan(a))
        ]
        disc_mag_s = float(np.mean(disc_vals_s)) if disc_vals_s else mag_s

        disc_vals_r = [
            np.sqrt(v**2 + a**2)
            for v, a in zip(disc_sub["shift_V_reader"], disc_sub["shift_A_reader"])
            if not (np.isnan(v) or np.isnan(a))
        ]
        disc_mag_r = float(np.mean(disc_vals_r)) if disc_vals_r else mag_r

        conf_vals_s = [
            np.sqrt(v**2 + a**2)
            for v, a in zip(conf_sub["shift_V_self"], conf_sub["shift_A_self"])
            if not (np.isnan(v) or np.isnan(a))
        ]
        conf_mag_s = float(np.mean(conf_vals_s)) if conf_vals_s else mag_s

        conf_vals_r = [
            np.sqrt(v**2 + a**2)
            for v, a in zip(conf_sub["shift_V_reader"], conf_sub["shift_A_reader"])
            if not (np.isnan(v) or np.isnan(a))
        ]
        conf_mag_r = float(np.mean(conf_vals_r)) if conf_vals_r else mag_r

        records.append(
            {
                "layer": l,
                "relative_depth": rel_d,
                "magnitude_self": mag_s,
                "magnitude_reader": mag_r,
                "shift_V_self": mean_sv,
                "shift_A_self": mean_sa,
                "shift_V_reader": mean_rv,
                "shift_A_reader": mean_ra,
                "directional_cosine_similarity": cos_sim_of_means,
                "mean_pairwise_directional_cosine": mean_pairwise_cos,
                "discovery_mag_self": disc_mag_s,
                "discovery_mag_reader": disc_mag_r,
                "confirmation_mag_self": conf_mag_s,
                "confirmation_mag_reader": conf_mag_r,
            }
        )

    df_out = pd.DataFrame(records).sort_values("layer")
    df_out.to_csv(out_map_path, index=False)
    print(f"Successfully repaired {out_map_path} with {len(df_out)} layers.")
    print("Non-null check:\n", df_out.notna().sum())
    print("Head:\n", df_out.head())

if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "v1/results/derived/v1_phase_c_prompt_end/gemma_base"
    pair_csv = os.path.join(target_dir, "e3_causal_map_pair_level.csv")
    out_csv = os.path.join(target_dir, "e3_causal_map.csv")
    repair_e3_map(pair_csv, out_csv)
