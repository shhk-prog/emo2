#!/usr/bin/env python3
"""
v1/primary/phase_c/summarize_phase_c.py

Comprehensive cross-model summary report generator for V1 Phase C.
Aggregates:
  - E3: Response-Onset Causal Map (Peak layers, relative depths, 2D directional cosine)
  - E4: Causal Interchangeability (Matched difference patching, specificity, Cohen's d_z)
  - E6: Double Dissociation (Targeted ablation, LMM interaction test)
"""

import argparse
import glob
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

MODELS = [
    ("Qwen/Qwen2.5-1.5B", "qwen2.5_1.5b_base", "Qwen 2.5 1.5B", "Base", 28),
    (
        "Qwen/Qwen2.5-1.5B-Instruct",
        "qwen2.5_1.5b_instruct",
        "Qwen 2.5 1.5B",
        "Instruct",
        28,
    ),
    ("meta-llama/Llama-3.2-1B", "llama3.2_1b_base", "Llama 3.2 1B", "Base", 16),
    (
        "meta-llama/Llama-3.2-1B-Instruct",
        "llama3.2_1b_instruct",
        "Llama 3.2 1B",
        "Instruct",
        16,
    ),
    ("google/gemma-2-2b", "gemma2_2b_base", "Gemma 2 2B", "Base", 26),
    (
        "google/gemma-2-2b-it",
        "gemma2_2b_instruct",
        "Gemma 2 2B",
        "Instruct",
        26,
    ),
    (
        "mistralai/Mistral-7B-v0.1",
        "mistral7b_v0.1_base",
        "Mistral 7B",
        "Base",
        32,
    ),
    (
        "mistralai/Mistral-7B-Instruct-v0.1",
        "mistral7b_v0.1_instruct",
        "Mistral 7B",
        "Instruct",
        32,
    ),
]


def load_model_data(base_dir: str, prefix: str):
    m_dir = os.path.join(base_dir, prefix)
    e3_csv = os.path.join(m_dir, "e3_causal_map.csv")
    e4_csv = os.path.join(m_dir, "e4_interchangeability_results.csv")
    e6_json = os.path.join(m_dir, "e6_lmm_results.json")

    df_e3 = pd.read_csv(e3_csv) if os.path.exists(e3_csv) else None
    df_e4 = pd.read_csv(e4_csv) if os.path.exists(e4_csv) else None
    dict_e6 = None
    if os.path.exists(e6_json):
        try:
            with open(e6_json, "r") as f:
                dict_e6 = json.load(f)
        except Exception:
            pass

    return df_e3, df_e4, dict_e6


def main():
    parser = argparse.ArgumentParser(
        description="Summarize V1 Phase C Cross-Model Results"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="v1/results/derived/v1_phase_c_prompt_end",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="v1/results/derived/v1_phase_c_summary",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summary_rows = []

    for full_name, prefix, display_name, m_type, total_layers in MODELS:
        df_e3, df_e4, dict_e6 = load_model_data(args.input_dir, prefix)
        if df_e3 is None and df_e4 is None:
            continue

        row = {
            "model": display_name,
            "type": m_type,
            "total_layers": total_layers,
        }

        if df_e3 is not None and not df_e3.empty:
            peak_r_idx = int(df_e3["magnitude_reader"].idxmax())
            peak_s_idx = int(df_e3["magnitude_self"].idxmax())
            row["e3_peak_reader_layer"] = int(df_e3.loc[peak_r_idx, "layer"])
            row["e3_peak_self_layer"] = int(df_e3.loc[peak_s_idx, "layer"])
            row["e3_peak_reader_depth"] = float(
                df_e3.loc[peak_r_idx, "relative_depth"]
            )
            row["e3_peak_self_depth"] = float(
                df_e3.loc[peak_s_idx, "relative_depth"]
            )
            row["e3_mean_directional_cosine"] = float(
                df_e3["directional_cosine_similarity"].mean()
            )

        if df_e4 is not None and not df_e4.empty:
            df_a1 = df_e4[df_e4["alpha"] == 1.0]
            if not df_a1.empty:
                row["e4_mean_specificity_V"] = float(
                    df_a1["specificity_V"].mean()
                )
                row["e4_mean_dz_V"] = float(df_a1["cohen_dz_V"].mean())

        if dict_e6 is not None:
            row["e6_p_interaction"] = dict_e6.get("p_interaction", np.nan)
            row["e6_has_crossover"] = dict_e6.get("has_crossover", False)

        summary_rows.append(row)

    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        out_csv = os.path.join(args.out_dir, "phase_c_cross_model_summary.csv")
        df_sum.to_csv(out_csv, index=False)
        print(f"Saved Phase C summary to {out_csv}")
    else:
        print(f"No Phase C model outputs found in {args.input_dir}")


if __name__ == "__main__":
    main()
