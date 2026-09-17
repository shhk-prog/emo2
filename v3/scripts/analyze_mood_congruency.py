#!/usr/bin/env python3
"""
Analysis script for Mood Congruency Experiment:
Estimating the mood congruency slope beta_mood and contrasting it with control conditions.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Analyze Mood Congruency Experiment Results")
    parser.add_argument("--input-file", type=str, required=True, help="Path to input jsonl results")
    parser.add_argument("--out-dir", type=str, default="v3/results/derived/mood_congruency")
    args = parser.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    input_path = os.path.join(PROJECT_ROOT, args.input_file)
    print(f"Loading results from {input_path}...")

    records = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} records.")

    # 1. Compute baseline (alpha = 0) for each stimulus, layer, direction
    df_base = df[df['alpha'] == 0.0][['layer', 'direction', 'stimulus_id', 'expected_v_rec']].rename(
        columns={'expected_v_rec': 'base_v_rec'}
    )
    df = df.merge(df_base, on=['layer', 'direction', 'stimulus_id'], how='left')
    df['delta_v_rec'] = df['expected_v_rec'] - df['base_v_rec']

    # 2. Layer-wise and Direction-wise regression: delta_v_rec ~ alpha
    summary_rows = []
    
    layers = sorted(df['layer'].unique())
    directions = df['direction'].unique()

    for l in layers:
        for d in directions:
            sub = df[(df['layer'] == l) & (df['direction'] == d)].copy()
            if len(sub) == 0:
                continue

            # Linear regression: delta_v ~ alpha
            res = stats.linregress(sub['alpha'], sub['delta_v_rec'])
            
            # Stimulus-averaged delta_v at extreme alphas
            neg3_mean = sub[sub['alpha'] == -3.0]['delta_v_rec'].mean() if len(sub[sub['alpha'] == -3.0]) > 0 else np.nan
            pos3_mean = sub[sub['alpha'] == 3.0]['delta_v_rec'].mean() if len(sub[sub['alpha'] == 3.0]) > 0 else np.nan
            
            summary_rows.append({
                "layer": l,
                "direction": d,
                "n_stimuli": sub['stimulus_id'].nunique(),
                "n_observations": len(sub),
                "beta_mood": res.slope,
                "se": res.stderr,
                "r_value": res.rvalue,
                "p_value": res.pvalue,
                "delta_v_at_minus3": neg3_mean,
                "delta_v_at_plus3": pos3_mean
            })

    df_summary = pd.DataFrame(summary_rows)

    # 3. Save Summary CSV & Markdown
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    csv_out = os.path.join(out_dir, f"{base_name}_summary.csv")
    md_out = os.path.join(out_dir, f"{base_name}_summary.md")

    df_summary.to_csv(csv_out, index=False)
    print(f"Summary table saved to: {csv_out}")

    # Generate Markdown Table
    md_content = f"# Mood Congruency Causal Effect Summary\n\n"
    md_content += f"- Source: `{args.input_file}`\n"
    md_content += f"- Total observations: {len(df)}\n\n"
    md_content += "## Linear Slope $\\beta_{\\mathrm{mood}}$ (Change in Recognized Valence per $\\sigma$ Steering)\n\n"
    md_content += "| Layer | Direction | N stimuli | $\\beta_{\\mathrm{mood}}$ (Slope) | SE | $t$-statistic | $p$-value | $\\Delta V$ ($-3\\sigma$) | $\\Delta V$ ($+3\\sigma$) |\n"
    md_content += "|---|---|---|---|---|---|---|---|---|\n"

    for _, row in df_summary.iterrows():
        t_stat = row['beta_mood'] / row['se'] if row['se'] > 0 else 0
        p_str = f"{row['p_value']:.2e}" if row['p_value'] < 0.001 else f"{row['p_value']:.4f}"
        md_content += (
            f"| {int(row['layer'])} | `{row['direction']}` | {row['n_stimuli']} | "
            f"**{row['beta_mood']:+.4f}** | {row['se']:.4f} | {t_stat:.2f} | {p_str} | "
            f"{row['delta_v_at_minus3']:+.3f} | {row['delta_v_at_plus3']:+.3f} |\n"
        )

    md_content += "\n\n### Interpretation Guide\n"
    md_content += "- **Mood Congruency Confirmed**: $\\beta_{\\mathrm{mood}} > 0$ with $p < 0.01$ in `valence` condition, while `random` condition $\\approx 0$.\n"
    md_content += "- **Negative finding**: $\\beta_{\\mathrm{mood}} \\approx 0$ or non-significant, indicating decoupling between internal mood state and recognition circuitry.\n"

    with open(md_out, "w") as f:
        f.write(md_content)
    print(f"Markdown report saved to: {md_out}")
    print("\n" + md_content)


if __name__ == "__main__":
    main()
