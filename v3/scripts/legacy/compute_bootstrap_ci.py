#!/usr/bin/env python3
"""
v3/scripts/compute_bootstrap_ci.py
==================================
Reproducible 95% Bootstrap Confidence Interval Computation
for causal recovery estimates and paired peak-site contrast.

Method:
  - Resampling: 2,000 (default) bootstrap iterations with replacement across matched test pairs
  - Percentile Method: [2.5th percentile, 97.5th percentile]
  - Random Seed: 42 (strictly fixed for reproducibility)

Inputs:
  - v3/results/focused_causal_sweep_39pairs_pair_level.csv (pair-level recovery data)
  - v3/results/generation_multilayer_residual_results.csv (multi-layer residual data)
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def bootstrap_ci(arr, n_boot=2000, seed=42, ci=95):
    """Computes empirical bootstrap confidence interval using the percentile method."""
    np.random.seed(seed)
    arr = np.array(arr)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    boots = []
    for _ in range(n_boot):
        sample = np.random.choice(arr, size=n, replace=True)
        boots.append(np.mean(sample))
    lower = np.percentile(boots, (100 - ci) / 2)
    upper = np.percentile(boots, 100 - (100 - ci) / 2)
    return float(lower), float(upper)

def main():
    parser = argparse.ArgumentParser(description="Reproducible Bootstrap CI Computation")
    parser.add_argument("--pair_level_path", type=str, default="v3/results/focused_causal_sweep_39pairs_pair_level.csv",
                        help="Path to pair-level focused causal recovery results")
    parser.add_argument("--multilayer_path", type=str, default="v3/results/generation_multilayer_residual_results.csv",
                        help="Path to multi-layer residual sweep results")
    parser.add_argument("--n_boot", type=int, default=2000, help="Number of bootstrap resamples (default: 2000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    print(f"=" * 70)
    print(f"Reproducible Bootstrap 95% CI Computation (N_boot={args.n_boot}, Seed={args.seed})")
    print(f"=" * 70)

    # 1. Pair-level Evaluation for Peak-Site Contrast
    pair_path = Path(args.pair_level_path)
    if pair_path.exists():
        df_pair = pd.read_csv(pair_path)
        print(f"\n[1] Focused 39-Pair Peak-Site Contrast Analysis ({len(df_pair)} pairs):")
        
        l15_mlp = df_pair["l15_mlp_recovery"].values
        l24_resid = df_pair["l24_resid_recovery"].values
        delta_g = df_pair["delta_g"].values if "delta_g" in df_pair.columns else (l24_resid - l15_mlp)

        ci_l15 = bootstrap_ci(l15_mlp, n_boot=args.n_boot, seed=args.seed)
        ci_l24 = bootstrap_ci(l24_resid, n_boot=args.n_boot, seed=args.seed)
        ci_delta = bootstrap_ci(delta_g, n_boot=args.n_boot, seed=args.seed)

        print(f"  * Layer 15 MLP (Probe Peak):")
        print(f"      Mean Recovery    : {np.mean(l15_mlp):+5.2f}%")
        print(f"      Median Recovery  : {np.median(l15_mlp):+5.2f}%")
        print(f"      95% Bootstrap CI : [{ci_l15[0]:+5.2f}%, {ci_l15[1]:+5.2f}%]")

        print(f"  * Layer 24 Residual (Late Causal Leverage):")
        print(f"      Mean Recovery    : {np.mean(l24_resid):+5.2f}%")
        print(f"      Median Recovery  : {np.median(l24_resid):+5.2f}%")
        print(f"      95% Bootstrap CI : [{ci_l24[0]:+5.2f}%, {ci_l24[1]:+5.2f}%]")

        print(f"  * Paired Peak-Site Contrast (Delta G = G_L24,RESID - G_L15,MLP):")
        print(f"      Mean Difference  : {np.mean(delta_g):+5.2f}%")
        print(f"      Median Difference: {np.median(delta_g):+5.2f}%")
        print(f"      95% Bootstrap CI : [{ci_delta[0]:+5.2f}%, {ci_delta[1]:+5.2f}%]")
    else:
        print(f"\n[!] Warning: {pair_path} not found.")

    # 2. Multi-Layer Residual Stream Patching Evaluation
    multi_path = Path(args.multilayer_path)
    if multi_path.exists():
        df_multi = pd.read_csv(multi_path)
        print(f"\n[2] Multi-Layer Residual Stream Patching (All 39 Pairs):")
        for _, row in df_multi.iterrows():
            print(f"  * {row['condition']:22s}: Mean = {row['mean_recovery']:5.2f}% (Med: {row['median_recovery']:5.2f}%), 95% CI = [{row['ci_95_low']:5.1f}%, {row['ci_95_high']:5.1f}%], Pos = {row['positive_fraction']:4.1f}%")
    else:
        print(f"\n[!] Warning: {multi_path} not found.")

    print(f"\n" + "=" * 70)
    print("Computation completed successfully.")

if __name__ == "__main__":
    main()
