#!/usr/bin/env python3
"""
v3/scripts/summarize_results.py
Summarizes the multi-layer patching and Mahalanobis diagnostics results.
"""

import json
import numpy as np
import pandas as pd

def summarize():
    with open("v3/results/multilayer_patching_results.json") as f:
        data = json.load(f)
        
    print(f"Total evaluated pairs: {len(data)}")
    
    # 1. Recovery by block configuration
    block_names = ["1-Layer (L15)", "2-Layer (L14-15)", "4-Layer (L13-16)", "8-Layer (L11-18)"]
    summary_blocks = []
    
    for bname in block_names:
        recs_aligned = [p["blocks"][bname]["rec_aligned"] for p in data]
        recs_within = [p["blocks"][bname]["rec_within"] for p in data]
        recs_raw = [p["blocks"][bname]["rec_raw"] for p in data]
        
        # Bootstrap 95% CI for aligned recovery
        boot_means = []
        for _ in range(1000):
            sample = np.random.choice(recs_aligned, size=len(recs_aligned), replace=True)
            boot_means.append(np.mean(sample))
        ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
        
        summary_blocks.append({
            "Block": bname,
            "Aligned Mean Rec (%)": np.mean(recs_aligned) * 100,
            "Aligned Median Rec (%)": np.median(recs_aligned) * 100,
            "Aligned 95% CI (%)": f"[{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]",
            "Raw Mean Rec (%)": np.mean(recs_raw) * 100,
            "Within Mean Rec (%)": np.mean(recs_within) * 100
        })
        
    df_blocks = pd.DataFrame(summary_blocks)
    print("\n=== MULTI-LAYER RECOVERY SUMMARY ===")
    print(df_blocks.to_string(index=False))
    
    # 2. Layer-wise Diagnostics (Mahalanobis Distance & Geometry)
    layers = sorted(list(data[0]["layer_diagnostics"].keys()), key=lambda x: int(x))
    diag_summary = []
    
    for l in layers:
        raw_dm = [p["layer_diagnostics"][l]["raw_mahalanobis"] for p in data]
        aligned_dm = [p["layer_diagnostics"][l]["aligned_mahalanobis"] for p in data]
        raw_cos = [p["layer_diagnostics"][l]["raw_cosine"] for p in data]
        aligned_cos = [p["layer_diagnostics"][l]["aligned_cosine"] for p in data]
        raw_norm = [p["layer_diagnostics"][l]["raw_norm_ratio"] for p in data]
        aligned_norm = [p["layer_diagnostics"][l]["aligned_norm_ratio"] for p in data]
        
        diag_summary.append({
            "Layer": l,
            "Raw D_M (Mean)": f"{np.mean(raw_dm):.1f}",
            "Aligned D_M (Mean)": f"{np.mean(aligned_dm):.2f}",
            "Raw Cosine": f"{np.mean(raw_cos):.3f}",
            "Aligned Cosine": f"{np.mean(aligned_cos):.3f}",
            "Aligned Norm Ratio": f"{np.mean(aligned_norm):.3f}"
        })
        
    df_diag = pd.DataFrame(diag_summary)
    print("\n=== MULTIVARIATE MAHALANOBIS & GEOMETRY DIAGNOSTICS ===")
    print(df_diag.to_string(index=False))
    
    def df_to_md(df):
        cols = list(df.columns)
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = []
        for _, r in df.iterrows():
            rows.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        return "\n".join([header, sep] + rows)

    # Save markdown summary
    with open("v3/results/summary_table.md", "w") as f:
        f.write("# Experimental Results Summary: Multi-layer Patching & Multivariate Diagnostics\n\n")
        f.write("### 1. Multi-Layer Simultaneous Patching Recovery (Normalized 2D EMD)\n\n")
        f.write(df_to_md(df_blocks) + "\n\n")
        f.write("### 2. Multivariate Mahalanobis Distance and Geometric In-Distribution Diagnostics\n\n")
        f.write(df_to_md(df_diag) + "\n")

if __name__ == "__main__":
    summarize()
