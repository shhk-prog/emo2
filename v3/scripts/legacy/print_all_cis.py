import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.compute_bootstrap_ci import bootstrap_ci

df = pd.read_csv("v3/results/focused_causal_sweep_39pairs_pair_level.csv")
layers = [10, 14, 15, 18, 20, 24]
comps = ['mlp', 'attn', 'resid']

print("Layer | Comp | Mean | Med | 95% Bootstrap CI (1-dec) | Exact")
for l in layers:
    for c in comps:
        col = f"gen_rec_L{l}_{c}"
        if col in df.columns:
            arr = df[col].dropna().values
            low, high = bootstrap_ci(arr, n_boot=2000, seed=42)
            print(f"L{l:2d} | {c.upper():5s} | {np.mean(arr):+6.2f}% | {np.median(arr):+6.2f}% | [{low:+5.1f}%, {high:+5.1f}%] | [{low:+6.2f}%, {high:+6.2f}%]")

# Also check l15_mlp_recovery and l24_resid_recovery
low15, high15 = bootstrap_ci(df["l15_mlp_recovery"].values, n_boot=2000, seed=42)
low24, high24 = bootstrap_ci(df["l24_resid_recovery"].values, n_boot=2000, seed=42)
low_d, high_d = bootstrap_ci(df["delta_g"].values, n_boot=2000, seed=42)
print("--- Direct columns ---")
print(f"L15 MLP  : [{low15:+5.1f}%, {high15:+5.1f}%] | [{low15:+6.2f}%, {high15:+6.2f}%]")
print(f"L24 RESID: [{low24:+5.1f}%, {high24:+5.1f}%] | [{low24:+6.2f}%, {high24:+6.2f}%]")
print(f"Delta G  : [{low_d:+5.1f}%, {high_d:+5.1f}%] | [{low_d:+6.2f}%, {high_d:+6.2f}%]")
