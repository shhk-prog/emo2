import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

summary_csv = "results/derived/scaling_summary.csv"
gap_dir = "results/derived/scaling"
out_dir = "results/figures/scaling"

os.makedirs(out_dir, exist_ok=True)

# 1. Load summary data
df = pd.read_csv(summary_csv)

# Plot 1: Clean Shift by Parameters (Base vs Instruct)
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
colors = {"Qwen2.5": "blue", "Llama-3.2": "green"}

for family in df['family'].unique():
    f_df = df[df['family'] == family].sort_values('parameters_b')
    plt.plot(f_df['parameters_b'], f_df['base_clean_shift_v'], marker='o', linestyle='--', color=colors[family], label=f"{family} Base")
    plt.plot(f_df['parameters_b'], f_df['instruct_clean_shift_v'], marker='s', linestyle='-', color=colors[family], label=f"{family} Instruct")

plt.xscale('log')
plt.xlabel("Model Parameters (Billion)")
plt.ylabel("Clean Shift (Delta E[V])")
plt.title("Scaling of Affective Reactivity (Clean Shift) by Model Size")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "scaling_clean_shift.png"))
plt.close()

# Plot 2: Behavioral Suppression Ratio
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

for family in df['family'].unique():
    f_df = df[df['family'] == family].sort_values('parameters_b')
    plt.plot(f_df['parameters_b'], f_df['behavioral_suppression_ratio'], marker='o', color=colors[family], label=family)

plt.xscale('log')
plt.xlabel("Model Parameters (Billion)")
plt.ylabel("Behavioral Suppression Ratio (%)")
plt.title("Behavioral Suppression Ratio by Model Size")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "scaling_suppression_ratio.png"))
plt.close()

# Plot 3: Layer-wise Suppression Gap (Recovery V)
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

for tag in df['tag']:
    gap_csv = os.path.join(gap_dir, tag, "suppression_gap.csv")
    if os.path.exists(gap_csv):
        gap_df = pd.read_csv(gap_csv)
        max_layer = gap_df['layer'].max()
        if max_layer > 0:
            norm_layers = gap_df['layer'] / max_layer
        else:
            norm_layers = gap_df['layer']
        
        plt.plot(norm_layers, gap_df['suppression_gap_recovery_V'], label=tag)

plt.xlabel("Normalized Layer Depth (0 to 1)")
plt.ylabel("Suppression Gap (Recovery V: Base - Instruct)")
plt.title("Layer-wise Causal Effect Suppression (Recovery V)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "suppression_gap_layers.png"))
plt.close()

print(f"All plots saved to {out_dir}/")
