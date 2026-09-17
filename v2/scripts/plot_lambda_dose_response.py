#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=str, default="v2/results/derived/phase9_advanced_patching")
    parser.add_argument("--out-dir", type=str, default="/mnt/nas/home/hiromi/.gemini/antigravity-ide/brain/5af8c3f8-4701-4aeb-abff-f62a3997e299/plots")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    records = []
    for f_name in os.listdir(args.in_dir):
        if f_name.startswith("lambda_patching_") and f_name.endswith(".jsonl"):
            path = os.path.join(args.in_dir, f_name)
            with open(path, 'r') as f:
                for line in f:
                    records.append(json.loads(line))
                    
    if not records:
        print("No data found.")
        return
        
    df = pd.DataFrame(records)
    
    # Calculate delta E_v relative to lambda=0 (Instruct baseline)
    baseline_df = df[df['lambda'] == 0.0].set_index(['id', 'layer', 'module'])[['E_v', 'E_a']]
    
    plot_data = []
    for idx, row in df.iterrows():
        b = baseline_df.loc[(row['id'], row['layer'], row['module'])]
        plot_data.append({
            'lambda': row['lambda'],
            'layer_module': f"L{row['layer']} {row['module'].upper()}",
            'delta_E_v': row['E_v'] - b['E_v'],
            'delta_E_a': row['E_a'] - b['E_a']
        })
        
    plot_df = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=plot_df, x='lambda', y='delta_E_v', hue='layer_module', marker='o', err_style="bars")
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Effect of Dose-Response Patching on Expected Valence")
    plt.xlabel("Interpolation Ratio $\lambda$ (0=Instruct, 1=Base)")
    plt.ylabel("$\Delta E_v$ (from Instruct baseline)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "lambda_dose_response_plot.png"), dpi=300)
    plt.close()
    print(f"Saved lambda_dose_response_plot.png to {args.out_dir}")

if __name__ == "__main__":
    main()
