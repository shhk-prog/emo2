#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def set_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12

def plot_confirmatory(df_base, df_instruct, out_dir):
    df_base['model_type'] = 'Base'
    df_instruct['model_type'] = 'Instruct'
    df = pd.concat([df_base, df_instruct], ignore_index=True)
    
    # We plot HIC, HBC, and Dissociation Gap
    metrics = ['HIC', 'HBC', 'Dissociation_Gap']
    titles = ['Human-Internal Coupling (HIC)', 'Human-Behavior Coupling (HBC)', 'Dissociation Gap (HIC - HBC)']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        sns.lineplot(data=df, x='layer', y=metric, hue='model_type', style='model_type', markers=True, ax=axes[i])
        axes[i].set_title(title)
        axes[i].set_xlabel('Layer')
        axes[i].set_ylabel('Pearson r' if metric != 'Dissociation_Gap' else '\u0394 r')
        axes[i].legend(title='Model')
        
    plt.tight_layout()
    out_path = os.path.join(out_dir, "confirmatory_gap_plot.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-file", type=str, default="v2/results/derived/phase1.5_confirmatory/Qwen_Qwen2.5-1.5B_confirmatory_results.csv")
    parser.add_argument("--instruct-file", type=str, default="v2/results/derived/phase1.5_confirmatory/Qwen_Qwen2.5-1.5B-Instruct_confirmatory_results.csv")
    parser.add_argument("--out-dir", type=str, default="v2/results/derived/phase1.5_confirmatory/plots")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    set_style()
    
    if os.path.exists(args.base_file) and os.path.exists(args.instruct_file):
        df_base = pd.read_csv(args.base_file)
        df_instruct = pd.read_csv(args.instruct_file)
        plot_confirmatory(df_base, df_instruct, args.out_dir)
    else:
        print("One or both input files not found.")

if __name__ == "__main__":
    main()
