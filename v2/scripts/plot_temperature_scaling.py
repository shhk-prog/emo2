#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def set_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12

def plot_temperature_scaling(df, out_dir):
    df_peak = df[df['intensity'] == 'peak'].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Temperature vs E[V] (Distribution) for Peak intensity
    sns.boxplot(data=df_peak, x='temperature', y='E_v', hue='is_instruct', ax=axes[0])
    axes[0].set_title('E[V] Distribution (Peak Intensity) across Temperatures')
    axes[0].set_xlabel('Temperature (τ)')
    axes[0].set_ylabel('E[V_self]')
    axes[0].legend(title='Is Instruct')
    
    # 2. Temperature vs p(5,5) (Neutral prior concentration) for Peak intensity
    sns.pointplot(data=df_peak, x='temperature', y='p_55', hue='is_instruct', dodge=True, ax=axes[1])
    axes[1].set_title('p(V=5, A=5) (Neutral Peak) across Temperatures')
    axes[1].set_xlabel('Temperature (τ)')
    axes[1].set_ylabel('p(5,5)')
    axes[1].legend(title='Is Instruct')
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, "temperature_scaling_plot.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, default="v2/results/derived/phase3b_stress_test/temperature_scaling_results.csv")
    parser.add_argument("--out-dir", type=str, default="v2/results/derived/phase3b_stress_test/plots")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    set_style()
    
    if os.path.exists(args.data_file):
        df = pd.read_csv(args.data_file)
        plot_temperature_scaling(df, args.out_dir)
    else:
        print(f"File not found: {args.data_file}")

if __name__ == "__main__":
    main()
