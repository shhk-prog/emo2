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

def plot_r2_scores(df, out_dir):
    """Plot R2 scores for different alignment methods across layers."""
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(data=df, x='layer', y='r2_direct', label='Direct', marker='o')
    sns.lineplot(data=df, x='layer', y='r2_ortho', label='Orthogonal Procrustes', marker='s')
    sns.lineplot(data=df, x='layer', y='r2_ridge', label='Ridge (Regularized)', marker='^')
    
    plt.title('Vector Reconstruction Accuracy (R^2) across layers')
    plt.xlabel('Layer')
    plt.ylabel('R^2 Score')
    plt.ylim(-2, 1) # Limit y-axis so Ridge is visible despite huge negative Direct scores
    plt.legend(title='Alignment Method')
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, "cross_decoding_r2.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved R2 plot to {out_path}")

def plot_cross_ibc(df, out_dir):
    """Plot Cross-Projection IBC for Valence and Arousal."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    # Plot Valence
    sns.lineplot(data=df, x='layer', y='ibc_V_target_inst', label='Target (Instruct Upper Bound)', linestyle='--', color='black', ax=axes[0])
    sns.lineplot(data=df, x='layer', y='ibc_V_direct', label='Direct', marker='o', ax=axes[0])
    sns.lineplot(data=df, x='layer', y='ibc_V_ortho', label='Orthogonal Procrustes', marker='s', ax=axes[0])
    sns.lineplot(data=df, x='layer', y='ibc_V_ridge', label='Ridge (Regularized)', marker='^', ax=axes[0])
    
    axes[0].set_title('Cross-Projection IBC (Valence)')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Pearson r (z_V vs E[V])')
    axes[0].legend(title='Alignment Method')
    
    # Plot Arousal
    sns.lineplot(data=df, x='layer', y='ibc_A_target_inst', label='Target (Instruct Upper Bound)', linestyle='--', color='black', ax=axes[1])
    sns.lineplot(data=df, x='layer', y='ibc_A_direct', label='Direct', marker='o', ax=axes[1])
    sns.lineplot(data=df, x='layer', y='ibc_A_ortho', label='Orthogonal Procrustes', marker='s', ax=axes[1])
    sns.lineplot(data=df, x='layer', y='ibc_A_ridge', label='Ridge (Regularized)', marker='^', ax=axes[1])
    
    axes[1].set_title('Cross-Projection IBC (Arousal)')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Pearson r (z_A vs E[A])')
    axes[1].legend(title='Alignment Method')
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, "cross_decoding_ibc.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Cross-IBC plot to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, default="v2/results/derived/phase2_cross_decoding/cross_decoding_results.csv")
    parser.add_argument("--out-dir", type=str, default="v2/results/derived/phase2_cross_decoding/plots")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    set_style()
    
    if os.path.exists(args.data_file):
        df = pd.read_csv(args.data_file)
        plot_r2_scores(df, args.out_dir)
        plot_cross_ibc(df, args.out_dir)
    else:
        print(f"File not found: {args.data_file}")

if __name__ == "__main__":
    main()
