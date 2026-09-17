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

def plot_ibc_curves(df, out_dir):
    """Plot IBC across layers for Base vs Instruct."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    # Filter for standard template
    df_std = df[df['template'] == 'standard'].copy()
    
    # Plot Valence
    sns.lineplot(data=df_std, x='layer', y='IBC_V', hue='model_type', marker='o', ax=axes[0], palette=['#1f77b4', '#d62728'])
    axes[0].set_title('Internal-Behavior Coupling (Valence) across layers')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Pearson r (z_V vs E[V])')
    
    # Plot Arousal
    sns.lineplot(data=df_std, x='layer', y='IBC_A', hue='model_type', marker='s', ax=axes[1], palette=['#1f77b4', '#d62728'])
    axes[1].set_title('Internal-Behavior Coupling (Arousal) across layers')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Pearson r (z_A vs E[A])')
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, "ibc_curves.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved IBC curves to {out_path}")

def plot_dose_response(df, layer, out_dir):
    """Plot Dose-Response (z_V and z_A) for a specific layer."""
    df_layer = df[(df['layer'] == layer) & (df['template'] == 'standard')].copy()
    
    if df_layer.empty:
        return
        
    # Restructure for seaborn barplot
    data = []
    for _, row in df_layer.iterrows():
        model = row['model_type']
        data.extend([
            {'Model': model, 'Intensity': 'Peak', 'Dimension': 'Valence', 'z_score': row['z_V_peak_mean']},
            {'Model': model, 'Intensity': 'Moderate', 'Dimension': 'Valence', 'z_score': row['z_V_moderate_mean']},
            {'Model': model, 'Intensity': 'Neutral', 'Dimension': 'Valence', 'z_score': row['z_V_neutral_mean']},
            {'Model': model, 'Intensity': 'Peak', 'Dimension': 'Arousal', 'z_score': row['z_A_peak_mean']},
            {'Model': model, 'Intensity': 'Moderate', 'Dimension': 'Arousal', 'z_score': row['z_A_moderate_mean']},
            {'Model': model, 'Intensity': 'Neutral', 'Dimension': 'Arousal', 'z_score': row['z_A_neutral_mean']},
        ])
        
    plot_df = pd.DataFrame(data)
    
    # Plot Valence
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=plot_df[plot_df['Dimension'] == 'Valence'], x='Intensity', y='z_score', hue='Model', 
                order=['Neutral', 'Moderate', 'Peak'], palette=['#1f77b4', '#d62728'], ax=ax)
    ax.set_title(f'Valence Projection Score (z_V) by Intensity (Layer {layer})')
    ax.set_ylabel('Mean Projection Score')
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"dose_response_V_layer{layer}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Arousal
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=plot_df[plot_df['Dimension'] == 'Arousal'], x='Intensity', y='z_score', hue='Model', 
                order=['Neutral', 'Moderate', 'Peak'], palette=['#1f77b4', '#d62728'], ax=ax)
    ax.set_title(f'Arousal Projection Score (z_A) by Intensity (Layer {layer})')
    ax.set_ylabel('Mean Projection Score')
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"dose_response_A_layer{layer}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Dose-Response barplots for layer {layer} to {out_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="v2/results/derived/phase1_preliminary")
    parser.add_argument("--out-dir", type=str, default="v2/results/derived/phase1_preliminary/plots")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    set_style()
    
    ibc_path = os.path.join(args.data_dir, "ibc_results.csv")
    dose_path = os.path.join(args.data_dir, "dose_response_results.csv")
    
    if os.path.exists(ibc_path):
        df_ibc = pd.read_csv(ibc_path)
        plot_ibc_curves(df_ibc, args.out_dir)
        
        # Identify the layer with max IBC in Base model for Dose-Response plot
        base_ibc = df_ibc[(df_ibc['model_type'] == 'Base') & (df_ibc['template'] == 'standard')]
        if not base_ibc.empty:
            max_v_layer = base_ibc.loc[base_ibc['IBC_V'].idxmax()]['layer']
            max_a_layer = base_ibc.loc[base_ibc['IBC_A'].idxmax()]['layer']
            
            if os.path.exists(dose_path):
                df_dose = pd.read_csv(dose_path)
                plot_dose_response(df_dose, layer=int(max_v_layer), out_dir=args.out_dir)
                if max_v_layer != max_a_layer:
                    plot_dose_response(df_dose, layer=int(max_a_layer), out_dir=args.out_dir)
    else:
        print(f"File not found: {ibc_path}")

if __name__ == "__main__":
    main()
