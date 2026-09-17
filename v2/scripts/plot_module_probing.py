#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from v2.scripts.run_steering_and_likelihood import compute_contrastive_directions

def set_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12

def load_npz(path):
    d = dict(np.load(path, allow_pickle=True))
    for k, v in d.items():
        if v.shape == (): d[k] = v.item()
    return d

def compute_module_projections(module_states, df_test, d_V, layer_count=28):
    # Returns DataFrame with layer, module, mean_z_V
    results = []
    
    # Pre-filter UIDs to avoid missing keys
    uids = df_test['id'].tolist()
    
    for l in range(layer_count):
        for mod in ['attn', 'mlp']:
            z_vals = []
            for uid in uids:
                key = f"{uid}_layer{l}_{mod}"
                if key in module_states:
                    rep = module_states[key]
                    z = np.dot(rep, d_V)
                    z_vals.append(z)
            if len(z_vals) > 0:
                results.append({
                    "layer": l,
                    "module": mod,
                    "mean_z_V": np.mean(z_vals)
                })
    return pd.DataFrame(results)

def plot_module_diff(df_base, df_inst, out_dir):
    df_base['model_type'] = 'Base'
    df_inst['model_type'] = 'Instruct'
    df = pd.concat([df_base, df_inst], ignore_index=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    # 1. Attention Module
    df_attn = df[df['module'] == 'attn']
    sns.lineplot(data=df_attn, x='layer', y='mean_z_V', hue='model_type', style='model_type', markers=True, ax=axes[0])
    axes[0].set_title('Attention Output Projection (Mean z_V)')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Mean Projection Score (z_V)')
    axes[0].legend(title='Model')
    
    # 2. MLP Module
    df_mlp = df[df['module'] == 'mlp']
    sns.lineplot(data=df_mlp, x='layer', y='mean_z_V', hue='model_type', style='model_type', markers=True, ax=axes[1])
    axes[1].set_title('MLP Output Projection (Mean z_V)')
    axes[1].set_xlabel('Layer')
    axes[1].legend(title='Model')
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, "module_probing_plot.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-modules", type=str, default="v2/results/derived/phase4_circuit/Qwen_Qwen2.5-1.5B_module_states.npz")
    parser.add_argument("--inst-modules", type=str, default="v2/results/derived/phase4_circuit/Qwen_Qwen2.5-1.5B-Instruct_module_states.npz")
    parser.add_argument("--train-hs", type=str, default="v2/results/raw/aipsy/Qwen_Qwen2.5-1.5B_hidden_states.npz")
    parser.add_argument("--data-dir", type=str, default="v2/data/processed/aipsy")
    parser.add_argument("--out-dir", type=str, default="v2/results/derived/phase4_circuit/plots")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    set_style()
    
    # 1. Load Data
    train_df = pd.read_csv(os.path.join(args.data_dir, "train_strict.csv"))
    test_df = pd.read_csv(os.path.join(args.data_dir, "test_strict.csv")).head(50) # Matching the limit used in extraction
    
    # 2. Compute Direction d_V from Base layer 27 (or whichever layer is stable)
    # Layer 24 is often used as a good representation layer
    hs_train = load_npz(args.train_hs)
    d_V, _, _, _ = compute_contrastive_directions(hs_train, train_df, layer=24)
    
    if d_V is None:
        print("Failed to compute d_V.")
        return
        
    # 3. Load Module States and compute projections
    mod_base = load_npz(args.base_modules)
    mod_inst = load_npz(args.inst_modules)
    
    df_proj_base = compute_module_projections(mod_base, test_df, d_V)
    df_proj_inst = compute_module_projections(mod_inst, test_df, d_V)
    
    # 4. Plot
    plot_module_diff(df_proj_base, df_proj_inst, args.out_dir)

if __name__ == "__main__":
    main()
