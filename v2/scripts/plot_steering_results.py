#!/usr/bin/env python3
import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def set_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def plot_steering_slope(df_base, df_instruct, direction, out_dir):
    df_base['model_type'] = 'Base'
    df_instruct['model_type'] = 'Instruct'
    df = pd.concat([df_base, df_instruct], ignore_index=True)
    
    layers = sorted(df['intervention_layer'].unique())
    
    # Target variable based on intervention direction
    target_var = 'E_v' if direction == 'valence' else 'E_a'
    target_label = 'E[V_{self}]' if direction == 'valence' else 'E[A_{self}]'
    
    for layer in layers:
        df_layer = df[df['intervention_layer'] == layer]
        
        plt.figure(figsize=(8, 6))
        sns.lineplot(
            data=df_layer, 
            x='intervention_alpha', 
            y=target_var, 
            hue='model_type',
            style='model_type',
            markers=True,
            err_style="bars", 
            errorbar=('ci', 95)
        )
        
        plt.title(f'Steering Slope (Layer {layer}, Direction: {direction.capitalize()})')
        plt.xlabel('Intervention Strength (α)')
        plt.ylabel(target_label)
        plt.legend(title='Model')
        
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"steering_slope_{direction}_layer{layer}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-file", type=str, required=True, help="Path to Base model results jsonl")
    parser.add_argument("--instruct-file", type=str, required=True, help="Path to Instruct model results jsonl")
    parser.add_argument("--direction", type=str, choices=["valence", "arousal"], default="valence")
    parser.add_argument("--out-dir", type=str, default="v2/results/derived/phase2_steering/plots")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    set_style()
    
    if os.path.exists(args.base_file) and os.path.exists(args.instruct_file):
        df_base = load_jsonl(args.base_file)
        df_instruct = load_jsonl(args.instruct_file)
        plot_steering_slope(df_base, df_instruct, args.direction, args.out_dir)
    else:
        print("One or both input files not found.")

if __name__ == "__main__":
    main()
