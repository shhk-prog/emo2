#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_gating():
    df_std = pd.DataFrame([json.loads(l) for l in open("v2/results/raw/aipsy/Qwen_Qwen2.5-1.5B-Instruct_standard_results.jsonl")])
    df_nat = pd.DataFrame([json.loads(l) for l in open("v2/results/derived/phase3b_stress_test/output_gating_natural_language.jsonl")])
    
    df_std['Condition'] = 'Instruct (JSON)'
    df_nat['Condition'] = 'Instruct (Natural)'
    
    df_all = pd.concat([df_std, df_nat])
    
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=df_all, x='Condition', y='E_v')
    plt.axhline(5, color='gray', linestyle='--')
    plt.title("Output-Gating Test: E_v Distribution")
    plt.ylabel("Expected Valence (E_v)")
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/output_gating_plot.png")
    plt.close()

def plot_patching():
    df_base = pd.DataFrame([json.loads(l) for l in open("v2/results/raw/aipsy/Qwen_Qwen2.5-1.5B_standard_results.jsonl")])
    df_inst = pd.DataFrame([json.loads(l) for l in open("v2/results/raw/aipsy/Qwen_Qwen2.5-1.5B-Instruct_standard_results.jsonl")])
    df_patch = pd.DataFrame([json.loads(l) for l in open("v2/results/derived/phase4_circuit/patching_layer20_mlp.jsonl")])
    
    df_base['Model'] = 'Base'
    df_inst['Model'] = 'Instruct'
    df_patch['Model'] = 'Instruct\n+ Base L20 MLP'
    
    df_all = pd.concat([df_base, df_inst, df_patch])
    
    from scipy.stats import wasserstein_distance
    import numpy as np
    
    # Calculate Wasserstein Distance to Base
    wd_inst = wasserstein_distance(df_inst['E_v'], df_base['E_v'])
    wd_patch = wasserstein_distance(df_patch['E_v'], df_base['E_v'])
    
    print(f"Wasserstein(Instruct, Base): {wd_inst:.4f}")
    print(f"Wasserstein(Patched, Base): {wd_patch:.4f}")
    
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df_base, x='E_v', label='Base', color='blue', fill=True, alpha=0.3)
    sns.kdeplot(data=df_inst, x='E_v', label='Instruct', color='red', fill=True, alpha=0.3)
    sns.kdeplot(data=df_patch, x='E_v', label='Instruct + Base L20 MLP', color='green', fill=True, alpha=0.3)
    plt.axvline(5, color='gray', linestyle='--')
    plt.title(f"Activation Patching: Distributional Recovery\nWD(Inst, Base)={wd_inst:.3f}, WD(Patch, Base)={wd_patch:.3f}")
    plt.xlabel("Expected Valence (E_v)")
    plt.legend()
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/activation_patching_plot.png")
    plt.close()

if __name__ == "__main__":
    plot_gating()
    plot_patching()
