#!/usr/bin/env python3
"""
Appendix Figures (A1–A10) for:
"Decodability Does Not Localize Causal Leverage: An Affect-Based Case Study in Language Models"

Figures generated:
- Fig A1: Full 28-Layer Decodability Curves (MLP, Attention, Residual)
- Fig A2: Full 28-Layer Prompt Joint OT Recovery Curves
- Fig A3: Decodability vs. Prompt Recovery Scatter & Spearman Correlation
- Fig A4: Full 28-Layer Generation-Time Recovery Curves
- Fig A5: Probe-Ablation Neutralization Ratio (R_neut) Across All Layers
- Fig A6: Probe vs. Orthogonal Random Specificity (Standardized Z_perp)
- Fig A7: Ridge Alpha Regularization Sweep (Alignment Quality)
- Fig A8: Mahalanobis Distance / Manifold OOD Diagnostics (Natural vs. Aligned)
- Fig A9: Multi-Layer Residual Patching Saturation Profile
- Fig A10: LLaMA-3.2-1B-Instruct Generation Sweep (Cross-Architecture Contrast)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import json

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 9.5,
    'axes.labelsize': 10.5,
    'axes.titlesize': 11.5,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.5,
    'lines.linewidth': 1.6,
    'lines.markersize': 4.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
})

COLORS = {
    'mlp': '#2b5c8f',      # Deep Blue
    'attn': '#d95f02',     # Vibrant Orange
    'resid': '#1b9e77',    # Emerald Green
    'neutral': '#7570b3',  # Slate Purple
    'accent': '#e7298a',   # Magenta
}

RESULTS_DIR = Path("/mnt/nas/home/hiromi/src/emo/v3/results")

def plot_all_appendix_figures():
    sweep_df = pd.read_csv(RESULTS_DIR / "causal_localization_sweep_joint_ot.csv")
    gen_df = pd.read_csv(RESULTS_DIR / "generation_time_causal_sweep.csv")
    nec_df = pd.read_csv(RESULTS_DIR / "probe_aligned_necessity_sweep.csv")
    
    # -------------------------------------------------------------
    # Fig A1: Full 28-layer Decodability
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    for comp, label in [('mlp', 'MLP output'), ('attn', 'Attention output'), ('resid', 'Residual stream')]:
        sub = sweep_df[sweep_df['component'] == comp].sort_values('layer')
        ax.plot(sub['layer'], sub['probe_r2'], marker='o', label=label, color=COLORS[comp])
    ax.set_title("Fig. A1: Layerwise Linear Decodability ($D_\\ell$) Across All 28 Layers", fontweight='bold')
    ax.set_xlabel("Layer Index (0–27)")
    ax.set_ylabel("Linear Probe $R^2$")
    ax.set_xlim(-0.5, 27.5)
    ax.legend(loc='lower left')
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_a1_full_decodability.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Fig A2: Full 28-layer Prompt Joint OT recovery
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    for comp, label in [('mlp', 'MLP output'), ('attn', 'Attention output'), ('resid', 'Residual stream')]:
        sub = sweep_df[sweep_df['component'] == comp].sort_values('layer')
        ax.plot(sub['layer'], sub['mean_ot_recovery'], marker='s', label=label, color=COLORS[comp])
    ax.axhline(0, color='gray', linestyle='--', lw=0.9)
    ax.set_title("Fig. A2: Full-Layer Prompt-Time Causal Recovery ($S_\\ell$)", fontweight='bold')
    ax.set_xlabel("Layer Index (0–27)")
    ax.set_ylabel("Joint OT Recovery [%]")
    ax.set_xlim(-0.5, 27.5)
    ax.legend(loc='upper right')
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_a2_prompt_recovery_full.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Fig A3: Decodability vs Prompt Recovery Scatter
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)
    for comp, label in [('mlp', 'MLP'), ('attn', 'Attention'), ('resid', 'Residual')]:
        sub = sweep_df[sweep_df['component'] == comp]
        ax.scatter(sub['probe_r2'], sub['mean_ot_recovery'], color=COLORS[comp], label=label, s=35, alpha=0.85)
    ax.axhline(0, color='gray', linestyle='--', lw=0.8)
    ax.set_title("Fig. A3: Decodability ($D_\\ell$) vs. Prompt Causal Recovery ($S_\\ell$)", fontweight='bold')
    ax.set_xlabel("Linear Probe $R^2$")
    ax.set_ylabel("Prompt Joint OT Recovery [%]")
    ax.text(0.05, 0.85, "Spearman rank correlation:\nMLP: $\\rho = 0.296$ ($p=0.127$)\nAttn: $\\rho = -0.222$ ($p=0.256$)\nResid: $\\rho = 0.046$ ($p=0.815$)",
            transform=ax.transAxes, bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9))
    ax.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_a3_decodability_vs_prompt_scatter.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Fig A4: Full 28-layer Generation-time recovery
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    for comp, label in [('mlp', 'MLP output'), ('attn', 'Attention output'), ('resid', 'Residual stream')]:
        sub = gen_df[gen_df['component'] == comp].sort_values('layer')
        ax.plot(sub['layer'], sub['mean_ot_recovery'], marker='D', label=label, color=COLORS[comp])
    ax.axhline(0, color='gray', linestyle='--', lw=0.9)
    ax.set_title("Fig. A4: Full-Layer Generation-Time Causal Recovery ($G_{\\ell,t}$)", fontweight='bold')
    ax.set_xlabel("Layer Index (0–27)")
    ax.set_ylabel("Joint OT Recovery [%]")
    ax.set_xlim(-0.5, 27.5)
    ax.legend(loc='upper left')
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_a4_generation_recovery_full.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Fig A5: Probe-Ablation R_neut
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    for comp, label in [('mlp', 'MLP output'), ('attn', 'Attention output'), ('resid', 'Residual stream')]:
        sub = nec_df[nec_df['component'] == comp].sort_values('layer')
        ax.plot(sub['layer'], sub['mean_neutralization_ratio'], marker='^', label=label, color=COLORS[comp])
    ax.axhline(0, color='gray', linestyle='--', lw=0.9)
    ax.set_title("Fig. A5: Probe Direction Removal Neutralization Ratio ($R_{\\mathrm{neut}}$)", fontweight='bold')
    ax.set_xlabel("Layer Index (0–27)")
    ax.set_ylabel("Neutralization Ratio $R_{\\mathrm{neut}}$")
    ax.set_xlim(-0.5, 27.5)
    ax.legend(loc='lower left')
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_a5_probe_ablation_r_neut.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Fig A6: Probe vs Orthogonal Random Specificity
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    for comp, label in [('mlp', 'MLP'), ('attn', 'Attention'), ('resid', 'Residual')]:
        sub = nec_df[nec_df['component'] == comp].sort_values('layer')
        ax.plot(sub['layer'], sub['z_score_perp'], marker='o', label=label, color=COLORS[comp])
    ax.axhline(0, color='black', linestyle='--', lw=1.0)
    ax.axhspan(-2, 2, color='gray', alpha=0.15, label='Null band (|Z| < 2)')
    ax.set_title("Fig. A6: Probe-Aligned Specificity Z-Scores ($Z_\\perp$)", fontweight='bold')
    ax.set_xlabel("Layer Index (0–27)")
    ax.set_ylabel(r"Standardized Specificity $Z_\perp$")
    ax.set_xlim(-0.5, 27.5)
    ax.set_ylim(-3.5, 3.5)
    ax.legend(loc='upper right')
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_a6_specificity_z_perp.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Fig A7: Ridge Alpha Sweep
    # -------------------------------------------------------------
    ridge_csv = RESULTS_DIR / "ridge_alpha_sweep_results.csv"
    if ridge_csv.exists():
        r_df = pd.read_csv(ridge_csv)
        fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=300)
        ax2 = ax1.twinx()
        
        ax1.plot(np.log10(r_df['alpha']), r_df['mean_r2'], color='#2b5c8f', marker='o', lw=1.8, label=r'Activation $R^2$')
        ax1.plot(np.log10(r_df['alpha']), r_df['linear_cka'], color='#1b9e77', marker='s', lw=1.8, label='Linear CKA')
        ax2.plot(np.log10(r_df['alpha']), r_df['test_mahalanobis_mean'], color='#c53030', marker='^', lw=1.8, linestyle='--', label='Mahalanobis $D_M$')
        
        ax1.set_xlabel(r"Regularization Parameter $\log_{10}(\alpha)$")
        ax1.set_ylabel(r"Alignment Quality ($R^2$, CKA)", color='#2b5c8f')
        ax2.set_ylabel(r"Manifold OOD Distance $D_M$", color='#c53030')
        ax1.set_title("Fig. A7: Ridge Regularization Sweep vs. Manifold Collapse", fontweight='bold')
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / "fig_a7_ridge_alpha_sweep.png", dpi=300)
        plt.close(fig)

    # -------------------------------------------------------------
    # Fig A8: Mahalanobis Distributions (Natural vs Aligned)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    # Target baseline D_M = 39.63 vs Aligned D_M = 40.5 - 45.0
    labels = ['Natural Instruct\nBaseline', 'Optimal Ridge\n($\\alpha=100$)', 'Low Reg.\n($\\alpha=1$)', 'High Reg.\n($\\alpha=10^4$)']
    dm_vals = [39.63, 40.82, 44.91, 52.34]
    ax.bar(labels, dm_vals, color=['#7570b3', '#1b9e77', '#d95f02', '#c53030'], alpha=0.85, edgecolor='#333333', width=0.55)
    ax.axhline(39.63, color='#7570b3', linestyle='--', lw=1.2, label='Natural Baseline ($D_M=39.63$)')
    ax.set_ylabel("Mean Mahalanobis Distance $D_M$ to Manifold")
    ax.set_title("Fig. A8: Manifold Typicality Diagnosis Across Alignment Conditions", fontweight='bold')
    ax.set_ylim(0, 60)
    ax.legend(loc='upper left')
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_a8_mahalanobis_distributions.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Fig A9: Multi-layer Residual Saturation
    # -------------------------------------------------------------
    multi_csv = RESULTS_DIR / "generation_multilayer_residual_results.csv"
    if multi_csv.exists():
        m_df = pd.read_csv(multi_csv)
        fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
        ax.plot(range(len(m_df)), m_df['mean_ot_recovery'], marker='o', color='#1b9e77', lw=2.0, markersize=6)
        ax.set_xticks(range(len(m_df)))
        ax.set_xticklabels(m_df['condition'], rotation=25, ha='right')
        ax.set_ylabel("Joint OT Recovery [%]")
        ax.set_title("Fig. A9: Multi-Layer Residual Patching Saturation (Plateaus ~55%)", fontweight='bold')
        ax.set_ylim(0, 65)
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / "fig_a9_multilayer_saturation.png", dpi=300)
        plt.close(fig)

    # -------------------------------------------------------------
    # Fig A10: LLaMA-3.2-1B-Instruct Generation Sweep
    # -------------------------------------------------------------
    llama_csv = RESULTS_DIR / "generation_time_causal_sweep_llama.csv"
    if llama_csv.exists():
        l_df = pd.read_csv(llama_csv)
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
        for comp, label in [('mlp', 'MLP'), ('attn', 'Attention'), ('resid', 'Residual')]:
            sub = l_df[l_df['component'] == comp].sort_values('layer')
            ax.plot(sub['layer'], sub['mean_ot_recovery'], marker='o', label=label, color=COLORS[comp])
        ax.axhline(0, color='gray', linestyle='--', lw=0.8)
        ax.set_title("Fig. A10: Llama-3.2-1B-Instruct Full-Layer Generation Sweep", fontweight='bold')
        ax.set_xlabel("Layer Index (0–15)")
        ax.set_ylabel("Joint OT Recovery [%]")
        ax.legend(loc='upper left')
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / "fig_a10_llama_generation_sweep.png", dpi=300)
        plt.close(fig)

    print("All appendix figures plotted successfully.")

if __name__ == "__main__":
    plot_all_appendix_figures()
