#!/usr/bin/env python3
"""
Publication-Grade Figures for:
"Decodability Does Not Localize Causal Leverage: An Affect-Based Case Study in Language Models"

Figures generated:
- Figure 1: Overall Experimental Framework & Core Hypothesis
- Figure 2: Four-Panel Representational-Causal Profile (D_l, S_l, N_l, G_{l,t})
- Figure 3: Direct Peak-Site Dissociation (L15 MLP vs L24 Residual)
- Figure 4: Greedy Collapse vs Distributional Sensitivity (Base vs Instruct)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from pathlib import Path

# Professional styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'lines.linewidth': 1.8,
    'lines.markersize': 5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
})

COLORS = {
    'mlp': '#2b5c8f',      # Deep Blue
    'attn': '#d95f02',     # Vibrant Vermilion / Orange
    'resid': '#1b9e77',    # Emerald Green
    'neutral': '#7570b3',  # Slate Purple
    'accent': '#e7298a',   # Magenta Accent
    'dark': '#222222',
    'light': '#f8f9fa',
    'box_bg': '#f0f4f8',
    'alert_bg': '#fdf2f2',
}

from pathlib import Path
from scipy import stats

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# ==============================================================================
# Figure 1: Overall Experimental Framework & Conceptual Hypothesis
# ==============================================================================
def plot_figure1():
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.5)

    # 1. Input Stimuli Box
    box_input = patches.FancyBboxPatch((0.4, 3.8), 2.2, 1.2, boxstyle="round,pad=0.1",
                                      fc="#e8edf5", ec="#2b5c8f", lw=1.5)
    ax.add_patch(box_input)
    ax.text(1.5, 4.6, "Input Stimuli", ha="center", va="center", fontweight="bold", fontsize=11, color="#1a365d")
    ax.text(1.5, 4.15, "AIPsy-Affect Strict\n(Affective vs. Neutral)", ha="center", va="center", fontsize=9, color="#333333")

    # Arrow to Model
    ax.annotate("", xy=(3.1, 4.4), xytext=(2.6, 4.4),
                arrowprops=dict(arrowstyle="->", lw=2, color="#2b5c8f"))

    # 2. Model Box
    box_model = patches.FancyBboxPatch((3.1, 3.6), 2.8, 1.5, boxstyle="round,pad=0.1",
                                       fc="#f0f4f8", ec="#333333", lw=2)
    ax.add_patch(box_model)
    ax.text(4.5, 4.65, "Qwen2.5-1.5B-Instruct", ha="center", va="center", fontweight="bold", fontsize=11, color="#111111")
    ax.text(4.5, 4.2, "28 Transformer Layers\n(MLP, Attention, Residual)", ha="center", va="center", fontsize=8.5, color="#555555")
    ax.text(4.5, 3.8, "Self-Report Evaluation Task", ha="center", va="center", fontsize=8, color="#2b5c8f", style="italic")

    # Arrow Downwards & Split
    ax.annotate("", xy=(4.5, 3.0), xytext=(4.5, 3.6),
                arrowprops=dict(arrowstyle="->", lw=2, color="#333333"))

    # 3. Prompt-time branch
    box_prompt = patches.FancyBboxPatch((0.5, 0.4), 4.2, 2.5, boxstyle="round,pad=0.1",
                                        fc="#ffffff", ec="#2b5c8f", lw=1.5)
    ax.add_patch(box_prompt)
    ax.text(2.6, 2.65, "Prompt-Time Interventions (Token $t_{\\mathrm{last}}$)", ha="center", va="center",
            fontweight="bold", fontsize=10, color="#2b5c8f")
    
    # Prompt interventions items
    ax.text(0.8, 2.05, "1. Linear Probing ($D_\\ell$):", fontweight="bold", fontsize=9, color="#333333")
    ax.text(1.1, 1.80, "$\to$ High decodability in middle layers ($R^2_{\mathrm{MLP},15}=0.561$)", fontsize=8.5, color="#555555")

    ax.text(0.8, 1.40, "2. Matched Substitution ($S_\\ell$):", fontweight="bold", fontsize=9, color="#333333")
    ax.text(1.1, 1.15, "$\to$ Local causal recovery $\approx 0\%$ across all 28 layers", fontsize=8.5, color="#555555")

    ax.text(0.8, 0.75, "3. Direction Ablation ($N_\\ell$):", fontweight="bold", fontsize=9, color="#333333")
    ax.text(1.1, 0.50, "$\to$ Probe removal fails to induce neutralization ($Z_\\perp \approx 0$)", fontsize=8.5, color="#555555")

    # Connector to Prompt branch
    ax.plot([4.5, 2.6, 2.6], [3.0, 3.0, 2.9], color="#333333", lw=1.5)

    # 4. Generation-time branch
    box_gen = patches.FancyBboxPatch((5.1, 0.4), 2.8, 2.5, boxstyle="round,pad=0.1",
                                     fc="#ffffff", ec="#1b9e77", lw=1.5)
    ax.add_patch(box_gen)
    ax.text(6.5, 2.65, "Generation-Time Interventions", ha="center", va="center",
            fontweight="bold", fontsize=10, color="#1b9e77")
    
    ax.text(5.3, 1.95, "Matched Substitution ($G_{\\ell,t}$):", fontweight="bold", fontsize=9, color="#333333")
    ax.text(5.4, 1.55, "$\to$ Patched at output token\n      prediction stages", fontsize=8.5, color="#555555")
    ax.text(5.4, 1.00, "$\to$ Late Residual emergence:\n      $G_{\\mathrm{RESID},24} = 53.24\%$",
            fontsize=8.5, fontweight="bold", color="#1b9e77")

    # Connector to Gen branch
    ax.plot([4.5, 6.5, 6.5], [3.0, 3.0, 2.9], color="#333333", lw=1.5)

    # 5. Core Hypothesis / Finding Box (Right Side)
    box_hyp = patches.FancyBboxPatch((8.2, 0.8), 2.5, 4.0, boxstyle="round,pad=0.15",
                                     fc="#fdf2f2", ec="#c53030", lw=2)
    ax.add_patch(box_hyp)
    ax.text(9.45, 4.4, "Central Finding", ha="center", va="center", fontweight="bold", fontsize=12, color="#9b2c2c")
    
    hyp_text = (
        "Decodability Does Not\n"
        "Localize Causal Leverage\n\n"
        r"$\mathbf{Decodability}\ (D_\ell)$" + "\n"
        r"$\neq$" + "\n"
        r"$\mathbf{Causal\ Leverage}\ (S_\ell, G_{\ell,t})$" + "\n\n"
        "• Representation is accessible\n"
        "  where it has no local lever\n"
        "  (L15 MLP: $R^2=0.56, S=0.5\%$)\n\n"
        "• Causal leverage emerges\n"
        "  late during generation\n"
        "  (L24 RESID: $R^2=0.15, G=53.2\%$)"
    )
    ax.text(9.45, 2.5, hyp_text, ha="center", va="center", fontsize=8.5, color="#2d3748", linespacing=1.2)

    # Big double arrow from branches to hypothesis box
    ax.annotate("", xy=(8.1, 2.8), xytext=(7.95, 2.8),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#c53030"))

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "figure1_overall_framework.png", dpi=300, bbox_inches='tight')
    fig.savefig(RESULTS_DIR / "figure1_overall_framework.pdf", bbox_inches='tight')
    plt.close(fig)
    print("Saved Figure 1.")

# ==============================================================================
# Figure 2: Four-Panel Representational-Causal Profile
# ==============================================================================
def plot_figure2():
    sweep_df = pd.read_csv(RESULTS_DIR / "causal_localization_sweep_joint_ot.csv")
    gen_df = pd.read_csv(RESULTS_DIR / "generation_time_causal_sweep.csv")
    nec_df = pd.read_csv(RESULTS_DIR / "probe_aligned_necessity_sweep.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
    plt.subplots_adjust(hspace=0.32, wspace=0.25)

    # -------------------------------------------------------------
    # Panel A: Layerwise Decodability (D_l)
    # -------------------------------------------------------------
    ax_a = axes[0, 0]
    for comp, label in [('mlp', 'MLP output'), ('attn', 'Attention output'), ('resid', 'Residual stream')]:
        sub = sweep_df[sweep_df['component'] == comp].sort_values('layer')
        ax_a.plot(sub['layer'], sub['probe_r2'], marker='o', label=label, color=COLORS[comp])

    # Peak markers
    ax_a.plot(15, 0.5610, marker='*', markersize=14, color='gold', markeredgecolor='#333333', zorder=5)
    ax_a.annotate(r"$\mathbf{L15\ MLP\ Peak}$" + "\n" + r"$R^2_{\mathrm{MLP},15} = 0.561$", 
                  xy=(15, 0.5610), xytext=(10.5, 0.61),
                  arrowprops=dict(arrowstyle="->", color=COLORS['mlp'], lw=1.2),
                  fontsize=8.5, fontweight='bold', color=COLORS['mlp'],
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=COLORS['mlp'], alpha=0.9))

    ax_a.plot(18, 0.5495, marker='s', markersize=7, color=COLORS['attn'], zorder=5)
    ax_a.annotate("L18 Attn ($R^2=0.550$)", xy=(18, 0.5495), xytext=(17.5, 0.47),
                  fontsize=8, color=COLORS['attn'])

    ax_a.plot(14, 0.5016, marker='^', markersize=7, color=COLORS['resid'], zorder=5)
    ax_a.annotate("L14 Resid ($R^2=0.502$)", xy=(14, 0.5016), xytext=(8, 0.44),
                  fontsize=8, color=COLORS['resid'])

    ax_a.set_title(r"$\mathbf{(a)\ Layerwise\ Linear\ Decodability\ (D_\ell)}$", loc='left')
    ax_a.set_xlabel("Layer Index (0–27)")
    ax_a.set_ylabel(r"Linear Probe $R^2$")
    ax_a.set_xlim(-0.5, 27.5)
    ax_a.set_ylim(0.15, 0.68)
    ax_a.legend(loc='lower left', frameon=True, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel B: Prompt-Time Local Causal Recovery (S_l)
    # -------------------------------------------------------------
    ax_b = axes[0, 1]
    for comp, label in [('mlp', 'MLP output'), ('attn', 'Attention output'), ('resid', 'Residual stream')]:
        sub = sweep_df[sweep_df['component'] == comp].sort_values('layer')
        ax_b.plot(sub['layer'], sub['mean_ot_recovery'], marker='s', label=label, color=COLORS[comp])

    ax_b.axhline(0, color='gray', linestyle='--', linewidth=0.9, alpha=0.8)
    
    # Gap annotation at L15 MLP
    ax_b.plot(15, 1.0008, marker='*', markersize=12, color='gold', markeredgecolor='#333333', zorder=5)
    ax_b.annotate(r"$\mathbf{L15\ MLP:}$" + "\n" + r"$D_{15}=0.561 \implies S_{15} \approx 0.5\text{--}1.0\%$", 
                  xy=(15, 1.00), xytext=(8, 4.2),
                  arrowprops=dict(arrowstyle="->", color=COLORS['mlp'], lw=1.2),
                  fontsize=8.5, fontweight='bold', color=COLORS['mlp'],
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=COLORS['mlp'], alpha=0.9))

    ax_b.set_title(r"$\mathbf{(b)\ Prompt\text{-}Time\ Causal\ Recovery\ (S_\ell)}$", loc='left')
    ax_b.set_xlabel("Layer Index (0–27)")
    ax_b.set_ylabel("Joint OT Recovery [%]")
    ax_b.set_xlim(-0.5, 27.5)
    ax_b.set_ylim(-6.0, 7.5)
    ax_b.legend(loc='upper right', frameon=True, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel C: Probe-Aligned Necessity (N_l)
    # -------------------------------------------------------------
    ax_c = axes[1, 0]
    for comp, label in [('mlp', 'MLP output'), ('attn', 'Attention output'), ('resid', 'Residual stream')]:
        sub = nec_df[nec_df['component'] == comp].sort_values('layer')
        ax_c.plot(sub['layer'], sub['z_score_perp'], marker='^', label=label, color=COLORS[comp], alpha=0.85)

    ax_c.axhline(0, color='black', linestyle='--', linewidth=1.0)
    ax_c.axhspan(-2, 2, color='gray', alpha=0.15, label='Null band (|Z| < 2)')

    ax_c.set_title(r"$\mathbf{(c)\ Probe\text{-}Aligned\ Necessity\ (N_\ell:\ Standardized\ Z_\perp)}$", loc='left')
    ax_c.set_xlabel("Layer Index (0–27)")
    ax_c.set_ylabel(r"Specificity $Z_\perp$ vs. Orthogonal")
    ax_c.set_xlim(-0.5, 27.5)
    ax_c.set_ylim(-3.5, 3.5)
    ax_c.text(0.5, -2.8, "Probe direction removal does not produce systematic neutralization\n(All 84 tests non-significant after FDR, q > 0.85)", 
              fontsize=7.5, color='#444444', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.8))
    ax_c.legend(loc='upper right', frameon=True, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel D: Generation-Time Causal Recovery (G_{l,t})
    # -------------------------------------------------------------
    ax_d = axes[1, 1]
    for comp, label in [('mlp', 'MLP output'), ('attn', 'Attention output'), ('resid', 'Residual stream')]:
        sub = gen_df[gen_df['component'] == comp].sort_values('layer')
        ax_d.plot(sub['layer'], sub['mean_ot_recovery'], marker='D', label=label, color=COLORS[comp])

    ax_d.axhline(0, color='gray', linestyle='--', linewidth=0.9, alpha=0.8)

    # Key points: L15 MLP vs L24 Residual
    ax_d.plot(15, -0.0597, marker='X', markersize=10, color='crimson', zorder=5)
    ax_d.annotate(r"$\mathbf{L15\ MLP:}\ -0.06\%$", xy=(15, 0), xytext=(10, -8),
                  arrowprops=dict(arrowstyle="->", color='crimson', lw=1.2),
                  fontsize=8.5, fontweight='bold', color='crimson',
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec='crimson', alpha=0.9))

    ax_d.plot(24, 53.2432, marker='*', markersize=15, color='gold', markeredgecolor='#1b9e77', lw=1.5, zorder=5)
    ax_d.annotate(r"$\mathbf{L24\ Resid\ Surge}$" + "\n" + r"$G_{\mathrm{RESID},24} = 53.24\%$", 
                  xy=(24, 53.24), xytext=(15, 48),
                  arrowprops=dict(arrowstyle="->", color=COLORS['resid'], lw=1.5),
                  fontsize=9, fontweight='bold', color=COLORS['resid'],
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=COLORS['resid'], alpha=0.9))

    ax_d.set_title(r"$\mathbf{(d)\ Generation\text{-}Time\ Causal\ Recovery\ (G_{\ell,t})}$", loc='left')
    ax_d.set_xlabel("Layer Index (0–27)")
    ax_d.set_ylabel("Joint OT Recovery [%]")
    ax_d.set_xlim(-0.5, 27.5)
    ax_d.set_ylim(-15, 65)
    ax_d.legend(loc='upper left', frameon=True, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "figure2_four_panel_profile.png", dpi=300, bbox_inches='tight')
    fig.savefig(RESULTS_DIR / "figure2_four_panel_profile.pdf", bbox_inches='tight')
    plt.close(fig)
    print("Saved Figure 2.")

# ==============================================================================
# Figure 3: Direct Peak-Site Dissociation (L15 MLP vs L24 Residual)
# ==============================================================================
def plot_figure3():
    pair_df = pd.read_csv(RESULTS_DIR / "focused_causal_sweep_39pairs_pair_level.csv")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300, gridspec_kw={'width_ratios': [1, 1.3]})
    plt.subplots_adjust(wspace=0.35)

    # Left: Direct Summary Bars / Contrast
    metrics = ['Decodability\n($R^2$)', 'Prompt Rec.\n($S_\\ell$, %)', 'Generation Rec.\n($G_{\\ell,t}$, %)']
    l15_vals = [0.5610, 0.5101, -0.0597]
    l24_vals = [0.1470, -0.2635, 53.2432]

    # Bar chart for Generation Recovery
    sites = ['Layer 15 MLP\n(Decodability Peak)', 'Layer 24 Residual\n(Causal Leverage Peak)']
    gen_means = [-0.06, 53.24]
    cis_low = [-2.0, 45.7]
    cis_high = [1.8, 60.5]
    err_low = [gen_means[0] - cis_low[0], gen_means[1] - cis_low[1]]
    err_high = [cis_high[0] - gen_means[0], cis_high[1] - gen_means[1]]

    colors_bar = [COLORS['mlp'], COLORS['resid']]
    bars = ax1.bar(sites, gen_means, yerr=[err_low, err_high], capsize=6, color=colors_bar, alpha=0.85, edgecolor='#333333', lw=1.2)
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax1.set_ylabel("Generation Causal Recovery [%] (95% Bootstrap CI)")
    ax1.set_ylim(-10, 70)
    ax1.set_title("(a) Representative-Site Recovery Contrast", fontweight='bold', loc='left')

    for bar, val, r2 in zip(bars, gen_means, [0.561, 0.147]):
        height = bar.get_height()
        y_pos = height + 8 if height >= 0 else 4
        ax1.text(bar.get_x() + bar.get_width()/2., y_pos,
                 f"{val:+.1f}%\n($R^2={r2:.3f}$)",
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Right: Paired 39-Point Slope Chart
    ax2.set_title(r"(b) Pairwise Dissociation across 39 Test Pairs", fontweight='bold', loc='left')
    
    l15_pts = pair_df["l15_mlp_recovery"].values
    l24_pts = pair_df["l24_resid_recovery"].values
    
    # Plot individual pair lines
    for y1, y2 in zip(l15_pts, l24_pts):
        color = '#1b9e77' if y2 > y1 else '#e7298a'
        alpha = 0.45
        ax2.plot([0, 1], [y1, y2], color=color, alpha=alpha, lw=1.2, marker='o', markersize=4)

    # Plot mean bold line
    ax2.plot([0, 1], [np.mean(l15_pts), np.mean(l24_pts)], color='#111111', lw=3.0, marker='s', markersize=7, zorder=6,
             label=f"Mean: {np.mean(l15_pts):+.1f}% $\\to$ {np.mean(l24_pts):+.1f}%")

    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['L15 MLP\n(Decodability Peak)', 'L24 Residual\n(Late Leverage)'], fontsize=9.5, fontweight='bold')
    ax2.set_ylabel("Pairwise Joint OT Recovery [%]")
    ax2.set_ylim(-25, 95)
    
    # Annotate effect size and CI
    delta_g = 53.30
    ax2.annotate(r"$\mathbf{\Delta G = +53.30\%}$" + "\n" + r"$\mathbf{95\%\ CI:\ [+45.34\%,\ +61.16\%]}$" + "\n" + r"$p < 10^{-12}$ (paired $t$ / Wilcoxon)",
                 xy=(0.5, 68), xytext=(0.15, 75),
                 fontsize=8.5, fontweight='bold', color='#1a365d',
                 bbox=dict(boxstyle="round,pad=0.3", fc="#e8edf5", ec="#2b5c8f", lw=1.2))

    ax2.legend(loc='lower right', frameon=True, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "figure3_direct_peak_dissociation.png", dpi=300, bbox_inches='tight')
    fig.savefig(RESULTS_DIR / "figure3_direct_peak_dissociation.pdf", bbox_inches='tight')
    plt.close(fig)
    print("Saved Figure 3.")

# ==============================================================================
# Figure 4: Greedy Collapse vs Distributional Sensitivity (RQ1)
# ==============================================================================
def plot_figure4():
    """Figure 4: Mechanistic Interpretation & Gating Summary from empirical data"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: Exact Neutral Argmax Rate from empirical summary if available
    summary_path = ROOT / "results" / "derived" / "v3_cross_model_replication_summary.json"
    models = ['Base', 'Instruct']
    rates = [0.0, 0.0]
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                sdata = json.load(f)
            # Load empirical exact_neutral_argmax_pct if available
            models = list(sdata.get("models", {}).keys())[:2] or models
            rates = [sdata.get("models", {}).get(m, {}).get("neutral_argmax_pct", 0.0) for m in models]
        except Exception as e:
            logger.warning(f"Failed to load empirical rates from {summary_path}: {e}")

    colors = ['#4A90E2', '#E94E77']
    bars = ax1.bar(models, rates, color=colors[:len(models)], alpha=0.85, edgecolor='#333333', lw=1.2, width=0.55)
    ax1.set_ylabel('Exact Neutral Argmax Rate (%)', fontsize=11)
    ax1.set_title('(A) Neutral Output Distribution', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, max(max(rates) * 1.25, 10.0))

    for bar, rate in zip(bars, rates):
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f'{rate:.1f}%',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Annotation conditional on empirical rate
    if len(rates) > 1 and rates[1] > 50.0:
        ax1.annotate("Higher exact neutral argmax", xy=(1, rates[1]), xytext=(0.2, rates[1] * 0.9),
                     fontsize=9, color='#d95f02', fontweight='bold')

    # Right: Distributional Sensitivity (Human Valence vs Expected Valence E[V])
    ax2.set_title(r"$\mathbf{(b)\ Distributional\ Sensitivity:\ Human\ vs.\ E[V]}$", loc='left')

    # Load empirical data from recognition_baseline
    emp_path = RESULTS_DIR / "recognition_baseline" / "qwen_emobank_recognition_post.csv"
    if emp_path.exists():
        emp_df = pd.read_csv(emp_path).dropna(subset=["human_v", "post_ev", "rec_ev"])
        # Subsample deterministically for clean visualization
        plot_df = emp_df.sample(min(120, len(emp_df)), random_state=42).sort_values("human_v")
        v_human = plot_df["human_v"].values
        v_self = plot_df["post_ev"].values
        v_reader = plot_df["rec_ev"].values

        r_self, _ = stats.pearsonr(v_human, v_self)
        r_reader, _ = stats.pearsonr(v_human, v_reader)

        # Plot Reader Estimation
        ax2.scatter(v_human, v_reader, alpha=0.55, color='#7570b3', s=24, edgecolors='none',
                    label=rf"$\mathbf{{Reader\ Estimation:}}\ r = {r_reader:.3f}$")
        m_r, b_r = np.polyfit(v_human, v_reader, 1)
        x_seq = np.linspace(v_human.min(), v_human.max(), 50)
        ax2.plot(x_seq, m_r * x_seq + b_r, color='#7570b3', lw=1.8, linestyle='--')

        # Plot Self-Report
        ax2.scatter(v_human, v_self, alpha=0.75, color='#d95f02', s=28, edgecolors='none',
                    label=rf"$\mathbf{{Self\text{{-}}Report:}}\ r = {r_self:.3f}$")
        m_s, b_s = np.polyfit(v_human, v_self, 1)
        ax2.plot(x_seq, m_s * x_seq + b_s, color='#d95f02', lw=2.2)
    else:
        ax2.text(0.5, 0.5, f"Empirical data file not found at\n{emp_path.name}", ha="center", va="center", transform=ax2.transAxes)

    ax2.set_xlabel("Human Valence (EmoBank Reader, 1–9)")
    ax2.set_ylabel(r"Expected Valence $E[V]$ from Likelihoods")
    ax2.set_xlim(1.0, 9.0)
    ax2.set_ylim(1.0, 9.0)
    ax2.legend(loc='upper left', frameon=True, framealpha=0.9)

    ax2.text(0.04, 0.12,
             "Empirical EmoBank Benchmark (N = 2,248)\n"
             "Both Reader and Self-report preserve monotonic affective sensitivity.",
             transform=ax2.transAxes, fontsize=8, color='#333333',
             bbox=dict(boxstyle='round,pad=0.25', fc='#f8f9fa', ec='gray', alpha=0.9))

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "figure4_greedy_collapse_vs_sensitivity.png", dpi=300, bbox_inches='tight')
    fig.savefig(RESULTS_DIR / "figure4_greedy_collapse_vs_sensitivity.pdf", bbox_inches='tight')
    plt.close(fig)
    print("Saved Figure 4.")

if __name__ == "__main__":
    plot_figure1()
    plot_figure2()
    plot_figure3()
    plot_figure4()
    print("All main paper figures plotted successfully.")
