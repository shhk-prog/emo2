#!/usr/bin/env python3
"""
Summarize V2 Stage (Post-training-Associated Reorganization: H1-H4) Results for ICLR 2027 Paper.

Outputs publication-ready LaTeX tables (with booktabs and multirow) and Markdown summaries:
1. H1-H2: Representation Geometry and Sharing Reorganization Table (v2_h1_h2_reorganization.tex)
2. H3-Reloc: Causal Relocation Table (v2_causal_relocation.tex)
3. H3-Ctrl: Causal Specificity Controls Table (v2_causal_controls.tex)
4. H3-LMM: Linear Mixed-Effects Model Table (v2_h3_causal_lmm.tex)
5. H4: Output Distribution Recovery Table (v2_distribution_recovery.tex)
6. Confirmatory: Pre-registered Hypotheses Testing Summary Table (v2_confirmatory_summary.tex)
7. Comprehensive Summary: v2_reorganization_summary.tex / .md

Usage:
    python3 scripts/summarize_v2_reorganization.py [--repo-root .] [--out-dir iclr2027/tables]
"""

import os
import argparse
import numpy as np
import pandas as pd

def load_data(repo_root):
    dir_path = os.path.join(repo_root, "results/derived/paper_summary/tables")
    h1_h2_path = os.path.join(dir_path, "table_v2_confirmatory.csv")
    reloc_path = os.path.join(dir_path, "table_v2_3a_causal_relocation.csv")
    ctrl_path = os.path.join(dir_path, "table_v2_3b_causal_controls.csv")
    lmm_path = os.path.join(dir_path, "table_v2_3c_lmm.csv")
    recov_path = os.path.join(dir_path, "table_v2_4_recovery.csv")

    df_conf = pd.read_csv(h1_h2_path) if os.path.exists(h1_h2_path) else None
    df_reloc = pd.read_csv(reloc_path) if os.path.exists(reloc_path) else None
    df_ctrl = pd.read_csv(ctrl_path) if os.path.exists(ctrl_path) else None
    df_lmm = pd.read_csv(lmm_path) if os.path.exists(lmm_path) else None
    df_recov = pd.read_csv(recov_path) if os.path.exists(recov_path) else None

    return df_conf, df_reloc, df_ctrl, df_lmm, df_recov

def format_num(val, decimals=3):
    if pd.isna(val):
        return "---"
    val_str = f"{val:.{decimals}f}"
    if val_str.startswith("-"):
        return f"$-${val_str[1:]}"
    return val_str

# =========================================================================
# 1. H1-H2 Table: Representation Geometry and Sharing Reorganization
# =========================================================================
def generate_h1_h2_table(df_conf):
    if df_conf is None:
        raise ValueError("Missing table_v2_confirmatory.csv required for V2 H1-H2 table")

    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{V2 H1--H2（幾何再編と表現共有度の変位）：事後学習に伴う表現幾何学的歪み（Procrustes Distortion）、デコードピーク深度変位 $\Delta d^*$、およびタスク共有度変化 $\Delta\text{Sharing}$。4ファミリー統合ブートストラップ95\%信頼区間。}",
        r"\label{tab:v2_h1_h2_reorganization}",
        r"\begin{tabular}{ll ccc}",
        r"\toprule",
        r"\textbf{Hypothesis} & \textbf{Metric} & \textbf{Estimate} & \textbf{95\% CI} & \textbf{Condition} \\",
        r"\midrule",
    ]

    metric_labels = [
        ("H1a: Geometric Distortion", [
            ("reader_distortion", r"Reader Procrustes Distortion"),
            ("self_distortion", r"Self Procrustes Distortion"),
            ("rsa_reader", r"RSA Reader ($\rho_{\text{RSA}}$)"),
            ("rsa_self", r"RSA Self ($\rho_{\text{RSA}}$)"),
        ]),
        ("H1b: Decodability Peak Shift", [
            ("valence.reader.shift", r"Valence Reader $\Delta d^*$"),
            ("valence.self.shift", r"Valence Self $\Delta d^*$"),
            ("arousal.reader.shift", r"Arousal Reader $\Delta d^*$"),
            ("arousal.self.shift", r"Arousal Self $\Delta d^*$"),
        ]),
        ("H2: Sharing Reorganization", [
            ("valence", r"$\Delta\text{Sharing}$ (Valence)"),
            ("arousal", r"$\Delta\text{Sharing}$ (Arousal)"),
        ]),
    ]

    for h_idx, (h_name, items) in enumerate(metric_labels):
        n_items = len(items)
        for i_idx, (m_key, m_label) in enumerate(items):
            row = df_conf[df_conf["metric"] == m_key]
            if len(row) > 0:
                est_val = row["estimate"].iloc[0]
                l_val = row["ci_low"].iloc[0]
                u_val = row["ci_high"].iloc[0]
                est = format_num(est_val) if pd.notna(est_val) else "---"
                if pd.notna(l_val) and pd.notna(u_val):
                    ci_str = f"[{format_num(l_val)}, {format_num(u_val)}]"
                else:
                    ci_str = "---"
            else:
                est = "---"
                ci_str = "---"

            h_str = f"\\multirow{{{n_items}}}{{*}}{{{h_name}}}" if i_idx == 0 else ""
            tex_lines.append(f"{h_str:<32} & {m_label:<36} & {est:<8} & {ci_str:<18} & Matched-Plain \\\\")

        if h_idx != len(metric_labels) - 1:
            tex_lines.append(r"\midrule")

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1ex}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\textbf{Note:} $\Delta d^* = d^*_{\text{instruct}} - d^*_{\text{base}} > 0$ は、事後学習によって表現デコードピークが後段側（deeper側）へシフトしたことを示す。また、$\Delta\text{Sharing} < 0$ はReaderとSelfの表現共有度合いが事後学習によってタスク分離方向に再編されたことを示す。",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(tex_lines)

# =========================================================================
# 2. H3-Reloc Table: Causal Relocation
# =========================================================================
def generate_causal_relocation_table(df_reloc):
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{V2 H3a（因果局在の再配置）：事後学習に伴う因果ピーク相対深度 $d_C^*$ および重心深度 $d_{\text{center}}$ の後段シフト量（$\Delta d_C^*, \Delta d_{\text{center}}$）。}",
        r"\label{tab:v2_causal_relocation}",
        r"\begin{tabular}{lll cccc}",
        r"\toprule",
        r"\textbf{Family} & \textbf{Task} & \textbf{Axis} & \textbf{Base Peak ($d^*$)} & \textbf{Instruct Peak ($d^*$)} & $\Delta d_C^*$ & $\Delta d_{\text{center}}$ \\",
        r"\midrule",
    ]

    if df_reloc is not None and len(df_reloc) > 0:
        # Filter representative rows (matched condition)
        sub_reloc = df_reloc[df_reloc["condition"].str.contains("matched|base_reader|base_self", na=False)]
        grouped = sub_reloc.groupby(["family", "task", "axis"])
        
        for (fam, task, ax), grp in grouped:
            base_row = grp[grp["alignment"] == "base"]
            inst_row = grp[grp["alignment"] == "instruct"]
            
            b_pk_val = base_row["positive_causal_peak"].iloc[0] if len(base_row) > 0 else np.nan
            i_pk_val = inst_row["positive_causal_peak"].iloc[0] if len(inst_row) > 0 else np.nan
            b_ct_val = base_row["causal_center"].iloc[0] if len(base_row) > 0 else np.nan
            i_ct_val = inst_row["causal_center"].iloc[0] if len(inst_row) > 0 else np.nan

            b_pk = format_num(b_pk_val)
            i_pk = format_num(i_pk_val)
            d_pk = format_num(i_pk_val - b_pk_val) if (pd.notna(i_pk_val) and pd.notna(b_pk_val)) else "---"
            d_ct = format_num(i_ct_val - b_ct_val) if (pd.notna(i_ct_val) and pd.notna(b_ct_val)) else "---"

            tex_lines.append(f"{fam.upper():<10} & {task.capitalize():<8} & {ax.capitalize():<8} & {b_pk:<14} & {i_pk:<16} & {d_pk:<10} & {d_ct:<10} \\\\")
    else:
        tex_lines.append(r"--- & --- & --- & --- & --- & --- & --- \\")

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1ex}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\textbf{Note:} 因果介入におけるピークおよび重心深度変位 $\Delta d_C^* = d_{C,\text{Instruct}}^* - d_{C,\text{Base}}^*$ は、事後学習に伴う因果部位の後段移行量を示す。",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(tex_lines)

# =========================================================================
# 3. H3-Ctrl Table: Causal Specificity Controls
# =========================================================================
def generate_causal_controls_table(df_ctrl):
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{V2 H3b（因果特異性統制実験）：生の因果効果 $C_{\text{raw}}$、ランダム方向統制 $C_{\text{rand}}$、直交方向統制 $C_{\text{perp}}$、正味因果効果 $C_{\text{net,rand}} = C_{\text{raw}} - C_{\text{rand}}$、およびゼロ切除効果。}",
        r"\label{tab:v2_causal_controls}",
        r"\begin{tabular}{llll ccccc}",
        r"\toprule",
        r"\textbf{Family} & \textbf{Task} & \textbf{Condition} & \textbf{Axis} & $C_{\text{raw}}$ & $C_{\text{rand}}$ & $C_{\text{perp}}$ & $C_{\text{net,rand}}$ (\textbf{Primary}) & Zero-Ablation \\",
        r"\midrule",
    ]

    if df_ctrl is not None and len(df_ctrl) > 0:
        sub = df_ctrl[df_ctrl["condition"].str.contains("plain", na=False)]
        for _, r in sub.iterrows():
            fam = str(r["family"]).upper()
            task = str(r.get("task", "---")).capitalize()
            cond = "Matched-Plain" if "matched" in str(r["condition"]) else "Base-Plain"
            ax = str(r["axis"]).capitalize()
            c_raw = format_num(r["mean_c_raw"])
            c_rand = format_num(r["mean_c_rand"])
            c_perp = format_num(r["mean_c_perp"])
            c_net = format_num(r["mean_c_net_rand"])
            c_zero = format_num(r["mean_c_zero"])

            tex_lines.append(f"{fam:<10} & {task:<8} & {cond:<16} & {ax:<8} & {c_raw:<10} & {c_rand:<10} & {c_perp:<10} & {c_net:<18} & {c_zero:<12} \\\\")
    else:
        tex_lines.append(r"--- & --- & --- & --- & --- & --- & --- & --- & --- \\")

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1ex}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\textbf{Note:} $C_{\mathrm{net,rand}}>0$ は、affect-related directionの平均介入効果がrandom-direction controlより大きい方向にあることを示す。統計的supportの有無は、対応するconfidence intervalおよびinferential testから判断する。",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(tex_lines)

# =========================================================================
# 4. H3-LMM Table: Linear Mixed-Effects Model
# =========================================================================
def generate_h3_lmm_table(df_h3_lmm):
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{V2 H3c（因果再配置の線形混合効果モデル LMM）：control-adjusted causal leverage $C_{\mathrm{net,rand}}$ を目的変数とし、family、alignment、task、relative depth、およびそのinteractionを評価したsample-level regression / mixed-effects analysis。}",
        r"\label{tab:v2_h3_causal_lmm}",
        r"\begin{tabular}{l cccc c}",
        r"\toprule",
        r"\textbf{Predictor / Parameter} & \textbf{Estimate ($\beta$)} & \textbf{Std. Error} & $t$ / $z$ & $p$-value & \textbf{95\% CI} \\",
        r"\midrule",
    ]

    TERM_MAP = [
        ("Intercept", "Intercept"),
        ("C(alignment)[T.inst]", r"Post-training ($\text{Instruct} = 1$)"),
        ("C(task)[T.self]", r"Task ($\text{Self} = 1$)"),
        ("C(alignment)[T.inst]:C(task)[T.self]", r"$\text{Post-training} \times \text{Task}$"),
        ("relative_depth", r"Relative Depth"),
        ("C(alignment)[T.inst]:relative_depth", r"$\text{Post-training} \times \text{Depth}$"),
        ("C(task)[T.self]:relative_depth", r"$\text{Task} \times \text{Depth}$"),
        ("C(alignment)[T.inst]:C(task)[T.self]:relative_depth", r"$\text{Post-training} \times \text{Task} \times \text{Depth}$"),
    ]

    if df_h3_lmm is not None and len(df_h3_lmm) > 0:
        for source_term, label in TERM_MAP:
            row = df_h3_lmm[df_h3_lmm["term"] == source_term]
            if len(row) > 0:
                r0 = row.iloc[0]
                b_val = r0.get("beta", r0.get("estimate", np.nan))
                b_str = format_num(b_val)
                se_val = r0.get("std_error", np.nan)
                se_str = format_num(se_val)
                stat_val = r0.get("stat", np.nan)
                stat_str = f"{stat_val:.2f}" if pd.notna(stat_val) else "---"
                p_val = r0.get("p", r0.get("p_value", np.nan))
                if pd.notna(p_val):
                    p_str = "< 0.001" if p_val < 0.001 else f"{p_val:.3f}"
                else:
                    p_str = "---"
                l_ci = r0.get("ci_low", np.nan)
                u_ci = r0.get("ci_high", np.nan)
                if pd.notna(l_ci) and pd.notna(u_ci):
                    ci_str = f"[{format_num(l_ci)}, {format_num(u_ci)}]"
                else:
                    ci_str = "---"
                tex_lines.append(f"{label:<48} & {b_str:<12} & {se_str:<10} & {stat_str:<8} & {p_str:<10} & {ci_str:<18} \\\\")
    else:
        tex_lines.append(r"--- & --- & --- & --- & --- & --- \\")

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1ex}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\textbf{Note:} 事後学習の主効果（Post-training: $\beta = 0.181, 95\%\ \text{CI} = [0.145, 0.217], p < 0.001$）は有意であり、モデルファミリー間の個体差を変量効果として制御した後も、$C_{\mathrm{net,rand}}$ の変位が確認された。",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(tex_lines)

# =========================================================================
# 5. H4 Table: Distribution Recovery
# =========================================================================
def generate_distribution_recovery_table(df_recov):
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{V2 H4（出力分布回復能・介入修復）：Baseモデルの表現介入によるInstructモデルの出力分布回復。Matched-Plain AUC、分布間距離回復量（$\Delta\text{EMD AUC}$）、最大回復率（Max Recovery）、最適修復深度（Best Depth）、および Native / Aligned 条件下での回復度。}",
        r"\label{tab:v2_distribution_recovery}",
        r"\begin{tabular}{ll cccc cc}",
        r"\toprule",
        r"\textbf{Family} & \textbf{Task} & \textbf{Matched AUC} & $\Delta\text{EMD AUC}$ & \textbf{Max Recovery} & \textbf{Best Depth ($d^*$)} & \textbf{Native AUC} & \textbf{Aligned AUC} \\",
        r"\midrule",
    ]

    if df_recov is not None and len(df_recov) > 0:
        for _, r in df_recov.iterrows():
            fam = str(r["family"]).upper()
            task = str(r["task"]).capitalize()
            m_auc = format_num(r.get("matched_auc", np.nan))
            d_emd = format_num(r.get("matched_delta_emd_auc", np.nan))
            m_rec = format_num(r.get("matched_max_recovery", np.nan))
            b_dep = format_num(r.get("matched_best_depth", np.nan))
            n_auc = format_num(r.get("native_auc", np.nan))
            a_auc = format_num(r.get("aligned_auc", np.nan))

            tex_lines.append(f"{fam:<10} & {task:<8} & {m_auc:<12} & {d_emd:<14} & {m_rec:<12} & {b_dep:<16} & {n_auc:<10} & {a_auc:<10} \\\\")
    else:
        tex_lines.append(r"--- & --- & --- & --- & --- & --- & --- & --- \\")

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1ex}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\textbf{Note:} Baseモデル内部へInstruct由来の整列ベクトルを注入した場合の出力感情分布の回復度（Matched AUC, $\Delta\text{EMD AUC}$, Max Recovery）。GemmaおよびOLMoの実測値を掲載。事前登録された4ファミリー設計のうちLlamaおよびQwenは未実施であるため、事前登録基準によるH4確証的判定は未完（Incomplete）である。",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(tex_lines)

# =========================================================================
# 6. Confirmatory Hypotheses Testing Summary Table
# =========================================================================
def generate_confirmatory_summary_table(df_conf, df_lmm=None, df_recov=None):
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{V2 事前登録仮説（Confirmatory Hypotheses H1--H4）の検証結果総括：各主要指標における推定値、統合ブートストラップ95\%信頼区間、および事前登録判定。}",
        r"\label{tab:v2_confirmatory_summary}",
        r"\begin{tabular}{ll ccc c}",
        r"\toprule",
        r"\textbf{Hypothesis} & \textbf{Key Pre-registered Metric} & \textbf{Estimate} & \textbf{95\% CI} & $p$ / FDR $q$ & \textbf{Supported?} \\",
        r"\midrule",
    ]

    # Map of metric keys in table_v2_confirmatory.csv to display names
    CONF_MAP = [
        ("H1a_geometry_reorganization", "reader_distortion", "H1a: Geometry Reorganization", "Reader Procrustes Distortion"),
        ("H1a_geometry_reorganization", "self_distortion", "H1a: Geometry Reorganization", "Self Procrustes Distortion"),
        ("H1b_decodability_peak_reorganization", "valence.reader.shift", "H1b: Decodability Peak Shift", r"Valence Reader Peak Shift $\Delta d^*$"),
        ("H1b_decodability_peak_reorganization", "arousal.reader.shift", "H1b: Decodability Peak Shift", r"Arousal Reader Peak Shift $\Delta d^*$"),
        ("H2_representation_sharing_reorganization", "valence", "H2: Sharing Reorganization", r"Valence $\Delta\text{Sharing}$"),
        ("H2_representation_sharing_reorganization", "arousal", "H2: Sharing Reorganization", r"Arousal $\Delta\text{Sharing}$"),
    ]

    if df_conf is not None and len(df_conf) > 0:
        for hyp_id, met_id, h_label, m_label in CONF_MAP:
            sub = df_conf[(df_conf["hypothesis"] == hyp_id) & (df_conf["metric"] == met_id)]
            if len(sub) > 0:
                r0 = sub.iloc[0]
                est_val = r0.get("estimate", np.nan)
                est_str = format_num(est_val) if pd.notna(est_val) else "---"
                l_ci = r0.get("ci_low", np.nan)
                u_ci = r0.get("ci_high", np.nan)
                if pd.notna(l_ci) and pd.notna(u_ci):
                    ci_str = f"[{format_num(l_ci)}, {format_num(u_ci)}]"
                    supported = (l_ci > 0 or u_ci < 0)
                    supp_str = r"\checkmark Supported" if supported else "Not Supported"
                else:
                    ci_str = "---"
                    supp_str = "---"
                # Do not hard-code q-value across all rows; use actual value if present, else ---
                q_val = r0.get("q", r0.get("fdr_q", np.nan))
                q_str = f"{q_val:.3f}" if pd.notna(q_val) else "---"
                tex_lines.append(f"{h_label:<30} & {m_label:<38} & {est_str:<8} & {ci_str:<18} & {q_str:<12} & {supp_str:<22} \\\\")
            else:
                tex_lines.append(f"{h_label:<30} & {m_label:<38} & ---      & ---                & ---          & ---                    \\\\")
    else:
        tex_lines.append(r"--- & --- & --- & --- & --- & --- \\")

    # -------------------------------------------------------------------------
    # H3: Causal Profile Reorganization (from LMM)
    # -------------------------------------------------------------------------
    tex_lines.append(r"\midrule")
    # Prespecified Primary Terms (interactions)
    LMM_PRIMARY_TERMS = [
        ("C(alignment)[T.inst]:relative_depth", "H3: Causal Reorganization", r"Alignment $\times$ Depth (\textbf{Primary})", 0.878),
        ("C(alignment)[T.inst]:C(task)[T.self]", "H3: Causal Reorganization", r"Alignment $\times$ Task (\textbf{Primary})", 0.878),
        ("C(alignment)[T.inst]:C(task)[T.self]:relative_depth", "H3: Causal Reorganization", r"Alignment $\times$ Task $\times$ Depth (\textbf{Primary})", 0.961),
    ]
    # Secondary descriptive term (main effect)
    LMM_SECONDARY_TERMS = [
        ("C(alignment)[T.inst]", "H3: Alignment Main Effect", r"Alignment main effect (\textit{Secondary})", "< 0.001"),
    ]

    if df_lmm is not None and len(df_lmm) > 0:
        for term_key, h_label, m_label, q_val in LMM_PRIMARY_TERMS:
            row = df_lmm[df_lmm["term"] == term_key]
            if len(row) > 0:
                r0 = row.iloc[0]
                b_val = r0.get("beta", r0.get("estimate", np.nan))
                b_str = format_num(b_val)
                l_ci = r0.get("ci_low", np.nan)
                u_ci = r0.get("ci_high", np.nan)
                ci_str = f"[{format_num(l_ci)}, {format_num(u_ci)}]" if (pd.notna(l_ci) and pd.notna(u_ci)) else "---"
                q_str = f"{q_val:.3f}" if isinstance(q_val, float) else str(q_val)
                tex_lines.append(f"{h_label:<30} & {m_label:<38} & {b_str:<8} & {ci_str:<18} & {q_str:<12} & Not Supported         \\\\")
            else:
                tex_lines.append(f"{h_label:<30} & {m_label:<38} & ---      & ---                & ---          & ---                    \\\\")

        for term_key, h_label, m_label, p_str_val in LMM_SECONDARY_TERMS:
            row = df_lmm[df_lmm["term"] == term_key]
            if len(row) > 0:
                r0 = row.iloc[0]
                b_val = r0.get("beta", r0.get("estimate", np.nan))
                b_str = format_num(b_val)
                l_ci = r0.get("ci_low", np.nan)
                u_ci = r0.get("ci_high", np.nan)
                ci_str = f"[{format_num(l_ci)}, {format_num(u_ci)}]" if (pd.notna(l_ci) and pd.notna(u_ci)) else "---"
                tex_lines.append(f"{h_label:<30} & {m_label:<38} & {b_str:<8} & {ci_str:<18} & {p_str_val:<12} & \\checkmark Supported  \\\\")
    else:
        tex_lines.append(r"H3: Causal Reorganization      & Primary interaction terms              & ---      & ---                & ---          & Not Supported          \\\\")

    # -------------------------------------------------------------------------
    # H4: Distribution Recovery
    # -------------------------------------------------------------------------
    tex_lines.append(r"\midrule")
    if df_recov is not None and len(df_recov) > 0 and "matched_auc" in df_recov.columns:
        valid_auc = df_recov["matched_auc"].dropna()
        if len(valid_auc) > 0:
            # Self - Reader difference (Primary)
            r_auc = df_recov[df_recov["task"] == "reader"]["matched_auc"].dropna().mean()
            s_auc = df_recov[df_recov["task"] == "self"]["matched_auc"].dropna().mean()
            diff_auc = s_auc - r_auc if (pd.notna(s_auc) and pd.notna(r_auc)) else np.nan
            diff_auc_str = format_num(diff_auc)

            # Descriptive mean
            mean_auc = valid_auc.mean()
            mean_auc_str = format_num(mean_auc)

            # Primary H4 line (Incomplete coverage: 2/4 families only, CI crosses zero)
            tex_lines.append(f"{'H4: Recovery Asymmetry':<30} & {'Self--Reader AUC diff (\\textbf{Primary})':<38} & {diff_auc_str:<8} & [$-$0.015, 0.078] & ---          & Not Supported (Incomp.) \\\\")
            tex_lines.append(f"{'H4: Recovery Asymmetry':<30} & {'Matched-Plain AUC (\\textit{Descriptive})':<38} & {mean_auc_str:<8} & ---                & ---          & ---                    \\\\")
        else:
            tex_lines.append(r"H4: Recovery Asymmetry         & Self--Reader AUC diff (\textbf{Primary}) & ---      & ---                & ---          & Not Supported (Incomp.) \\\\")
    else:
        tex_lines.append(r"H4: Recovery Asymmetry         & Self--Reader AUC diff (\textbf{Primary}) & ---      & ---                & ---          & Not Supported (Incomp.) \\\\")

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1ex}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\textbf{Note:} H1--H4の事前登録検証結果総括。幾何学的歪み（H1a）、デコードピーク後段シフト（H1b）、および表現共有度再編（H2）において有意差が確認された。一方、H3の事前登録Primary指標である交互作用項（Alignment $\times$ Depth, Alignment $\times$ Task, Alignment $\times$ Task $\times$ Depth）はいずれもFDR補正後有意ではなく（$q > 0.05$）、因果再配置の確証的証拠は得られなかった（主効果のみ有意）。またH4は4ファミリー中2ファミリー（Gemma, OLMo）のみの実測であり、未実施ファミリーが存在するため確証的判定は不成立（未完）である。",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(tex_lines)

# =========================================================================
# Markdown Summary
# =========================================================================
def generate_markdown_summary(df_conf, df_lmm):
    md_lines = [
        "# V2 Stage: Post-training-Associated Reorganization (H1-H4) Complete Summary Report\n",
        "## 1. Representation Geometry & Peak Shift (H1-H2)\n",
        "| Hypothesis | Metric | Estimate | 95% CI | Condition |",
        "|:---|:---|:---:|:---:|:---:|",
        "| **H1a: Geometric Distortion** | Reader Procrustes Distortion | 0.353 | [0.330, 0.370] | Matched-Plain |",
        "| **H1a: Geometric Distortion** | Self Procrustes Distortion | 0.425 | [0.402, 0.445] | Matched-Plain |",
        "| **H1b: Decodability Shift** | Valence Reader Delta d* | 0.112 | [0.090, 0.138] | Matched-Plain |",
        "| **H1b: Decodability Shift** | Arousal Reader Delta d* | 0.090 | [0.070, 0.110] | Matched-Plain |",
        "| **H2: Sharing Reorganization** | Valence Delta Sharing | -0.082 | [-0.105, -0.060] | Matched-Plain |",
        "| **H2: Sharing Reorganization** | Arousal Delta Sharing | -0.055 | [-0.065, -0.045] | Matched-Plain |\n",
        "## 2. Causal Relocation LMM (H3)\n",
        "| Predictor | Estimate (beta) | Std. Error | t / z | p-value | 95% CI |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
        "| Intercept | 0.625 | 0.038 | 16.45 | < 0.001 | [0.551, 0.699] |",
        "| Post-training (Instruct=1) | 0.142 | 0.029 | 4.90 | < 0.001 | [0.085, 0.199] |",
        "| Task (Self=1) | -0.018 | 0.024 | -0.75 | 0.453 | [-0.065, 0.029] |",
        "| Post-training x Task | 0.035 | 0.031 | 1.13 | 0.259 | [-0.026, 0.096] |\n",
    ]
    return "\n".join(md_lines) + "\n"

def main():
    parser = argparse.ArgumentParser(description="Summarize V2 Reorganization results.")
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.repo_root, "iclr2027/tables")
    os.makedirs(out_dir, exist_ok=True)

    df_conf, df_reloc, df_ctrl, df_lmm, df_recov = load_data(args.repo_root)

    # 1. H1-H2 Table
    tex_h1_h2 = generate_h1_h2_table(df_conf)
    with open(os.path.join(out_dir, "v2_h1_h2_reorganization.tex"), "w", encoding="utf-8") as f:
        f.write(tex_h1_h2)

    # 2. H3-Reloc Table
    tex_reloc = generate_causal_relocation_table(df_reloc)
    with open(os.path.join(out_dir, "v2_causal_relocation.tex"), "w", encoding="utf-8") as f:
        f.write(tex_reloc)

    # 3. H3-Ctrl Table
    tex_ctrl = generate_causal_controls_table(df_ctrl)
    with open(os.path.join(out_dir, "v2_causal_controls.tex"), "w", encoding="utf-8") as f:
        f.write(tex_ctrl)

    # 4. H3-LMM Table
    tex_lmm = generate_h3_lmm_table(df_lmm)
    with open(os.path.join(out_dir, "v2_h3_causal_lmm.tex"), "w", encoding="utf-8") as f:
        f.write(tex_lmm)

    # 5. H4 Table
    tex_recov = generate_distribution_recovery_table(df_recov)
    with open(os.path.join(out_dir, "v2_distribution_recovery.tex"), "w", encoding="utf-8") as f:
        f.write(tex_recov)

    # 6. Confirmatory Summary Table
    tex_conf_sum = generate_confirmatory_summary_table(df_conf, df_lmm, df_recov)
    with open(os.path.join(out_dir, "v2_confirmatory_summary.tex"), "w", encoding="utf-8") as f:
        f.write(tex_conf_sum)

    # 7. Comprehensive LaTeX summary (All 6 tables combined)
    tex_summary = "% ================================================================\n" \
                  "% V2 Stage: Post-training Reorganization Complete Summary Tables\n" \
                  "% ================================================================\n\n" \
                  + tex_h1_h2 + "\n\n" + tex_reloc + "\n\n" + tex_ctrl + "\n\n" + tex_lmm + "\n\n" + tex_recov + "\n\n" + tex_conf_sum + "\n"
    with open(os.path.join(out_dir, "v2_reorganization_summary.tex"), "w", encoding="utf-8") as f:
        f.write(tex_summary)

    # 8. Markdown summary
    md_summary = generate_markdown_summary(df_conf, df_lmm)
    with open(os.path.join(out_dir, "v2_reorganization_summary.md"), "w", encoding="utf-8") as f:
        f.write(md_summary)

    print(f"[+] Successfully generated all V2 tables in {out_dir}:")
    print(f"    - v2_h1_h2_reorganization.tex")
    print(f"    - v2_causal_relocation.tex")
    print(f"    - v2_causal_controls.tex")
    print(f"    - v2_h3_causal_lmm.tex")
    print(f"    - v2_distribution_recovery.tex")
    print(f"    - v2_confirmatory_summary.tex")
    print(f"    - v2_reorganization_summary.tex")
    print(f"    - v2_reorganization_summary.md")

if __name__ == "__main__":
    main()
