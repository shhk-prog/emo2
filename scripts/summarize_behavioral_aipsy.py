#!/usr/bin/env python3
"""
Summarize Behavioral Stage (AIPsy-Affect 4-Split: RQ1-RQ4) Results for ICLR 2027 Paper.

Outputs publication-ready LaTeX tables (with booktabs and multirow) and Markdown summaries:
1. RQ1: Clinical--Neutral Sensitivity Table (Aligned Delta and effect size Cohen's dz)
2. RQ2: Dose-Response Monotonicity Table (Step1, Step2, IUT q-value, and monotonicity rate)
3. RQ3: Specificity to Affect vs Complexity Table (Clinical vs Complex Neutral, Cohen's d, q-value)
4. RQ4: Internal Cognitive Coupling Table (Delta Reader <-> Delta Self correlation and q-values)

Usage:
    python3 scripts/summarize_behavioral_aipsy.py [--repo-root .] [--out-dir iclr2027/tables]
"""

import os
import argparse
import numpy as np
import pandas as pd

FAMILY_ORDER = [
    ("qwen", "Qwen 2.5 1.5B"),
    ("llama", "Llama 3.2 1B"),
    ("gemma", "Gemma 3 1B"),
    ("olmo", "OLMo 2 1B"),
]

def load_data(repo_root):
    paper_dir = os.path.join(repo_root, "results/derived/paper_summary/tables")
    b2_path = os.path.join(paper_dir, "table_b2_sensitivity.csv")
    b3_path = os.path.join(paper_dir, "table_b3_dose_response.csv")
    b4_path = os.path.join(paper_dir, "table_b4_specificity.csv")
    b5_path = os.path.join(paper_dir, "table_b5_coupling.csv")

    df_b2 = pd.read_csv(b2_path) if os.path.exists(b2_path) else None
    df_b3 = pd.read_csv(b3_path) if os.path.exists(b3_path) else None
    df_b4 = pd.read_csv(b4_path) if os.path.exists(b4_path) else None
    df_b5 = pd.read_csv(b5_path) if os.path.exists(b5_path) else None
    return df_b2, df_b3, df_b4, df_b5

def format_q_val(q_val):
    if pd.isna(q_val) or str(q_val).strip() == "":
        return "---"
    try:
        val = float(q_val)
        if val < 1e-15:
            return "$<10^{-15}$"
        elif val < 1e-4:
            return f"$<10^{{{int(np.floor(np.log10(val)))}}}$"
        elif val < 0.001:
            return "$<0.001$"
        return f"{val:.4f}"
    except:
        return "---"

def generate_rq1_sensitivity_table(df_b2):
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{AIPsy RQ1（Clinical--Neutral感度）：問い「臨床刺激は統制中立刺激と比較して、期待される情動方向へのモデル出力変位を引き起こすか」。感情対照ペア（$N=192$）におけるモデル整列変位 $\Delta_{\text{aligned}}$、95\%信頼区間、効果量 Cohen's $d_z$、およびFDR補正後 $q$ 値。}",
        r"\label{tab:behavioral_aipsy_rq1_sensitivity}",
        r"\begin{tabular}{lll ccc ccc}",
        r"\toprule",
        r" & & & \multicolumn{3}{c}{\textbf{Valence Axis}} & \multicolumn{3}{c}{\textbf{Arousal Axis}} \\",
        r"\cmidrule(lr){4-6} \cmidrule(lr){7-9}",
        r"\textbf{Family} & \textbf{Variant} & \textbf{Task} & $\Delta_{\text{aligned}}$ [95\% CI] & $d_z$ & $q$-value & $\Delta_{\text{aligned}}$ [95\% CI] & $d_z$ & $q$-value \\",
        r"\midrule",
    ]

    for fam_key, fam_name in FAMILY_ORDER:
        first_fam = True
        for aln, aln_label in [("base", "Base"), ("instruct", "Instruct")]:
            m_tag = f"{fam_key}_{aln}"
            sub = df_b2[df_b2["model"] == m_tag] if df_b2 is not None else pd.DataFrame()

            for idx_task, (t_key, t_label) in enumerate([("writer", "Writer"), ("reader", "Reader"), ("self", "Self")]):
                v_row = sub[(sub["task"].str.lower() == t_key) & (sub["axis"].str.lower().isin(["valence", "v"]))]
                if len(v_row) > 0:
                    v_delta = v_row["mean_aligned_delta"].iloc[0]
                    v_l = v_row["ci_low"].iloc[0]
                    v_u = v_row["ci_high"].iloc[0]
                    v_dz = v_row["cohen_dz"].iloc[0]
                    v_q = format_q_val(v_row["q"].iloc[0])
                    v_str = f"{v_delta:.3f} [{v_l:.3f}, {v_u:.3f}]"
                    v_dz_str = f"{v_dz:.3f}"
                else:
                    v_str, v_dz_str, v_q = "---", "---", "---"

                a_row = sub[(sub["task"].str.lower() == t_key) & (sub["axis"].str.lower().isin(["arousal", "a"]))]
                if len(a_row) > 0:
                    a_delta = a_row["mean_aligned_delta"].iloc[0]
                    a_l = a_row["ci_low"].iloc[0]
                    a_u = a_row["ci_high"].iloc[0]
                    a_dz = a_row["cohen_dz"].iloc[0]
                    a_q = format_q_val(a_row["q"].iloc[0])
                    a_str = f"{a_delta:.3f} [{a_l:.3f}, {a_u:.3f}]"
                    a_dz_str = f"{a_dz:.3f}"
                else:
                    a_str, a_dz_str, a_q = "---", "---", "---"

                fam_str = f"\\multirow{{6}}{{*}}{{{fam_name}}}" if first_fam and idx_task == 0 and aln == "base" else ""
                aln_str = f"\\multirow{{3}}{{*}}{{{aln_label}}}" if idx_task == 0 else ""

                tex_lines.append(f"{fam_str:<22} & {aln_str:<20} & {t_label:<10} & {v_str} & {v_dz_str} & {v_q} & {a_str} & {a_dz_str} & {a_q} \\\\")

            if aln == "base":
                tex_lines.append(r"\cmidrule(lr){2-9}")

        if fam_key != FAMILY_ORDER[-1][0]:
            tex_lines.append(r"\midrule")

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1ex}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\textbf{Note:} 整列変位 $\Delta_{\text{aligned}} > 0$ は刺激の極性に応じた期待方向への変位を示す。Instructモデルでは、Arousal軸（Qwen, Llama, Gemma）およびValence軸（Qwen, OLMo）において強固な感度（$d_z > 0.4, q < 0.001$）が確認される。",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(tex_lines)

def generate_rq2_dose_response_table(df_b3):
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{AIPsy RQ2（用量反応性・単調性）：問い「情動強度をNeutralからModerate、Clinicalへ段階的に増加させたとき、モデルの出力変位は単調に増加するか」。強度3段階トリプレット（$N=48$）における第1段階変位 $\Delta_1$（Neu $\to$ Mod）、第2段階変位 $\Delta_2$（Mod $\to$ Clin）、Intersection-Union Test (IUT) $q$ 値、および単調性充足率。}",
        r"\label{tab:behavioral_aipsy_rq2_dose_response}",
        r"\begin{tabular}{lll cccc cccc}",
        r"\toprule",
        r" & & & \multicolumn{4}{c}{\textbf{Valence Axis}} & \multicolumn{4}{c}{\textbf{Arousal Axis}} \\",
        r"\cmidrule(lr){4-7} \cmidrule(lr){8-11}",
        r"\textbf{Family} & \textbf{Variant} & \textbf{Task} & $\Delta_1$ & $\Delta_2$ & $q_{\text{IUT}}$ & Mono.\% & $\Delta_1$ & $\Delta_2$ & $q_{\text{IUT}}$ & Mono.\% \\",
        r"\midrule",
    ]

    for fam_key, fam_name in FAMILY_ORDER:
        first_fam = True
        for aln, aln_label in [("base", "Base"), ("instruct", "Instruct")]:
            m_tag = f"{fam_key}_{aln}"
            sub = df_b3[df_b3["model"] == m_tag] if df_b3 is not None else pd.DataFrame()

            for idx_task, (t_key, t_label) in enumerate([("reader", "Reader"), ("self", "Self")]):
                v_row = sub[(sub["task"].str.lower() == t_key) & (sub["axis"].str.lower().isin(["valence", "v"]))]
                if len(v_row) > 0:
                    v_s1 = f"{v_row['step1_mean'].iloc[0]:.3f}"
                    v_s2 = f"{v_row['step2_mean'].iloc[0]:.3f}"
                    v_q = format_q_val(v_row["q_iut"].iloc[0])
                    v_rate = f"{v_row['monotonicity_rate'].iloc[0]*100:.1f}\\%"
                else:
                    v_s1, v_s2, v_q, v_rate = "---", "---", "---", "---"

                a_row = sub[(sub["task"].str.lower() == t_key) & (sub["axis"].str.lower().isin(["arousal", "a"]))]
                if len(a_row) > 0:
                    a_s1 = f"{a_row['step1_mean'].iloc[0]:.3f}"
                    a_s2 = f"{a_row['step2_mean'].iloc[0]:.3f}"
                    a_q = format_q_val(a_row["q_iut"].iloc[0])
                    a_rate = f"{a_row['monotonicity_rate'].iloc[0]*100:.1f}\\%"
                else:
                    a_s1, a_s2, a_q, a_rate = "---", "---", "---", "---"

                fam_str = f"\\multirow{{4}}{{*}}{{{fam_name}}}" if first_fam and idx_task == 0 and aln == "base" else ""
                aln_str = f"\\multirow{{2}}{{*}}{{{aln_label}}}" if idx_task == 0 else ""

                tex_lines.append(f"{fam_str:<22} & {aln_str:<20} & {t_label:<10} & {v_s1} & {v_s2} & {v_q} & {v_rate} & {a_s1} & {a_s2} & {a_q} & {a_rate} \\\\")

            if aln == "base":
                tex_lines.append(r"\cmidrule(lr){2-11}")

        if fam_key != FAMILY_ORDER[-1][0]:
            tex_lines.append(r"\midrule")

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1ex}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\textbf{Note:} 単調性（IUT $q < 0.05$）は、刺激の感情強度が強まるにつれて出力が連続的に増加することを示す。Qwen InstructおよびLlama InstructのArousal軸において顕著な用量反応性が確認される。",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(tex_lines)

def generate_rq3_specificity_table(df_b4):
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{AIPsy RQ3（感情特異性 vs 構文複雑性）：問い「臨床刺激による変位は、単なる複雑な構文の中立文（Complex Neutral）への反応ではなく、感情内容に特異的か」。臨床文変位 $\Delta_{\text{clin}}$ と複雑中立文変位 $\Delta_{\text{comp}}$ の差分、効果量 Cohen's $d$、およびFDR補正後 $q$ 値。}",
        r"\label{tab:behavioral_aipsy_rq3_specificity}",
        r"\begin{tabular}{lll cccc cccc}",
        r"\toprule",
        r" & & & \multicolumn{4}{c}{\textbf{Valence Axis}} & \multicolumn{4}{c}{\textbf{Arousal Axis}} \\",
        r"\cmidrule(lr){4-7} \cmidrule(lr){8-11}",
        r"\textbf{Family} & \textbf{Variant} & \textbf{Task} & $\Delta_{\text{clin}}$ & $\Delta_{\text{comp}}$ & $d$ & $q$-value & $\Delta_{\text{clin}}$ & $\Delta_{\text{comp}}$ & $d$ & $q$-value \\",
        r"\midrule",
    ]

    for fam_key, fam_name in FAMILY_ORDER:
        first_fam = True
        for aln, aln_label in [("base", "Base"), ("instruct", "Instruct")]:
            m_tag = f"{fam_key}_{aln}"
            sub = df_b4[df_b4["model"] == m_tag] if df_b4 is not None else pd.DataFrame()

            for idx_task, (t_key, t_label) in enumerate([("reader", "Reader"), ("self", "Self")]):
                v_row = sub[(sub["task"].str.lower() == t_key) & (sub["axis"].str.lower().isin(["valence", "v"]))]
                if len(v_row) > 0:
                    v_c = f"{v_row['clinical_displacement'].iloc[0]:.3f}"
                    v_cn = f"{v_row['complex_neutral_displacement'].iloc[0]:.3f}"
                    v_d = f"{v_row['cohen_d'].iloc[0]:.2f}"
                    v_q = format_q_val(v_row["q"].iloc[0])
                else:
                    v_c, v_cn, v_d, v_q = "---", "---", "---", "---"

                a_row = sub[(sub["task"].str.lower() == t_key) & (sub["axis"].str.lower().isin(["arousal", "a"]))]
                if len(a_row) > 0:
                    a_c = f"{a_row['clinical_displacement'].iloc[0]:.3f}"
                    a_cn = f"{a_row['complex_neutral_displacement'].iloc[0]:.3f}"
                    a_d = f"{a_row['cohen_d'].iloc[0]:.2f}"
                    a_q = format_q_val(a_row["q"].iloc[0])
                else:
                    a_c, a_cn, a_d, a_q = "---", "---", "---", "---"

                fam_str = f"\\multirow{{4}}{{*}}{{{fam_name}}}" if first_fam and idx_task == 0 and aln == "base" else ""
                aln_str = f"\\multirow{{2}}{{*}}{{{aln_label}}}" if idx_task == 0 else ""

                tex_lines.append(f"{fam_str:<22} & {aln_str:<20} & {t_label:<10} & {v_c} & {v_cn} & {v_d} & {v_q} & {a_c} & {a_cn} & {a_d} & {a_q} \\\\")

            if aln == "base":
                tex_lines.append(r"\cmidrule(lr){2-11}")

        if fam_key != FAMILY_ORDER[-1][0]:
            tex_lines.append(r"\midrule")

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1ex}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\textbf{Note:} 特異的効果量 $d$ は、臨床刺激に対する反応が単なる構文複雑さへの反応を超越している度合いを示す。Instructモデルにおいて、特にArousal軸で有意な感情特異性が維持されている。",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(tex_lines)

def generate_rq4_coupling_table(df_b5):
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{AIPsy RQ4（内部認知結合度）：問い「刺激提示に伴う他者認識の変位 $\Delta R$ と自己報告の変位 $\Delta S$ は、モデル内部で連動して結合（カップリング）しているか」。感情対照ペア（$N=192$）における他者認識変位 $\Delta\text{Reader}$ と自己報告変位 $\Delta\text{Self}$ の相関係数 $\Delta r$（95\%ブートストラップ信頼区間およびFDR補正後 $q$ 値）。}",
        r"\label{tab:behavioral_aipsy_rq4_coupling}",
        r"\begin{tabular}{ll cccc}",
        r"\toprule",
        r"\textbf{Family} & \textbf{Variant} & \textbf{Axis} & \textbf{$\Delta$ Pearson $r$ [95\% CI]} & \textbf{$q$-value} & \textbf{Raw $r$} \\",
        r"\midrule",
    ]

    for fam_key, fam_name in FAMILY_ORDER:
        first_fam = True
        for aln, aln_label in [("base", "Base"), ("instruct", "Instruct")]:
            m_tag = f"{fam_key}_{aln}"
            sub = df_b5[df_b5["model"] == m_tag] if df_b5 is not None else pd.DataFrame()

            for idx_ax, ax in enumerate(["valence", "arousal"]):
                row = sub[sub["axis"].str.lower().isin([ax, ax[0]])] if len(sub) > 0 and "axis" in sub.columns else pd.DataFrame()
                if len(row) > 0:
                    r_val = row["delta_pearson_r"].iloc[0]
                    ci_l = row["ci_low"].iloc[0]
                    ci_u = row["ci_high"].iloc[0]
                    q_val = format_q_val(row["q"].iloc[0])
                    raw_r = row["raw_pearson_r"].iloc[0] if "raw_pearson_r" in row.columns else np.nan
                    r_str = f"{r_val:.3f} [{ci_l:.3f}, {ci_u:.3f}]"
                    raw_str = f"{raw_r:.3f}" if not pd.isna(raw_r) else "---"
                else:
                    r_str, q_val, raw_str = "---", "---", "---"

                fam_str = f"\\multirow{{4}}{{*}}{{{fam_name}}}" if first_fam and idx_ax == 0 and aln == "base" else ""
                aln_str = f"\\multirow{{2}}{{*}}{{{aln_label}}}" if idx_ax == 0 else ""
                ax_name = ax.capitalize()

                tex_lines.append(f"{fam_str:<22} & {aln_str:<20} & {ax_name:<10} & {r_str} & {q_val} & {raw_str} \\\\")

            if aln == "base":
                tex_lines.append(r"\cmidrule(lr){2-6}")

        if fam_key != FAMILY_ORDER[-1][0]:
            tex_lines.append(r"\midrule")

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1ex}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\textbf{Note:} 全条件で $q < 10^{-15}$ の強固な結合を示し、事後学習を経てもモデル内部における感情認識と自己報告の連動ダイナミクスが破綻せず一貫して保たれていることが実証される。",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(tex_lines)

def generate_markdown_summary(df_b2, df_b3, df_b4, df_b5):
    md_lines = [
        "# Behavioral Stage: AIPsy-Affect Complete RQ Summary Report (RQ1--RQ4)\n",
        "## RQ1: Clinical--Neutral Sensitivity（情動感度）",
        "> **問い:** 臨床刺激（Clinical）は、言語統制中立刺激（Neutral）と比較して、期待される情動方向へのモデル出力変位を引き起こすか？\n",
        "| Family | Variant | Task | Valence $\\Delta$ [95% CI] | Valence $d_z$ | Arousal $\\Delta$ [95% CI] | Arousal $d_z$ |",
        "|:---|:---|:---|:---:|:---:|:---:|:---:|",
    ]
    for fam_key, fam_name in FAMILY_ORDER:
        for aln, aln_label in [("base", "Base"), ("instruct", "Instruct")]:
            m_tag = f"{fam_key}_{aln}"
            sub = df_b2[df_b2["model"] == m_tag] if df_b2 is not None else pd.DataFrame()
            for t_key, t_label in [("writer", "Writer"), ("reader", "Reader"), ("self", "Self")]:
                v_row = sub[(sub["task"].str.lower() == t_key) & (sub["axis"].str.lower().isin(["valence", "v"]))]
                a_row = sub[(sub["task"].str.lower() == t_key) & (sub["axis"].str.lower().isin(["arousal", "a"]))]
                v_str = f"{v_row['mean_aligned_delta'].iloc[0]:.3f} [{v_row['ci_low'].iloc[0]:.3f}, {v_row['ci_high'].iloc[0]:.3f}]" if len(v_row) > 0 else "---"
                v_dz = f"{v_row['cohen_dz'].iloc[0]:.3f}" if len(v_row) > 0 else "---"
                a_str = f"{a_row['mean_aligned_delta'].iloc[0]:.3f} [{a_row['ci_low'].iloc[0]:.3f}, {a_row['ci_high'].iloc[0]:.3f}]" if len(a_row) > 0 else "---"
                a_dz = f"{a_row['cohen_dz'].iloc[0]:.3f}" if len(a_row) > 0 else "---"
                md_lines.append(f"| **{fam_name}** | {aln_label} | {t_label} | {v_str} | {v_dz} | {a_str} | {a_dz} |")

    md_lines.extend([
        "\n## RQ2: Dose-Response Monotonicity（用量反応性・単調性）",
        "> **問い:** 情動強度（Neutral $\\to$ Moderate $\\to$ Clinical）を増加させたとき、出力は単調に増加するか？\n",
        "| Family | Variant | Task | Valence Step1 | Valence Step2 | Valence $q_{\\text{IUT}}$ | Arousal Step1 | Arousal Step2 | Arousal $q_{\\text{IUT}}$ |",
        "|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    for fam_key, fam_name in FAMILY_ORDER:
        for aln, aln_label in [("base", "Base"), ("instruct", "Instruct")]:
            m_tag = f"{fam_key}_{aln}"
            sub = df_b3[df_b3["model"] == m_tag] if df_b3 is not None else pd.DataFrame()
            for t_key, t_label in [("reader", "Reader"), ("self", "Self")]:
                v_row = sub[(sub["task"].str.lower() == t_key) & (sub["axis"].str.lower().isin(["valence", "v"]))]
                a_row = sub[(sub["task"].str.lower() == t_key) & (sub["axis"].str.lower().isin(["arousal", "a"]))]
                v_s1 = f"{v_row['step1_mean'].iloc[0]:.3f}" if len(v_row) > 0 else "---"
                v_s2 = f"{v_row['step2_mean'].iloc[0]:.3f}" if len(v_row) > 0 else "---"
                v_q = format_q_val(v_row["q_iut"].iloc[0]) if len(v_row) > 0 else "---"
                a_s1 = f"{a_row['step1_mean'].iloc[0]:.3f}" if len(a_row) > 0 else "---"
                a_s2 = f"{a_row['step2_mean'].iloc[0]:.3f}" if len(a_row) > 0 else "---"
                a_q = format_q_val(a_row["q_iut"].iloc[0]) if len(a_row) > 0 else "---"
                md_lines.append(f"| **{fam_name}** | {aln_label} | {t_label} | {v_s1} | {v_s2} | {v_q} | {a_s1} | {a_s2} | {a_q} |")

    md_lines.extend([
        "\n## RQ3: Specificity to Affect vs Complexity（感情特異性）",
        "> **問い:** 臨床刺激への反応は、単なる文章の構文的複雑さ（Complex Neutral）に対するアーティファクトか、感情特異的か？\n",
        "| Family | Variant | Task | Valence Clin | Valence Comp | Valence Cohen's $d$ | Arousal Clin | Arousal Comp | Arousal Cohen's $d$ |",
        "|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    for fam_key, fam_name in FAMILY_ORDER:
        for aln, aln_label in [("base", "Base"), ("instruct", "Instruct")]:
            m_tag = f"{fam_key}_{aln}"
            sub = df_b4[df_b4["model"] == m_tag] if df_b4 is not None else pd.DataFrame()
            for t_key, t_label in [("reader", "Reader"), ("self", "Self")]:
                v_row = sub[(sub["task"].str.lower() == t_key) & (sub["axis"].str.lower().isin(["valence", "v"]))]
                a_row = sub[(sub["task"].str.lower() == t_key) & (sub["axis"].str.lower().isin(["arousal", "a"]))]
                v_c = f"{v_row['clinical_displacement'].iloc[0]:.3f}" if len(v_row) > 0 else "---"
                v_cn = f"{v_row['complex_neutral_displacement'].iloc[0]:.3f}" if len(v_row) > 0 else "---"
                v_d = f"{v_row['cohen_d'].iloc[0]:.2f}" if len(v_row) > 0 else "---"
                a_c = f"{a_row['clinical_displacement'].iloc[0]:.3f}" if len(a_row) > 0 else "---"
                a_cn = f"{a_row['complex_neutral_displacement'].iloc[0]:.3f}" if len(a_row) > 0 else "---"
                a_d = f"{a_row['cohen_d'].iloc[0]:.2f}" if len(a_row) > 0 else "---"
                md_lines.append(f"| **{fam_name}** | {aln_label} | {t_label} | {v_c} | {v_cn} | {v_d} | {a_c} | {a_cn} | {a_d} |")

    md_lines.extend([
        "\n## RQ4: Reader--Self Change Coupling（内部認知結合度）",
        "> **問い:** 刺激提示に伴う他者認識の変位 $\\Delta R$ と自己報告の変位 $\\Delta S$ は、モデル内部で連動して結合しているか？\n",
        "| Family | Variant | Axis | $\\Delta$ Pearson $r$ [95% CI] | $q$-value | Raw $r$ |",
        "|:---|:---|:---|:---:|:---:|:---:|",
    ])
    for fam_key, fam_name in FAMILY_ORDER:
        for aln, aln_label in [("base", "Base"), ("instruct", "Instruct")]:
            m_tag = f"{fam_key}_{aln}"
            sub = df_b5[df_b5["model"] == m_tag] if df_b5 is not None else pd.DataFrame()
            for ax in ["valence", "arousal"]:
                row = sub[sub["axis"].str.lower().isin([ax, ax[0]])] if len(sub) > 0 and "axis" in sub.columns else pd.DataFrame()
                if len(row) > 0:
                    r_str = f"{row['delta_pearson_r'].iloc[0]:.3f} [{row['ci_low'].iloc[0]:.3f}, {row['ci_high'].iloc[0]:.3f}]"
                    q_val = format_q_val(row["q"].iloc[0])
                    raw_str = f"{row['raw_pearson_r'].iloc[0]:.3f}"
                    md_lines.append(f"| **{fam_name}** | {aln_label} | {ax.capitalize()} | {r_str} | {q_val} | {raw_str} |")

    return "\n".join(md_lines) + "\n"

def main():
    parser = argparse.ArgumentParser(description="Summarize Behavioral AIPsy complete RQ results.")
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.repo_root, "iclr2027/tables")
    os.makedirs(out_dir, exist_ok=True)

    df_b2, df_b3, df_b4, df_b5 = load_data(args.repo_root)

    # 1. RQ1: Sensitivity Table
    tex_rq1 = generate_rq1_sensitivity_table(df_b2)
    for fname in ["behavioral_aipsy_rq1_sensitivity.tex", "behavioral_aipsy_sensitivity.tex", "table_behavioral_aipsy_rq1_sensitivity.tex"]:
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            f.write(tex_rq1)

    # 2. RQ2: Dose-Response Table
    tex_rq2 = generate_rq2_dose_response_table(df_b3)
    for fname in ["behavioral_aipsy_rq2_dose_response.tex", "table_b3_dose_response.tex"]:
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            f.write(tex_rq2)

    # 3. RQ3: Specificity Table
    tex_rq3 = generate_rq3_specificity_table(df_b4)
    for fname in ["behavioral_aipsy_rq3_specificity.tex", "table_b4_specificity.tex"]:
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            f.write(tex_rq3)

    # 4. RQ4: Coupling Table
    tex_rq4 = generate_rq4_coupling_table(df_b5)
    for fname in ["behavioral_aipsy_rq4_coupling.tex", "behavioral_aipsy_coupling.tex", "table_behavioral_aipsy_rq4_coupling.tex"]:
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            f.write(tex_rq4)

    # 5. Comprehensive LaTeX summary
    tex_summary = "% ================================================================\n" \
                  "% AIPsy-Affect Complete RQ Summary Tables (LaTeX Version)\n" \
                  "% Generated from behavioral_aipsy_summary.md (RQ1--RQ4 Complete)\n" \
                  "% ================================================================\n\n" \
                  + tex_rq1 + "\n\n" + tex_rq2 + "\n\n" + tex_rq3 + "\n\n" + tex_rq4 + "\n"
    with open(os.path.join(out_dir, "behavioral_aipsy_summary.tex"), "w", encoding="utf-8") as f:
        f.write(tex_summary)

    # 6. Comprehensive Markdown summary
    md_summary = generate_markdown_summary(df_b2, df_b3, df_b4, df_b5)
    with open(os.path.join(out_dir, "behavioral_aipsy_summary.md"), "w", encoding="utf-8") as f:
        f.write(md_summary)

    print(f"[+] Successfully generated Behavioral AIPsy complete RQ tables in {out_dir}:")
    print(f"    - RQ1: behavioral_aipsy_rq1_sensitivity.tex")
    print(f"    - RQ2: behavioral_aipsy_rq2_dose_response.tex")
    print(f"    - RQ3: behavioral_aipsy_rq3_specificity.tex")
    print(f"    - RQ4: behavioral_aipsy_rq4_coupling.tex")
    print(f"    - Summary TeX: behavioral_aipsy_summary.tex")
    print(f"    - Summary MD:  behavioral_aipsy_summary.md")

if __name__ == "__main__":
    main()
