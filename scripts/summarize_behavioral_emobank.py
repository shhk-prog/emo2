#!/usr/bin/env python3
"""
Summarize Behavioral Stage (EmoBank 3-Way VAD Correspondence & Coupling) Results for ICLR 2027 Paper.

Outputs publication-ready LaTeX tables (with booktabs and multirow) and Markdown summaries:
1. 3-Way Ground-Truth Correspondence Table (Writer, Reader, Self across V, A, D for Base vs Instruct)
2. Internal Cognitive Coupling Table: Corr(Delta R, Delta S) across dimensions

Usage:
    python3 scripts/summarize_behavioral_emobank.py [--repo-root .] [--out-dir iclr2027/tables]
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
    b1_path = os.path.join(paper_dir, "table_b1_emobank_correspondence.csv")
    b5_path = os.path.join(paper_dir, "table_b5_coupling.csv")

    df_b1 = pd.read_csv(b1_path) if os.path.exists(b1_path) else None
    df_b5 = pd.read_csv(b5_path) if os.path.exists(b5_path) else None
    return df_b1, df_b5

def format_p_stars(p_val):
    if pd.isna(p_val):
        return ""
    if p_val < 0.001:
        return "^{***}"
    elif p_val < 0.01:
        return "^{**}"
    elif p_val < 0.05:
        return "^{*}"
    return ""

def generate_family_3way_table(df_b1):
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{EmoBank 3者間VADアライメント：問い「人間の言語表現（Writer）および読者評価（Reader）のVADグラウンドトゥルースに対して、LLMの認識・自己報告（Self）はどの程度整合するか」。4モデルファミリーのBaseおよびInstructモデルにおける人間正解ラベル（Writer / Reader）および自己報告（Self）のピアソン相関係数 $r$（$N=1000$）。}",
        r"\label{tab:behavioral_emobank_3way_vad}",
        r"\begin{tabular}{lll ccc}",
        r"\toprule",
        r"\textbf{Family} & \textbf{Model Variant} & \textbf{Perspective / Task} & \textbf{Valence ($r$)} & \textbf{Arousal ($r$)} & \textbf{Dominance ($r$)} \\",
        r"\midrule",
    ]

    for fam_key, fam_name in FAMILY_ORDER:
        first_fam = True
        for aln, aln_label in [("base", "Base"), ("instruct", "Instruct")]:
            m_tag = f"{fam_key}_{aln}"
            sub = df_b1[df_b1["model"] == m_tag] if df_b1 is not None else pd.DataFrame()

            for idx_task, (t_key, t_label) in enumerate([("writer", "Writer"), ("reader", "Reader"), ("self", "Self")]):
                vals = []
                for ax in ["valence", "arousal", "dominance"]:
                    row = sub[(sub["task"].str.lower() == t_key) & 
                              (sub["axis"].str.lower().isin([ax, ax[0]]))] if len(sub) > 0 else pd.DataFrame()
                    if len(row) > 0:
                        r_col = "pearson_r" if "pearson_r" in row.columns else "r_continuous"
                        p_col = "pearson_p" if "pearson_p" in row.columns else "p_continuous"
                        r_val = row[r_col].iloc[0]
                        p_val = row[p_col].iloc[0] if p_col in row.columns else np.nan
                        stars = format_p_stars(p_val)
                        vals.append(f"{r_val:.3f}${stars}$")
                    else:
                        vals.append("---")

                fam_str = f"\\multirow{{6}}{{*}}{{{fam_name}}}" if first_fam and idx_task == 0 and aln == "base" else ""
                aln_str = f"\\multirow{{3}}{{*}}{{{aln_label}}}" if idx_task == 0 else ""

                tex_lines.append(f"{fam_str:<22} & {aln_str:<20} & {t_label:<10} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")

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
        r"\textbf{Note:} $^{***}: p < 0.001, ^{**}: p < 0.01, ^{*}: p < 0.05$。BaseモデルではQwenおよびLlamaのみが正の相関を示したが、事後学習（Instruct）により全ファミリーにおいて認識（Writer/Reader）および自己報告（Self）のアライメントが系統的に向上した。",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(tex_lines)

def generate_internal_coupling_table(df_b5):
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{EmoBank 内部認知結合度（Cognitive Coupling）：問い「他者の感情を評価した変位 $\Delta\text{Reader}$ と、自身の状態として報告した変位 $\Delta\text{Self}$ はモデル内部で連動しているか」。モデルの他者認識変位と自己報告変位の相関 $\Delta r$（$N=192$ ペア、95\%ブートストラップ信頼区間）。}",
        r"\label{tab:behavioral_emobank_coupling}",
        r"\begin{tabular}{ll ccc}",
        r"\toprule",
        r"\textbf{Family} & \textbf{Model Variant} & \textbf{Valence ($\Delta r$ [95\% CI])} & \textbf{Arousal ($\Delta r$ [95\% CI])} & \textbf{Dominance ($\Delta r$ [95\% CI])} \\",
        r"\midrule",
    ]

    for fam_key, fam_name in FAMILY_ORDER:
        first_fam = True
        for aln, aln_label in [("base", "Base"), ("instruct", "Instruct")]:
            m_tag = f"{fam_key}_{aln}"
            sub = df_b5[df_b5["model"] == m_tag] if df_b5 is not None else pd.DataFrame()

            cells = []
            for ax in ["valence", "arousal", "dominance"]:
                row = sub[sub["axis"].str.lower().isin([ax, ax[0]])] if len(sub) > 0 and "axis" in sub.columns else pd.DataFrame()
                if len(row) > 0:
                    r_val = row["delta_pearson_r"].iloc[0] if "delta_pearson_r" in row.columns else row["r_RS"].iloc[0]
                    ci_l = row["ci_low"].iloc[0] if "ci_low" in row.columns else np.nan
                    ci_u = row["ci_high"].iloc[0] if "ci_high" in row.columns else np.nan
                    if not pd.isna(ci_l) and not pd.isna(ci_u):
                        cells.append(f"{r_val:.3f} [{ci_l:.3f}, {ci_u:.3f}]")
                    else:
                        cells.append(f"{r_val:.3f}")
                else:
                    cells.append("---")

            fam_str = f"\\multirow{{2}}{{*}}{{{fam_name}}}" if first_fam else ""
            first_fam = False
            tex_lines.append(f"{fam_str:<22} & {aln_label:<10} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")

        if fam_key != FAMILY_ORDER[-1][0]:
            tex_lines.append(r"\midrule")

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1ex}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\textbf{Note:} 全てのファミリーおよびバリアントにおいて、他者認識と自己報告の結合相関 $\Delta r$ は極めて高く（$r > 0.55, p < 10^{-16}$）、モデルが他者の感情を高く評価した刺激に対して自身の自己報告も連動して高く報告する内部的一貫性（認知結合）が確認された。",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(tex_lines)

def generate_markdown_summary(df_b1, df_b5):
    md_lines = [
        "# Behavioral Stage: EmoBank 3-Way VAD Summary Report\n",
        "## 1. 3-Way Ground-Truth Correspondence（人間評価アライメント）",
        "> **問い:** 人間の言語表現（Writer）および読者感情評価（Reader）のVADグラウンドトゥルースに対して、LLMの認識・自己報告（Self）はどの程度整合するか？\n",
        "| Family | Model Variant | Perspective | Valence ($r$) | Arousal ($r$) | Dominance ($r$) |",
        "|:---|:---|:---|:---:|:---:|:---:|",
    ]
    for fam_key, fam_name in FAMILY_ORDER:
        for aln, aln_label in [("base", "Base"), ("instruct", "Instruct")]:
            m_tag = f"{fam_key}_{aln}"
            sub = df_b1[df_b1["model"] == m_tag] if df_b1 is not None else pd.DataFrame()
            for t_key, t_label in [("writer", "Writer"), ("reader", "Reader"), ("self", "Self")]:
                vals = []
                for ax in ["valence", "arousal", "dominance"]:
                    row = sub[(sub["task"].str.lower() == t_key) & 
                              (sub["axis"].str.lower().isin([ax, ax[0]]))] if len(sub) > 0 else pd.DataFrame()
                    if len(row) > 0:
                        r_val = row["pearson_r"].iloc[0]
                        p_val = row["pearson_p"].iloc[0]
                        stars = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
                        vals.append(f"{r_val:.3f}{stars}")
                    else:
                        vals.append("---")
                md_lines.append(f"| **{fam_name}** | {aln_label} | {t_label} | {vals[0]} | {vals[1]} | {vals[2]} |")

    md_lines.append("\n## 2. 内部認知結合度 $\\Delta r$ [95% CI]")
    md_lines.append("> **問い:** 他者の感情を評価した変位 $\\Delta\\text{Reader}$ と、自身の状態として報告した変位 $\\Delta\\text{Self}$ はモデル内部で連動しているか？\n")
    md_lines.append("| Family | Model Variant | Valence $\\Delta r$ [95% CI] | Arousal $\\Delta r$ [95% CI] | Dominance $\\Delta r$ [95% CI] |")
    md_lines.append("|:---|:---|:---:|:---:|:---:|")
    for fam_key, fam_name in FAMILY_ORDER:
        for aln, aln_label in [("base", "Base"), ("instruct", "Instruct")]:
            m_tag = f"{fam_key}_{aln}"
            sub = df_b5[df_b5["model"] == m_tag] if df_b5 is not None else pd.DataFrame()
            cells = []
            for ax in ["valence", "arousal", "dominance"]:
                row = sub[sub["axis"].str.lower().isin([ax, ax[0]])] if len(sub) > 0 and "axis" in sub.columns else pd.DataFrame()
                if len(row) > 0:
                    r_val = row["delta_pearson_r"].iloc[0]
                    ci_l = row["ci_low"].iloc[0]
                    ci_u = row["ci_high"].iloc[0]
                    cells.append(f"{r_val:.3f} [{ci_l:.3f}, {ci_u:.3f}]")
                else:
                    cells.append("---")
            md_lines.append(f"| **{fam_name}** | {aln_label} | {cells[0]} | {cells[1]} | {cells[2]} |")

    return "\n".join(md_lines) + "\n"

def main():
    parser = argparse.ArgumentParser(description="Summarize Behavioral EmoBank results.")
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.repo_root, "iclr2027/tables")
    os.makedirs(out_dir, exist_ok=True)

    df_b1, df_b5 = load_data(args.repo_root)

    # 1. 3-Way Table
    tex_3way = generate_family_3way_table(df_b1)
    for fname in ["behavioral_emobank_3way_vad.tex", "table_b_emobank_family_3way.tex"]:
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            f.write(tex_3way)

    # 2. Internal Coupling Table
    tex_coupling = generate_internal_coupling_table(df_b5)
    for fname in ["behavioral_emobank_coupling.tex", "table_b_emobank_coupling.tex"]:
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            f.write(tex_coupling)

    # 3. Comprehensive LaTeX summary
    tex_summary = "% ================================================================\n" \
                  "% EmoBank Complete Summary Tables (LaTeX Version)\n" \
                  "% Generated from behavioral_emobank_summary.md\n" \
                  "% ================================================================\n\n" \
                  + tex_3way + "\n\n" + tex_coupling + "\n"
    with open(os.path.join(out_dir, "behavioral_emobank_summary.tex"), "w", encoding="utf-8") as f:
        f.write(tex_summary)

    # 4. Markdown summary
    md_summary = generate_markdown_summary(df_b1, df_b5)
    with open(os.path.join(out_dir, "behavioral_emobank_summary.md"), "w", encoding="utf-8") as f:
        f.write(md_summary)

    print(f"[+] Successfully generated Behavioral EmoBank tables in {out_dir}:")
    print(f"    - behavioral_emobank_3way_vad.tex")
    print(f"    - behavioral_emobank_coupling.tex")
    print(f"    - behavioral_emobank_summary.tex")
    print(f"    - behavioral_emobank_summary.md")

if __name__ == "__main__":
    main()
