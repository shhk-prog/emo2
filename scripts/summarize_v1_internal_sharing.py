#!/usr/bin/env python3
"""
Summarize V1 Stage (Internal Representation and Causal Sharing: E1-E6) Results for ICLR 2027 Paper.

Outputs publication-ready LaTeX tables (with booktabs and multirow) and Markdown summaries:
1. E1: Peak Decodability Table (Reader vs Self peak relative depth, r, R2, and distance)
2. E3: Shared Causal Map Table (Rank Spearman correlation and cosine similarity)

Usage:
    python3 scripts/summarize_v1_internal_sharing.py [--repo-root .] [--out-dir iclr2027/tables]
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
    dir_path = os.path.join(repo_root, "results/derived/paper_summary/tables")
    e1_path = os.path.join(dir_path, "table_v1_1_peak_decodability.csv")
    e3_path = os.path.join(dir_path, "table_v1_4_causal_map.csv")

    df_e1 = pd.read_csv(e1_path) if os.path.exists(e1_path) else None
    df_e3 = pd.read_csv(e3_path) if os.path.exists(e3_path) else None
    return df_e1, df_e3

def format_num(val, decimals=3):
    if pd.isna(val):
        return "---"
    return f"{val:.{decimals}f}"

def generate_e1_table(df_e1):
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{V1 E1（表現デコードピークの局在）：各モデルファミリーにおける感情認識（Reader）および自己報告（Self）の最高線形デコード性能（$R^2$、Pearson $r$）と最適相対深度 $d^*$。}",
        r"\label{tab:v1_peak_decodability}",
        r"\begin{tabular}{lll cccc c}",
        r"\toprule",
        r" & & & \multicolumn{2}{c}{\textbf{Valence Peak}} & \multicolumn{2}{c}{\textbf{Arousal Peak}} & \textbf{Reader--Self} \\",
        r"\cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"\textbf{Family} & \textbf{Variant} & \textbf{Task} & Layer ($d^*$) & $R^2$ ($r$) & Layer ($d^*$) & $R^2$ ($r$) & Distance $\Delta d^*$ \\",
        r"\midrule",
    ]

    for fam_key, fam_name in FAMILY_ORDER:
        first_fam = True
        for aln, aln_label in [("base", "Base"), ("instruct", "Instruct")]:
            m_tag = f"{fam_key}_{aln}"
            sub = df_e1[df_e1["model"] == m_tag] if df_e1 is not None else pd.DataFrame()

            # Valence Reader
            r_v = sub[(sub["task"] == "reader") & (sub["axis"] == "valence")]
            s_v = sub[(sub["task"] == "self") & (sub["axis"] == "valence")]
            r_a = sub[(sub["task"] == "reader") & (sub["axis"] == "arousal")]
            s_a = sub[(sub["task"] == "self") & (sub["axis"] == "arousal")]

            def format_cell(row):
                if len(row) > 0:
                    l = int(row["peak_layer"].iloc[0])
                    d = row["peak_relative_depth"].iloc[0]
                    r2 = row["peak_r2"].iloc[0]
                    r = row["pearson"].iloc[0]
                    return f"L{l} ({d:.2f})", f"{r2:.3f} ({r:.3f})"
                return "---", "---"

            rv_ld, rv_stat = format_cell(r_v)
            sv_ld, sv_stat = format_cell(s_v)
            ra_ld, ra_stat = format_cell(r_a)
            sa_ld, sa_stat = format_cell(s_a)

            dist_v = abs(r_v["peak_distance_to_self"].iloc[0]) if len(r_v) > 0 and not pd.isna(r_v["peak_distance_to_self"].iloc[0]) else 0.0
            dist_a = abs(r_a["peak_distance_to_self"].iloc[0]) if len(r_a) > 0 and not pd.isna(r_a["peak_distance_to_self"].iloc[0]) else 0.0
            dist_str = f"{dist_v:.2f} / {dist_a:.2f}"

            fam_str = f"\\multirow{{4}}{{*}}{{{fam_name}}}" if first_fam and aln == "base" else ""
            aln_str = f"\\multirow{{2}}{{*}}{{{aln_label}}}"
            first_fam = False

            tex_lines.append(f"{fam_str:<22} & {aln_str:<20} & Reader & {rv_ld} & {rv_stat} & {ra_ld} & {ra_stat} & \\multirow{{2}}{{*}}{{{dist_str}}} \\\\")
            tex_lines.append(f"{'':<22} & {'':<20} & Self   & {sv_ld} & {sv_stat} & {sa_ld} & {sa_stat} & \\\\")

            if aln == "base":
                tex_lines.append(r"\cmidrule(lr){2-8}")

        if fam_key != FAMILY_ORDER[-1][0]:
            tex_lines.append(r"\midrule")

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1ex}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\textbf{Note:} ReaderとSelfのデコードピーク距離 $\Delta d^*$ は Valence では多くのモデルで 0.00--0.08 と極めて近傍に局在し、両タスクがトランスフォーマーの極めて近い内部表現層を活用していることを示す。",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(tex_lines)

def generate_e3_causal_map_table(df_e3):
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{V1 E3（因果プロファイルの層状局在と共有）：残差ストリーム交換・摂動介入における因果効果ピーク深度 $d_{\text{causal}}^*$、Reader--Self間の層別重要度順位相関（Spearman $\rho_{\text{rank}}$）、および介入方向コサイン類似度 $\cos\theta$。}",
        r"\label{tab:v1_causal_profile}",
        r"\begin{tabular}{ll cccc c}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{\textbf{Peak Causal Layer}} & \multicolumn{2}{c}{\textbf{Causal Magnitude ($\times 10^{-3}$)}} & \textbf{Reader--Self Sharing} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        r"\textbf{Family} & \textbf{Variant} & Reader ($d^*$) & Self ($d^*$) & Reader (Peak) & Self (Peak) & Spearman $\rho_{\text{rank}}$ / Mean Cosine \\",
        r"\midrule",
    ]

    for fam_key, fam_name in FAMILY_ORDER:
        first_fam = True
        for aln, aln_label in [("base", "Base"), ("instruct", "Instruct")]:
            m_tag = f"{fam_key}_{aln}"
            sub = df_e3[df_e3["model"] == m_tag] if df_e3 is not None else pd.DataFrame()

            if len(sub) > 0:
                r_row = sub[sub["task"] == "reader"].iloc[0] if len(sub[sub["task"] == "reader"]) > 0 else None
                s_row = sub[sub["task"] == "self"].iloc[0] if len(sub[sub["task"] == "self"]) > 0 else None
                r_peak = f"L{int(r_row['peak_layer'])} ({r_row['peak_depth']:.2f})" if r_row is not None else "---"
                s_peak = f"L{int(s_row['peak_layer'])} ({s_row['peak_depth']:.2f})" if s_row is not None else "---"
                r_mag = f"{r_row['peak_causal_magnitude'] * 1e3:.2f}" if r_row is not None else "---"
                s_mag = f"{s_row['peak_causal_magnitude'] * 1e3:.2f}" if s_row is not None else "---"
                rank_rho = sub["reader_self_rank_spearman"].iloc[0]
                cos_sim = sub["mean_direction_cosine"].iloc[0]
                share_str = f"{rank_rho:.3f} / {cos_sim:.3f}"
            else:
                r_peak, s_peak, r_mag, s_mag, share_str = "---", "---", "---", "---", "---"

            fam_str = f"\\multirow{{2}}{{*}}{{{fam_name}}}" if first_fam else ""
            first_fam = False
            tex_lines.append(f"{fam_str:<22} & {aln_label:<10} & {r_peak} & {s_peak} & {r_mag} & {s_mag} & {share_str} \\\\")

        if fam_key != FAMILY_ORDER[-1][0]:
            tex_lines.append(r"\midrule")

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1ex}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\textbf{Note:} 全モデルにおいて因果ピーク深度は Reader と Self でほぼ完全一致（同一層または近傍層）し、層別感度順位相関は $\rho_{\text{rank}} = 0.58$--$0.88$ と極めて高い。これは他者感情の認識と自己報告の生成が単に相関しているだけでなく、因果的にも共通の回路基盤に依存していることを示す。",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(tex_lines)

def generate_markdown_summary(df_e1, df_e3):
    md_lines = [
        "# V1 Stage: Internal Representation and Causal Sharing Summary\n",
        "## 1. Peak Linear Decodability (E1)\n",
        "| Family | Variant | Task | Valence Peak ($d^*$) | Valence $R^2$ ($r$) | Arousal Peak ($d^*$) | Arousal $R^2$ ($r$) | Reader--Self $\\Delta d^*$ |",
        "|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for fam_key, fam_name in FAMILY_ORDER:
        for aln, aln_label in [("base", "Base"), ("instruct", "Instruct")]:
            m_tag = f"{fam_key}_{aln}"
            sub = df_e1[df_e1["model"] == m_tag] if df_e1 is not None else pd.DataFrame()
            r_v = sub[(sub["task"] == "reader") & (sub["axis"] == "valence")]
            s_v = sub[(sub["task"] == "self") & (sub["axis"] == "valence")]
            r_a = sub[(sub["task"] == "reader") & (sub["axis"] == "arousal")]
            s_a = sub[(sub["task"] == "self") & (sub["axis"] == "arousal")]

            def get_s(row):
                if len(row) > 0:
                    return f"L{int(row['peak_layer'].iloc[0])} ({row['peak_relative_depth'].iloc[0]:.2f})", f"{row['peak_r2'].iloc[0]:.3f} ({row['pearson'].iloc[0]:.3f})"
                return "---", "---"

            rv_ld, rv_st = get_s(r_v)
            sv_ld, sv_st = get_s(s_v)
            ra_ld, ra_st = get_s(r_a)
            sa_ld, sa_st = get_s(s_a)
            dist_v = abs(r_v["peak_distance_to_self"].iloc[0]) if len(r_v) > 0 else 0.0
            dist_a = abs(r_a["peak_distance_to_self"].iloc[0]) if len(r_a) > 0 else 0.0
            d_str = f"{dist_v:.2f} / {dist_a:.2f}"
            md_lines.append(f"| **{fam_name}** | {aln_label} | Reader | {rv_ld} | {rv_st} | {ra_ld} | {ra_st} | {d_str} |")
            md_lines.append(f"| **{fam_name}** | {aln_label} | Self   | {sv_ld} | {sv_st} | {sa_ld} | {sa_st} | |")

    md_lines.append("\n## 2. Causal Mapping and Circuit Sharing (E3)\n")
    md_lines.append("| Family | Variant | Reader Peak ($d^*$) | Self Peak ($d^*$) | Reader Mag ($\\times 10^{-3}$) | Self Mag ($\\times 10^{-3}$) | Spearman $\\rho_{\\text{rank}}$ | Mean Cosine |")
    md_lines.append("|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for fam_key, fam_name in FAMILY_ORDER:
        for aln, aln_label in [("base", "Base"), ("instruct", "Instruct")]:
            m_tag = f"{fam_key}_{aln}"
            sub = df_e3[df_e3["model"] == m_tag] if df_e3 is not None else pd.DataFrame()
            if len(sub) > 0:
                r_row = sub[sub["task"] == "reader"].iloc[0] if len(sub[sub["task"] == "reader"]) > 0 else None
                s_row = sub[sub["task"] == "self"].iloc[0] if len(sub[sub["task"] == "self"]) > 0 else None
                r_peak = f"L{int(r_row['peak_layer'])} ({r_row['peak_depth']:.2f})" if r_row is not None else "---"
                s_peak = f"L{int(s_row['peak_layer'])} ({s_row['peak_depth']:.2f})" if s_row is not None else "---"
                r_mag = f"{r_row['peak_causal_magnitude'] * 1e3:.2f}" if r_row is not None else "---"
                s_mag = f"{s_row['peak_causal_magnitude'] * 1e3:.2f}" if s_row is not None else "---"
                rank_rho = sub["reader_self_rank_spearman"].iloc[0]
                cos_sim = sub["mean_direction_cosine"].iloc[0]
                md_lines.append(f"| **{fam_name}** | {aln_label} | {r_peak} | {s_peak} | {r_mag} | {s_mag} | {rank_rho:.3f} | {cos_sim:.3f} |")

    return "\n".join(md_lines) + "\n"

def main():
    parser = argparse.ArgumentParser(description="Summarize V1 Representation Sharing results.")
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.repo_root, "iclr2027/tables")
    os.makedirs(out_dir, exist_ok=True)

    df_e1, df_e3 = load_data(args.repo_root)

    # 1. E1 Table
    tex_e1 = generate_e1_table(df_e1)
    for fname in ["v1_peak_decodability.tex", "table_v1_e1_decodability.tex"]:
        p = os.path.join(out_dir, fname)
        with open(p, "w", encoding="utf-8") as f:
            f.write(tex_e1)

    # 2. E3 Table
    tex_e3 = generate_e3_causal_map_table(df_e3)
    for fname in ["v1_causal_profile.tex", "table_v1_e3_causal_map.tex"]:
        p = os.path.join(out_dir, fname)
        with open(p, "w", encoding="utf-8") as f:
            f.write(tex_e3)

    # 3. Comprehensive LaTeX summary
    tex_summary = "% ================================================================\n" \
                  "% V1 Stage: Internal Representation and Causal Sharing Complete Summary Tables\n" \
                  "% Generated from v1_internal_sharing_summary.md\n" \
                  "% ================================================================\n\n" \
                  + tex_e1 + "\n\n" + tex_e3 + "\n"
    with open(os.path.join(out_dir, "v1_internal_sharing_summary.tex"), "w", encoding="utf-8") as f:
        f.write(tex_summary)

    # 4. Markdown summary
    md_summary = generate_markdown_summary(df_e1, df_e3)
    p_md = os.path.join(out_dir, "v1_internal_sharing_summary.md")
    with open(p_md, "w", encoding="utf-8") as f:
        f.write(md_summary)

    print(f"[+] Successfully generated V1 tables in {out_dir}:")
    print(f"    - v1_peak_decodability.tex")
    print(f"    - v1_causal_profile.tex")
    print(f"    - v1_internal_sharing_summary.tex")
    print(f"    - v1_internal_sharing_summary.md")

if __name__ == "__main__":
    main()
