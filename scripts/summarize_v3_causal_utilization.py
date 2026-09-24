#!/usr/bin/env python3
"""
Summarize V3 Stage (Causal Utilization and Spatiotemporal Dynamics) Results for ICLR 2027 Paper.

Outputs publication-ready LaTeX tables (with booktabs) and Markdown summaries:
1. Gate Criteria Evaluation Table (v3_gate_decision.tex)
2. Spatiotemporal Dissociation Table (v3_spatiotemporal_dynamics.tex)
3. Mediated Attenuation Table (v3_mediated_attenuation.tex)
4. Confirmatory Replication Details Table (v3_confirmatory_details.tex)
5. Confirmatory Replication Matrix Table (v3_confirmatory_matrix.tex)
6. Comprehensive Summary: v3_causal_utilization_summary.tex / .md

Usage:
    python3 scripts/summarize_v3_causal_utilization.py [--repo-root .] [--out-dir iclr2027/tables]
"""

import os
import argparse
import pandas as pd
import numpy as np

def load_data(repo_root):
    dir_path = os.path.join(repo_root, "results/derived/paper_summary/tables")
    gate_path = os.path.join(dir_path, "table_v3_1_gate.csv")
    spatio_path = os.path.join(dir_path, "table_v3_2_spatiotemporal_summary.csv")
    atten_path = os.path.join(dir_path, "table_v3_3_mediated_attenuation.csv")
    conf_path = os.path.join(dir_path, "table_v3_4_confirmatory.csv")
    conf_matrix_path = os.path.join(dir_path, "table_v3_confirmatory_matrix.csv")

    df_gate = pd.read_csv(gate_path) if os.path.exists(gate_path) else None
    df_spatio = pd.read_csv(spatio_path) if os.path.exists(spatio_path) else None
    df_atten = pd.read_csv(atten_path) if os.path.exists(atten_path) else None
    df_conf = pd.read_csv(conf_path) if os.path.exists(conf_path) else None
    df_conf_matrix = pd.read_csv(conf_matrix_path) if os.path.exists(conf_matrix_path) else None

    return df_gate, df_spatio, df_atten, df_conf, df_conf_matrix

def format_num(val, decimals=4):
    if pd.isna(val):
        return "---"
    if abs(val) < 1e-3 and val != 0:
        return f"{val:.2e}"
    val_str = f"{val:.{decimals}f}"
    if val_str.startswith("-"):
        return f"$-${val_str[1:]}"
    return val_str

# =========================================================================
# 1. Gate Table
# =========================================================================
def generate_gate_table(df_gate):
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{V3 事前登録 Gate 判定結果：因果介入パイプライン継続可否の評価規約。各判定軸の信頼区間下限／上限が 1--9 raw scale 閾値（0.05 / 0.10）および TVD 上限（0.15）を満たすかを判定。}",
        r"\label{tab:v3_gate_decision}",
        r"\begin{tabular}{l ccc ccc c}",
        r"\toprule",
        r"\textbf{Axis} & \multicolumn{2}{c}{\textbf{Sufficiency ($\beta_1$)}} & \multicolumn{2}{c}{\textbf{Specificity ($\beta_1$)}} & \multicolumn{2}{c}{\textbf{Topic Control (TVD)}} & \textbf{Gate Decision} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r" & \textbf{CI Low} & \textbf{Pass?} & \textbf{CI Low} & \textbf{Pass?} & \textbf{CI High} & \textbf{Pass?} & \\",
        r"\midrule",
    ]

    for _, row in df_gate.iterrows():
        axis_name = str(row["axis"]).capitalize()
        suff_ci = format_num(row["sufficiency_slope_ci_low"])
        suff_pass = r"\checkmark" if row["sufficiency_pass"] else r"$\times$"
        spec_ci = format_num(row["specificity_ci_low"])
        spec_pass = r"\checkmark" if row["specificity_pass"] else r"$\times$"
        topic_ci = format_num(row["topic_tvd_ci_high"])
        topic_pass = r"\checkmark" if row["topic_pass"] else r"$\times$"
        decision = r"\textbf{NO\_GO}" if row["axis_decision"] == "NO_GO" else r"\textbf{GO}"

        tex_lines.append(
            f"{axis_name:<10} & {suff_ci:<10} & {suff_pass:<8} & {spec_ci:<10} & {spec_pass:<8} & {topic_ci:<10} & {topic_pass:<8} & {decision} \\\\"
        )

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1ex}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\textbf{Note:} 事前登録規約に従い、$\text{CI}_{\text{low}} > 0.05$（Sufficiency / Specificity）未達のため、パイプライン判定は \textbf{NO\_GO}（\texttt{pipeline\_continues = False}）となり、以降の分析は探索的・定性的分析として位置づけられる。",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(tex_lines)

# =========================================================================
# 2. Spatiotemporal Dynamics Table
# =========================================================================
def generate_spatiotemporal_table(df_spatio, df_atten):
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{V3 探索的時空間ダイナミクス（Qwen 2.5 1.5B Discovery）：直接読み出しピーク深度 $d_D$ と因果介入ピーク深度 $d_C$ の空間的解離、および下流層媒介減衰率。}",
        r"\label{tab:v3_spatiotemporal_dynamics}",
        r"\begin{tabular}{l ccccc ccc}",
        r"\toprule",
        r" & \multicolumn{5}{c}{\textbf{Peak \& Center-of-Mass Depth}} & \multicolumn{3}{c}{\textbf{Mediated Attenuation}} \\",
        r"\cmidrule(lr){2-6} \cmidrule(lr){7-9}",
        r"\textbf{Axis} & $d_D^{\text{peak}}$ & $d_C^{\text{peak}}$ & $\Delta d^{\text{peak}}$ & $d_D^{\text{center}}$ & $d_C^{\text{center}}$ & Layer ($d$) & Atten. Ratio & 95\% CI \\",
        r"\midrule",
    ]

    merged = pd.merge(df_spatio, df_atten, on="axis", how="left")
    for _, row in merged.iterrows():
        axis_name = str(row["axis"]).capitalize()
        d_d_p = format_num(row["d_D_peak"], 3)
        d_c_p = format_num(row["d_C_peak"], 3)
        delta_p = format_num(row["delta_d_peak"], 3)
        d_d_c = format_num(row["d_D_center"], 3)
        d_c_c = format_num(row["d_C_center"], 3)

        m_layer = f"L{int(row['mediator_layer'])} ({row['mediator_depth']:.2f})"
        at_ratio = f"{row['attenuation_ratio'] * 100:.2f}" + r"\%"
        at_ci = f"[{row['ratio_ci_low'] * 100:.2f}, {row['ratio_ci_high'] * 100:.2f}]" + r"\%"

        tex_lines.append(
            f"{axis_name:<10} & {d_d_p:<8} & {d_c_p:<8} & {delta_p:<8} & {d_d_c:<8} & {d_c_c:<8} & {m_layer:<12} & {at_ratio:<12} & {at_ci} \\\\"
        )

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1ex}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\textbf{Note:} 全軸で $\Delta d^{\text{peak}} < 0$ であり、直接読み出しが可能な表現層よりも前の層で因果効果が最大化する「時空間的解離」が観測された。しかし、媒介減衰率はいずれも微小（$< 1.0\%$）であり、下流媒介仮説の成立は限定的である。",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(tex_lines)

# =========================================================================
# 3. Mediated Attenuation Detailed Table
# =========================================================================
def generate_mediated_attenuation_table(df_atten):
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{V3 探索的媒介減衰詳細（Mediated Attenuation Analysis）：中間層（Mediator Layer）への介入による総変位量 $T$、残差変位 $R$、絶対減衰量 $M = T - R$、減衰率（Attenuation Ratio）、およびブートストラップ95\%信頼区間。}",
        r"\label{tab:v3_mediated_attenuation}",
        r"\begin{tabular}{l ccccc c}",
        r"\toprule",
        r"\textbf{Axis} & \textbf{Mediator Layer ($d$)} & \textbf{Total Shift $T$} & \textbf{Residual $R$} & \textbf{Attenuation $M$} & \textbf{Attenuation Ratio (\%)} & \textbf{95\% CI (\%)} \\",
        r"\midrule",
    ]

    if df_atten is not None and len(df_atten) > 0:
        for _, row in df_atten.iterrows():
            ax = str(row["axis"]).capitalize()
            layer_str = f"L{int(row['mediator_layer'])} ({row['mediator_depth']:.2f})"
            atten_val = row["mediated_attenuation"]
            # Total shift approximate / representation
            t_shift = f"{atten_val / (row['attenuation_ratio'] + 1e-8):.4f}" if row['attenuation_ratio'] > 0 else "---"
            r_shift = f"{(atten_val / (row['attenuation_ratio'] + 1e-8)) - atten_val:.4f}" if row['attenuation_ratio'] > 0 else "---"
            m_val = format_num(atten_val)
            ratio_str = f"{row['attenuation_ratio'] * 100:.2f}" + r"\%"
            ci_str = f"[{row['ratio_ci_low'] * 100:.2f}, {row['ratio_ci_high'] * 100:.2f}]" + r"\%"

            tex_lines.append(f"{ax:<10} & {layer_str:<18} & {t_shift:<14} & {r_shift:<12} & {m_val:<14} & {ratio_str:<22} & {ci_str} \\\\")
    else:
        tex_lines.append(r"--- & --- & --- & --- & --- & --- & --- \\")

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1ex}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\textbf{Note:} $M = T - R$ は媒介層の潜在表現を統制したことによる自己報告変位の減衰量。Valenceでは 95\% CI が負値を含み（$[-1.77\%, 3.42\%]$）、媒介効果が統計的に確証されないことを示す。",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(tex_lines)

# =========================================================================
# 4. Confirmatory Details Table
# =========================================================================
def generate_confirmatory_details_table(df_conf):
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{V3 独立ファミリー検証詳細（Confirmatory Replication Details）：Llama 3.2、Gemma 3、OLMo 2 における事前登録仮説（H1--H4）の推定値、95\%信頼区間、判定閾値、および合否判定（Pass/Fail）。}",
        r"\label{tab:v3_confirmatory_details}",
        r"\begin{tabular}{ll cccc c}",
        r"\toprule",
        r"\textbf{Family} & \textbf{Hypothesis} & \textbf{Axis} & \textbf{Estimate} & \textbf{95\% CI} & \textbf{Threshold} & \textbf{Pass/Fail} \\",
        r"\midrule",
    ]

    hypo_names = {
        "H1_dissociation": r"H1: Dissociation ($\Delta d$)",
        "H2_sufficiency": r"H2: Sufficiency ($\beta_1$)",
        "H3_mediation": r"H3: Mediation ($M$)",
        "H4_temporal_contrast": r"H4: Contrast ($\Delta\tau$)",
    }

    if df_conf is not None and len(df_conf) > 0:
        prev_fam = ""
        for _, row in df_conf.iterrows():
            fam = str(row["family"])
            h_key = str(row["hypothesis"])
            h_label = hypo_names.get(h_key, h_key)
            ax = str(row["axis"]).capitalize()
            est = format_num(row["estimate"])
            ci = f"[{format_num(row['ci_low'])}, {format_num(row['ci_high'])}]"
            thresh = str(row["threshold"])
            p_bool = row["pass"]
            p_str = r"\checkmark \textbf{PASS}" if p_bool else r"$\times$ FAIL"

            fam_str = fam if fam != prev_fam else ""
            if prev_fam != "" and fam != prev_fam:
                tex_lines.append(r"\midrule")
            prev_fam = fam

            tex_lines.append(f"{fam_str:<14} & {h_label:<24} & {ax:<8} & {est:<10} & {ci:<18} & {thresh:<12} & {p_str} \\\\")
    else:
        tex_lines.append(r"--- & --- & --- & --- & --- & --- & --- \\")

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1ex}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\textbf{Note:} 時空間解離（H1）は Llama 3.2 および OLMo 2 で再現されたが、因果充足性（H2）や媒介性（H3）は全モデルで事前登録閾値（$\text{CI}_{\text{low}} > 0$）を満たさず FAIL となった。",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(tex_lines)

# =========================================================================
# 5. Confirmatory Replication Matrix Table
# =========================================================================
def generate_confirmatory_table(df_conf, df_conf_matrix):
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{V3 独立ファミリー再現性マトリックス：探索段階（Qwen）で得られた知見の独立3ファミリー（Llama 3.2, Gemma 3, OLMo 2）における検証結果総括。}",
        r"\label{tab:v3_confirmatory_matrix}",
        r"\begin{tabular}{l cccc c}",
        r"\toprule",
        r"\textbf{Family} & \textbf{H1 (Dissociation)} & \textbf{H2 (Sufficiency)} & \textbf{H3 (Mediation)} & \textbf{H4 (Temporal Contrast)} & \textbf{All Confirmed?} \\",
        r"\midrule",
    ]

    for _, row in df_conf_matrix.iterrows():
        fam = row["family"]
        h1 = r"\checkmark" if row["H1"] else r"$\times$"
        h2 = r"\checkmark" if row["H2"] else r"$\times$"
        h3 = r"\checkmark" if row["H3"] else r"$\times$"
        h4 = r"\checkmark" if row["H4"] else r"$\times$"
        all_c = r"\textbf{YES}" if row["all_confirmed"] else r"\textbf{NO}"

        tex_lines.append(
            f"{fam:<12} & {h1:<12} & {h2:<12} & {h3:<12} & {h4:<12} & {all_c} \\\\"
        )

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1ex}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\textbf{Note:} H1（時空間解離 $\Delta d < 0$）は Llama 3.2 および OLMo 2 において再現されたが、効果量閾値および媒介性仮説（H2--H4）はいずれの独立モデルでも満たされず、因果的普遍性の成立には限定的であることが確認された。",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(tex_lines)

# =========================================================================
# Markdown Summary
# =========================================================================
def generate_markdown_summary(df_gate, df_spatio, df_atten, df_conf, df_conf_matrix):
    md_lines = [
        "# V3 Stage: Causal Utilization and Spatiotemporal Dynamics Summary",
        "",
        "## 1. Pre-registered Gate Criteria Evaluation",
        "",
        "| Axis | Sufficiency CI Low | Sufficiency Pass | Specificity CI Low | Specificity Pass | Topic TVD CI High | Topic Pass | Overall Decision |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, row in df_gate.iterrows():
        md_lines.append(
            f"| {str(row['axis']).capitalize()} | {format_num(row['sufficiency_slope_ci_low'])} | {row['sufficiency_pass']} | "
            f"{format_num(row['specificity_ci_low'])} | {row['specificity_pass']} | "
            f"{format_num(row['topic_tvd_ci_high'])} | {row['topic_pass']} | **{row['axis_decision']}** |"
        )
    md_lines.extend([
        "",
        "> **Protocol Note:** CI low failed the raw scale thresholds (0.05/0.10). In accordance with the pre-registered protocol, the causal pipeline decision is **NO_GO** (`pipeline_continues = False`). All downstream analyses are interpreted as exploratory spatiotemporal discovery.",
        "",
        "## 2. Spatiotemporal Dissociation (Qwen 2.5 1.5B Discovery)",
        "",
        "| Axis | Readout Peak $d_D$ | Causal Peak $d_C$ | $\\Delta d$ | Readout Center | Causal Center | Mediator Layer ($d$) | Attenuation Ratio (%) | Atten. 95% CI (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    merged = pd.merge(df_spatio, df_atten, on="axis", how="left")
    for _, row in merged.iterrows():
        md_lines.append(
            f"| {str(row['axis']).capitalize()} | {format_num(row['d_D_peak'], 3)} | {format_num(row['d_C_peak'], 3)} | "
            f"{format_num(row['delta_d_peak'], 3)} | {format_num(row['d_D_center'], 3)} | {format_num(row['d_C_center'], 3)} | "
            f"L{int(row['mediator_layer'])} ({row['mediator_depth']:.2f}) | {row['attenuation_ratio']*100:.2f}% | "
            f"[{row['ratio_ci_low']*100:.2f}, {row['ratio_ci_high']*100:.2f}]% |"
        )
    md_lines.extend([
        "",
        "## 3. Confirmatory Replication Matrix across Independent Families",
        "",
        "| Family | H1 (Dissociation $\\Delta d < 0$) | H2 (Sufficiency $\\beta > 0$) | H3 (Mediation $\\beta > 0$) | H4 (Temporal Contrast $\\beta > 0$) | All Confirmed? |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for _, row in df_conf_matrix.iterrows():
        md_lines.append(
            f"| {row['family']} | {'PASS' if row['H1'] else 'FAIL'} | {'PASS' if row['H2'] else 'FAIL'} | "
            f"{'PASS' if row['H3'] else 'FAIL'} | {'PASS' if row['H4'] else 'FAIL'} | **{row['all_confirmed']}** |"
        )
    md_lines.append("")
    return "\n".join(md_lines)

def main():
    parser = argparse.ArgumentParser(description="Generate V3 Causal Tables for Paper")
    parser.add_argument("--repo-root", default=".", help="Root directory of the repo")
    parser.add_argument("--out-dir", default="iclr2027/tables", help="Directory to save LaTeX and Markdown tables")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df_gate, df_spatio, df_atten, df_conf, df_conf_matrix = load_data(args.repo_root)

    if df_gate is None:
        print("Warning: V3 data files not found.")
        return

    # 1. Gate Table
    gate_tex = generate_gate_table(df_gate)
    with open(os.path.join(args.out_dir, "v3_gate_decision.tex"), "w", encoding="utf-8") as f:
        f.write(gate_tex)

    # 2. Spatiotemporal Dynamics Table
    spatio_tex = generate_spatiotemporal_table(df_spatio, df_atten)
    with open(os.path.join(args.out_dir, "v3_spatiotemporal_dynamics.tex"), "w", encoding="utf-8") as f:
        f.write(spatio_tex)

    # 3. Mediated Attenuation Detailed Table
    atten_tex = generate_mediated_attenuation_table(df_atten)
    with open(os.path.join(args.out_dir, "v3_mediated_attenuation.tex"), "w", encoding="utf-8") as f:
        f.write(atten_tex)

    # 4. Confirmatory Details Table
    conf_det_tex = generate_confirmatory_details_table(df_conf)
    with open(os.path.join(args.out_dir, "v3_confirmatory_details.tex"), "w", encoding="utf-8") as f:
        f.write(conf_det_tex)

    # 5. Confirmatory Replication Matrix Table
    conf_tex = generate_confirmatory_table(df_conf, df_conf_matrix)
    with open(os.path.join(args.out_dir, "v3_confirmatory_matrix.tex"), "w", encoding="utf-8") as f:
        f.write(conf_tex)

    # 6. Comprehensive LaTeX summary (All 5 tables combined)
    tex_summary = "% ================================================================\n" \
                  "% V3 Stage: Causal Utilization and Spatiotemporal Dynamics Complete Summary Tables\n" \
                  "% ================================================================\n\n" \
                  + gate_tex + "\n\n" + spatio_tex + "\n\n" + atten_tex + "\n\n" + conf_det_tex + "\n\n" + conf_tex + "\n"
    with open(os.path.join(args.out_dir, "v3_causal_utilization_summary.tex"), "w", encoding="utf-8") as f:
        f.write(tex_summary)

    # 7. Markdown Summary
    md_summary = generate_markdown_summary(df_gate, df_spatio, df_atten, df_conf, df_conf_matrix)
    with open(os.path.join(args.out_dir, "v3_causal_utilization_summary.md"), "w", encoding="utf-8") as f:
        f.write(md_summary)

    print(f"[+] Successfully generated all V3 tables in {args.out_dir}:")
    print(f"    - v3_gate_decision.tex")
    print(f"    - v3_spatiotemporal_dynamics.tex")
    print(f"    - v3_mediated_attenuation.tex")
    print(f"    - v3_confirmatory_details.tex")
    print(f"    - v3_confirmatory_matrix.tex")
    print(f"    - v3_causal_utilization_summary.tex")
    print(f"    - v3_causal_utilization_summary.md")

if __name__ == "__main__":
    main()
