#!/usr/bin/env python3
"""
Summarize V3 Stage (Causal Utilization and Spatiotemporal Dynamics) Results for ICLR 2027 Paper.

Outputs publication-ready LaTeX tables (with booktabs) and Markdown summaries:
1. Gate Criteria Evaluation Table (Sufficiency, Specificity, Mediated Attenuation, Topic Control) -> NO_GO Decision
2. Spatiotemporal Dissociation Table (Direct Readout vs Causal Peak Dynamics, Mediated Attenuation)
3. Confirmatory Replication Matrix Table across Independent Families (Llama 3.2, Gemma 3, OLMo 2)

Usage:
    python3 scripts/summarize_v3_causal_utilization.py [--out-dir iclr2027/tables]
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
    return f"{val:.{decimals}f}"

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
        tvd_ci = format_num(row["topic_tvd_ci_high"])
        tvd_pass = r"\checkmark" if row["topic_pass"] else r"$\times$"
        decision = r"\textbf{NO\_GO}" if row["axis_decision"] == "NO_GO" else row["axis_decision"]

        line = f"{axis_name} & {suff_ci} & {suff_pass} & {spec_ci} & {spec_pass} & {tvd_ci} & {tvd_pass} & {decision} \\\\"
        tex_lines.append(line)

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
        d_d_peak = format_num(row["d_D_peak"], 3)
        d_c_peak = format_num(row["d_C_peak"], 3)
        delta_d = format_num(row["delta_d_peak"], 3)
        d_d_center = format_num(row["d_D_center"], 3)
        d_c_center = format_num(row["d_C_center"], 3)

        med_layer = f"L{int(row['mediator_layer'])} ({row['mediator_depth']:.2f})"
        atten_ratio = f"{row['attenuation_ratio'] * 100:.2f}\\%"
        ci_str = f"[{row['ratio_ci_low'] * 100:.2f}, {row['ratio_ci_high'] * 100:.2f}]\\%"

        line = f"{axis_name} & {d_d_peak} & {d_c_peak} & {delta_d} & {d_d_center} & {d_c_center} & {med_layer} & {atten_ratio} & {ci_str} \\\\"
        tex_lines.append(line)

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1ex}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\textbf{Note:} $\Delta d^{\text{peak}} = d_C^{\text{peak}} - d_D^{\text{peak}} < 0$ は、因果効果ピークがプローブ最適デコード深度よりも浅い（上流の）中間層に位置する「時空間解離」を示す。下流アブレーションによる減衰率はわずかであり、単一媒介経路への過度な依存がないことを示唆する。",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(tex_lines)

def generate_confirmatory_table(df_conf, df_conf_matrix):
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{V3 独立ファミリー検証再現性マトリックス：探索的発見（Qwen 2.5）で得られた時空間仮説 H1--H4 を、独立した3つの未見モデルファミリーで検証した結果。}",
        r"\label{tab:v3_confirmatory_matrix}",
        r"\begin{tabular}{l cccc c}",
        r"\toprule",
        r"\textbf{Family} & \textbf{H1: Dissociation} & \textbf{H2: Sufficiency} & \textbf{H3: Mediation} & \textbf{H4: Contrast} & \textbf{All Confirmed?} \\",
        r" & ($\Delta d < 0$) & ($\beta_{\text{suff}} > 0$) & ($\beta_{\text{med}} > 0$) & ($\beta_{\text{contrast}} > 0$) & \\",
        r"\midrule",
    ]

    for _, row in df_conf_matrix.iterrows():
        fam = row["family"]
        h1 = r"\checkmark \textbf{Pass}" if row["H1"] else r"$\times$ Fail"
        h2 = r"\checkmark \textbf{Pass}" if row["H2"] else r"$\times$ Fail"
        h3 = r"\checkmark \textbf{Pass}" if row["H3"] else r"$\times$ Fail"
        h4 = r"\checkmark \textbf{Pass}" if row["H4"] else r"$\times$ Fail"
        all_c = r"\textbf{True}" if row["all_confirmed"] else r"\textbf{False}"

        line = f"{fam} & {h1} & {h2} & {h3} & {h4} & {all_c} \\\\"
        tex_lines.append(line)

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
    gate_tex_path = os.path.join(args.out_dir, "v3_gate_decision.tex")
    with open(gate_tex_path, "w", encoding="utf-8") as f:
        f.write(gate_tex)
    print(f"Generated: {gate_tex_path}")

    # 2. Spatiotemporal Dynamics Table
    spatio_tex = generate_spatiotemporal_table(df_spatio, df_atten)
    spatio_tex_path = os.path.join(args.out_dir, "v3_spatiotemporal_dynamics.tex")
    with open(spatio_tex_path, "w", encoding="utf-8") as f:
        f.write(spatio_tex)
    print(f"Generated: {spatio_tex_path}")

    # 3. Confirmatory Replication Matrix Table
    conf_tex = generate_confirmatory_table(df_conf, df_conf_matrix)
    conf_tex_path = os.path.join(args.out_dir, "v3_confirmatory_matrix.tex")
    with open(conf_tex_path, "w", encoding="utf-8") as f:
        f.write(conf_tex)
    print(f"Generated: {conf_tex_path}")

    # 4. Comprehensive LaTeX summary
    tex_summary = "% ================================================================\n" \
                  "% V3 Stage: Causal Utilization and Spatiotemporal Dynamics Complete Summary Tables\n" \
                  "% Generated from v3_causal_utilization_summary.md\n" \
                  "% ================================================================\n\n" \
                  + gate_tex + "\n\n" + spatio_tex + "\n\n" + conf_tex + "\n"
    tex_summary_path = os.path.join(args.out_dir, "v3_causal_utilization_summary.tex")
    with open(tex_summary_path, "w", encoding="utf-8") as f:
        f.write(tex_summary)
    print(f"Generated: {tex_summary_path}")

    # 5. Markdown Summary
    md_summary = generate_markdown_summary(df_gate, df_spatio, df_atten, df_conf, df_conf_matrix)
    md_path = os.path.join(args.out_dir, "v3_causal_utilization_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_summary)
    print(f"Generated: {md_path}")

if __name__ == "__main__":
    main()
