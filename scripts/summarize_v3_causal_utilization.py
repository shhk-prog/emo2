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
        r"\caption{V3 事前登録 Gate 判定結果：因果介入パイプライン継続可否の評価規約。各判定軸の信頼区間下限／上限が 1--9 raw scale 閾値（Sufficiency: 0.10, Specificity: 0.05, Endogenous: 0.05）および TVD 上限（0.15）を満たすかを判定。}",
        r"\label{tab:v3_gate_decision}",
        r"\begin{tabular}{l cccc cccc c}",
        r"\toprule",
        r"\textbf{Axis} & \multicolumn{2}{c}{\textbf{Sufficiency ($\beta_1$)}} & \multicolumn{2}{c}{\textbf{Specificity ($\beta_1$)}} & \multicolumn{2}{c}{\textbf{Endogenous ($M$)}} & \multicolumn{2}{c}{\textbf{Topic Control (TVD)}} & \textbf{Gate Decision} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}",
        r" & \textbf{CI Low} & \textbf{Pass?} & \textbf{CI Low} & \textbf{Pass?} & \textbf{CI Low} & \textbf{Pass?} & \textbf{CI High} & \textbf{Pass?} & \\",
        r"\midrule",
    ]

    for _, row in df_gate.iterrows():
        axis_name = str(row["axis"]).capitalize()
        suff_ci = format_num(row["sufficiency_slope_ci_low"])
        suff_pass = r"\checkmark" if row["sufficiency_pass"] else r"$\times$"
        spec_ci = format_num(row["specificity_ci_low"])
        spec_pass = r"\checkmark" if row["specificity_pass"] else r"$\times$"
        atten_ci = format_num(row["attenuation_ci_low"])
        atten_pass = r"\checkmark" if row["attenuation_pass"] else r"$\times$"
        topic_ci = format_num(row["topic_tvd_ci_high"])
        topic_pass = r"\checkmark" if row["topic_pass"] else r"$\times$"
        decision = r"\textbf{NO\_GO}" if row["axis_decision"] == "NO_GO" else r"\textbf{GO}"

        tex_lines.append(
            f"{axis_name:<10} & {suff_ci:<10} & {suff_pass:<8} & {spec_ci:<10} & {spec_pass:<8} & {atten_ci:<10} & {atten_pass:<8} & {topic_ci:<10} & {topic_pass:<8} & {decision} \\\\"
        )

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1ex}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\textbf{Note:} 事前登録 Gate 判定基準（Sufficiency: $\text{CI}_{\text{low}} > 0.10$, Specificity: $\text{CI}_{\text{low}} > 0.05$, Endogenous Relevance: $\text{CI}_{\text{low}} > 0.05$, Topic TVD: $\text{CI}_{\text{high}} < 0.15$）。全軸でSufficiency / Specificity / Endogenousの事前定義thresholdを満たさなかったため、パイプライン判定は \textbf{NO\_GO}（\texttt{pipeline\_continues = False}）となり、以降の分析は探索的・定性的分析として位置づけられる。",
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
        r"\caption{V3 探索的媒介減衰詳細（Mediated Attenuation Analysis）：中間層（Mediator Layer）への介入による総変位量 $T$、残差変位 $R$、絶対減衰量 $M = T - R$、ランダム統制 $M_{\text{rand}}$、正味媒介減衰量 $M_{\text{net}} = M - M_{\text{rand}}$、減衰率（Attenuation Ratio）、およびブートストラップ95\%信頼区間。}",
        r"\label{tab:v3_mediated_attenuation}",
        r"\begin{tabular}{ll cccccc cc}",
        r"\toprule",
        r"\textbf{Axis} & \textbf{Mediator ($d$)} & $T$ & $R$ & $M$ & $M_{\text{rand}}$ & $M_{\text{net}}$ & \textbf{Atten. Ratio} & \textbf{95\% CI ($M$)} & \textbf{95\% CI ($M_{\text{net}}$)} \\",
        r"\midrule",
    ]

    if df_atten is not None and len(df_atten) > 0:
        for _, row in df_atten.iterrows():
            ax = str(row["axis"]).capitalize()
            layer_str = f"L{int(row['mediator_layer'])} ({row['mediator_depth']:.2f})"
            t_shift = format_num(row.get("total_shift", np.nan))
            r_shift = format_num(row.get("residual_shift", np.nan))
            m_val = format_num(row.get("mediated_attenuation", np.nan))
            m_rand = format_num(row.get("attenuation_random", np.nan))
            m_net = format_num(row.get("net_attenuation", np.nan))
            ratio_val = row.get("attenuation_ratio", np.nan)
            ratio_str = f"{ratio_val * 100:.2f}" + r"\%" if pd.notna(ratio_val) else "---"
            m_ci_l = row.get("mediated_ci_low", row.get("attenuation_ci_low", np.nan))
            m_ci_h = row.get("mediated_ci_high", row.get("attenuation_ci_high", np.nan))
            ci_m = f"[{format_num(m_ci_l)}, {format_num(m_ci_h)}]" if (pd.notna(m_ci_l) and pd.notna(m_ci_h)) else "---"
            net_ci_l = row.get("net_ci_low", np.nan)
            net_ci_h = row.get("net_ci_high", np.nan)
            ci_net = f"[{format_num(net_ci_l)}, {format_num(net_ci_h)}]" if (pd.notna(net_ci_l) and pd.notna(net_ci_h)) else "---"
            tex_lines.append(f"{ax:<10} & {layer_str:<16} & {t_shift:<8} & {r_shift:<8} & {m_val:<10} & {m_rand:<10} & {m_net:<10} & {ratio_str:<14} & {ci_m:<22} & {ci_net} \\\\")
    else:
        tex_lines.append(r"--- & --- & --- & --- & --- & --- & --- & --- & --- & --- \\")

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1ex}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\textbf{Note:} $M = T - R$ は媒介層の潜在表現統制による変位減衰量、$M_{\text{net}} = M - M_{\text{rand}}$ はランダム部分空間統制を差し引いた正味媒介減衰量。ValenceおよびArousalともに 95\% CI がゼロを跨ぎ、統計的に堅牢な媒介効果は支持されない。",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(tex_lines)

# =========================================================================
# 4. Confirmatory Details Table
# =========================================================================
def generate_confirmatory_details_table(df_conf, repo_root="."):
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{V3 独立ファミリー検証詳細（Confirmatory Replication Details）：Llama 3.2、Gemma 3、OLMo 2 における事前登録仮説（H1--H4）のValenceおよびArousal双方の推定値、ファミリー固有95\%信頼区間、判定閾値、および事前登録CI基準に基づく合否判定（Pass/Fail）。}",
        r"\label{tab:v3_confirmatory_details}",
        r"\begin{tabular}{ll cccc c}",
        r"\toprule",
        r"\textbf{Family} & \textbf{Hypothesis} & \textbf{Axis} & \textbf{Estimate} & \textbf{Family 95\% CI} & \textbf{Threshold} & \textbf{Pass/Fail} \\",
        r"\midrule",
    ]

    if df_conf is not None and len(df_conf) > 0:
        fams = ["Llama 3.2", "Gemma 3", "OLMo 2"]
        # In case family names differ slightly in CSV
        csv_fams = df_conf["family"].unique()
        ordered_fams = [f for f in fams if f in csv_fams] or list(csv_fams)

        METRIC_ORDER = [
            ("H1_dissociation", "peak_dissociation", r"H1 Peak Dissoc. ($\Delta d^*$)", 2, r"$\text{CI}_{\text{low}} > 0$"),
            ("H1_dissociation", "center_dissociation", r"H1 Center Dissoc. ($\Delta\bar{d}$)", 2, r"$\text{CI}_{\text{low}} > 0$"),
            ("H2_sufficiency", "sufficiency_slope", r"H2: Sufficiency ($\beta_1$)", 4, r"$\text{CI}_{\text{low}} > 0.10$"),
            ("H3_mediation", "mediated_M", r"H3: Mediation ($M$)", 4, r"$\text{CI}_{\text{low}}(M) > 0$"),
            ("H3_mediation", "net_vs_random", r"H3: Net vs. Random ($M_{\text{net}}$)", 4, r"$\text{CI}_{\text{low}}(M_{\text{net}}) > 0$"),
            ("H4_temporal_contrast", "temporal_contrast", r"H4: Contrast ($\Delta C$)", 4, r"$\text{CI}_{\text{low}} > 0$"),
        ]

        for idx_fam, fam in enumerate(ordered_fams):
            sub_fam = df_conf[df_conf["family"] == fam]
            fam_str = f"\\multirow{{12}}{{*}}{{\\textbf{{{fam}}}}}"
            first_row = True

            for hyp, met, h_label, prec, th_str in METRIC_ORDER:
                for ax in ["valence", "arousal"]:
                    match = sub_fam[(sub_fam["hypothesis"] == hyp) & (sub_fam["axis"] == ax)]
                    if "metric" in sub_fam.columns:
                        match = match[match["metric"] == met]

                    if len(match) > 0:
                        r0 = match.iloc[0]
                        est_val = r0.get("estimate", np.nan)
                        est_str = format_num(est_val, prec)
                        l_ci = r0.get("ci_low", np.nan)
                        u_ci = r0.get("ci_high", np.nan)
                        ci_str = f"[{format_num(l_ci, prec)}, {format_num(u_ci, prec)}]" if (pd.notna(l_ci) and pd.notna(u_ci)) else "---"
                        passed = bool(r0.get("pass", False))
                        p_str = r"\checkmark \textbf{PASS}" if passed else r"$\times$ FAIL"
                    else:
                        est_str = "---"
                        ci_str = "---"
                        p_str = "---"

                    f_label = fam_str if first_row else ""
                    first_row = False
                    tex_lines.append(f"{f_label:<28} & {h_label:<38} & {ax.capitalize():<8} & {est_str:<10} & {ci_str:<18} & {th_str:<26} & {p_str} \\\\")

            if idx_fam < len(ordered_fams) - 1:
                tex_lines.append(r"\midrule")
    else:
        tex_lines.append(r"--- & --- & --- & --- & --- & --- & --- \\")

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1ex}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\textbf{Note:} 各ファミリー固有の95\%ブートストラップ信頼区間および事前登録判定基準（H1: $\text{CI}_{\text{low}} > 0$; H2: $\text{CI}_{\text{low}} > 0.10$; H3: $\text{CI}_{\text{low}}(M) > 0$ かつ $\text{CI}_{\text{low}}(M_{\text{net}}) > 0$; H4: $\text{CI}_{\text{low}} > 0$）に基づく評価。H3の支持には内因性変位減衰量 $M$ の信頼区間下限が正であること（$\text{CI}_{\text{low}}(M) > 0$）に加え、ランダム部分空間統制を差し引いた正味減衰量 $M_{\text{net}}$ の信頼区間下限も正であること（$\text{CI}_{\text{low}}(M_{\text{net}}) > 0$）が要求される。H1--H3はいずれのモデル・軸でも事前登録基準を満たさなかった（FAIL）。H4では各モデルで軸レベルの正の効果量（$\text{CI}_{\text{low}} > 0$）が観測されたが、Gate判定がNO\_GOであるため事前登録パイプライン全体のConfirmationは不成立となった。",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(tex_lines)

# =========================================================================
# 5. Confirmatory Replication Matrix Table
# =========================================================================
def generate_confirmatory_table(df_conf, df_conf_matrix, repo_root="."):
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{V3 独立ファミリー再現性マトリックス：探索段階（Qwen）で得られた知見の独立3ファミリー（Llama 3.2, Gemma 3, OLMo 2）における検証結果総括。各仮説はValenceおよびArousalの双方が基準を満たした場合のみ支持（$\checkmark$）とされる。}",
        r"\label{tab:v3_confirmatory_matrix}",
        r"\begin{tabular}{l cccc c}",
        r"\toprule",
        r"\textbf{Family} & \textbf{H1 (Dissociation)} & \textbf{H2 (Sufficiency)} & \textbf{H3 (Mediation)} & \textbf{H4 (Temporal Contrast)} & \textbf{All Confirmed?} \\",
        r"\midrule",
    ]

    if df_conf_matrix is not None and len(df_conf_matrix) > 0:
        for _, r in df_conf_matrix.iterrows():
            fam = str(r["family"])
            h1 = r"\checkmark" if r["H1"] else r"$\times$"
            h2 = r"\checkmark" if r["H2"] else r"$\times$"
            h3 = r"\checkmark" if r["H3"] else r"$\times$"
            h4 = r"\checkmark" if r["H4"] else r"$\times$"
            all_c = r"\textbf{YES}" if r["all_confirmed"] else r"\textbf{NO}"
            tex_lines.append(f"{fam:<14} & {h1:<12} & {h2:<12} & {h3:<12} & {h4:<12} & {all_c} \\\\")
    else:
        tex_lines.append(r"--- & --- & --- & --- & --- & --- \\")

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1ex}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\textbf{Note:} 全4仮説の総合判定マトリックス。各セルは該当仮説においてValenceおよびArousalの双方が事前登録基準（95\% CI）を満たした場合に合致（$\checkmark$）とする。全モデルファミリーにおいてH1--H3は棄却され、H4のみが支持されたため、全体結論として情動表現の機能的・因果的活用仮説は支持されなかった（All Confirmed = NO）。",
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
        "| Family | H1 (Dissociation $\\Delta d > 0$) | H2 (Sufficiency $\\beta_1 > 0.10$) | H3 (Mediation $M > 0$) | H4 (Temporal Contrast $\\Delta C > 0$) | All Confirmed? |",
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
    conf_det_tex = generate_confirmatory_details_table(df_conf, repo_root=args.repo_root)
    with open(os.path.join(args.out_dir, "v3_confirmatory_details.tex"), "w", encoding="utf-8") as f:
        f.write(conf_det_tex)

    # 5. Confirmatory Replication Matrix Table
    conf_tex = generate_confirmatory_table(df_conf, df_conf_matrix, repo_root=args.repo_root)
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
