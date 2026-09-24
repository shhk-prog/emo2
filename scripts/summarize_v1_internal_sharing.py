#!/usr/bin/env python3
"""
Summarize V1 Stage (Internal Representation and Causal Sharing: E1-E6) Results for ICLR 2027 Paper.

Outputs publication-ready LaTeX tables (with booktabs and multirow) and Markdown summaries:
1. E1: Peak Decodability Table (v1_peak_decodability.tex)
2. E2: Shared Representation Geometry Table (v1_shared_geometry.tex)
3. Phase B: Semantic and Contextual Controls Table (v1_semantic_controls.tex)
4. E3: Shared Causal Map Table (v1_causal_profile.tex)
5. E4: Causal Interchangeability Table (v1_interchangeability.tex)
6. E6: Task-Specific Specialization Table (v1_specialization.tex)
7. Comprehensive Summary: v1_internal_sharing_summary.tex / .md

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
    e2_path = os.path.join(dir_path, "table_v1_2_shared_geometry.csv")
    e3_ctrl_path = os.path.join(dir_path, "table_v1_3_semantic_controls.csv")
    e3_path = os.path.join(dir_path, "table_v1_4_causal_map.csv")
    e4_path = os.path.join(dir_path, "table_v1_5_interchangeability.csv")
    e6_path = os.path.join(dir_path, "table_v1_6_specialization.csv")

    df_e1 = pd.read_csv(e1_path) if os.path.exists(e1_path) else None
    df_e2 = pd.read_csv(e2_path) if os.path.exists(e2_path) else None
    df_ctrl = pd.read_csv(e3_ctrl_path) if os.path.exists(e3_ctrl_path) else None
    df_e3 = pd.read_csv(e3_path) if os.path.exists(e3_path) else None
    df_e4 = pd.read_csv(e4_path) if os.path.exists(e4_path) else None
    df_e6 = pd.read_csv(e6_path) if os.path.exists(e6_path) else None

    return df_e1, df_e2, df_ctrl, df_e3, df_e4, df_e6

def format_num(val, decimals=3):
    if pd.isna(val):
        return "---"
    val_str = f"{val:.{decimals}f}"
    if val_str.startswith("-"):
        return f"$-${val_str[1:]}"
    return val_str

def compute_fdr_bh(p_values):
    p_vals = np.asarray(p_values, dtype=float)
    n = len(p_vals)
    if n == 0:
        return np.array([])
    order = np.argsort(p_vals)
    ranked = p_vals[order]
    q_vals = ranked * n / np.arange(1, n + 1)
    q_vals = np.minimum.accumulate(q_vals[::-1])[::-1]
    q_vals = np.clip(q_vals, 0.0, 1.0)
    out = np.empty_like(q_vals)
    out[order] = q_vals
    return out

# =========================================================================
# 1. E1: Peak Decodability Table
# =========================================================================
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

            r_v = sub[(sub["task"] == "reader") & (sub["axis"] == "valence")]
            s_v = sub[(sub["task"] == "self") & (sub["axis"] == "valence")]
            r_a = sub[(sub["task"] == "reader") & (sub["axis"] == "arousal")]
            s_a = sub[(sub["task"] == "self") & (sub["axis"] == "arousal")]

            def format_cell(row):
                if len(row) > 0:
                    l = int(row["peak_layer"].iloc[0])
                    d = row["peak_relative_depth"].iloc[0]
                    r2 = row["peak_r2"].iloc[0]
                    r = row["pearson"].iloc[0] if "pearson" in row.columns else row.get("peak_r", row.get("r", pd.Series([np.nan]))).iloc[0]
                    return f"L{l} ({d:.2f})", f"{r2:.3f} ({r:.3f})"
                return "---", "---"

            rv_ld, rv_st = format_cell(r_v)
            sv_ld, sv_st = format_cell(s_v)
            ra_ld, ra_st = format_cell(r_a)
            sa_ld, sa_st = format_cell(s_a)

            dist_v = f"{abs(r_v['peak_relative_depth'].iloc[0] - s_v['peak_relative_depth'].iloc[0]):.2f}" if len(r_v) > 0 and len(s_v) > 0 else "---"
            dist_a = f"{abs(r_a['peak_relative_depth'].iloc[0] - s_a['peak_relative_depth'].iloc[0]):.2f}" if len(r_a) > 0 and len(s_a) > 0 else "---"
            d_str = f"V:{dist_v}, A:{dist_a}"

            fam_str = f"\\multirow{{4}}{{*}}{{\\textbf{{{fam_name}}}}}" if first_fam else ""
            first_fam = False

            tex_lines.append(f"{fam_str:<32} & \\multirow{{2}}{{*}}{{{aln_label}}} & Reader & {rv_ld} & {rv_st} & {ra_ld} & {ra_st} & \\multirow{{2}}{{*}}{{{d_str}}} \\\\")
            tex_lines.append(f"{'':<32} &  & Self   & {sv_ld} & {sv_st} & {sa_ld} & {sa_st} &  \\\\")

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
        r"\textbf{Note:} ReaderとSelfのdecodability peakは多くの条件で近接したが、一部のmodel / axisでは相対深度0.2以上の乖離も観測された。したがって両taskはaffect-relevant informationを共有する一方、その最適なreadout depthは完全には一致しない。",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(tex_lines)

# =========================================================================
# 2. E2: Shared Representation Geometry Table
# =========================================================================
def generate_e2_geometry_table(df_e2):
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{V1 E2（共有表現幾何）：ReaderとSelfの間での幾何アライメント。直接転移性能（Direct Transfer $R^2$）、表現類似度（RSA $\rho$）、Procrustes回転前後の転移決定係数および幾何整列ゲイン（Alignment Gain）。}",
        r"\label{tab:v1_shared_geometry}",
        r"\begin{tabular}{lll cccc c}",
        r"\toprule",
        r"\textbf{Family} & \textbf{Variant} & \textbf{Axis} & \textbf{Direct R$\to$S $R^2$} & \textbf{Direct S$\to$R $R^2$} & \textbf{RSA $\rho$} & \textbf{Aligned $R^2$} & \textbf{Alignment Gain} \\",
        r"\midrule",
    ]

    for fam_key, fam_name in FAMILY_ORDER:
        first_fam = True
        for aln, aln_label in [("base", "Base"), ("instruct", "Instruct")]:
            m_tag = f"{fam_key}_{aln}"
            sub = df_e2[df_e2["model"] == m_tag] if df_e2 is not None else pd.DataFrame()

            for idx_ax, ax in enumerate(["valence", "arousal"]):
                sub_ax = sub[sub["axis"].str.lower() == ax] if len(sub) > 0 else pd.DataFrame()
                if len(sub_ax) > 0:
                    best_row = sub_ax.sort_values(by="alignment_gain", ascending=False).iloc[0]
                    c_rs = format_num(best_row["cross_r_to_s"])
                    c_sr = format_num(best_row["cross_s_to_r"])
                    rsa_v = format_num(best_row["rsa"])
                    al_r2 = format_num(best_row["aligned_r2"])
                    gain = format_num(best_row["alignment_gain"])
                else:
                    c_rs = c_sr = rsa_v = al_r2 = gain = "---"

                fam_str = f"\\multirow{{4}}{{*}}{{\\textbf{{{fam_name}}}}}" if first_fam else ""
                first_fam = False
                var_str = f"\\multirow{{2}}{{*}}{{{aln_label}}}" if idx_ax == 0 else ""

                tex_lines.append(f"{fam_str:<32} & {var_str:<18} & {ax.capitalize():<8} & {c_rs:<12} & {c_sr:<12} & {rsa_v:<10} & {al_r2:<10} & {gain:<10} \\\\")

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
        r"\textbf{Note:} ReaderとSelfのstimulus geometryには全条件で正のRSAが観測され、多くの条件で高いrank-level similarityを示した。一方、direct cross-decodingは大きく負となる条件が多く、raw coordinate systemが直接交換可能であることは支持されない。Procrustes alignment後には性能改善がみられ、両taskのgeometryが線形変換によって部分的に整列可能であることと整合する。",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(tex_lines)

# =========================================================================
# 3. Phase B: Semantic and Contextual Controls Table
# =========================================================================
def generate_semantic_controls_table(df_ctrl):
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{V1 Phase B（意味論・文脈制御統制実験）：語彙・構文の表層的手がかりを統制したミニマルペアにおけるプロービング精度。元データ（Original）、パラフレーズ（Paraphrase）、単語シャッフル（Shuffle Drop）、文脈逆転（Outcome Reversal）における非フォールバック対数（$N_{\text{nonfallback}}$）。}",
        r"\label{tab:v1_semantic_controls}",
        r"\begin{tabular}{lll ccccc}",
        r"\toprule",
        r"\textbf{Family} & \textbf{Variant} & \textbf{Task} & \textbf{Original Acc} & \textbf{Paraphrase ($N$)} & \textbf{Word Shuffle} & \textbf{Shuffle Drop} & \textbf{Reversal Drop ($N$)} \\",
        r"\midrule",
    ]

    for fam_key, fam_name in FAMILY_ORDER:
        first_fam = True
        for aln, aln_label in [("base", "Base"), ("instruct", "Instruct")]:
            m_tag = f"{fam_key}_{aln}"
            sub = df_ctrl[df_ctrl["model"] == m_tag] if df_ctrl is not None else pd.DataFrame()

            for idx_task, (t_key, t_label) in enumerate([("reader", "Reader"), ("self", "Self")]):
                row = sub[sub["task"] == t_key] if len(sub) > 0 else pd.DataFrame()
                if len(row) > 0:
                    r0 = row.iloc[0]
                    orig = format_num(r0["original_bal_acc"])
                    para = f"{r0['paraphrase_bal_acc']:.3f} ($N={int(r0['n_paraphrase_nonfallback'])}$)"
                    shuf = format_num(r0["shuffle_bal_acc"])
                    drop = format_num(r0["shuffle_drop"])
                    rev = f"{r0['reversal_delta_probability']:.3f} ($N={int(r0['n_reversal_nonfallback'])}$)"
                else:
                    orig = para = shuf = drop = rev = "---"

                fam_str = f"\\multirow{{4}}{{*}}{{\\textbf{{{fam_name}}}}}" if first_fam else ""
                first_fam = False
                var_str = f"\\multirow{{2}}{{*}}{{{aln_label}}}" if idx_task == 0 else ""

                tex_lines.append(f"{fam_str:<32} & {var_str:<18} & {t_label:<8} & {orig:<12} & {para:<24} & {shuf:<12} & {drop:<12} & {rev:<24} \\\\")

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
        r"\textbf{Note:} Word Shuffleに対する一貫したaccuracy dropは、decodabilityが完全なbag-of-words shortcutだけでは説明しにくいことを支持する。一方、ParaphraseおよびOutcome Reversalはnonfallback sample数が小さいため、semantic validityに関する補助的evidenceとして解釈する。",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(tex_lines)

# =========================================================================
# 4. E3: Shared Causal Map Table
# =========================================================================
def generate_e3_causal_map_table(df_e3):
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{V1 E3（共有因果マップ）：刺激関連差分ベクトルへの介入に対するReaderおよびSelfの出力応答プロファイル。ピーク介入層、因果効果量（$\times 10^{-3}$）、プロファイル間の順位相関（Spearman $\rho_{\text{rank}}$）、および方向余弦類似度。}",
        r"\label{tab:v1_causal_profile}",
        r"\begin{tabular}{lll cccc cc}",
        r"\toprule",
        r" & & & \multicolumn{2}{c}{\textbf{Reader Peak}} & \multicolumn{2}{c}{\textbf{Self Peak}} & \multicolumn{2}{c}{\textbf{Circuit Sharing}} \\",
        r"\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}",
        r"\textbf{Family} & \textbf{Variant} & \textbf{Axis} & Layer ($d^*$) & Mag ($\times 10^{-3}$) & Layer ($d^*$) & Mag ($\times 10^{-3}$) & Spearman $\rho_{\text{rank}}$ & Mean Cosine \\",
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

                rank_rho = format_num(sub["reader_self_rank_spearman"].iloc[0])
                cos_sim = format_num(sub["mean_direction_cosine"].iloc[0])
            else:
                r_peak = s_peak = r_mag = s_mag = rank_rho = cos_sim = "---"

            fam_str = f"\\multirow{{2}}{{*}}{{\\textbf{{{fam_name}}}}}" if first_fam else ""
            first_fam = False

            tex_lines.append(f"{fam_str:<32} & {aln_label:<10} & Valence & {r_peak:<12} & {r_mag:<10} & {s_peak:<12} & {s_mag:<10} & {rank_rho:<10} & {cos_sim:<10} \\\\")

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
        r"\textbf{Note:} ReaderとSelfのlayer-wise causal profilesには中程度から高い正の順位相関が観測されたが、その強さはmodel間で異なり、intervention directionのcosine similarityがほぼ0となる条件も存在した。したがって、両taskはlayer-level causal sensitivityを部分的に共有するが、同一のcausal directionまたは同一回路を利用するとは結論しない。",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(tex_lines)

# =========================================================================
# 5. E4: Causal Interchangeability Table
# =========================================================================
def generate_interchangeability_table(df_e4):
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{V1 E4（因果交換可能性）：Readerタスクで形成された差分ベクトルをSelf計算へ移植した際の因果効果。適合効果（Matched Effect）、ランダム統制効果（Random Mean）、特異性（Specificity $= M - R$）、95\%信頼区間、有意確率（$p$ / FDR $q$）、および転移比率（Transfer Ratio）。}",
        r"\label{tab:v1_interchangeability}",
        r"\begin{tabular}{lll ccccc c}",
        r"\toprule",
        r"\textbf{Family} & \textbf{Variant} & \textbf{Axis} & \textbf{Matched ($\times 10^{-3}$)} & \textbf{Random ($\times 10^{-3}$)} & \textbf{Specificity ($\times 10^{-3}$)} & \textbf{95\% CI ($\times 10^{-3}$)} & $p$ / $q$ & \textbf{Transfer Ratio} \\",
        r"\midrule",
    ]

    for fam_key, fam_name in FAMILY_ORDER:
        first_fam = True
        for aln, aln_label in [("base", "Base"), ("instruct", "Instruct")]:
            m_tag = f"{fam_key}_{aln}"
            sub = df_e4[df_e4["model"] == m_tag] if df_e4 is not None else pd.DataFrame()

            for idx_ax, ax in enumerate(["valence", "arousal"]):
                sub_ax = sub[sub["axis"].str.lower() == ax] if len(sub) > 0 else pd.DataFrame()
                if len(sub_ax) > 0:
                    r0 = sub_ax.iloc[0]
                    m_eff = f"{r0['matched_effect'] * 1e3:.3f}"
                    rand = f"{r0['random_mean'] * 1e3:.3f}"
                    spec = f"{r0['specificity'] * 1e3:.3f}"
                    ci = f"[{r0['ci_low'] * 1e3:.2f}, {r0['ci_high'] * 1e3:.2f}]"
                    pq = f"{r0['p']:.3f} / {r0['q']:.3f}"
                    tr = f"{r0['transfer_ratio']:.3f}" if not pd.isna(r0['transfer_ratio']) else "---"
                else:
                    m_eff = rand = spec = ci = pq = tr = "---"

                fam_str = f"\\multirow{{4}}{{*}}{{\\textbf{{{fam_name}}}}}" if first_fam else ""
                first_fam = False
                var_str = f"\\multirow{{2}}{{*}}{{{aln_label}}}" if idx_ax == 0 else ""

                tex_lines.append(f"{fam_str:<32} & {var_str:<18} & {ax.capitalize():<8} & {m_eff:<16} & {rand:<16} & {spec:<16} & {ci:<20} & {pq:<14} & {tr:<12} \\\\")

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
        r"\textbf{Note:} Reader由来のstimulus-associated hidden-state differenceをSelfへ移植した際のmatched-minus-random specificityは、いずれのPrimary conditionでもFDR補正後に有意ではなかった。したがって、本解析からReader由来representationのcross-task causal interchangeabilityを支持するrobust evidenceは得られなかった。",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(tex_lines)

# =========================================================================
# 6. E6: Task-Specific Specialization Table
# =========================================================================
def generate_specialization_table(df_e6):
    if df_e6 is not None and "interaction_p" in df_e6.columns:
        df_e6 = df_e6.copy()
        df_e6["interaction_q"] = compute_fdr_bh(df_e6["interaction_p"])
    else:
        df_e6 = df_e6.copy() if df_e6 is not None else None

    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{V1 E6（タスク特異的因果特殊化）：ReaderおよびSelfに特異的な因果局所部位（Selective Depth）の同定、各部位における切除効果（Ablation Effect）、および Task $\times$ SiteType 交互作用効果（$\beta_{\text{int}}, p$, FDR $q$）。}",
        r"\label{tab:v1_specialization}",
        r"\begin{tabular}{lll cccc ccc}",
        r"\toprule",
        r" & & & \multicolumn{2}{c}{\textbf{Reader Site Effect}} & \multicolumn{2}{c}{\textbf{Self Site Effect}} & \multicolumn{3}{c}{\textbf{Interaction}} \\",
        r"\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-10}",
        r"\textbf{Family} & \textbf{Variant} & \textbf{Selective Sites} & On Reader & On Self & On Reader & On Self & $\beta_{\text{int}}$ & $p$ & FDR $q$ \\",
        r"\midrule",
    ]

    for fam_key, fam_name in FAMILY_ORDER:
        first_fam = True
        for aln, aln_label in [("base", "Base"), ("instruct", "Instruct")]:
            m_tag = f"{fam_key}_{aln}"
            sub = df_e6[df_e6["model"] == m_tag] if df_e6 is not None else pd.DataFrame()

            if len(sub) > 0:
                r0 = sub.iloc[0]
                site_str = str(r0["site_selection"]).replace("Identified ", "")
                r_on_r = f"{r0['reader_ablation_on_reader']:.3f}"
                r_on_s = f"{r0['reader_ablation_on_self']:.3f}"
                s_on_r = f"{r0['self_ablation_on_reader']:.3f}"
                s_on_s = f"{r0['self_ablation_on_self']:.3f}"
                b_int = f"{r0['interaction_beta']:.4f}"
                p_int = f"{r0['interaction_p']:.2e}" if r0['interaction_p'] < 0.001 else f"{r0['interaction_p']:.3f}"
                q_int = f"{r0['interaction_q']:.2e}" if r0['interaction_q'] < 0.001 else f"{r0['interaction_q']:.3f}"
            else:
                site_str = "No distinct sites"
                r_on_r = r_on_s = s_on_r = s_on_s = b_int = p_int = q_int = "---"

            fam_str = f"\\multirow{{2}}{{*}}{{\\textbf{{{fam_name}}}}}" if first_fam else ""
            first_fam = False

            tex_lines.append(f"{fam_str:<32} & {aln_label:<10} & {site_str:<18} & {r_on_r:<10} & {r_on_s:<10} & {s_on_r:<10} & {s_on_s:<10} & {b_int:<10} & {p_int:<10} & {q_int:<10} \\\\")

            if aln == "base":
                tex_lines.append(r"\cmidrule(lr){2-10}")

        if fam_key != FAMILY_ORDER[-1][0]:
            tex_lines.append(r"\midrule")

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1ex}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\textbf{Note:} Task $\times$ SiteType interactionはQwen Base / Instruct、Llama Base / Instruct、OLMo Baseでnominally significantであった一方、Gemma Base / InstructおよびOLMo Instructでは明確なsupportが得られなかった。したがってtask-specific causal specializationは一部modelで観測されたが、全familyに共通する性質ではなかった。",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(tex_lines)

# =========================================================================
# Markdown Summary
# =========================================================================
def generate_markdown_summary(df_e1, df_e2, df_ctrl, df_e3, df_e4, df_e6):
    md_lines = [
        "# V1 Stage: Internal Representation and Causal Sharing Complete Summary Report\n",
        "## 1. Linear Probe Decodability and Peak Localization (E1)\n",
        "| Family | Variant | Task | Valence Layer (d*) | Valence R2 (r) | Arousal Layer (d*) | Arousal R2 (r) | Reader-Self Delta d* |",
        "|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|",
    ]
    # (既存フォーマット維持)
    for fam_key, fam_name in FAMILY_ORDER:
        for aln, aln_label in [("base", "Base"), ("instruct", "Instruct")]:
            m_tag = f"{fam_key}_{aln}"
            sub = df_e1[df_e1["model"] == m_tag] if df_e1 is not None else pd.DataFrame()
            r_v = sub[(sub["task"] == "reader") & (sub["axis"] == "valence")]
            s_v = sub[(sub["task"] == "self") & (sub["axis"] == "valence")]
            r_a = sub[(sub["task"] == "reader") & (sub["axis"] == "arousal")]
            s_a = sub[(sub["task"] == "self") & (sub["axis"] == "arousal")]
            def fc(row):
                return (f"L{int(row['peak_layer'].iloc[0])} ({row['peak_relative_depth'].iloc[0]:.2f})",
                        f"{row['peak_r2'].iloc[0]:.3f} ({row['pearson'].iloc[0]:.3f})") if len(row) > 0 else ("---", "---")
            rv_ld, rv_st = fc(r_v); sv_ld, sv_st = fc(s_v)
            ra_ld, ra_st = fc(r_a); sa_ld, sa_st = fc(s_a)
            dist_v = f"{abs(r_v['peak_relative_depth'].iloc[0] - s_v['peak_relative_depth'].iloc[0]):.2f}" if len(r_v) > 0 and len(s_v) > 0 else "---"
            dist_a = f"{abs(r_a['peak_relative_depth'].iloc[0] - s_a['peak_relative_depth'].iloc[0]):.2f}" if len(r_a) > 0 and len(s_a) > 0 else "---"
            d_str = f"V:{dist_v}, A:{dist_a}"
            md_lines.append(f"| **{fam_name}** | {aln_label} | Reader | {rv_ld} | {rv_st} | {ra_ld} | {ra_st} | {d_str} |")
            md_lines.append(f"| **{fam_name}** | {aln_label} | Self   | {sv_ld} | {sv_st} | {sa_ld} | {sa_st} | |")

    md_lines.append("\n## 2. Shared Causal Map and Circuit Sharing (E3)\n")
    md_lines.append("| Family | Variant | Reader Peak (d*) | Self Peak (d*) | Reader Mag (x10^-3) | Self Mag (x10^-3) | Spearman rho | Mean Cosine |")
    md_lines.append("|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for fam_key, fam_name in FAMILY_ORDER:
        for aln, aln_label in [("base", "Base"), ("instruct", "Instruct")]:
            m_tag = f"{fam_key}_{aln}"
            sub = df_e3[df_e3["model"] == m_tag] if df_e3 is not None else pd.DataFrame()
            if len(sub) > 0:
                r_row = sub[sub["task"] == "reader"].iloc[0] if len(sub[sub["task"] == "reader"]) > 0 else None
                s_row = sub[sub["task"] == "self"].iloc[0] if len(sub[sub["task"] == "self"]) > 0 else None
                r_p = f"L{int(r_row['peak_layer'])} ({r_row['peak_depth']:.2f})" if r_row is not None else "---"
                s_p = f"L{int(s_row['peak_layer'])} ({s_row['peak_depth']:.2f})" if s_row is not None else "---"
                r_m = f"{r_row['peak_causal_magnitude'] * 1e3:.2f}" if r_row is not None else "---"
                s_m = f"{s_row['peak_causal_magnitude'] * 1e3:.2f}" if s_row is not None else "---"
                rank_rho = sub["reader_self_rank_spearman"].iloc[0]
                cos_sim = sub["mean_direction_cosine"].iloc[0]
                md_lines.append(f"| **{fam_name}** | {aln_label} | {r_p} | {s_p} | {r_m} | {s_m} | {rank_rho:.3f} | {cos_sim:.3f} |")

    return "\n".join(md_lines) + "\n"

def main():
    parser = argparse.ArgumentParser(description="Summarize V1 Representation Sharing results.")
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.repo_root, "iclr2027/tables")
    os.makedirs(out_dir, exist_ok=True)

    df_e1, df_e2, df_ctrl, df_e3, df_e4, df_e6 = load_data(args.repo_root)

    # 1. E1 Table
    tex_e1 = generate_e1_table(df_e1)
    for fname in ["v1_peak_decodability.tex", "table_v1_e1_decodability.tex"]:
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            f.write(tex_e1)

    # 2. E2 Table
    tex_e2 = generate_e2_geometry_table(df_e2)
    with open(os.path.join(out_dir, "v1_shared_geometry.tex"), "w", encoding="utf-8") as f:
        f.write(tex_e2)

    # 3. Phase B Semantic Controls Table
    tex_ctrl = generate_semantic_controls_table(df_ctrl)
    with open(os.path.join(out_dir, "v1_semantic_controls.tex"), "w", encoding="utf-8") as f:
        f.write(tex_ctrl)

    # 4. E3 Table
    tex_e3 = generate_e3_causal_map_table(df_e3)
    for fname in ["v1_causal_profile.tex", "table_v1_e3_causal_map.tex"]:
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            f.write(tex_e3)

    # 5. E4 Interchangeability Table
    tex_e4 = generate_interchangeability_table(df_e4)
    with open(os.path.join(out_dir, "v1_interchangeability.tex"), "w", encoding="utf-8") as f:
        f.write(tex_e4)

    # 6. E6 Specialization Table
    tex_e6 = generate_specialization_table(df_e6)
    with open(os.path.join(out_dir, "v1_specialization.tex"), "w", encoding="utf-8") as f:
        f.write(tex_e6)

    # 7. Comprehensive LaTeX summary (All 6 tables combined)
    tex_summary = "% ================================================================\n" \
                  "% V1 Stage: Internal Representation and Causal Sharing Complete Summary Tables\n" \
                  "% ================================================================\n\n" \
                  + tex_e1 + "\n\n" + tex_e2 + "\n\n" + tex_ctrl + "\n\n" + tex_e3 + "\n\n" + tex_e4 + "\n\n" + tex_e6 + "\n"
    with open(os.path.join(out_dir, "v1_internal_sharing_summary.tex"), "w", encoding="utf-8") as f:
        f.write(tex_summary)

    # 8. Markdown summary
    md_summary = generate_markdown_summary(df_e1, df_e2, df_ctrl, df_e3, df_e4, df_e6)
    with open(os.path.join(out_dir, "v1_internal_sharing_summary.md"), "w", encoding="utf-8") as f:
        f.write(md_summary)

    print(f"[+] Successfully generated all V1 tables in {out_dir}:")
    print(f"    - v1_peak_decodability.tex")
    print(f"    - v1_shared_geometry.tex")
    print(f"    - v1_semantic_controls.tex")
    print(f"    - v1_causal_profile.tex")
    print(f"    - v1_interchangeability.tex")
    print(f"    - v1_specialization.tex")
    print(f"    - v1_internal_sharing_summary.tex")
    print(f"    - v1_internal_sharing_summary.md")

if __name__ == "__main__":
    main()
