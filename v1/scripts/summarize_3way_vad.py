#!/usr/bin/env python3
"""
Comprehensive Summary and Comparison for EmoBank 3-Way VAD Evaluation.

Framework:
  1. ① Writer-State Estimation (W)   vs Human Writer VAD
  2. ② Reader-Response Prediction (R) vs Human Reader VAD
  3. ③ Self-Report (S)               vs Human Reader VAD (External Reference)
     Evaluated via 3 Tripartite Dimensions:
       - Alignment   : Correlation r (tracking across stimuli)
       - Calibration : MAE (scale proximity to human levels)
       - Collapse    : P(Greedy=(5,5,5)) (surface neutrality fixation)
  4. ④ Internal Coupling: Corr(R, S)
     (How strongly self-report couples to the model's own reader prediction)

Outputs:
  - Per-model 9-table breakdown (3 tasks x 3 dimensions: V, A, D)
  - Per-model 3x3 Summary Matrix (Continuous r, Greedy r, MAE)
  - Cross-model 9 individual comparison tables (Task x Dimension)
  - Cross-model Task-wise aggregated tables (Writer, Reader, Self)
  - Cross-model Dimension-wise aggregated tables (Valence, Arousal, Dominance)
  - Cross-model Self-Report Alignment / Calibration / Collapse Tripartite Summary Table
  - Cross-model Internal Cognitive Coupling Corr(R, S) Table
  - Unified Markdown report and CSV summaries
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Add scripts directory to path for md_to_tex
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from md_to_tex import convert_md_file_to_tex
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

TASKS = [
    ("writer", "① Writer-State Estimation (W)", "human_writer"),
    ("reader", "② Reader-Response Prediction (R)", "human_reader"),
    ("self",   "③ Self-Report (S)",              "human_reader"),
]

DIMS = [
    ("v", "Valence"),
    ("a", "Arousal"),
    ("d", "Dominance")
]

def format_p(p):
    if np.isnan(p):
        return "N/A"
    if p < 0.001:
        return "<.001"
    return f"{p:.3f}"

def safe_corr(x, y):
    """Computes Pearson and Spearman correlations, safely handling zero variance."""
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, np.nan, np.nan
    r, p = pearsonr(x, y)
    rho, rho_p = spearmanr(x, y)
    return float(r), float(p), float(rho), float(rho_p)

def bootstrap_corr_ci(x, y, n_boot=1000, ci_level=0.95, seed=42):
    """Computes 95% pair bootstrap CI for Pearson r and Spearman rho."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(x)
    r_boots, rho_boots = [], []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        bx, by = x[idx], y[idx]
        if np.std(bx) > 0 and np.std(by) > 0:
            br, _ = pearsonr(bx, by)
            brho, _ = spearmanr(bx, by)
            if not np.isnan(br): r_boots.append(br)
            if not np.isnan(brho): rho_boots.append(brho)
    alpha = 1.0 - ci_level
    r_low = float(np.percentile(r_boots, 100 * (alpha / 2.0))) if r_boots else np.nan
    r_high = float(np.percentile(r_boots, 100 * (1.0 - alpha / 2.0))) if r_boots else np.nan
    rho_low = float(np.percentile(rho_boots, 100 * (alpha / 2.0))) if rho_boots else np.nan
    rho_high = float(np.percentile(rho_boots, 100 * (1.0 - alpha / 2.0))) if rho_boots else np.nan
    return r_low, r_high, rho_low, rho_high

def format_corr_val(val):
    if np.isnan(val):
        return "N/A (std=0)"
    return f"{val:.4f}"

def analyze_model_csv(csv_path):
    """Analyzes a single model's 3-way VAD predictions CSV file."""
    df = pd.read_csv(csv_path)
    base_name = os.path.basename(csv_path).replace("_3way_vad.csv", "")
    
    is_instruct = "instruct" in base_name.lower() or "chat" in base_name.lower()
    
    stats = {
        "tag": base_name,
        "is_instruct": is_instruct,
        "num_samples": len(df),
        "cells": {},
        "exact_neutral_rates": {}
    }
    
    # Analyze 3 Tasks x 3 Dimensions = 9 Cells
    for t_key, t_label, ref_prefix in TASKS:
        prefix = t_key[0]
        # Task-wide exact neutral argmax rate (5,5,5)
        task_555 = ((df[f"{prefix}_gv"] == 5) & (df[f"{prefix}_ga"] == 5) & (df[f"{prefix}_gd"] == 5)).mean() * 100.0
        stats["exact_neutral_rates"][t_key] = float(task_555)

        for d_key, d_label in DIMS:
            pred_col = f"{prefix}_e{d_key}"
            greedy_col = f"{prefix}_g{d_key}"
            target_col = f"{ref_prefix}_{d_key}"
            
            y_pred = df[pred_col].values
            y_greedy = df[greedy_col].values
            y_true = df[target_col].values
            
            # Rescale LLM predictions (1-9 scale, neutral 5) to human EmoBank scale (1-5 scale, neutral 3)
            # Formula: y_scaled = (y + 1.0) / 2.0 (Maps 1->1.0, 5->3.0, 9->5.0)
            y_pred_scaled = (y_pred + 1.0) / 2.0
            y_greedy_scaled = (y_greedy + 1.0) / 2.0

            # Continuous likelihood correlations (Scale-invariant, unchanged)
            r, p_val, rho, rho_p = safe_corr(y_pred_scaled, y_true)
            
            # Greedy prediction correlations (Scale-invariant, unchanged)
            r_g, p_g, rho_g, rho_g_p = safe_corr(y_greedy_scaled, y_true)
            
            # Calibration metrics on 1-5 human scale
            mae = float(mean_absolute_error(y_true, y_pred_scaled))
            rmse = float(root_mean_squared_error(y_true, y_pred_scaled))
            
            # Dimension-specific exact neutral argmax rate (5 in LLM scale = 3.0 in human scale)
            pct_dim_5 = float((y_greedy == 5).mean() * 100.0)
            
            stats["cells"][(t_key, d_key)] = {
                "task": t_label,
                "dim": d_label,
                "r": r,
                "p_val": p_val,
                "rho": rho,
                "rho_p": rho_p,
                "r_greedy": r_g,
                "p_greedy": p_g,
                "rho_greedy": rho_g,
                "mae": mae,
                "rmse": rmse,
                "pred_mean": float(np.mean(y_pred_scaled)),
                "pred_std": float(np.std(y_pred_scaled)),
                "greedy_mean": float(np.mean(y_greedy_scaled)),
                "greedy_std": float(np.std(y_greedy_scaled)),
                "pct_greedy_5": pct_dim_5,
                "true_mean": float(np.mean(y_true)),
                "true_std": float(np.std(y_true)),
            }
            
    # Internal cognitive coupling: Reader vs Self
    stats["internal_rs"] = {}
    for d_key, d_label in DIMS:
        r_pred = df[f"r_e{d_key}"].values
        s_pred = df[f"s_e{d_key}"].values
        r_rs, p_rs, rho_rs, p_rho_rs = safe_corr(r_pred, s_pred)
        r_ci_l, r_ci_u, rho_ci_l, rho_ci_u = bootstrap_corr_ci(r_pred, s_pred, n_boot=1000, seed=42)
        stats["internal_rs"][d_key] = {
            "dim": d_label,
            "r": r_rs,
            "p_val": p_rs,
            "r_ci_lower": r_ci_l,
            "r_ci_upper": r_ci_u,
            "rho": rho_rs,
            "p_rho": p_rho_rs,
            "rho_ci_lower": rho_ci_l,
            "rho_ci_upper": rho_ci_u,
        }
        
    return stats

def generate_markdown_report(all_models_stats, out_md_path):
    lines = []
    lines.append("# EmoBank 3-Way VAD (Valence-Arousal-Dominance) Evaluation Report\n")
    lines.append("本レポートは、EmoBank テストセットにおける各LLMの **3軸の感情測定課題** を **Valence・Arousal・Dominance (VAD)** の3次元で多角的に評価・比較した結果です。\n")
    
    lines.append("### 評価体系の整理（3軸の明確な概念分離 ＋ 内部結合度）")
    lines.append("1. **① Writer-State Estimation ($W$)**: テキスト書き手の情動状態の推定（正解基準: EmoBank **Writer VAD**）")
    lines.append("   - プロンプト: `\"Read the following text and estimate the affective state of the writer who wrote it.\"`")
    lines.append("2. **② Reader-Response Prediction ($R$)**: 一般読者が受ける情動反応の予測（正解基準: EmoBank **Reader VAD**）")
    lines.append("   - プロンプト: `\"Read the following text and estimate the affective response that this text is likely to evoke in an average human reader.\"`")
    lines.append("   - *意味*: 他者認識能力の測定。正解ラベルに対する予測精度を表す。")
    lines.append("3. **③ Self-Report ($S$)**: 刺激提示に対するLLM自身の自己報告情動状態（外部参照: EmoBank **Reader VAD**）")
    lines.append("   - プロンプト: `\"Read the following text and report your affective state.\"`")
    lines.append("   - *意味*: **Human-Affect Correspondence (人間情動対応付け / アライメント)**。正解精度（Accuracy）ではなく、人間読者反応との行動的対応関係の度合いとして解釈する。")
    lines.append("   - **評価の3大観測量**: ")
    lines.append("     - **Human-Affect Alignment ($r$)**: 人間の刺激間情動変化への追従度（相関係数）")
    lines.append("     - **Calibration ($\\text{MAE}$)**: 人間評価尺度との絶対距離")
    lines.append("     - **Exact Neutral Argmax Rate ($P(\\text{Greedy}=(5,5,5))$)**: 最尤出力が完全中立に位置する割合（観測量）")
    lines.append("4. **④ 内部認知結合度 (Internal Cognitive Coupling $\\text{Corr}(R, S)$)**: 同一モデルにおける Reader予測とSelf自己報告の連動性（ペアブートストラップ 95% CI および Spearman $\\rho$ 併記）\n")
    lines.append("---\n")

    # =========================================================================
    # Part 1: 各モデルの詳細分析（各モデル9表 ＋ 3x3マトリクス）
    # =========================================================================
    lines.append("## Behavioral Executive Summary: Four Methodological Pillars\n")
    lines.append("行動実験の最終結果は、以下の **4大評価ブロック（Four Methodological Pillars）** に基づき報告されます：\n")
    lines.append("1. **Human Grounding**: 人間評価値に対するモデル出力の外部妥当性・相関 ($r_{\\mathrm{ext}}$)\n")
    lines.append("2. **Affective Sensitivity**: 情動刺激に対する出力変位・感度 (Paired Cohen's $d_z$)\n")
    lines.append("3. **Dose-Response / Complexity Control**: 感情強度・共変量に対する用量反応性結合勾配\n")
    lines.append("4. **Reader–Self Coupling**: 他者認識 (Reader) と自己報告 (Self) の結合・アライメント ($r_{\\mathrm{reader-self}}$)\n\n")
    lines.append("---\n\n")
    lines.append("## Part 1: 各モデル別 詳細分析（3タスク × 3次元 ＝ 9表 ＋ 3×3マトリクス）\n")
    
    for m in all_models_stats:
        tag = m["tag"]
        m_type = "Instruct" if m["is_instruct"] else "Base"
        n_samples = m["num_samples"]
        
        lines.append(f"### ■ モデル: `{tag}` ({m_type}, N={n_samples})\n")
        
        # 1. 3x3 Summary Matrix
        lines.append(f"#### 【サマリー】 3×3 相関 & 誤差マトリクス")
        lines.append("| タスク \\ 感情次元 | Valence ($V$) | Arousal ($A$) | Dominance ($D$) | 完全中立最尤率 $(5,5,5)$ |")
        lines.append("|:---|:---:|:---:|:---:|:---:|")
        for t_key, t_label, _ in TASKS:
            cv = m["cells"][(t_key, "v")]
            ca = m["cells"][(t_key, "a")]
            cd = m["cells"][(t_key, "d")]
            col_rate = m["exact_neutral_rates"][t_key]
            
            lines.append(
                f"| **{t_label}** | "
                f"**$r={cv['r']:.3f}$** (MAE {cv['mae']:.2f}) | "
                f"**$r={ca['r']:.3f}$** (MAE {ca['mae']:.2f}) | "
                f"**$r={cd['r']:.3f}$** (MAE {cd['mae']:.2f}) | "
                f"`{col_rate:.1f}%` |"
            )
        lines.append("")
        
        # Internal Alignment
        iv = m['internal_rs']['v']
        ia = m['internal_rs']['a']
        id_ = m['internal_rs']['d']
        lines.append(f"- **内部認知結合度 (Internal Coupling $\\text{{Corr}}(R, S)$)**:")
        lines.append(f"  - Valence: $r={iv['r']:.3f}$ [95% CI {iv['r_ci_lower']:.3f}, {iv['r_ci_upper']:.3f}], $\\rho={iv['rho']:.3f}$ ($p={format_p(iv['p_val'])}$)")
        lines.append(f"  - Arousal: $r={ia['r']:.3f}$ [95% CI {ia['r_ci_lower']:.3f}, {ia['r_ci_upper']:.3f}], $\\rho={ia['rho']:.3f}$ ($p={format_p(ia['p_val'])}$)")
        lines.append(f"  - Dominance: $r={id_['r']:.3f}$ [95% CI {id_['r_ci_lower']:.3f}, {id_['r_ci_upper']:.3f}], $\\rho={id_['rho']:.3f}$ ($p={format_p(id_['p_val'])}$)\n")
        
        # 9 Individual Tables
        lines.append(f"#### 【9個の詳細表】 タスク別・次元別詳細メトリクス\n")
        
        table_num = 1
        for t_key, t_label, ref_label in TASKS:
            lines.append(f"##### {t_label}")
            lines.append(f"*評価基準: {ref_label} VAD*\n")
            
            for d_key, d_label in DIMS:
                cell = m["cells"][(t_key, d_key)]
                lines.append(f"**表 {table_num}: {t_label} — {d_label}**")
                lines.append("| 評価軸 / 指標 (Metric) | 値 (Value) | 学術的意味・備考 |")
                lines.append("|:---|:---:|:---|")
                
                # Alignment
                lines.append(f"| **[Alignment] 連続相関 ($r$)** | **{format_corr_val(cell['r'])}** | $p = {format_p(cell['p_val'])} $ (刺激間パターンの追従度) |")
                lines.append(f"| **[Alignment] 順位相関 ($\\rho$)** | {format_corr_val(cell['rho'])} | $p = {format_p(cell['rho_p'])} $ (単調増加性) |")
                lines.append(f"| **[Alignment] Greedy相関 ($r_{{greedy}}$)** | {format_corr_val(cell['r_greedy'])} | 最尤トークン出力ベースの一致度 |")
                
                # Calibration
                lines.append(f"| **[Calibration] 平均絶対誤差 (MAE)** | **{cell['mae']:.4f}** | 人間正解値との絶対スケール距離 |")
                lines.append(f"| **[Calibration] 二乗平均平方根誤差 (RMSE)** | {cell['rmse']:.4f} | 大きな乖離に対する感度 |")
                
                # Distributions
                lines.append(f"| **[分布] モデル予測期待値平均 $\\pm$ SD** | {cell['pred_mean']:.3f} $\\pm$ {cell['pred_std']:.3f} | $E[{d_key.upper()}]$ の連続分布 |")
                lines.append(f"| **[分布] 人間アノテーション平均 $\\pm$ SD** | {cell['true_mean']:.3f} $\\pm$ {cell['true_std']:.3f} | 人間正解値の分布 |")
                
                # Exact Neutral Rate
                lines.append(f"| **[中立観測量] 単一次元 Greedy=5 最尤率** | `{cell['pct_greedy_5']:.2f}%` | 当該次元の最尤出力がスケール中央値（5）に位置する割合 |")
                lines.append("")
                table_num += 1
                
        lines.append("---\n")

    # =========================================================================
    # Part 2: 4ファミリー別 Base vs. Instruct 9大直接対比表（3タスク × 3次元）
    # =========================================================================
    lines.append("## Part 2: 4ファミリー別 Base vs. Instruct 9大直接対比表（3タスク × 3次元）\n")
    lines.append("各感情測定課題・各次元において、4つの主要モデルファミリー（Qwen 2.5, Mistral 7B, Llama 3.2, Gemma 2）の **BaseモデルとInstructモデルを同一表内で直接比較** した9枚の表です。\n")
    
    FAMILIES = [
        ("Qwen 2.5 1.5B", "qwen"),
        ("Mistral 7B v0.1", "mistral"),
        ("Llama 3.2 1B", "llama"),
        ("Gemma 2 2B", "gemma"),
    ]

    def get_family_pair(all_stats, fam_key):
        base_m = next((m for m in all_stats if fam_key in m["tag"].lower() and not m["is_instruct"]), None)
        inst_m = next((m for m in all_stats if fam_key in m["tag"].lower() and m["is_instruct"]), None)
        return base_m, inst_m

    t_idx = 1
    for t_key, t_label, ref_label in TASKS:
        for d_key, d_label in DIMS:
            lines.append(f"### 表 2-{t_idx}: 【{t_label}】 — {d_label} ($r_{d_key.upper()}$)")
            lines.append(f"*評価対象: {d_label} / 正解基準: {ref_label} VAD (1〜5人間尺度)*\n")
            lines.append("| モデルファミリー | Base $r$ | Instruct $r$ | アライメント変位 $\\Delta r$ | Base Greedy $r$ | Instruct Greedy $r$ | Base MAE | Instruct MAE | Base 完全中立率 | Instruct 完全中立率 |")
            lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

            for fam_name, fam_key in FAMILIES:
                bm, im = get_family_pair(all_models_stats, fam_key)
                if not bm or not im:
                    continue
                bc = bm["cells"][(t_key, d_key)]
                ic = im["cells"][(t_key, d_key)]
                
                br = bc["r"]
                ir = ic["r"]
                diff_r = ir - br if not (np.isnan(ir) or np.isnan(br)) else np.nan
                diff_str = f"**{diff_r:+.3f}**" if not np.isnan(diff_r) else "N/A"
                
                b_gr = format_corr_val(bc["r_greedy"])
                i_gr = format_corr_val(ic["r_greedy"])
                
                lines.append(
                    f"| **{fam_name}** | "
                    f"{format_corr_val(br)} | **{format_corr_val(ir)}** | {diff_str} | "
                    f"{b_gr} | {i_gr} | "
                    f"{bc['mae']:.3f} | {ic['mae']:.3f} | "
                    f"`{bc['pct_greedy_5']:.1f}%` | `{ic['pct_greedy_5']:.1f}%` |"
                )
            lines.append("")
            t_idx += 1
            
    lines.append("---\n")

    # Helper to get models ordered by family pairs (Base, then Instruct)
    ordered_models = []
    for _, fam_key in FAMILIES:
        bm, im = get_family_pair(all_models_stats, fam_key)
        if bm: ordered_models.append(bm)
        if im: ordered_models.append(im)
    # Append any remaining models not in FAMILIES
    for m in all_models_stats:
        if m not in ordered_models:
            ordered_models.append(m)

    # =========================================================================
    # Part 3: 全モデル横断 タスク別 統合表（3表）
    # =========================================================================
    lines.append("## Part 3: 全モデル横断 タスク別 統合表（3表）\n")
    
    # 3-1: Writer
    lines.append("### 表 3-1: ① Writer-State Estimation ($W$) 統合表")
    lines.append("| モデル名 (Tag) | 種別 | Valence $r_V^W$ | Arousal $r_A^W$ | Dominance $r_D^W$ | MAE $(V, A, D)$ | 完全中立最尤率 $(5,5,5)$ |")
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for m in ordered_models:
        m_type = "Instruct" if m["is_instruct"] else "Base"
        cv, ca, cd = m["cells"][("writer", "v")], m["cells"][("writer", "a")], m["cells"][("writer", "d")]
        maes = f"({cv['mae']:.2f}, {ca['mae']:.2f}, {cd['mae']:.2f})"
        lines.append(f"| **{m['tag']}** | {m_type} | **{cv['r']:.3f}** | {ca['r']:.3f} | {cd['r']:.3f} | {maes} | `{m['exact_neutral_rates']['writer']:.1f}%` |")
    lines.append("")

    # 3-2: Reader
    lines.append("### 表 3-2: ② Reader-Response Prediction ($R$) 統合表")
    lines.append("| モデル名 (Tag) | 種別 | Valence $r_V^R$ | Arousal $r_A^R$ | Dominance $r_D^R$ | MAE $(V, A, D)$ | 完全中立最尤率 $(5,5,5)$ |")
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for m in ordered_models:
        m_type = "Instruct" if m["is_instruct"] else "Base"
        cv, ca, cd = m["cells"][("reader", "v")], m["cells"][("reader", "a")], m["cells"][("reader", "d")]
        maes = f"({cv['mae']:.2f}, {ca['mae']:.2f}, {cd['mae']:.2f})"
        lines.append(f"| **{m['tag']}** | {m_type} | **{cv['r']:.3f}** | {ca['r']:.3f} | {cd['r']:.3f} | {maes} | `{m['exact_neutral_rates']['reader']:.1f}%` |")
    lines.append("")

    # 3-3: Self-Report
    lines.append("### 表 3-3: ③ Self-Report ($S$) 統合表")
    lines.append("| モデル名 (Tag) | 種別 | Valence $r_V^S$ | Arousal $r_A^S$ | Dominance $r_D^S$ | MAE $(V, A, D)$ | 完全中立最尤率 $(5,5,5)$ | 内部結合度 $R \\leftrightarrow S (V, A, D)$ |")
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for m in ordered_models:
        m_type = "Instruct" if m["is_instruct"] else "Base"
        cv, ca, cd = m["cells"][("self", "v")], m["cells"][("self", "a")], m["cells"][("self", "d")]
        maes = f"({cv['mae']:.2f}, {ca['mae']:.2f}, {cd['mae']:.2f})"
        rs_str = f"({m['internal_rs']['v']['r']:.2f}, {m['internal_rs']['a']['r']:.2f}, {m['internal_rs']['d']['r']:.2f})"
        lines.append(
            f"| **{m['tag']}** | {m_type} | **{cv['r']:.3f}** | {ca['r']:.3f} | {cd['r']:.3f} | "
            f"{maes} | `{m['exact_neutral_rates']['self']:.1f}%` | {rs_str} |"
        )
    lines.append("\n---\n")

    # =========================================================================
    # Part 4: 全モデル横断 感情次元別 統合表（3表）
    # =========================================================================
    lines.append("## Part 4: 全モデル横断 感情次元別 統合表（3表）\n")
    
    for d_key, d_label in DIMS:
        lines.append(f"### 表 4-{d_key.upper()}: 【{d_label} 次元】 各タスク間の比較")
        lines.append(f"| モデル名 (Tag) | 種別 | ① Writer $r_{d_key.upper()}^W$ | ② Reader $r_{d_key.upper()}^R$ | ③ Self $r_{d_key.upper()}^S$ | 内部結合度 $\\text{{Corr}}(R, S)$ [95% CI] | 予測-自己差分 ($r^R - r^S$) |")
        lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
        for m in ordered_models:
            m_type = "Instruct" if m["is_instruct"] else "Base"
            rw = m["cells"][("writer", d_key)]["r"]
            rr = m["cells"][("reader", d_key)]["r"]
            rs = m["cells"][("self", d_key)]["r"]
            irs = m["internal_rs"][d_key]
            r_rs = irs["r"]
            ci_str = f"[{irs['r_ci_lower']:.2f}, {irs['r_ci_upper']:.2f}]" if not np.isnan(irs['r_ci_lower']) else ""
            diff = rr - rs if not (np.isnan(rr) or np.isnan(rs)) else np.nan
            diff_str = f"{diff:+.3f}" if not np.isnan(diff) else "N/A"
            lines.append(
                f"| **{m['tag']}** | {m_type} | {format_corr_val(rw)} | {format_corr_val(rr)} | **{format_corr_val(rs)}** | "
                f"{format_corr_val(r_rs)} {ci_str} | {diff_str} |"
            )
        lines.append("")

    # =========================================================================
    # Part 5: Self-Report 特化: Alignment vs Calibration vs Exact Neutral Argmax
    # =========================================================================
    lines.append("---\n")
    lines.append("## Part 5: ③ Self-Report 特化分析（Human-Affect Alignment vs Calibration vs Exact Neutral Rate）\n")
    lines.append("自己報告において、「人間情動への追従度（Alignment）」、「絶対値の校正（Calibration）」、「出力の完全中立最尤率（Exact Neutral Rate）」を対比する専門表です。\n")
    lines.append(r"| モデル名 (Tag) | 種別 | Alignment ($r_V^S$) | Alignment ($r_A^S$) | Calibration ($\text{MAE}_V$) | Calibration ($\text{MAE}_A$) | 期待値平均 $(E_V, E_A)$ | 完全中立最尤率 $(5,5,5)$ |")
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for m in ordered_models:
        m_type = "Instruct" if m["is_instruct"] else "Base"
        cv = m["cells"][("self", "v")]
        ca = m["cells"][("self", "a")]
        means = f"({cv['pred_mean']:.2f}, {ca['pred_mean']:.2f})"
        lines.append(
            f"| **{m['tag']}** | {m_type} | "
            f"**{cv['r']:.3f}** | {ca['r']:.3f} | "
            f"{cv['mae']:.3f} | {ca['mae']:.3f} | "
            f"{means} | `{m['exact_neutral_rates']['self']:.1f}%` |"
        )
    lines.append("")

    report_content = "\n".join(lines) + "\n"
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    return report_content

def build_summary_dataframe(all_models_stats):
    rows = []
    for m in all_models_stats:
        row = {
            "tag": m["tag"],
            "is_instruct": m["is_instruct"],
            "num_samples": m["num_samples"],
            "exact_555_pct_writer": m["exact_neutral_rates"]["writer"],
            "exact_555_pct_reader": m["exact_neutral_rates"]["reader"],
            "exact_555_pct_self":   m["exact_neutral_rates"]["self"],
        }
        for (t_key, d_key), cell in m["cells"].items():
            row[f"{t_key}_{d_key}_r"] = cell["r"]
            row[f"{t_key}_{d_key}_p"] = cell["p_val"]
            row[f"{t_key}_{d_key}_rho"] = cell["rho"]
            row[f"{t_key}_{d_key}_r_greedy"] = cell["r_greedy"]
            row[f"{t_key}_{d_key}_mae"] = cell["mae"]
            row[f"{t_key}_{d_key}_rmse"] = cell["rmse"]
            row[f"{t_key}_{d_key}_pred_mean"] = cell["pred_mean"]
            row[f"{t_key}_{d_key}_pred_std"] = cell["pred_std"]
            row[f"{t_key}_{d_key}_pct_greedy_5"] = cell["pct_greedy_5"]
        for d_key, irs in m["internal_rs"].items():
            row[f"internal_rs_{d_key}_r"] = irs["r"]
            row[f"internal_rs_{d_key}_r_ci_lower"] = irs["r_ci_lower"]
            row[f"internal_rs_{d_key}_r_ci_upper"] = irs["r_ci_upper"]
            row[f"internal_rs_{d_key}_rho"] = irs["rho"]
            row[f"internal_rs_{d_key}_p_rho"] = irs["p_rho"]
            row[f"internal_rs_{d_key}_rho_ci_lower"] = irs["rho_ci_lower"]
            row[f"internal_rs_{d_key}_rho_ci_upper"] = irs["rho_ci_upper"]
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description="Summarize 3-way VAD evaluation on EmoBank.")
    parser.add_argument("--results-dir", type=str, default="v1/results/emobank_3way_vad_test1k",
                        help="Directory containing *_3way_vad.csv results")
    parser.add_argument("--out-md", type=str, default=None,
                        help="Path for output Markdown report")
    parser.add_argument("--out-tex", type=str, default=None,
                        help="Path for output LaTeX report")
    parser.add_argument("--out-csv", type=str, default=None,
                        help="Path for output summary CSV")
    args = parser.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.results_dir, "*_3way_vad.csv")))
    if not csv_files:
        print(f"[!] No *_3way_vad.csv files found in {args.results_dir}")
        return

    print(f"Found {len(csv_files)} model result CSVs in {args.results_dir}")
    all_models_stats = []
    for cf in csv_files:
        print(f"Analyzing {os.path.basename(cf)}...")
        stats = analyze_model_csv(cf)
        all_models_stats.append(stats)

    # Sort models by tag
    all_models_stats.sort(key=lambda x: x["tag"])

    out_md = args.out_md or os.path.join(args.results_dir, "3way_vad_detailed_report.md")
    out_tex = args.out_tex or os.path.splitext(out_md)[0] + ".tex"
    out_csv = args.out_csv or os.path.join(args.results_dir, "3way_vad_summary.csv")

    # Generate Markdown Report
    generate_markdown_report(all_models_stats, out_md)

    # Generate LaTeX Report
    convert_md_file_to_tex(out_md, out_tex)

    # Generate CSV Summary
    df_summary = build_summary_dataframe(all_models_stats)
    df_summary.to_csv(out_csv, index=False)

    print("\n" + "=" * 60)
    print("SUCCESS: 3-Way VAD Enhanced Summary Generated!")
    print(f"  Markdown Report : {out_md}")
    print(f"  LaTeX Report    : {out_tex}")
    print(f"  CSV Summary     : {out_csv}")
    print("=" * 60)

if __name__ == "__main__":
    main()

