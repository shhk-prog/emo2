#!/usr/bin/env python3
"""
Comprehensive Summary and Analysis for AIPsy-Affect 4-Split Evaluation.

Addresses the 4 Core Research Questions:
  - RQ1: Sensitivity (Clinical vs. Neutral)
  - RQ2: Dose-Response (Neutral -> Moderate -> Clinical)
  - RQ3: Specificity (Complex Neutral vs. Neutral / Clinical Control)
  - RQ4: Recognition-Self-Report Coupling (Delta VA_R <-> Delta VA_S)
Plus 8-Emotion Granular Profiles and Cross-Model Family Comparisons.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, spearmanr, pearsonr

# Add scripts directory to path for md_to_tex
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from md_to_tex import convert_md_file_to_tex

EMOTIONS = [
    ("grief", "V-"),
    ("terror", "V- A+"),
    ("rage", "V- A+"),
    ("loathing", "V-"),
    ("ecstasy", "V+ A+"),
    ("admiration", "V+"),
    ("amazement", "A+"),
    ("vigilance", "A+")
]

CLUSTERS = {
    "negative": ["grief", "terror", "rage", "loathing"],
    "positive": ["ecstasy", "admiration"],
    "alert": ["amazement", "vigilance"],
}

def safe_corr(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, np.nan, np.nan
    r, p = pearsonr(x, y)
    rho, rho_p = spearmanr(x, y)
    return float(r), float(p), float(rho), float(rho_p)

def bootstrap_corr_ci(x, y, n_boot=1000, ci_level=0.95, seed=42):
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

def fit_lmm_dose_response(triplets_df, task_prefix, dim):
    """
    Fits Y ~ Intensity + (1 | pair_id) on valid triplets.
    Intensity: neutral=0, moderate=1, clinical=2.
    """
    try:
        import statsmodels.formula.api as smf
        sub = triplets_df.dropna(subset=[f"{task_prefix}_e{dim}", "triplet_id"]).copy()
        split_map = {"neutral": 0, "moderate": 1, "clinical": 2}
        sub["intensity"] = sub["split"].map(split_map)
        sub = sub.dropna(subset=["intensity"])
        
        md = smf.mixedlm(f"{task_prefix}_e{dim} ~ intensity", sub, groups=sub["triplet_id"])
        mdf = md.fit(method=["lbfgs", "cg"], maxiter=200)
        slope = float(mdf.params.get("intensity", 0.0))
        se = float(mdf.bse.get("intensity", 0.0))
        pval = float(mdf.pvalues.get("intensity", 1.0))
        return {"slope": slope, "se": se, "p_val": pval, "converged": bool(mdf.converged)}
    except Exception:
        # Fallback: OLS slope
        sub = triplets_df.dropna(subset=[f"{task_prefix}_e{dim}", "triplet_id"]).copy()
        split_map = {"neutral": 0, "moderate": 1, "clinical": 2}
        sub["intensity"] = sub["split"].map(split_map)
        sub = sub.dropna(subset=["intensity"])
        if len(sub) < 3:
            return {"slope": 0.0, "se": 0.0, "p_val": 1.0, "converged": False}
        slope, intercept = np.polyfit(sub["intensity"], sub[f"{task_prefix}_e{dim}"], 1)
        return {"slope": float(slope), "se": 0.0, "p_val": np.nan, "converged": False}

def analyze_single_model(csv_path):
    df = pd.read_csv(csv_path)
    base_name = os.path.basename(csv_path).replace("_aipsy_4split.csv", "")
    is_instruct = any(k in base_name.lower() for k in ["instruct", "chat", "_it", "-it"])

    # Standardize LLM 1-9 scale to human 1-5 scale (Neutral = 3.0)
    # E_1..5 = (E_1..9 + 1.0) / 2.0
    val_cols = [c for c in df.columns if c.endswith("_ev") or c.endswith("_ea") or c.endswith("_ed")]
    for col in val_cols:
        df[col] = (df[col] + 1.0) / 2.0

    # Split subsets
    clin = df[df["split"] == "clinical"].copy().set_index("pair_id")
    neut = df[df["split"] == "neutral"].copy().set_index("pair_id")
    comp = df[df["split"] == "complex_neutral"].copy()
    
    # Common pairs between clinical and neutral
    common_pairs = clin.index.intersection(neut.index)
    c_pairs = clin.loc[common_pairs]
    n_pairs = neut.loc[common_pairs]

    stats = {
        "tag": base_name,
        "is_instruct": is_instruct,
        "num_total": len(df),
        "num_pairs": len(common_pairs),
    }

    # =========================================================================
    # RQ1: Sensitivity (Clinical vs. Neutral) - Overall & 3 Emotion Clusters
    # =========================================================================
    stats["rq1"] = {}
    p_vals_to_correct = []
    keys_for_p = []

    for task in ["reader", "self", "writer"]:
        t_p = task[0]
        t_res = {"overall": {}, "clusters": {}}
        
        # Overall
        for dim in ["v", "a"]:
            diff = c_pairs[f"{t_p}_e{dim}"].values - n_pairs[f"{t_p}_e{dim}"].values
            mean_diff = float(np.mean(diff))
            std_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else float(np.std(diff))
            t_stat, p_val = ttest_rel(c_pairs[f"{t_p}_e{dim}"].values, n_pairs[f"{t_p}_e{dim}"].values)
            cohen_dz = mean_diff / std_diff if std_diff > 1e-9 else 0.0
            t_res["overall"][dim] = {
                "mean_delta": mean_diff,
                "std_delta": std_diff,
                "t_stat": float(t_stat),
                "p_val": float(p_val),
                "q_val": float(p_val), # will be updated
                "cohen_dz": float(cohen_dz),
                "is_primary": (task in ["reader", "self"])
            }
            p_vals_to_correct.append(float(p_val))
            keys_for_p.append((task, "overall", dim))
            
        # 3 Emotion Clusters
        for c_name, emos in CLUSTERS.items():
            sub_mask = c_pairs["emotion"].isin(emos)
            sub_c = c_pairs[sub_mask]
            sub_n = n_pairs[sub_mask]
            c_res = {}
            for dim in ["v", "a"]:
                c_diff = sub_c[f"{t_p}_e{dim}"].values - sub_n[f"{t_p}_e{dim}"].values
                m_diff = float(np.mean(c_diff)) if len(c_diff) > 0 else 0.0
                s_diff = float(np.std(c_diff, ddof=1)) if len(c_diff) > 1 else (float(np.std(c_diff)) if len(c_diff) > 0 else 1.0)
                if len(c_diff) > 1 and s_diff > 0:
                    t_st, p_v = ttest_rel(sub_c[f"{t_p}_e{dim}"].values, sub_n[f"{t_p}_e{dim}"].values)
                else:
                    t_st, p_v = 0.0, 1.0
                cd_z = m_diff / s_diff if s_diff > 1e-9 else 0.0
                c_res[dim] = {
                    "mean_delta": m_diff,
                    "std_delta": s_diff,
                    "t_stat": float(t_st),
                    "p_val": float(p_v),
                    "q_val": float(p_v), # will be updated
                    "cohen_dz": float(cd_z),
                    "count": len(c_diff)
                }
                p_vals_to_correct.append(float(p_v))
                keys_for_p.append((task, c_name, dim))
            t_res["clusters"][c_name] = c_res
            
        stats["rq1"][task] = t_res

    # Apply FDR Correction across all RQ1 tests
    if p_vals_to_correct:
        try:
            from statsmodels.stats.multitest import multipletests
            _, q_vals, _, _ = multipletests(p_vals_to_correct, method="fdr_bh")
            for idx, (t, c, d) in enumerate(keys_for_p):
                if c == "overall":
                    stats["rq1"][t]["overall"][d]["q_val"] = float(q_vals[idx])
                else:
                    stats["rq1"][t]["clusters"][c][d]["q_val"] = float(q_vals[idx])
        except Exception:
            pass

    # =========================================================================
    # RQ2: Dose-Response (Neutral -> Moderate -> Clinical, N=48 triplets)
    # =========================================================================
    triplets = df[df["triplet_id"].notna()].copy()
    t_groups = triplets.groupby("triplet_id")
    
    rq2_res = {}
    for task in ["reader", "self"]:
        t_p = task[0]
        cluster_dr = {
            "negative": {"mn_v": [], "pm_v": [], "mn_a": [], "pm_a": [], "mono": 0, "total": 0},
            "positive": {"mn_v": [], "pm_v": [], "mn_a": [], "pm_a": [], "mono": 0, "total": 0},
            "alert": {"mn_v": [], "pm_v": [], "mn_a": [], "pm_a": [], "mono": 0, "total": 0},
        }
        overall_mono = 0
        valid_triplets = 0
        
        for tid, grp in t_groups:
            if len(grp) != 3:
                continue
            row_neu = grp[grp["split"] == "neutral"]
            row_mod = grp[grp["split"] == "moderate"]
            row_cli = grp[grp["split"] == "clinical"]
            if len(row_neu) == 0 or len(row_mod) == 0 or len(row_cli) == 0:
                continue
            
            valid_triplets += 1
            v_neu = row_neu[f"{t_p}_ev"].values[0]
            v_mod = row_mod[f"{t_p}_ev"].values[0]
            v_cli = row_cli[f"{t_p}_ev"].values[0]
            
            a_neu = row_neu[f"{t_p}_ea"].values[0]
            a_mod = row_mod[f"{t_p}_ea"].values[0]
            a_cli = row_cli[f"{t_p}_ea"].values[0]
            
            emo = row_cli["emotion"].values[0] if pd.notna(row_cli["emotion"].values[0]) else "unknown"
            
            d_mn_v = v_mod - v_neu
            d_pm_v = v_cli - v_mod
            d_mn_a = a_mod - a_neu
            d_pm_a = a_cli - a_mod
            
            # Identify cluster
            c_key = None
            if emo in CLUSTERS["negative"]:
                c_key = "negative"
                is_m = (v_neu >= v_mod and v_mod >= v_cli)
            elif emo in CLUSTERS["positive"]:
                c_key = "positive"
                is_m = (v_neu <= v_mod and v_mod <= v_cli)
            else:
                c_key = "alert"
                is_m = (a_neu <= a_mod and a_mod <= a_cli)
                
            if is_m:
                overall_mono += 1
                
            if c_key:
                cluster_dr[c_key]["mn_v"].append(d_mn_v)
                cluster_dr[c_key]["pm_v"].append(d_pm_v)
                cluster_dr[c_key]["mn_a"].append(d_mn_a)
                cluster_dr[c_key]["pm_a"].append(d_pm_a)
                cluster_dr[c_key]["total"] += 1
                if is_m:
                    cluster_dr[c_key]["mono"] += 1

        # Summary for RQ2: LMM (Primary) + Monotonicity Rate (Secondary)
        lmm_v = fit_lmm_dose_response(triplets, t_p, "v")
        lmm_a = fit_lmm_dose_response(triplets, t_p, "a")

        c_summary = {}
        for c_k, cd in cluster_dr.items():
            c_summary[c_k] = {
                "mean_mn_v": float(np.mean(cd["mn_v"])) if cd["mn_v"] else 0.0,
                "mean_pm_v": float(np.mean(cd["pm_v"])) if cd["pm_v"] else 0.0,
                "mean_mn_a": float(np.mean(cd["mn_a"])) if cd["mn_a"] else 0.0,
                "mean_pm_a": float(np.mean(cd["pm_a"])) if cd["pm_a"] else 0.0,
                "mono_rate": float(cd["mono"] / cd["total"] * 100.0) if cd["total"] else 0.0,
                "count": cd["total"]
            }
        rq2_res[task] = {
            "lmm_v": lmm_v,
            "lmm_a": lmm_a,
            "overall_mono_rate": float(overall_mono / valid_triplets * 100.0) if valid_triplets else 0.0,
            "clusters": c_summary
        }
    stats["rq2"] = rq2_res

    # =========================================================================
    # RQ3: Specificity (Complexity Control: Contrast + Bootstrap CI)
    # =========================================================================
    rq3_res = {}
    for task in ["reader", "self"]:
        t_p = task[0]
        mean_neu_v = float(df[df["split"] == "neutral"][f"{t_p}_ev"].mean())
        mean_neu_a = float(df[df["split"] == "neutral"][f"{t_p}_ea"].mean())
        mean_comp_v = float(df[df["split"] == "complex_neutral"][f"{t_p}_ev"].mean())
        mean_comp_a = float(df[df["split"] == "complex_neutral"][f"{t_p}_ea"].mean())
        
        delta_comp_v = mean_comp_v - mean_neu_v
        delta_comp_a = mean_comp_a - mean_neu_a
        
        # Cluster-specific contrast: |Delta_affect| > |Delta_complexity|
        c_spec = {}
        for c_name, emos in CLUSTERS.items():
            sub_cli = df[(df["split"] == "clinical") & (df["emotion"].isin(emos))]
            if len(sub_cli) == 0:
                continue
            
            # Pair-level affect differences
            sub_pairs = c_pairs[c_pairs["emotion"].isin(emos)]
            matched_neu = n_pairs.loc[sub_pairs.index]
            
            pair_delta_v = np.abs(sub_pairs[f"{t_p}_ev"].values - matched_neu[f"{t_p}_ev"].values)
            pair_delta_a = np.abs(sub_pairs[f"{t_p}_ea"].values - matched_neu[f"{t_p}_ea"].values)
            
            # Contrast: |Delta_affect| - |Delta_complexity|
            contrast_v = pair_delta_v - abs(delta_comp_v)
            contrast_a = pair_delta_a - abs(delta_comp_a)
            
            # Bootstrap 95% CI for contrast
            rng = np.random.default_rng(42)
            n_boot = 1000
            n_sub = len(contrast_v)
            boots_v, boots_a = [], []
            for _ in range(n_boot):
                b_idx = rng.choice(n_sub, size=n_sub, replace=True)
                boots_v.append(np.mean(contrast_v[b_idx]))
                boots_a.append(np.mean(contrast_a[b_idx]))
            
            ci_v_l = float(np.percentile(boots_v, 2.5))
            ci_v_u = float(np.percentile(boots_v, 97.5))
            ci_a_l = float(np.percentile(boots_a, 2.5))
            ci_a_u = float(np.percentile(boots_a, 97.5))
            
            raw_v = float(np.mean(pair_delta_v))
            raw_a = float(np.mean(pair_delta_a))
            
            c_spec[c_name] = {
                "mean_raw_affect_v": raw_v,
                "mean_raw_affect_a": raw_a,
                "contrast_v": float(np.mean(contrast_v)),
                "contrast_v_ci": (ci_v_l, ci_v_u),
                "contrast_v_sig": bool(ci_v_l > 0),
                "contrast_a": float(np.mean(contrast_a)),
                "contrast_a_ci": (ci_a_l, ci_a_u),
                "contrast_a_sig": bool(ci_a_l > 0),
                "count": n_sub
            }
            
        rq3_res[task] = {
            "delta_complexity_v": delta_comp_v,
            "delta_complexity_a": delta_comp_a,
            "clusters": c_spec
        }
    stats["rq3"] = rq3_res

    # =========================================================================
    # RQ4: Recognition-Self-Report Coupling (Delta R vs. Delta S)
    # =========================================================================
    delta_r_v = c_pairs["r_ev"].values - n_pairs["r_ev"].values
    delta_s_v = c_pairs["s_ev"].values - n_pairs["s_ev"].values
    delta_r_a = c_pairs["r_ea"].values - n_pairs["r_ea"].values
    delta_s_a = c_pairs["s_ea"].values - n_pairs["s_ea"].values
    
    corr_v, p_v, rho_v, rho_p_v = safe_corr(delta_r_v, delta_s_v)
    corr_a, p_a, rho_a, rho_p_a = safe_corr(delta_r_a, delta_s_a)
    
    r_ci_vl, r_ci_vu, rho_ci_vl, rho_ci_vu = bootstrap_corr_ci(delta_r_v, delta_s_v)
    r_ci_al, r_ci_au, rho_ci_al, rho_ci_au = bootstrap_corr_ci(delta_r_a, delta_s_a)
    
    # Cluster-specific correlations with CI
    cluster_corrs = {}
    for c_name, emos in CLUSTERS.items():
        sub_mask = c_pairs["emotion"].isin(emos)
        sub_rv = delta_r_v[sub_mask]
        sub_sv = delta_s_v[sub_mask]
        sub_ra = delta_r_a[sub_mask]
        sub_sa = delta_s_a[sub_mask]
        cv, pv, rhov, _ = safe_corr(sub_rv, sub_sv)
        ca, pa, rhoa, _ = safe_corr(sub_ra, sub_sa)
        cv_l, cv_u, _, _ = bootstrap_corr_ci(sub_rv, sub_sv)
        ca_l, ca_u, _, _ = bootstrap_corr_ci(sub_ra, sub_sa)
        cluster_corrs[c_name] = {
            "corr_v": cv, "p_v": pv, "ci_v": (cv_l, cv_u), "rho_v": rhov,
            "corr_a": ca, "p_a": pa, "ci_a": (ca_l, ca_u), "rho_a": rhoa
        }
    
    amp_ratio_v = float(np.mean(np.abs(delta_s_v)) / (np.mean(np.abs(delta_r_v)) + 1e-6))
    amp_ratio_a = float(np.mean(np.abs(delta_s_a)) / (np.mean(np.abs(delta_r_a)) + 1e-6))
    
    exact_neutral_argmax_pct = float(((df["s_gv"] == 5) & (df["s_ga"] == 5) & (df["s_gd"] == 5)).mean() * 100.0)

    stats["rq4"] = {
        "overall_corr_v": corr_v,
        "overall_p_v": p_v,
        "overall_r_ci_v": (r_ci_vl, r_ci_vu),
        "overall_rho_v": rho_v,
        "overall_rho_p_v": rho_p_v,
        "overall_corr_a": corr_a,
        "overall_p_a": p_a,
        "overall_r_ci_a": (r_ci_al, r_ci_au),
        "overall_rho_a": rho_a,
        "overall_rho_p_a": rho_p_a,
        "cluster_corrs": cluster_corrs,
        "amplitude_ratio_v": amp_ratio_v,
        "amplitude_ratio_a": amp_ratio_a,
        "exact_neutral_argmax_pct": exact_neutral_argmax_pct,
    }

    # =========================================================================
    # 8 Emotion Granular Profiles (Clinical vs. Neutral per Emotion)
    # =========================================================================
    emo_stats = {}
    for emo, exp_pattern in EMOTIONS:
        sub_c = c_pairs[c_pairs["emotion"] == emo]
        sub_n = n_pairs[n_pairs["target_emotion"] == emo]
        common = sub_c.index.intersection(sub_n.index)
        if len(common) == 0:
            continue
        sc = sub_c.loc[common]
        sn = sub_n.loc[common]
        
        emo_stats[emo] = {
            "count": len(common),
            "pattern": exp_pattern,
            "r_delta_v": float(np.mean(sc["r_ev"] - sn["r_ev"])),
            "r_delta_a": float(np.mean(sc["r_ea"] - sn["r_ea"])),
            "r_delta_d": float(np.mean(sc["r_ed"] - sn["r_ed"])),
            "s_delta_v": float(np.mean(sc["s_ev"] - sn["s_ev"])),
            "s_delta_a": float(np.mean(sc["s_ea"] - sn["s_ea"])),
            "s_delta_d": float(np.mean(sc["s_ed"] - sn["s_ed"])),
        }
    stats["emotion_profiles"] = emo_stats

    return stats

def format_p(p):
    if np.isnan(p):
        return "N/A"
    if p < 0.001:
        return "<.001"
    return f"{p:.3f}"

def get_family_pairs(all_models):
    """Pairs models into 4 families: Qwen 2.5 1.5B, Mistral 7B, Llama 3.2 1B, Gemma 2 2B"""
    family_defs = [
        ("Qwen 2.5 1.5B", ["qwen2.5_1.5b", "qwen"]),
        ("Mistral 7B", ["mistral_7b", "mistral7b", "mistral"]),
        ("Llama 3.2 1B", ["llama3.2_1b", "llama"]),
        ("Gemma 2 2B", ["gemma2_2b", "gemma"]),
    ]
    family_pairs = []
    for fam_name, patterns in family_defs:
        base_m = None
        inst_m = None
        for m in all_models:
            t = m["tag"].lower()
            if any(p in t for p in patterns):
                if m["is_instruct"]:
                    inst_m = m
                else:
                    base_m = m
        family_pairs.append({
            "name": fam_name,
            "base": base_m,
            "instruct": inst_m
        })
    return family_pairs

def generate_markdown_report(all_models, out_path):
    lines = []
    lines.append("# AIPsy-Affect 4-Split Comprehensive Evaluation Report\n")
    lines.append("本レポートは、AIPsy-Affectの4つのsplit（clinical, neutral, moderate, complex_neutral）を活用し、")
    lines.append("行動実験（Behavioral Stage）における主要成果を以下の **4大Primary解析の柱（Four Methodological Pillars）** に整理して報告した結果です。\n")
    lines.append("※ **スケール標準化**: 全数値を 1〜5 尺度（中立点 3.0）に標準化し、**VA（Valence-Arousal）2次元**に焦点を当てて集計しています。\n")
    lines.append("1. **① Human Grounding**: 人間感情空間（EmoBank）への対応付け・外部接地")
    lines.append("2. **② Sensitivity (RQ1)**: 感情刺激と中立対の弁別（対応のある効果量 $d_z = \\bar\\Delta / s_\\Delta$, $\\text{ddof}=1$ ＋ FDR多重比較補正 $q$ 値）")
    lines.append("3. **③ Dose-Response & Specificity (RQ2 & RQ3)**: 混合効果モデル $Y \\sim \\text{Intensity} + (1|\\text{pair})$ の傾き ＆ 複雑性統制コントラスト $|\\Delta_{\\rm affect}| > |\\Delta_{\\rm complexity}|$ (Bootstrap 95% CI)")
    lines.append("4. **④ Reader–Self Coupling (RQ4)**: 他者認識と自己報告の連動性（Pearson $r$ [Pair Bootstrap 95% CI] ＆ Spearman $\\rho$ 併記）\n")
    lines.append("---\n")

    fam_pairs = get_family_pairs(all_models)

    # =========================================================================
    # Part 1: 4大Primary解析 Base vs. Instruct 統合対比表
    # =========================================================================
    lines.append("## Part 1: 4大Primary解析 Base vs. Instruct 統合対比表 (1-5尺度)\n")
    lines.append("> **感情群クラスタ**:\n"
                 "> - **Negative（4感情）**: `grief`, `terror`, `rage`, `loathing` (理論期待: $V-$)\n"
                 "> - **Positive（2感情）**: `ecstasy`, `admiration` (理論期待: $V+$)\n"
                 "> - **Alert（2感情）**: `amazement`, `vigilance` (理論期待: $A+$)\n")

    # -------------------------------------------------------------------------
    # 柱1 & 柱2: RQ1 Sensitivity
    # -------------------------------------------------------------------------
    lines.append("### 1. 【Sensitivity: 他者認識 (Reader) ＆ 自己報告 (Self)】 Paired $d_z$ ＆ FDR $q$-value")
    lines.append("> **仕様**: $d_z = \\frac{\\bar\\Delta}{s_\\Delta}$ (`ddof=1`), Benjamini-Hochberg FDR補正済 $q$ 値")
    lines.append("")
    lines.append("| モデルファミリー | 種別 | タスク | Negative $\\Delta V$ ($d_z, q$) | Positive $\\Delta V$ ($d_z, q$) | Alert $\\Delta A$ ($d_z, q$) |")
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|")
    for fp in fam_pairs:
        fam_name = fp["name"]
        for m_type, m in [("Base", fp["base"]), ("Instruct", fp["instruct"])]:
            if m is None:
                lines.append(f"| **{fam_name}** | {m_type} | Reader / Self | - | - | - |")
                continue
            for t_key, t_label in [("reader", "Reader"), ("self", "Self")]:
                c = m["rq1"][t_key]["clusters"]
                neg = c["negative"]["v"]
                pos = c["positive"]["v"]
                alt = c["alert"]["a"]
                lines.append(
                    f"| **{fam_name}** | {m_type} | {t_label} | "
                    f"**{neg['mean_delta']:+.2f}** ($d_z={neg['cohen_dz']:+.2f}, q={format_p(neg['q_val'])}$) | "
                    f"**{pos['mean_delta']:+.2f}** ($d_z={pos['cohen_dz']:+.2f}, q={format_p(pos['q_val'])}$) | "
                    f"**{alt['mean_delta']:+.2f}** ($d_z={alt['cohen_dz']:+.2f}, q={format_p(alt['q_val'])}$) |"
                )
    lines.append("")

    # -------------------------------------------------------------------------
    # 柱3: RQ2 & RQ3 Dose-Response & Specificity
    # -------------------------------------------------------------------------
    lines.append("### 2. 【Dose-Response ＆ Specificity】 LMM傾き $\\beta_{\\text{Intensity}}$ ＆ 複雑性コントラスト (Bootstrap CI)")
    lines.append("> **仕様**: Primary解析として $Y \\sim \\text{Intensity} + (1|\\text{pair})$ の傾き $\\beta$ と、純感情コントラスト $|\\Delta_{\\rm affect}| - |\\Delta_{\\rm complexity}|$ [95% CI] を採用")
    lines.append("")
    lines.append("| モデルファミリー | 種別 | タスク | Dose-Resp LMM $\\beta_V$ ($p$) | Dose-Resp 単調率 | 複雑性 $\\Delta V_{comp}$ | Negative純感情コントラスト [95% CI] |")
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for fp in fam_pairs:
        fam_name = fp["name"]
        for m_type, m in [("Base", fp["base"]), ("Instruct", fp["instruct"])]:
            if m is None:
                lines.append(f"| **{fam_name}** | {m_type} | Reader / Self | - | - | - | - |")
                continue
            for t_key, t_label in [("reader", "Reader"), ("self", "Self")]:
                r2 = m["rq2"][t_key]
                r3 = m["rq3"][t_key]
                neg_spec = r3["clusters"]["negative"]
                lmm_v = r2["lmm_v"]
                ci_str = f"[{neg_spec['contrast_v_ci'][0]:+.2f}, {neg_spec['contrast_v_ci'][1]:+.2f}]"
                sig_mark = "*" if neg_spec["contrast_v_sig"] else ""
                lines.append(
                    f"| **{fam_name}** | {m_type} | {t_label} | "
                    f"$\\beta={lmm_v['slope']:+.3f}$ (`{format_p(lmm_v['p_val'])}`) | "
                    f"{r2['overall_mono_rate']:.1f}% | "
                    f"{r3['delta_complexity_v']:+.2f} | "
                    f"**{neg_spec['contrast_v']:+.2f}** {ci_str}{sig_mark} |"
                )
    lines.append("")

    # -------------------------------------------------------------------------
    # 柱4: RQ4 Reader-Self Coupling
    # -------------------------------------------------------------------------
    lines.append("### 3. 【Reader–Self Coupling】 他者認識と自己報告の連動性 ($\\Delta VA_R \\leftrightarrow \\Delta VA_S$)")
    lines.append("> **仕様**: Pearson $r$ [Pair Bootstrap 95% CI]、Spearman $\\rho$、および客観的中立最尤率（Exact Neutral Rate）")
    lines.append("")
    lines.append("| モデルファミリー | 種別 | 全体連動 $r_V$ [95% CI] | 順位連動 $\\rho_V$ | 全体連動 $r_A$ [95% CI] | 順位連動 $\\rho_A$ | 振幅比率 $\\frac{\\|\\Delta_S\\|}{\\|\\Delta_R\\|}_V$ | 完全中立最尤率 $(5,5,5)$ |")
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for fp in fam_pairs:
        fam_name = fp["name"]
        for m_type, m in [("Base", fp["base"]), ("Instruct", fp["instruct"])]:
            if m is None:
                lines.append(f"| **{fam_name}** | {m_type} | - | - | - | - | - | - |")
                continue
            r4 = m["rq4"]
            ci_v = f"[{r4['overall_r_ci_v'][0]:.2f}, {r4['overall_r_ci_v'][1]:.2f}]"
            ci_a = f"[{r4['overall_r_ci_a'][0]:.2f}, {r4['overall_r_ci_a'][1]:.2f}]"
            lines.append(
                f"| **{fam_name}** | {m_type} | "
                f"**{r4['overall_corr_v']:.3f}** {ci_v} | {r4['overall_rho_v']:.3f} | "
                f"**{r4['overall_corr_a']:.3f}** {ci_a} | {r4['overall_rho_a']:.3f} | "
                f"{r4['amplitude_ratio_v']:.2f} | "
                f"`{r4['exact_neutral_argmax_pct']:.1f}%` |"
            )
    lines.append("\n---\n")

    # =========================================================================
    # Part 2: モデル個別詳細レポート
    # =========================================================================
    lines.append("## Part 2: モデル個別詳細レポート\n")
    for m in all_models:
        tag = m["tag"]
        m_type = "Instruct" if m["is_instruct"] else "Base"
        lines.append(f"### ■ モデル: `{tag}` ({m_type})\n")

        # Table 1: RQ1 Sensitivity
        lines.append("#### 【RQ1: Sensitivity】 3大感情刺激に対する反応性 (Paired $d_z$ & FDR $q$)")
        lines.append("| タスク | 感情群 (Cluster) | $\\Delta V$ ($d_z$) | $t$-value ($p$, $q$) | $\\Delta A$ ($d_z$) | $t$-value ($p$, $q$) |")
        lines.append("|:---|:---:|:---:|:---:|:---:|:---:|")
        for t_name in ["reader", "self"]:
            t_label = "Reader (R)" if t_name == "reader" else "Self (S)"
            cls = m["rq1"][t_name]["clusters"]
            for c_name, c_label in [("negative", "Negative (4感情)"), ("positive", "Positive (2感情)"), ("alert", "Alert (2感情)")]:
                cd = cls[c_name]
                lines.append(
                    f"| **{t_label}** | {c_label} | "
                    f"**{cd['v']['mean_delta']:+.2f}** ($d_z={cd['v']['cohen_dz']:+.2f}$) | {cd['v']['t_stat']:.2f} (`p={format_p(cd['v']['p_val'])}, q={format_p(cd['v']['q_val'])}`) | "
                    f"**{cd['a']['mean_delta']:+.2f}** ($d_z={cd['a']['cohen_dz']:+.2f}$) | {cd['a']['t_stat']:.2f} (`p={format_p(cd['a']['p_val'])}, q={format_p(cd['a']['q_val'])}`) |"
                )
        lines.append("")

        # Table 2: RQ2 Dose-Response
        lines.append("#### 【RQ2: Dose-Response】 感情強度の段階性 (LMM Primary)")
        for t_name in ["reader", "self"]:
            t_label = "Reader (R)" if t_name == "reader" else "Self (S)"
            r2 = m["rq2"][t_name]
            lv = r2["lmm_v"]
            la = r2["lmm_a"]
            lines.append(f"- **{t_label}**: Valence LMM $\\beta={lv['slope']:+.3f}$ ($p={format_p(lv['p_val'])}$), Arousal LMM $\\beta={la['slope']:+.3f}$ ($p={format_p(la['p_val'])}$), 単調性成立率 {r2['overall_mono_rate']:.1f}%\n")

        # Table 3: RQ3 Specificity
        lines.append("#### 【RQ3: Specificity】 複雑性統制コントラスト ($|\\Delta_{\\rm affect}| - |\\Delta_{\\rm complexity}|$ [95% CI])")
        lines.append("| タスク | 複雑性 $\\Delta V_{comp}$ | Negativeコントラスト [95% CI] | Positiveコントラスト [95% CI] | Alertコントラスト [95% CI] |")
        lines.append("|:---|:---:|:---:|:---:|:---:|")
        for t_name in ["reader", "self"]:
            t_label = "Reader (R)" if t_name == "reader" else "Self (S)"
            r3 = m["rq3"][t_name]
            cn = r3["clusters"]["negative"]
            cp = r3["clusters"]["positive"]
            ca = r3["clusters"]["alert"]
            lines.append(
                f"| **{t_label}** | {r3['delta_complexity_v']:+.2f} | "
                f"**{cn['contrast_v']:+.2f}** [{cn['contrast_v_ci'][0]:+.2f}, {cn['contrast_v_ci'][1]:+.2f}] | "
                f"**{cp['contrast_v']:+.2f}** [{cp['contrast_v_ci'][0]:+.2f}, {cp['contrast_v_ci'][1]:+.2f}] | "
                f"**{ca['contrast_a']:+.2f}** [{ca['contrast_a_ci'][0]:+.2f}, {ca['contrast_a_ci'][1]:+.2f}] |"
            )
        lines.append("")

        # Table 4: RQ4 Coupling
        lines.append("#### 【RQ4: Coupling】 認識と自己報告の連動性")
        r4 = m["rq4"]
        cc = r4["cluster_corrs"]
        lines.append("| 指標 | 全体 (Overall) [95% CI] | Spearman $\\rho$ | Negative群 [95% CI] | Positive群 [95% CI] | Alert群 [95% CI] |")
        lines.append("|:---|:---:|:---:|:---:|:---:|:---:|")
        civ = f"[{r4['overall_r_ci_v'][0]:.2f}, {r4['overall_r_ci_v'][1]:.2f}]"
        cia = f"[{r4['overall_r_ci_a'][0]:.2f}, {r4['overall_r_ci_a'][1]:.2f}]"
        lines.append(f"| **連動相関 $r_V$** | **{r4['overall_corr_v']:.3f}** {civ} | {r4['overall_rho_v']:.3f} | {cc['negative']['corr_v']:.3f} [{cc['negative']['ci_v'][0]:.2f}, {cc['negative']['ci_v'][1]:.2f}] | {cc['positive']['corr_v']:.3f} | {cc['alert']['corr_v']:.3f} |")
        lines.append(f"| **連動相関 $r_A$** | **{r4['overall_corr_a']:.3f}** {cia} | {r4['overall_rho_a']:.3f} | {cc['negative']['corr_a']:.3f} [{cc['negative']['ci_a'][0]:.2f}, {cc['negative']['ci_a'][1]:.2f}] | {cc['positive']['corr_a']:.3f} | {cc['alert']['corr_a']:.3f} |")
        lines.append(f"- **完全中立最尤率**: `{r4['exact_neutral_argmax_pct']:.1f}%`, 振幅比率 $V={r4['amplitude_ratio_v']:.2f}, A={r4['amplitude_ratio_a']:.2f}$\n")
        lines.append("---\n")

    report_content = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    return report_content

def build_summary_csv(all_models, out_csv):
    rows = []
    for m in all_models:
        r_neg = m["rq1"]["reader"]["clusters"]["negative"]
        r_pos = m["rq1"]["reader"]["clusters"]["positive"]
        s_neg = m["rq1"]["self"]["clusters"]["negative"]
        s_pos = m["rq1"]["self"]["clusters"]["positive"]
        
        row = {
            "tag": m["tag"],
            "is_instruct": m["is_instruct"],
            "num_pairs": m["num_pairs"],
            "rq1_reader_neg_v_delta": r_neg["v"]["mean_delta"],
            "rq1_reader_neg_v_dz": r_neg["v"]["cohen_dz"],
            "rq1_reader_neg_v_q": r_neg["v"]["q_val"],
            "rq1_self_neg_v_delta": s_neg["v"]["mean_delta"],
            "rq1_self_neg_v_dz": s_neg["v"]["cohen_dz"],
            "rq1_self_neg_v_q": s_neg["v"]["q_val"],
            "rq2_reader_mono_rate": m["rq2"]["reader"]["overall_mono_rate"],
            "rq2_reader_lmm_slope_v": m["rq2"]["reader"]["lmm_v"]["slope"],
            "rq2_reader_lmm_p_v": m["rq2"]["reader"]["lmm_v"]["p_val"],
            "rq2_self_mono_rate": m["rq2"]["self"]["overall_mono_rate"],
            "rq2_self_lmm_slope_v": m["rq2"]["self"]["lmm_v"]["slope"],
            "rq2_self_lmm_p_v": m["rq2"]["self"]["lmm_v"]["p_val"],
            "rq3_reader_delta_complexity_v": m["rq3"]["reader"]["delta_complexity_v"],
            "rq3_reader_contrast_v": m["rq3"]["reader"]["clusters"]["negative"]["contrast_v"],
            "rq3_self_delta_complexity_v": m["rq3"]["self"]["delta_complexity_v"],
            "rq3_self_contrast_v": m["rq3"]["self"]["clusters"]["negative"]["contrast_v"],
            "rq4_coupling_v_r": m["rq4"]["overall_corr_v"],
            "rq4_coupling_v_ci_lower": m["rq4"]["overall_r_ci_v"][0],
            "rq4_coupling_v_ci_upper": m["rq4"]["overall_r_ci_v"][1],
            "rq4_coupling_v_rho": m["rq4"]["overall_rho_v"],
            "rq4_coupling_a_r": m["rq4"]["overall_corr_a"],
            "rq4_coupling_a_ci_lower": m["rq4"]["overall_r_ci_a"][0],
            "rq4_coupling_a_ci_upper": m["rq4"]["overall_r_ci_a"][1],
            "rq4_coupling_a_rho": m["rq4"]["overall_rho_a"],
            "exact_neutral_argmax_pct": m["rq4"]["exact_neutral_argmax_pct"],
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df

def main():
    parser = argparse.ArgumentParser(description="Summarize AIPsy-Affect 4-split evaluation results.")
    parser.add_argument("--results-dir", type=str, default="v1/results/aipsy_4split_eval")
    parser.add_argument("--out-md", type=str, default=None,
                        help="Path for output Markdown report")
    parser.add_argument("--out-tex", type=str, default=None,
                        help="Path for output LaTeX report")
    parser.add_argument("--out-csv", type=str, default=None,
                        help="Path for output summary CSV")
    args = parser.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.results_dir, "*_aipsy_4split.csv")))
    if not csv_files:
        print(f"[!] No *_aipsy_4split.csv files found in {args.results_dir}")
        return

    print(f"Found {len(csv_files)} model result CSVs in {args.results_dir}")
    all_models = []
    for cf in csv_files:
        print(f"Analyzing {os.path.basename(cf)}...")
        m_stats = analyze_single_model(cf)
        all_models.append(m_stats)

    all_models.sort(key=lambda x: x["tag"])

    out_md = args.out_md or os.path.join(args.results_dir, "aipsy_4split_detailed_report.md")
    out_tex = getattr(args, 'out_tex', None) or os.path.splitext(out_md)[0] + ".tex"
    out_csv = args.out_csv or os.path.join(args.results_dir, "aipsy_4split_summary.csv")

    generate_markdown_report(all_models, out_md)
    convert_md_file_to_tex(out_md, out_tex)
    build_summary_csv(all_models, out_csv)

    print("\n" + "=" * 60)
    print("SUCCESS: AIPsy 4-Split Comprehensive Summary Generated!")
    print(f"  Markdown Report : {out_md}")
    print(f"  LaTeX Report    : {out_tex}")
    print(f"  CSV Summary     : {out_csv}")
    print("=" * 60)

if __name__ == "__main__":
    main()
