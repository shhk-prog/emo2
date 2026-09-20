#!/usr/bin/env python3
"""
scripts/reaggregate_v2_summary.py
V2-RQ1 & RQ2 クロスファミリー集計の再計算スクリプト
v2/results/raw/v2_geometry_*.json から Primary (matched-plain) と Secondary (native-chat) を分離して
v2/results/derived/v2_cross_family_summary.json を再生成する。
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats

def compute_center_of_mass(values, depths):
    v = np.array(values, dtype=np.float64)
    d = np.array(depths, dtype=np.float64)
    v_pos = np.clip(v, 0, None)
    if np.sum(v_pos) > 1e-9:
        return float(np.sum(v_pos * d) / np.sum(v_pos))
    return float(np.mean(d))

def compute_peak_depth(values, depths):
    v = np.array(values, dtype=np.float64)
    d = np.array(depths, dtype=np.float64)
    idx = int(np.argmax(v))
    return float(d[idx])

def compute_bootstrap_ci(values, n_boot=1000, ci=0.95, seed=42):
    v = np.array(values, dtype=np.float64)
    if len(v) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.RandomState(seed)
    boot_means = [np.mean(rng.choice(v, size=len(v), replace=True)) for _ in range(n_boot)]
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, 100 * alpha))
    up = float(np.percentile(boot_means, 100 * (1.0 - alpha)))
    return float(np.mean(v)), low, up

def paired_family_comparison(inst_vals, base_vals):
    diffs = np.array(inst_vals, dtype=np.float64) - np.array(base_vals, dtype=np.float64)
    mean_diff = float(np.mean(diffs))
    if len(diffs) > 1 and np.std(diffs) > 1e-9:
        t_stat, p_val = stats.ttest_1samp(diffs, 0.0)
        try:
            w_stat, w_p_val = stats.wilcoxon(diffs)
        except Exception:
            w_stat, w_p_val = None, None
    else:
        t_stat, p_val = 0.0, 1.0
        w_stat, w_p_val = None, None
    return {
        "mean_difference": mean_diff,
        "t_statistic": float(t_stat) if t_stat is not None else None,
        "p_value": float(p_val) if p_val is not None else None,
        "wilcoxon_p_value": float(w_p_val) if w_p_val is not None else None,
    }

def main():
    repo_root = Path(__file__).resolve().parent.parent
    raw_dir = repo_root / "v2" / "results" / "raw"
    derived_dir = repo_root / "v2" / "results" / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)

    family_files = sorted(raw_dir.glob("v2_geometry_*.json"))
    if not family_files:
        raise FileNotFoundError(f"No geometry files found in {raw_dir}")

    per_family_summary = {}
    fams = []
    
    # Primary (matched-plain) lists
    com_dist_r_prim = []
    com_dist_s_prim = []
    pk_base_cross_prim = []
    pk_inst_cross_prim = []
    mean_delta_share_prim = []

    # Secondary (native-chat) lists
    com_dist_r_sec = []
    com_dist_s_sec = []
    pk_inst_cross_sec = []
    mean_delta_share_sec = []

    for fpath in family_files:
        fam_id = fpath.stem.replace("v2_geometry_", "")
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        fams.append(fam_id)
        depths = data["relative_depths"]
        
        # Geometry
        r_dist = data["rq1_geometry"]["reader_distortion"]
        s_dist = data["rq1_geometry"]["self_distortion"]
        r_dist_m = data["rq1_geometry"].get("reader_distortion_matched", r_dist)
        s_dist_m = data["rq1_geometry"].get("self_distortion_matched", s_dist)

        # Cross-decoding
        p_cross = data["rq2_cross_decoding"]["valence"]
        base_cross = p_cross["base_cross_r_to_s"]
        inst_native_cross = p_cross["inst_cross_r_to_s"]
        inst_matched_cross = p_cross.get("inst_matched_cross_r_to_s", inst_native_cross)

        # Sharing
        p_sharing = data["rq2_sharing"]["valence"]
        delta_native = p_sharing["delta_sharing"]
        delta_matched = p_sharing.get("delta_sharing_matched", delta_native)
        delta_format = p_sharing.get("delta_sharing_format", [0.0] * len(depths))

        # Primary Matched Plain
        prim = {
            "com_distortion_reader": compute_center_of_mass(r_dist_m, depths),
            "com_distortion_self": compute_center_of_mass(s_dist_m, depths),
            "peak_depth_base_cross": compute_peak_depth(base_cross, depths),
            "peak_depth_inst_cross": compute_peak_depth(inst_matched_cross, depths),
            "peak_depth_delta_share": compute_peak_depth(delta_matched, depths),
            "mean_delta_sharing": float(np.mean(delta_matched)),
        }

        # Secondary Native Chat
        sec = {
            "com_distortion_reader": compute_center_of_mass(r_dist, depths),
            "com_distortion_self": compute_center_of_mass(s_dist, depths),
            "peak_depth_base_cross": compute_peak_depth(base_cross, depths),
            "peak_depth_inst_cross": compute_peak_depth(inst_native_cross, depths),
            "peak_depth_delta_share": compute_peak_depth(delta_native, depths),
            "mean_delta_sharing": float(np.mean(delta_native)),
        }

        format_eff = {
            "mean_delta_sharing_format": float(np.mean(delta_format)),
        }

        per_family_summary[fam_id] = {
            "primary_matched_plain": prim,
            "secondary_native_chat": sec,
            "format_effect": format_eff,
            **prim,
            "mean_delta_sharing_matched": prim["mean_delta_sharing"],
            "mean_delta_sharing_format": format_eff["mean_delta_sharing_format"],
        }

        # Collect for bootstrap
        com_dist_r_prim.append(prim["com_distortion_reader"])
        com_dist_s_prim.append(prim["com_distortion_self"])
        pk_base_cross_prim.append(prim["peak_depth_base_cross"])
        pk_inst_cross_prim.append(prim["peak_depth_inst_cross"])
        mean_delta_share_prim.append(prim["mean_delta_sharing"])

        com_dist_r_sec.append(sec["com_distortion_reader"])
        com_dist_s_sec.append(sec["com_distortion_self"])
        pk_inst_cross_sec.append(sec["peak_depth_inst_cross"])
        mean_delta_share_sec.append(sec["mean_delta_sharing"])

    n_boot = 1000
    pt_com_r, com_r_low, com_r_up = compute_bootstrap_ci(com_dist_r_prim, n_boot=n_boot)
    pt_com_s, com_s_low, com_s_up = compute_bootstrap_ci(com_dist_s_prim, n_boot=n_boot)
    pt_base_pk, base_pk_low, base_pk_up = compute_bootstrap_ci(pk_base_cross_prim, n_boot=n_boot)
    pt_inst_pk, inst_pk_low, inst_pk_up = compute_bootstrap_ci(pk_inst_cross_prim, n_boot=n_boot)
    pt_share, share_low, share_up = compute_bootstrap_ci(mean_delta_share_prim, n_boot=n_boot)

    pt_com_r_sec, com_r_sec_low, com_r_sec_up = compute_bootstrap_ci(com_dist_r_sec, n_boot=n_boot)
    pt_com_s_sec, com_s_sec_low, com_s_sec_up = compute_bootstrap_ci(com_dist_s_sec, n_boot=n_boot)
    pt_inst_pk_sec, inst_pk_sec_low, inst_pk_sec_up = compute_bootstrap_ci(pk_inst_cross_sec, n_boot=n_boot)
    pt_share_sec, share_sec_low, share_sec_up = compute_bootstrap_ci(mean_delta_share_sec, n_boot=n_boot)

    paired_peak_comp = paired_family_comparison(pk_inst_cross_prim, pk_base_cross_prim)

    summary_data = {
        "families": fams,
        "per_family_summary": per_family_summary,
        "primary_matched_plain": {
            "bootstrap_ci_95": {
                "com_distortion_reader": {"mean": pt_com_r, "ci_lower": com_r_low, "ci_upper": com_r_up},
                "com_distortion_self": {"mean": pt_com_s, "ci_lower": com_s_low, "ci_upper": com_s_up},
                "peak_depth_base_cross": {"mean": pt_base_pk, "ci_lower": base_pk_low, "ci_upper": base_pk_up},
                "peak_depth_inst_cross": {"mean": pt_inst_pk, "ci_lower": inst_pk_low, "ci_upper": inst_pk_up},
                "mean_delta_sharing": {"mean": pt_share, "ci_lower": share_low, "ci_upper": share_up},
            },
            "paired_peak_depth_comparison": paired_peak_comp,
        },
        "secondary_native_chat": {
            "bootstrap_ci_95": {
                "com_distortion_reader": {"mean": pt_com_r_sec, "ci_lower": com_r_sec_low, "ci_upper": com_r_sec_up},
                "com_distortion_self": {"mean": pt_com_s_sec, "ci_lower": com_s_sec_low, "ci_upper": com_s_sec_up},
                "peak_depth_inst_cross": {"mean": pt_inst_pk_sec, "ci_lower": inst_pk_sec_low, "ci_upper": inst_pk_sec_up},
                "mean_delta_sharing": {"mean": pt_share_sec, "ci_lower": share_sec_low, "ci_upper": share_sec_up},
            },
        },
        "bootstrap_ci_95": {
            "com_distortion_reader": {"mean": pt_com_r, "ci_lower": com_r_low, "ci_upper": com_r_up},
            "com_distortion_self": {"mean": pt_com_s, "ci_lower": com_s_low, "ci_upper": com_s_up},
            "peak_depth_base_cross": {"mean": pt_base_pk, "ci_lower": base_pk_low, "ci_upper": base_pk_up},
            "peak_depth_inst_cross": {"mean": pt_inst_pk, "ci_lower": inst_pk_low, "ci_upper": inst_pk_up},
            "mean_delta_sharing": {"mean": pt_share, "ci_lower": share_low, "ci_upper": share_up},
        },
        "paired_peak_depth_comparison": paired_peak_comp,
    }

    out_path = derived_dir / "v2_cross_family_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Successfully reaggregated V2 summary -> {out_path}")

if __name__ == "__main__":
    main()
