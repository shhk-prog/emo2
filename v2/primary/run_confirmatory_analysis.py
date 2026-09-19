#!/usr/bin/env python3
"""
v2/primary/run_confirmatory_analysis.py

V2 確証的統合統計解析 (Confirmatory Statistical Analysis)
4モデルファミリー (Qwen 2.5, Llama 3.2, Gemma 3, OLMo 2) の成果物を集約し、
以下の確証的仮説検定を実行する:
  - H1: Post-training-Associated Reorganization of Affect Decodability Peak (Δd* shift)
  - H2: Post-training-Associated Alteration of Reader-Self Representation Sharing (ΔSharing)
  - H3: Causal Profile Reorganization across Alignment & Tasks (LMM with Family Fixed Effects)
  - H4: Distributional Recovery Asymmetry (Self vs. Reader Recovery Ratios)
"""

import argparse
import glob
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import yaml

from affective_empathy_eval.geometry import compute_peak_depth
from affective_empathy_eval.statistics import (
    apply_benjamini_hochberg,
    compute_bootstrap_ci,
    fit_sample_level_lmm,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run V2 Confirmatory Statistical Analysis (LMM + FDR)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/v2_experiments.yaml",
        help="Path to V2 config",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="v2/results/raw",
        help="Directory containing raw JSON outputs",
    )
    parser.add_argument(
        "--derived-dir",
        type=str,
        default="v2/results/derived",
        help="Directory for saving derived confirmatory results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate synthetic inputs for fast pipeline validation",
    )
    return parser.parse_args()


def run_confirmatory_analysis(
    raw_dir: Path,
    derived_dir: Path,
    is_dry_run: bool = False,
) -> Dict[str, Any]:
    if is_dry_run:
        raw_dir = raw_dir / "dry_run" if raw_dir.name != "dry_run" else raw_dir
        derived_dir = derived_dir / "dry_run" if derived_dir.name != "dry_run" else derived_dir

    raw_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    families = ["qwen", "llama", "gemma", "olmo"]
    confirmatory_report: Dict[str, Any] = {
        "status": "success",
        "dry_run": is_dry_run,
        "families": families,
        "hypotheses": {},
    }

    # =========================================================================
    # H1: Post-training-Associated Reorganization of Affect Decodability Peak
    # =========================================================================
    # 出力構造: valence.reader.shift, valence.self.shift, arousal.reader.shift, arousal.self.shift
    axes = ["valence", "arousal"]
    sub_keys = ["reader", "self"]

    h1_shifts: Dict[str, Dict[str, List[float]]] = {
        ax: {k: [] for k in sub_keys} for ax in axes
    }
    per_family_h1: Dict[str, Dict[str, float]] = {}

    if is_dry_run:
        # モックH1データ: 4ファミリー分のシフト
        mock_shifts = {
            "qwen": {"valence.reader.shift": 0.12, "valence.self.shift": 0.08, "arousal.reader.shift": 0.10, "arousal.self.shift": 0.05},
            "llama": {"valence.reader.shift": 0.08, "valence.self.shift": 0.05, "arousal.reader.shift": 0.06, "arousal.self.shift": 0.04},
            "gemma": {"valence.reader.shift": 0.15, "valence.self.shift": 0.11, "arousal.reader.shift": 0.12, "arousal.self.shift": 0.09},
            "olmo": {"valence.reader.shift": 0.10, "valence.self.shift": 0.07, "arousal.reader.shift": 0.08, "arousal.self.shift": 0.06},
        }
        for fam_id, s_dict in mock_shifts.items():
            per_family_h1[fam_id] = s_dict
            for ax in axes:
                for k in sub_keys:
                    h1_shifts[ax][k].append(s_dict[f"{ax}.{k}.shift"])
    else:
        geom_files = list(raw_dir.glob("v2_geometry_*.json"))
        if len(geom_files) < 2:
            raise FileNotFoundError(f"Required geometry files not found in {raw_dir} (found {len(geom_files)})")
        for gf in geom_files:
            try:
                fam_id = gf.stem.replace("v2_geometry_", "")
                with open(gf, "r", encoding="utf-8") as f:
                    gdata = json.load(f)

                rq2_sharing = gdata.get("rq2_sharing", {})
                fam_dict: Dict[str, float] = {}

                for ax in axes:
                    axis_data = rq2_sharing.get(ax, {})
                    base_reader = axis_data.get("base_r2_reader", [])
                    base_self = axis_data.get("base_r2_self", [])
                    # Primary: matched-plain if available, else native-chat
                    inst_reader = axis_data.get("inst_matched_r2_reader") or axis_data.get("inst_native_r2_reader", [])
                    inst_self = axis_data.get("inst_matched_r2_self") or axis_data.get("inst_native_r2_self", [])

                    L = len(base_reader)
                    if L == 0:
                        continue
                    depths = np.asarray(gdata.get("relative_depths") or np.linspace(0.0, 1.0, L))

                    base_pk_r = compute_peak_depth(base_reader, depths)
                    inst_pk_r = compute_peak_depth(inst_reader, depths)
                    base_pk_s = compute_peak_depth(base_self, depths)
                    inst_pk_s = compute_peak_depth(inst_self, depths)

                    shift_r = float(inst_pk_r - base_pk_r) if not (np.isnan(base_pk_r) or np.isnan(inst_pk_r)) else np.nan
                    shift_s = float(inst_pk_s - base_pk_s) if not (np.isnan(base_pk_s) or np.isnan(inst_pk_s)) else np.nan

                    if not np.isnan(shift_r):
                        h1_shifts[ax]["reader"].append(shift_r)
                    if not np.isnan(shift_s):
                        h1_shifts[ax]["self"].append(shift_s)

                    fam_dict[f"{ax}.reader.shift"] = shift_r
                    fam_dict[f"{ax}.self.shift"] = shift_s

                per_family_h1[fam_id] = fam_dict
            except Exception as e:
                logger.warning(f"Failed to parse {gf} for H1: {e}")

        # 最低2つのファミリーで有効な値が取れているか検証
        valid_v_r = len(h1_shifts["valence"]["reader"])
        if valid_v_r < 2:
            raise RuntimeError(f"Insufficient valid geometry data for H1 in {raw_dir}: {valid_v_r}")

    h1_report: Dict[str, Any] = {
        "interpretation": "post-training-associated reorganization of decodability peak",
        "per_family_shifts": per_family_h1,
        "effects": {},
    }
    for ax in axes:
        for k in sub_keys:
            shifts = h1_shifts[ax][k]
            if len(shifts) >= 2:
                pt, ci_low, ci_high = compute_bootstrap_ci(shifts, n_boot=1000)
                reorg_supported = bool(ci_low > 0.0 or ci_high < 0.0)
            else:
                pt, ci_low, ci_high, reorg_supported = np.nan, np.nan, np.nan, False
            h1_report["effects"][f"{ax}.{k}.shift"] = {
                "mean_shift": float(pt) if not np.isnan(pt) else None,
                "bootstrap_ci_95": [float(ci_low), float(ci_high)] if not np.isnan(ci_low) else None,
                "reorganization_supported": reorg_supported,
            }

    confirmatory_report["hypotheses"]["H1_decodability_peak_reorganization"] = h1_report

    # =========================================================================
    # H2: Post-training-Associated Alteration of Reader-Self Representation Sharing
    # =========================================================================
    h2_diffs_primary: Dict[str, List[float]] = {ax: [] for ax in axes}
    h2_diffs_secondary: Dict[str, List[float]] = {ax: [] for ax in axes}
    per_family_h2: Dict[str, Dict[str, float]] = {}

    if is_dry_run:
        mock_h2 = {
            "qwen": {"valence.primary_delta_sharing": -0.08, "valence.secondary_delta_sharing": -0.06, "arousal.primary_delta_sharing": -0.05, "arousal.secondary_delta_sharing": -0.04},
            "llama": {"valence.primary_delta_sharing": -0.12, "valence.secondary_delta_sharing": -0.10, "arousal.primary_delta_sharing": -0.07, "arousal.secondary_delta_sharing": -0.06},
            "gemma": {"valence.primary_delta_sharing": -0.05, "valence.secondary_delta_sharing": -0.04, "arousal.primary_delta_sharing": -0.04, "arousal.secondary_delta_sharing": -0.03},
            "olmo": {"valence.primary_delta_sharing": -0.09, "valence.secondary_delta_sharing": -0.08, "arousal.primary_delta_sharing": -0.06, "arousal.secondary_delta_sharing": -0.05},
        }
        for fam_id, h2_dict in mock_h2.items():
            per_family_h2[fam_id] = h2_dict
            for ax in axes:
                h2_diffs_primary[ax].append(h2_dict[f"{ax}.primary_delta_sharing"])
                h2_diffs_secondary[ax].append(h2_dict[f"{ax}.secondary_delta_sharing"])
    else:
        for gf in geom_files:
            try:
                fam_id = gf.stem.replace("v2_geometry_", "")
                with open(gf, "r", encoding="utf-8") as f:
                    gdata = json.load(f)
                rq2_sharing = gdata.get("rq2_sharing", {})
                fam_h2_dict: Dict[str, float] = {}

                for ax in axes:
                    axis_data = rq2_sharing.get(ax, {})
                    # Primary: Base plain vs Instruct matched-plain (delta_sharing_matched)
                    delta_share_matched = axis_data.get("delta_sharing_matched")
                    # Secondary: Base plain vs Instruct native-chat (delta_sharing_native or delta_sharing)
                    delta_share_native = axis_data.get("delta_sharing_native") or axis_data.get("delta_sharing")

                    if delta_share_matched is not None and len(delta_share_matched) > 0:
                        m_val = float(np.mean(delta_share_matched))
                        h2_diffs_primary[ax].append(m_val)
                        fam_h2_dict[f"{ax}.primary_delta_sharing_matched"] = m_val

                    if delta_share_native is not None and len(delta_share_native) > 0:
                        n_val = float(np.mean(delta_share_native))
                        h2_diffs_secondary[ax].append(n_val)
                        fam_h2_dict[f"{ax}.secondary_delta_sharing_native"] = n_val

                per_family_h2[fam_id] = fam_h2_dict
            except Exception as e:
                logger.warning(f"Failed to parse {gf} for H2: {e}")

        if len(h2_diffs_primary["valence"]) < 2 and len(h2_diffs_secondary["valence"]) < 2:
            raise RuntimeError(f"Insufficient valid geometry files for H2 in {raw_dir}")

    h2_report: Dict[str, Any] = {
        "interpretation": "post-training-associated reorganization of Reader-Self representation sharing",
        "per_family_sharing": per_family_h2,
        "primary_contrast": "Base plain vs Instruct matched-plain",
        "secondary_contrast": "Base plain vs Instruct native-chat",
        "effects": {},
    }
    for ax in axes:
        # Primary
        p_diffs = h2_diffs_primary[ax] if len(h2_diffs_primary[ax]) >= 2 else h2_diffs_secondary[ax]
        contrast_type = "primary_matched" if len(h2_diffs_primary[ax]) >= 2 else "secondary_native_fallback"
        pt, ci_low, ci_high = compute_bootstrap_ci(p_diffs, n_boot=1000)
        reorg_supported = bool(ci_high < 0.0 or ci_low > 0.0)
        h2_report["effects"][ax] = {
            "contrast_type": contrast_type,
            "mean_delta_sharing": float(pt),
            "bootstrap_ci_95": [float(ci_low), float(ci_high)],
            "sharing_alteration_supported": reorg_supported,
        }

    confirmatory_report["hypotheses"]["H2_representation_sharing_reorganization"] = h2_report

    # =========================================================================
    # H3: Causal Profile Reorganization LMM (with Family Fixed Effects)
    # =========================================================================
    pair_csv_path = derived_dir / "v2_causal_pair_level.csv"
    if is_dry_run:
        # Mock pair-level data for dry run only
        rng = np.random.default_rng(42)
        mock_rows = []
        for fam in families:
            for align in ["base", "inst"]:
                for task in ["reader", "self"]:
                    for l in range(8):
                        depth = l / 7.0
                        for p in range(10):
                            base_cv = 1.0 / (1.0 + np.exp(-8.0 * (depth - 0.7)))
                            cv = max(0.0, float(base_cv + (0.2 if align == "inst" else 0.0) + rng.normal(0, 0.05)))
                            ca = max(0.0, float(base_cv * 0.9 + rng.normal(0, 0.05)))
                            mock_rows.append({
                                "family": fam,
                                "alignment": align,
                                "task": task,
                                "pair_id": f"pair_{p}",
                                "layer": l,
                                "relative_depth": depth,
                                "c_v": cv,
                                "c_a": ca,
                                "c_v_zero": cv * 1.2,
                                "c_a_zero": ca * 1.2,
                            })
        df_pair = pd.DataFrame(mock_rows)
    else:
        if not pair_csv_path.exists():
            raise FileNotFoundError(f"Required result not found for real run: {pair_csv_path}")
        df_pair = pd.read_csv(pair_csv_path)

    logger.info(f"Fitting Confirmatory LMM on {len(df_pair)} observations with C(family) fixed effects...")
    lmm_results: Dict[str, Any] = {}

    PRIMARY_TERMS = [
        "C(alignment)[T.inst]:relative_depth",
        "C(alignment)[T.inst]:C(task)[T.self]",
        "C(alignment)[T.inst]:C(task)[T.self]:relative_depth",
    ]

    primary_p_values: List[float] = []
    primary_p_keys: List[str] = []

    for axis_name, col_name in [("valence", "c_v"), ("arousal", "c_a")]:
        formula = f"{col_name} ~ C(family) + C(alignment) * C(task) * relative_depth"
        try:
            lmm_fit = fit_sample_level_lmm(
                df=df_pair,
                formula=formula,
                groups="pair_id",
            )
            lmm_results[axis_name] = {
                "converged": lmm_fit["converged"],
                "formula": formula,
                "params": lmm_fit["params"],
                "pvalues": lmm_fit["pvalues"],
                "conf_int": lmm_fit["conf_int"],
            }
            # Collect primary interaction terms for FDR
            for term in PRIMARY_TERMS:
                if term in lmm_fit["pvalues"]:
                    pval = lmm_fit["pvalues"][term]
                    if not np.isnan(pval):
                        primary_p_values.append(float(pval))
                        primary_p_keys.append(f"{axis_name}_{term}")
        except Exception as e:
            logger.warning(f"LMM {axis_name.capitalize()} fit failed: {e}")
            lmm_results[axis_name] = {"converged": False, "error": str(e), "formula": formula}

    # Primary FDR correction only on prespecified interaction terms
    if primary_p_values:
        fdr_adjusted = apply_benjamini_hochberg(np.array(primary_p_values))
        primary_fdr_map = dict(zip(primary_p_keys, [float(x) for x in fdr_adjusted]))
    else:
        primary_fdr_map = {}

    confirmatory_report["hypotheses"]["H3_causal_profile_reorganization_lmm"] = {
        "interpretation": "post-training-associated alteration of the depth profile of interventionally measured causal leverage",
        "formula": "c ~ C(family) + C(alignment) * C(task) * relative_depth",
        "primary_terms": PRIMARY_TERMS,
        "models": lmm_results,
        "primary_fdr_adjusted_p_values": primary_fdr_map,
    }
    # Backward compatible alias
    confirmatory_report["hypotheses"]["H3_causal_dissociation_lmm"] = confirmatory_report["hypotheses"]["H3_causal_profile_reorganization_lmm"]

    # =========================================================================
    # H4: Distribution Recovery Comparison (Self vs. Reader)
    # =========================================================================
    if is_dry_run:
        self_ratios = [0.72, 0.81, 0.69, 0.77]
        reader_ratios = [0.55, 0.62, 0.58, 0.60]
    else:
        recovery_files = list(raw_dir.glob("v2_recovery_*.json"))
        self_ratios = []
        reader_ratios = []
        for rf in recovery_files:
            try:
                with open(rf, "r", encoding="utf-8") as f:
                    rdata = json.load(f)
                if "self" in rdata and "reader" in rdata:
                    self_ratios.append(rdata["self"]["max_recovery_ratio"])
                    reader_ratios.append(rdata["reader"]["max_recovery_ratio"])
            except Exception as e:
                logger.warning(f"Failed to parse recovery file {rf}: {e}")
        if len(self_ratios) < 2:
            raise RuntimeError(
                f"Insufficient valid recovery files found in {raw_dir} (found {len(self_ratios)}). "
                "Real runs require valid recovery artifacts."
            )

    diffs = np.array(self_ratios) - np.array(reader_ratios)
    pt_diff, ci_low, ci_high = compute_bootstrap_ci(diffs.tolist(), n_boot=1000)

    confirmatory_report["hypotheses"]["H4_recovery_asymmetry"] = {
        "interpretation": "post-training-associated distribution recovery asymmetry",
        "self_max_recovery_mean": float(np.mean(self_ratios)),
        "reader_max_recovery_mean": float(np.mean(reader_ratios)),
        "diff_self_minus_reader_mean": float(pt_diff),
        "diff_bootstrap_ci_95": [float(ci_low), float(ci_high)],
        "reorganization_supported": bool(ci_low > 0.0 or ci_high < 0.0),
    }

    # Save final report
    out_file = derived_dir / "v2_lmm_confirmatory.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(confirmatory_report, f, indent=2)
    logger.info(f"Saved V2 Confirmatory Analysis report to {out_file}")

    return confirmatory_report


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    derived_dir = Path(args.derived_dir)
    run_confirmatory_analysis(raw_dir=raw_dir, derived_dir=derived_dir, is_dry_run=args.dry_run)


if __name__ == "__main__":
    main()
