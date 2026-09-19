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
        "n_families": len(families),
        "inference_scope": "descriptive_cross_family_bootstrap",
        "sample_inference_scope": "lmm_with_family_fixed_effects",
        "hypotheses": {},
    }

    # =========================================================================
    # H1a: Geometric Reorganization (Procrustes Distortion & RSA)
    # =========================================================================
    h1a_metrics: Dict[str, List[float]] = {
        "reader_distortion": [],
        "self_distortion": [],
        "rsa_reader": [],
        "rsa_self": [],
    }
    per_family_h1a: Dict[str, Dict[str, float]] = {}

    if is_dry_run:
        mock_h1a = {
            "qwen": {"reader_distortion": 0.35, "self_distortion": 0.42, "rsa_reader": 0.65, "rsa_self": 0.58},
            "llama": {"reader_distortion": 0.38, "self_distortion": 0.45, "rsa_reader": 0.62, "rsa_self": 0.55},
            "gemma": {"reader_distortion": 0.32, "self_distortion": 0.39, "rsa_reader": 0.68, "rsa_self": 0.61},
            "olmo": {"reader_distortion": 0.36, "self_distortion": 0.44, "rsa_reader": 0.64, "rsa_self": 0.56},
        }
        for fam_id, m_dict in mock_h1a.items():
            per_family_h1a[fam_id] = m_dict
            for mk in h1a_metrics:
                h1a_metrics[mk].append(m_dict[mk])
    else:
        geom_files = list(raw_dir.glob("v2_geometry_*.json"))
        if len(geom_files) < 2:
            raise FileNotFoundError(f"Required geometry files not found in {raw_dir} (found {len(geom_files)})")
        for gf in geom_files:
            try:
                fam_id = gf.stem.replace("v2_geometry_", "")
                with open(gf, "r", encoding="utf-8") as f:
                    gdata = json.load(f)
                rq1_geom = gdata.get("rq1_geometry", {})
                fam_h1a: Dict[str, float] = {}
                for mk, key_name in [
                    ("reader_distortion", "reader_distortion_matched"),
                    ("self_distortion", "self_distortion_matched"),
                    ("rsa_reader", "rsa_reader_matched"),
                    ("rsa_self", "rsa_self_matched"),
                ]:
                    vals = rq1_geom.get(key_name)
                    if vals is None:
                        raise RuntimeError(
                            f"Required matched geometric metric '{key_name}' missing in {gf}. "
                            "Confirmatory H1a requires matched-plain geometry."
                        )
                    if vals:
                        mean_v = float(np.mean(vals))
                        h1a_metrics[mk].append(mean_v)
                        fam_h1a[mk] = mean_v
                per_family_h1a[fam_id] = fam_h1a
            except Exception as e:
                logger.warning(f"Failed to parse {gf} for H1a: {e}")
                if not is_dry_run:
                    raise

    h1a_report: Dict[str, Any] = {
        "interpretation": "post-training-associated geometric distortion and representational similarity under matched-plain conditions",
        "rsa_metric_type": "rsa_similarity",
        "per_family_geometry": per_family_h1a,
        "effects": {},
    }
    for mk, vals in h1a_metrics.items():
        if len(vals) >= 2:
            pt, ci_low, ci_high = compute_bootstrap_ci(vals, n_boot=1000)
        else:
            pt, ci_low, ci_high = np.nan, np.nan, np.nan
        h1a_report["effects"][mk] = {
            "mean": float(pt) if not np.isnan(pt) else None,
            "bootstrap_ci_95": [float(ci_low), float(ci_high)] if not np.isnan(ci_low) else None,
        }
    confirmatory_report["hypotheses"]["H1a_geometry_reorganization"] = h1a_report

    # =========================================================================
    # H1b: Post-training-Associated Reorganization of Affect Decodability Peak
    # =========================================================================
    axes = ["valence", "arousal"]
    sub_keys = ["reader", "self"]

    h1_shifts: Dict[str, Dict[str, List[float]]] = {
        ax: {k: [] for k in sub_keys} for ax in axes
    }
    h1_shifts_secondary_native: Dict[str, Dict[str, List[float]]] = {
        ax: {k: [] for k in sub_keys} for ax in axes
    }
    per_family_h1: Dict[str, Dict[str, float]] = {}

    if is_dry_run:
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
                    h1_shifts_secondary_native[ax][k].append(s_dict[f"{ax}.{k}.shift"] * 1.1)
    else:
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

                    # Primary: matched-plain only
                    inst_reader = axis_data.get("inst_matched_r2_reader")
                    inst_self = axis_data.get("inst_matched_r2_self")
                    if inst_reader is None or len(inst_reader) == 0 or inst_self is None or len(inst_self) == 0:
                        raise RuntimeError(
                            f"Primary contrast 'inst_matched_r2_*' missing in {gf} for {fam_id}. "
                            "Confirmatory H1 requires matched-plain control."
                        )

                    # Secondary: native-chat
                    inst_reader_native = axis_data.get("inst_native_r2_reader", [])
                    inst_self_native = axis_data.get("inst_native_r2_self", [])

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

                    # Secondary native shift
                    if inst_reader_native and inst_self_native:
                        inst_pk_r_n = compute_peak_depth(inst_reader_native, depths)
                        inst_pk_s_n = compute_peak_depth(inst_self_native, depths)
                        shift_r_n = float(inst_pk_r_n - base_pk_r) if not (np.isnan(base_pk_r) or np.isnan(inst_pk_r_n)) else np.nan
                        shift_s_n = float(inst_pk_s_n - base_pk_s) if not (np.isnan(base_pk_s) or np.isnan(inst_pk_s_n)) else np.nan
                        if not np.isnan(shift_r_n):
                            h1_shifts_secondary_native[ax]["reader"].append(shift_r_n)
                        if not np.isnan(shift_s_n):
                            h1_shifts_secondary_native[ax]["self"].append(shift_s_n)
                        fam_dict[f"{ax}.reader.native_shift"] = shift_r_n
                        fam_dict[f"{ax}.self.native_shift"] = shift_s_n

                per_family_h1[fam_id] = fam_dict
            except Exception as e:
                logger.warning(f"Failed to parse {gf} for H1: {e}")
                if not is_dry_run:
                    raise

        valid_v_r = len(h1_shifts["valence"]["reader"])
        if valid_v_r < 2:
            raise RuntimeError(f"Insufficient valid geometry data for H1 in {raw_dir}: {valid_v_r}")

    h1_report: Dict[str, Any] = {
        "interpretation": "post-training-associated reorganization of decodability peak (Primary: matched-plain)",
        "per_family_shifts": per_family_h1,
        "primary_effects": {},
        "secondary_native_effects": {},
    }
    for ax in axes:
        for k in sub_keys:
            shifts = h1_shifts[ax][k]
            if len(shifts) >= 2:
                pt, ci_low, ci_high = compute_bootstrap_ci(shifts, n_boot=1000)
                reorg_supported = bool(ci_low > 0.0 or ci_high < 0.0)
            else:
                pt, ci_low, ci_high, reorg_supported = np.nan, np.nan, np.nan, False
            h1_report["primary_effects"][f"{ax}.{k}.shift"] = {
                "mean_shift": float(pt) if not np.isnan(pt) else None,
                "bootstrap_ci_95": [float(ci_low), float(ci_high)] if not np.isnan(ci_low) else None,
                "reorganization_supported": reorg_supported,
            }

            # Secondary
            shifts_n = h1_shifts_secondary_native[ax][k]
            if len(shifts_n) >= 2:
                pt_n, ci_l_n, ci_h_n = compute_bootstrap_ci(shifts_n, n_boot=1000)
            else:
                pt_n, ci_l_n, ci_h_n = np.nan, np.nan, np.nan
            h1_report["secondary_native_effects"][f"{ax}.{k}.native_shift"] = {
                "mean_shift": float(pt_n) if not np.isnan(pt_n) else None,
                "bootstrap_ci_95": [float(ci_l_n), float(ci_h_n)] if not np.isnan(ci_l_n) else None,
            }

    confirmatory_report["hypotheses"]["H1b_decodability_peak_reorganization"] = h1_report
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
                    # Primary: Base plain vs Instruct matched-plain (delta_sharing_matched only)
                    delta_share_matched = axis_data.get("delta_sharing_matched")
                    if delta_share_matched is None or len(delta_share_matched) == 0:
                        raise RuntimeError(
                            f"Missing primary 'delta_sharing_matched' for {fam_id} ({ax}) in {gf}. "
                            "Confirmatory H2 requires matched-plain sharing contrast."
                        )
                    m_val = float(np.mean(delta_share_matched))
                    h2_diffs_primary[ax].append(m_val)
                    fam_h2_dict[f"{ax}.primary_delta_sharing_matched"] = m_val

                    # Secondary: Base plain vs Instruct native-chat
                    delta_share_native = axis_data.get("delta_sharing_native") or axis_data.get("delta_sharing")
                    if delta_share_native is not None and len(delta_share_native) > 0:
                        n_val = float(np.mean(delta_share_native))
                        h2_diffs_secondary[ax].append(n_val)
                        fam_h2_dict[f"{ax}.secondary_delta_sharing_native"] = n_val

                per_family_h2[fam_id] = fam_h2_dict
            except Exception as e:
                logger.warning(f"Failed to parse {gf} for H2: {e}")
                if not is_dry_run:
                    raise

        if len(h2_diffs_primary["valence"]) < 2:
            raise RuntimeError(f"Insufficient valid geometry files for H2 in {raw_dir}")

    h2_report: Dict[str, Any] = {
        "interpretation": "post-training-associated reorganization of Reader-Self representation sharing",
        "per_family_sharing": per_family_h2,
        "primary_contrast": "Base plain vs Instruct matched-plain",
        "secondary_contrast": "Base plain vs Instruct native-chat",
        "primary_effects": {},
        "secondary_effects": {},
    }
    for ax in axes:
        # Primary
        p_diffs = h2_diffs_primary[ax]
        pt, ci_low, ci_high = compute_bootstrap_ci(p_diffs, n_boot=1000)
        reorg_supported = bool(ci_high < 0.0 or ci_low > 0.0)
        h2_report["primary_effects"][ax] = {
            "contrast_type": "primary_matched_plain",
            "mean_delta_sharing": float(pt),
            "bootstrap_ci_95": [float(ci_low), float(ci_high)],
            "sharing_alteration_supported": reorg_supported,
        }

        # Secondary
        s_diffs = h2_diffs_secondary[ax]
        if len(s_diffs) >= 2:
            s_pt, s_low, s_high = compute_bootstrap_ci(s_diffs, n_boot=1000)
            h2_report["secondary_effects"][ax] = {
                "contrast_type": "secondary_native_chat",
                "mean_delta_sharing": float(s_pt),
                "bootstrap_ci_95": [float(s_low), float(s_high)],
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
                fmt = "plain" if align == "base" else "matched_plain"
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
                                 "format_condition": fmt,
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

    # Primary analysis: Base plain vs Instruct matched-plain
    if "format_condition" in df_pair.columns:
        df_pair_primary = df_pair[
            ((df_pair["alignment"] == "base") & (df_pair["format_condition"] == "plain"))
            | ((df_pair["alignment"] == "inst") & (df_pair["format_condition"] == "matched_plain"))
        ].copy()
        df_pair_secondary = df_pair[
            ((df_pair["alignment"] == "base") & (df_pair["format_condition"] == "plain"))
            | ((df_pair["alignment"] == "inst") & (df_pair["format_condition"] == "native_chat"))
        ].copy()
    else:
        df_pair_primary = df_pair.copy()
        df_pair_secondary = pd.DataFrame()

    logger.info(f"Fitting Confirmatory Primary LMM on {len(df_pair_primary)} observations (matched-plain) with C(family) fixed effects...")
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
                df=df_pair_primary,
                formula=formula,
                groups="pair_id",
            )
            lmm_results[axis_name] = {
                "converged": lmm_fit["converged"],
                "formula": formula,
                "contrast": "primary_matched_plain",
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
            logger.warning(f"Primary LMM {axis_name.capitalize()} fit failed: {e}")
            lmm_results[axis_name] = {"converged": False, "error": str(e), "formula": formula}

    # Primary FDR correction only on prespecified interaction terms
    if primary_p_values:
        fdr_adjusted = apply_benjamini_hochberg(np.array(primary_p_values))
        primary_fdr_map = dict(zip(primary_p_keys, [float(x) for x in fdr_adjusted]))
    else:
        primary_fdr_map = {}

    # Secondary LMM on native-chat
    secondary_lmm_results: Dict[str, Any] = {}
    if not df_pair_secondary.empty and len(df_pair_secondary["alignment"].unique()) >= 2:
        for axis_name, col_name in [("valence", "c_v"), ("arousal", "c_a")]:
            formula = f"{col_name} ~ C(family) + C(alignment) * C(task) * relative_depth"
            try:
                lmm_fit_sec = fit_sample_level_lmm(
                    df=df_pair_secondary,
                    formula=formula,
                    groups="pair_id",
                )
                secondary_lmm_results[axis_name] = {
                    "converged": lmm_fit_sec["converged"],
                    "formula": formula,
                    "contrast": "secondary_native_chat",
                    "params": lmm_fit_sec["params"],
                    "pvalues": lmm_fit_sec["pvalues"],
                    "conf_int": lmm_fit_sec["conf_int"],
                }
            except Exception as e:
                logger.warning(f"Secondary LMM {axis_name.capitalize()} fit failed: {e}")

    confirmatory_report["hypotheses"]["H3_causal_profile_reorganization_lmm"] = {
        "interpretation": "post-training-associated alteration of the depth profile of interventionally measured causal leverage (Primary: matched-plain)",
        "formula": "c ~ C(family) + C(alignment) * C(task) * relative_depth",
        "primary_terms": PRIMARY_TERMS,
        "primary_contrast": "Base plain vs Instruct matched-plain",
        "secondary_contrast": "Base plain vs Instruct native-chat",
        "models": lmm_results,
        "secondary_models": secondary_lmm_results,
        "primary_fdr_adjusted_p_values": primary_fdr_map,
    }
    confirmatory_report["hypotheses"]["H3_causal_dissociation_lmm"] = confirmatory_report["hypotheses"]["H3_causal_profile_reorganization_lmm"]

    # =========================================================================
    # H4: Distribution Recovery Comparison (Self vs. Reader)
    # =========================================================================
    trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))

    if is_dry_run:
        self_ratios = [0.72, 0.81, 0.69, 0.77]
        reader_ratios = [0.55, 0.62, 0.58, 0.60]
        self_aucs = [0.45, 0.52, 0.43, 0.48]
        reader_aucs = [0.32, 0.38, 0.35, 0.36]
        self_aucs_matched = [0.47, 0.54, 0.45, 0.50]
        reader_aucs_matched = [0.34, 0.40, 0.37, 0.38]
        self_aucs_aligned = [0.42, 0.49, 0.40, 0.45]
        reader_aucs_aligned = [0.30, 0.35, 0.33, 0.34]
    else:
        recovery_files = list(raw_dir.glob("v2_recovery_*.json"))
        self_ratios = []
        reader_ratios = []
        self_aucs = []
        reader_aucs = []
        self_aucs_matched = []
        reader_aucs_matched = []
        self_aucs_aligned = []
        reader_aucs_aligned = []
        for rf in recovery_files:
            try:
                with open(rf, "r", encoding="utf-8") as f:
                    rdata = json.load(f)
                if "self" in rdata and "reader" in rdata:
                    s_dat = rdata["self"]
                    r_dat = rdata["reader"]
                    # Strict matched-plain metrics for Primary analysis (no native fallback allowed)
                    s_max_m = s_dat.get("max_recovery_ratio_matched_plain")
                    if s_max_m is None and "recovery_ratios_matched_plain" in s_dat:
                        s_max_m = float(np.max(s_dat["recovery_ratios_matched_plain"]))
                    r_max_m = r_dat.get("max_recovery_ratio_matched_plain")
                    if r_max_m is None and "recovery_ratios_matched_plain" in r_dat:
                        r_max_m = float(np.max(r_dat["recovery_ratios_matched_plain"]))

                    if s_max_m is None or r_max_m is None:
                        raise RuntimeError(
                            f"V2 H4 Primary requires max_recovery_ratio_matched_plain in {rf}. "
                            "Fallback to native is strictly prohibited in confirmatory analysis."
                        )
                    self_ratios.append(s_max_m)
                    reader_ratios.append(r_max_m)

                    depths = rdata.get("relative_depths") or np.linspace(0.0, 1.0, len(s_dat.get("recovery_ratios_matched_plain", s_dat.get("recovery_ratios", []))))

                    # Primary metric: AUC recovery matched-plain
                    s_auc_m = s_dat.get("auc_recovery_matched_plain")
                    if s_auc_m is None and "recovery_ratios_matched_plain" in s_dat:
                        s_auc_m = float(trapz_func(s_dat["recovery_ratios_matched_plain"], depths))
                    r_auc_m = r_dat.get("auc_recovery_matched_plain")
                    if r_auc_m is None and "recovery_ratios_matched_plain" in r_dat:
                        r_auc_m = float(trapz_func(r_dat["recovery_ratios_matched_plain"], depths))

                    if s_auc_m is None or r_auc_m is None:
                        raise RuntimeError(
                            f"V2 H4 Primary requires auc_recovery_matched_plain in {rf}. "
                            "Fallback to native is strictly prohibited in confirmatory analysis."
                        )

                    self_aucs_matched.append(s_auc_m)
                    reader_aucs_matched.append(r_auc_m)

                    # Secondary: native chat AUC
                    s_auc = s_dat.get("auc_recovery") or float(trapz_func(s_dat["recovery_ratios"], depths))
                    r_auc = r_dat.get("auc_recovery") or float(trapz_func(r_dat["recovery_ratios"], depths))
                    self_aucs.append(s_auc)
                    reader_aucs.append(r_auc)

                    # Mechanistic control: aligned AUC
                    s_auc_al = s_dat.get("auc_recovery_aligned")
                    r_auc_al = r_dat.get("auc_recovery_aligned")
                    if s_auc_al is not None and r_auc_al is not None:
                        self_aucs_aligned.append(s_auc_al)
                        reader_aucs_aligned.append(r_auc_al)
            except Exception as e:
                logger.warning(f"Failed to parse recovery file {rf}: {e}")
        if len(self_aucs_matched) < 2:
            raise RuntimeError(
                f"Insufficient valid recovery files found in {raw_dir} (found {len(self_aucs_matched)}). "
                "Real runs require valid recovery artifacts."
            )

    # Primary: AUC recovery difference on matched-plain
    diffs_auc_matched = np.array(self_aucs_matched) - np.array(reader_aucs_matched)
    pt_auc_m, am_low, am_high = compute_bootstrap_ci(diffs_auc_matched.tolist(), n_boot=1000)

    # Secondary: Native chat AUC recovery difference
    diffs_auc = np.array(self_aucs) - np.array(reader_aucs)
    pt_auc, a_low, a_high = compute_bootstrap_ci(diffs_auc.tolist(), n_boot=1000)

    # Mechanistic control: Procrustes aligned AUC recovery difference
    if len(self_aucs_aligned) >= 2:
        diffs_auc_al = np.array(self_aucs_aligned) - np.array(reader_aucs_aligned)
        pt_auc_al, al_low, al_high = compute_bootstrap_ci(diffs_auc_al.tolist(), n_boot=1000)
    else:
        pt_auc_al, al_low, al_high = np.nan, np.nan, np.nan

    # Peak recovery ratio difference
    diffs_max = np.array(self_ratios) - np.array(reader_ratios)
    pt_max, m_low, m_high = compute_bootstrap_ci(diffs_max.tolist(), n_boot=1000)

    confirmatory_report["hypotheses"]["H4_recovery_asymmetry"] = {
        "interpretation": "post-training-associated distribution recovery asymmetry (Primary: matched-plain AUC recovery; Secondary: native-chat AUC recovery; Mechanistic control: Procrustes-aligned AUC recovery)",
        "primary_auc_recovery": {
            "contrast_type": "primary_matched_plain",
            "self_mean_auc": float(np.mean(self_aucs_matched)),
            "reader_mean_auc": float(np.mean(reader_aucs_matched)),
            "diff_self_minus_reader_mean": float(pt_auc_m),
            "diff_bootstrap_ci_95": [float(am_low), float(am_high)],
            "reorganization_supported": bool(am_low > 0.0 or am_high < 0.0),
        },
        "secondary_native_chat_auc_recovery": {
            "contrast_type": "secondary_native_chat",
            "self_mean_auc": float(np.mean(self_aucs)),
            "reader_mean_auc": float(np.mean(reader_aucs)),
            "diff_self_minus_reader_mean": float(pt_auc),
            "diff_bootstrap_ci_95": [float(a_low), float(a_high)],
            "reorganization_supported": bool(a_low > 0.0 or a_high < 0.0),
        },
        "mechanistic_control_aligned_auc_recovery": {
            "contrast_type": "mechanistic_control_procrustes_aligned",
            "diff_self_minus_reader_mean": float(pt_auc_al) if not np.isnan(pt_auc_al) else None,
            "diff_bootstrap_ci_95": [float(al_low), float(al_high)] if not np.isnan(al_low) else None,
        },
        "secondary_max_recovery_ratio": {
            "self_max_recovery_mean": float(np.mean(self_ratios)),
            "reader_max_recovery_mean": float(np.mean(reader_ratios)),
            "diff_self_minus_reader_mean": float(pt_max),
            "diff_bootstrap_ci_95": [float(m_low), float(m_high)],
            "reorganization_supported": bool(m_low > 0.0 or m_high < 0.0),
        },
        # Backwards compatible top-level fields pointing to Primary
        "diff_self_minus_reader_mean": float(pt_auc_m),
        "diff_bootstrap_ci_95": [float(am_low), float(am_high)],
        "reorganization_supported": bool(am_low > 0.0 or am_high < 0.0),
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
