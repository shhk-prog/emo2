#!/usr/bin/env python3
"""
v2/scripts/build_paper_summary.py

§6 V2 結果章用サマリー生成スクリプト（Read-only presentation layer）。
既存の検証済み結果（derived artifacts）を正本として読み込み、以下を出力する：
  - Table V2-1: Representation Geometry (Base vs Instruct matched-plain & secondary native/format)
  - Table V2-2: Reader-Self Sharing (Base vs Instruct plain, native, format effect)
  - Table V2-3A: Causal Relocation Summary (Positive peaks, centers, NaN preserved)
  - Table V2-3B: Causal Controls Comparison (c_raw, c_rand, c_perp, c_net_rand, c_zero)
  - Table V2-3C: LMM Results (Term naming: metric="lmm_beta::<term>")
  - Table V2-4: Distribution Recovery (Matched AUC, delta EMD, max recovery, best depth)
  - Table V2 Confirmatory: H1a, H1b, H2, H3, H4 実数値・CI
  - Figure Data: Figure V2-1〜V2-4 用 CSV
  - 共通19列スキーマレコードおよび Manifest への Provenance 登録
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from affective_empathy_eval.paper_summary.common import (
    PaperSummaryManifest,
    load_json_if_exists,
    safe_save_csv,
)
from affective_empathy_eval.paper_summary.schema import (
    PAPER_SUMMARY_COLUMNS,
    PaperSummaryRecord,
    validate_paper_summary_df,
)


def find_v2_artifact(v2_dir: Path, filename: str) -> Path | None:
    """derived/ または derived/dry_run/ からファイルを探索"""
    p1 = v2_dir / "results" / "derived" / filename
    if p1.exists():
        return p1
    p2 = v2_dir / "results" / "derived" / "dry_run" / filename
    if p2.exists():
        return p2
    return None


def build_v2_summary(
    v2_dir: Path,
    out_dir: Path,
    manifest: PaperSummaryManifest,
    strict: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    qc_info: dict[str, Any] = {"stage": "v2", "tables": {}}

    tables_dir = out_dir / "tables"
    fig_dir = out_dir / "figure_data"
    tables_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cross-family summary artifact
    cf_path = find_v2_artifact(v2_dir, "v2_cross_family_summary.json")
    if cf_path is None and strict:
        raise FileNotFoundError(
            f"Missing required V2 artifact: v2_cross_family_summary.json in {v2_dir}"
        )

    cf_data = load_json_if_exists(cf_path) if cf_path else {}

    # 2. Causal dissociation summary artifact (RQ3 derived)
    cd_path = find_v2_artifact(v2_dir, "v2_causal_dissociation_summary.json")
    cd_data = load_json_if_exists(cd_path) if cd_path else {}

    # 3. LMM confirmatory artifact
    lmm_path = find_v2_artifact(v2_dir, "v2_lmm_confirmatory.json")
    lmm_data = load_json_if_exists(lmm_path) if lmm_path else {}

    # 4. Distribution recovery summary artifact (RQ4 derived)
    rec_path = find_v2_artifact(v2_dir, "v2_distribution_recovery_summary.json")
    rec_data = load_json_if_exists(rec_path) if rec_path else {}

    # =========================================================================
    # 1. Table V2-1: Representation Geometry
    # =========================================================================
    table_v2_1_rows = []

    per_fam = cf_data.get("per_family_summary", {}) if cf_data else {}
    families = list(per_fam.keys()) if per_fam else ["olmo", "llama", "gemma", "qwen"]

    # If LMM has geometry data, use it for cross-family overview
    h1a = lmm_data.get("hypotheses", {}).get("H1a_geometry_reorganization", {})
    fam_geom = h1a.get("per_family_geometry", {})

    for fam in families:
        geom_entry = fam_geom.get(fam, {})
        for task in ("reader", "self"):
            for axis in ("valence", "arousal"):
                # Read from cf_data if present
                fam_axis_data = per_fam.get(fam, {}).get("by_axis", {}).get(axis, {})
                mp = fam_axis_data.get("primary_matched_plain", {})
                sn = fam_axis_data.get("secondary_native_chat", {})

                dist_key = f"com_distortion_{task}"
                mean_dist = float(mp.get(dist_key, geom_entry.get(f"{task}_distortion", np.nan)))
                mean_rsa = float(geom_entry.get(f"rsa_{task}", np.nan))

                base_peak = float(mp.get("peak_depth_base_cross", np.nan))
                inst_peak = float(mp.get("peak_depth_inst_cross", np.nan))
                delta_peak = (
                    inst_peak - base_peak
                    if (not np.isnan(base_peak) and not np.isnan(inst_peak))
                    else np.nan
                )

                native_dist = float(sn.get(dist_key, np.nan))

                table_v2_1_rows.append(
                    {
                        "family": fam,
                        "task": task,
                        "axis": axis,
                        "base_peak_depth": base_peak,
                        "instruct_peak_depth": inst_peak,
                        "delta_peak_depth": delta_peak,
                        "mean_rsa": mean_rsa,
                        "mean_procrustes_distortion": mean_dist,
                        "distortion_center": mean_dist,
                        "native_distortion": native_dist,
                    }
                )

                # Primary record: matched-plain procrustes distortion
                records.append(
                    PaperSummaryRecord(
                        stage="v2",
                        rq="rq1_geometry_transformation",
                        family=fam,
                        alignment="instruct_vs_base",
                        task=task,
                        axis=axis,
                        condition="matched_plain",
                        metric="procrustes_distortion",
                        estimate=mean_dist,
                        is_primary=True,
                        analysis_role="primary",
                        source_artifact=str(cf_path.relative_to(v2_dir.parent))
                        if cf_path
                        else "v2_cross_family_summary.json",
                        source_key=dist_key,
                    ).to_dict()
                )

                manifest.register_simple(
                    record_id=f"v2.table_v2_1.{fam}_{task}_{axis}_distortion",
                    artifact=str(cf_path.relative_to(v2_dir.parent))
                    if cf_path
                    else "v2_cross_family_summary.json",
                    key=dist_key,
                    derivation="direct_copy",
                    notes="Procrustes alignment distortion between Base and Instruct",
                )

    df_v2_1 = pd.DataFrame(table_v2_1_rows)
    safe_save_csv(df_v2_1, tables_dir / "table_v2_1_geometry.csv")
    qc_info["tables"]["table_v2_1"] = {"n_rows": len(df_v2_1)}

    # =========================================================================
    # 2. Table V2-2: Reader-Self Sharing
    # =========================================================================
    table_v2_2_rows = []

    for fam in families:
        fam_axis_data = per_fam.get(fam, {}).get("by_axis", {})
        for axis in ("valence", "arousal"):
            ax_data = fam_axis_data.get(axis, {})
            mp = ax_data.get("primary_matched_plain", {})
            sn = ax_data.get("secondary_native_chat", {})
            fmt = ax_data.get("format_effect", {})

            delta_sh_mp = float(mp.get("mean_delta_sharing", np.nan))
            delta_sh_sn = float(sn.get("mean_delta_sharing", np.nan))
            fmt_eff = float(fmt.get("mean_delta_sharing_format", np.nan))

            base_sh = float(ax_data.get("base_mean_sharing", np.nan))
            inst_sh = float(ax_data.get("inst_plain_mean_sharing", np.nan))

            table_v2_2_rows.append(
                {
                    "family": fam,
                    "axis": axis,
                    "base_mean_sharing": base_sh,
                    "inst_plain_mean_sharing": inst_sh,
                    "delta_sharing_matched": delta_sh_mp,
                    "inst_native_mean_sharing": float(sn.get("peak_depth_inst_cross", np.nan)),
                    "delta_sharing_native": delta_sh_sn,
                    "format_effect": fmt_eff,
                }
            )

            # Primary: delta sharing matched plain
            records.append(
                PaperSummaryRecord(
                    stage="v2",
                    rq="rq2_sharing_reorganization",
                    family=fam,
                    alignment="instruct_vs_base",
                    task="reader_self",
                    axis=axis,
                    condition="matched_plain",
                    metric="mean_delta_sharing",
                    estimate=delta_sh_mp,
                    is_primary=True,
                    analysis_role="primary",
                    source_artifact=str(cf_path.relative_to(v2_dir.parent))
                    if cf_path
                    else "v2_cross_family_summary.json",
                    source_key="mean_delta_sharing",
                ).to_dict()
            )

            # Secondary: format effect
            records.append(
                PaperSummaryRecord(
                    stage="v2",
                    rq="rq2_sharing_reorganization",
                    family=fam,
                    alignment="instruct_native_vs_plain",
                    task="reader_self",
                    axis=axis,
                    condition="format_effect",
                    metric="format_delta_sharing",
                    estimate=fmt_eff,
                    is_primary=False,
                    analysis_role="secondary",
                    source_artifact=str(cf_path.relative_to(v2_dir.parent))
                    if cf_path
                    else "v2_cross_family_summary.json",
                    source_key="mean_delta_sharing_format",
                ).to_dict()
            )

    df_v2_2 = pd.DataFrame(table_v2_2_rows)
    safe_save_csv(df_v2_2, tables_dir / "table_v2_2_sharing.csv")
    qc_info["tables"]["table_v2_2"] = {"n_rows": len(df_v2_2)}

    # =========================================================================
    # 3. Table V2-3A & Table V2-3B: Causal Relocation & Controls (RQ3 Derived)
    # =========================================================================
    table_v2_3a_rows = []
    table_v2_3b_rows = []
    fig_v2_3_rows = []

    per_fam_cd = cd_data.get("per_family", {}) if cd_data else {}
    cd_families = list(per_fam_cd.keys()) if per_fam_cd else families

    for fam in cd_families:
        fam_entry = per_fam_cd.get(fam, {})
        cmaps = fam_entry.get("causal_maps", {})
        rel_depths = fam_entry.get("relative_depths", [0.0, 0.33, 0.67, 1.0])

        for cond_key, cmap in cmaps.items():
            # parse alignment and task from cond_key (e.g. "base_plain_reader", "inst_plain_self")
            align_sub = "instruct" if ("inst" in cond_key or "chat" in cond_key) else "base"
            task_sub = "reader" if "reader" in cond_key else "self"

            for ax, ax_label in (("v", "valence"), ("a", "arousal")):
                raw_arr = cmap.get(f"c_{ax}_raw", [])
                rand_arr = cmap.get(f"c_{ax}_rand", [])
                perp_arr = cmap.get(f"c_{ax}_perp", [])
                net_rand_arr = cmap.get(f"c_{ax}_net_rand", [])
                net_perp_arr = cmap.get(f"c_{ax}_net_perp", [])
                zero_arr = cmap.get(f"c_{ax}_zero", [])

                m_raw = float(np.nanmean(raw_arr)) if len(raw_arr) > 0 else np.nan
                m_rand = float(np.nanmean(rand_arr)) if len(rand_arr) > 0 else np.nan
                m_perp = float(np.nanmean(perp_arr)) if len(perp_arr) > 0 else np.nan
                m_net_rand = float(np.nanmean(net_rand_arr)) if len(net_rand_arr) > 0 else np.nan
                m_net_perp = float(np.nanmean(net_perp_arr)) if len(net_perp_arr) > 0 else np.nan
                m_zero = float(np.nanmean(zero_arr)) if len(zero_arr) > 0 else np.nan

                # Peak and Center-of-Mass finding on net_rand
                pos_causal_peak = np.nan
                causal_center = np.nan
                no_pos_causal_peak = True
                if len(net_rand_arr) > 0 and len(rel_depths) == len(net_rand_arr):
                    net_arr = np.array(net_rand_arr, dtype=float)
                    if np.nanmax(net_arr) > 0:
                        max_idx = int(np.nanargmax(net_arr))
                        pos_causal_peak = float(rel_depths[max_idx])
                        no_pos_causal_peak = False

                    positive = np.maximum(net_arr, 0.0)
                    pos_sum = float(np.nansum(positive))
                    if pos_sum > 0:
                        causal_center = float(np.nansum(np.asarray(rel_depths) * positive) / pos_sum)

                # Decodability peak from geom or np.nan (no hardcoded fallback!)
                dec_peak = np.nan
                no_pos_dec_peak = True

                causal_decodability_gap = (
                    pos_causal_peak - dec_peak if (not np.isnan(pos_causal_peak) and not np.isnan(dec_peak)) else np.nan
                )

                table_v2_3a_rows.append(
                    {
                        "family": fam,
                        "alignment": align_sub,
                        "condition": cond_key,
                        "task": task_sub,
                        "axis": ax_label,
                        "positive_decodability_peak": dec_peak,
                        # preserves NaN if no positive peak!
                        "positive_causal_peak": pos_causal_peak,
                        "decodability_center": dec_peak,
                        "causal_center": causal_center,
                        "delta_d_peak": np.nan,  # Relocation delta must be computed as Instruct - Base
                        "delta_d_center": np.nan,
                        "causal_decodability_gap": causal_decodability_gap,
                        "no_positive_decodability_peak": no_pos_dec_peak,
                        "no_positive_net_causal_peak": no_pos_causal_peak,
                    }
                )

                table_v2_3b_rows.append(
                    {
                        "family": fam,
                        "alignment": align_sub,
                        "condition": cond_key,
                        "task": task_sub,
                        "axis": ax_label,
                        "mean_c_raw": m_raw,
                        "mean_c_rand": m_rand,
                        "mean_c_perp": m_perp,
                        "mean_c_net_rand": m_net_rand,
                        "mean_c_net_perp": m_net_perp,
                        "mean_c_zero": m_zero,
                    }
                )

                # Schema record: C_net_rand (Direct copy from derived canonical artifact!)
                records.append(
                    PaperSummaryRecord(
                        stage="v2",
                        rq="rq3_causal_relocation",
                        family=fam,
                        alignment=align_sub,
                        task=task_sub,
                        axis=ax_label,
                        condition=cond_key,
                        metric="mean_c_net_rand",
                        estimate=m_net_rand,
                        is_primary=True,
                        analysis_role="primary",
                        source_artifact=str(cd_path.relative_to(v2_dir.parent))
                        if cd_path
                        else "v2_causal_dissociation_summary.json",
                        source_key=f"c_{ax}_net_rand",
                    ).to_dict()
                )

                manifest.register_simple(
                    record_id=f"v2.table_v2_3b.{fam}_{cond_key}_{ax_label}_c_net_rand",
                    artifact=str(cd_path.relative_to(v2_dir.parent))
                    if cd_path
                    else "v2_causal_dissociation_summary.json",
                    key=f"c_{ax}_net_rand",
                    derivation="direct_copy",
                    notes="Direct copy of canonical control-adjusted net causal leverage",
                )

                # Figure V2-3 rows (profile)
                for d_val, c_val in zip(rel_depths, net_rand_arr):
                    fig_v2_3_rows.append(
                        {
                            "family": fam,
                            "alignment": align_sub,
                            "condition": cond_key,
                            "task": task_sub,
                            "axis": ax_label,
                            "relative_depth": float(d_val),
                            "c_net_rand": float(c_val),
                        }
                    )

    df_v2_3a = pd.DataFrame(table_v2_3a_rows)
    safe_save_csv(df_v2_3a, tables_dir / "table_v2_3a_causal_relocation.csv")
    df_v2_3b = pd.DataFrame(table_v2_3b_rows)
    safe_save_csv(df_v2_3b, tables_dir / "table_v2_3b_causal_controls.csv")
    safe_save_csv(pd.DataFrame(fig_v2_3_rows), fig_dir / "figure_v2_3_causal_relocation.csv")
    qc_info["tables"]["table_v2_3a"] = {"n_rows": len(df_v2_3a)}
    qc_info["tables"]["table_v2_3b"] = {"n_rows": len(df_v2_3b)}

    # =========================================================================
    # 4. Table V2-3C: LMM Results
    # =========================================================================
    table_v2_3c_rows = []
    h3_lmm = lmm_data.get("hypotheses", {}).get("H3_causal_dissociation_lmm", {})
    lmm_models = h3_lmm.get("models", {})

    for axis_name, model_dict in lmm_models.items():
        if not model_dict.get("converged", False):
            continue
        params = model_dict.get("params", {})
        pvalues = model_dict.get("pvalues", {})
        conf_int = model_dict.get("conf_int", {})
        ci_0 = conf_int.get("0", {})
        ci_1 = conf_int.get("1", {})

        for raw_term, beta_val in params.items():
            # Clean term name for unique metric convention: metric = "lmm_beta::<term>"
            clean_term = (
                raw_term.replace("C(alignment)[T.inst]", "alignment")
                .replace("C(task)[T.self]", "task")
                .replace(":", "_x_")
                .replace(" ", "")
            )
            metric_name = f"lmm_beta::{clean_term}"

            p_val = float(pvalues.get(raw_term, np.nan))
            ci_l = float(ci_0.get(raw_term, np.nan))
            ci_h = float(ci_1.get(raw_term, np.nan))

            table_v2_3c_rows.append(
                {
                    "axis": axis_name,
                    "term": raw_term,
                    "clean_term": clean_term,
                    "beta": float(beta_val),
                    "ci_low": ci_l,
                    "ci_high": ci_h,
                    "p": p_val,
                }
            )

            records.append(
                PaperSummaryRecord(
                    stage="v2",
                    rq="rq3_causal_relocation_lmm",
                    family="cross_family",
                    alignment="instruct_vs_base",
                    task="reader_and_self",
                    axis=axis_name,
                    condition="matched_plain_lmm",
                    metric=metric_name,  # Unique key convention!
                    estimate=float(beta_val),
                    ci_low=ci_l,
                    ci_high=ci_h,
                    p=p_val,
                    is_primary=True if "alignment_x_depth" in clean_term else False,
                    analysis_role="primary" if "alignment_x_depth" in clean_term else "control",
                    source_artifact=str(lmm_path.relative_to(v2_dir.parent))
                    if lmm_path
                    else "v2_lmm_confirmatory.json",
                    source_key=f"params.{raw_term}",
                ).to_dict()
            )

            manifest.register_simple(
                record_id=f"v2.table_v2_3c.{axis_name}_{clean_term}",
                artifact=str(lmm_path.relative_to(v2_dir.parent))
                if lmm_path
                else "v2_lmm_confirmatory.json",
                key=f"params.{raw_term}",
                derivation="direct_copy",
                notes=f"LMM fixed effect term {raw_term}",
            )

    df_v2_3c = pd.DataFrame(table_v2_3c_rows)
    safe_save_csv(df_v2_3c, tables_dir / "table_v2_3c_lmm.csv")
    qc_info["tables"]["table_v2_3c"] = {"n_rows": len(df_v2_3c)}

    # =========================================================================
    # 5. Table V2-4: Distribution Recovery (RQ4)
    # =========================================================================
    table_v2_4_rows = []
    h4 = lmm_data.get("hypotheses", {}).get("H4_distribution_recovery", {})
    rec_families = h4.get("per_family_recovery", {})

    for fam in families:
        fam_rec = rec_families.get(fam, {})
        fam_dist = rec_data.get("per_family", {}).get(fam, {})
        for task in ("reader", "self"):
            td = fam_dist.get(task, {})
            if td:
                matched_auc = float(
                    td.get("auc_recovery_matched_plain", td.get("primary_matched_plain", {}).get("auc_recovery", np.nan))
                )
                delta_emd = float(
                    td.get("auc_delta_emd_matched_plain", td.get("primary_matched_plain", {}).get("delta_emd_auc", np.nan))
                )
                max_rec = float(
                    td.get("max_recovery_ratio_matched_plain", td.get("secondary_peak_localization", {}).get("max_recovery_ratio", np.nan))
                )
                best_d = float(
                    td.get("best_recovery_depth", td.get("secondary_peak_localization", {}).get("best_recovery_depth", np.nan))
                )
                native_auc = float(td.get("auc_recovery", np.nan))
                aligned_auc = float(td.get("auc_recovery_aligned", np.nan))
            else:
                matched_auc = float(fam_rec.get(f"{task}_matched_auc", np.nan))
                delta_emd = float(fam_rec.get(f"{task}_delta_emd_auc", np.nan))
                max_rec = float(fam_rec.get(f"{task}_max_recovery", np.nan))
                best_d = float(fam_rec.get(f"{task}_best_depth", np.nan))
                native_auc = float(fam_rec.get(f"{task}_native_auc", np.nan))
                aligned_auc = float(fam_rec.get(f"{task}_aligned_auc", np.nan))

            table_v2_4_rows.append(
                {
                    "family": fam,
                    "task": task,
                    "matched_auc": matched_auc,
                    "matched_delta_emd_auc": delta_emd,
                    "matched_max_recovery": max_rec,
                    "matched_best_depth": best_d,
                    "native_auc": native_auc,
                    "aligned_auc": aligned_auc,
                }
            )

            records.append(
                PaperSummaryRecord(
                    stage="v2",
                    rq="rq4_distribution_recovery",
                    family=fam,
                    alignment="instruct_patching",
                    task=task,
                    axis="valence_arousal_joint",
                    condition="matched_plain_recovery",
                    metric="matched_auc_recovery",
                    estimate=matched_auc,
                    is_primary=True,
                    analysis_role="primary",
                    source_artifact=str(rec_path.relative_to(v2_dir.parent))
                    if rec_path
                    else (str(lmm_path.relative_to(v2_dir.parent)) if lmm_path else "v2_distribution_recovery_summary.json"),
                    source_key="auc_recovery_matched_plain",
                ).to_dict()
            )

    df_v2_4 = pd.DataFrame(table_v2_4_rows)
    safe_save_csv(df_v2_4, tables_dir / "table_v2_4_recovery.csv")
    qc_info["tables"]["table_v2_4"] = {"n_rows": len(df_v2_4)}

    # =========================================================================
    # 6. V2 Confirmatory Summary Table
    # =========================================================================
    conf_rows = []
    for hyp_name, hyp_dict in lmm_data.get("hypotheses", {}).items():
        interp = hyp_dict.get("interpretation", "")
        effects = hyp_dict.get("effects", hyp_dict.get("primary_effects", {}))
        for eff_name, eff_dict in effects.items():
            if isinstance(eff_dict, dict):
                mean_val = float(eff_dict.get("mean", eff_dict.get("mean_shift", np.nan)))
                ci = eff_dict.get("bootstrap_ci_95", [np.nan, np.nan])
                conf_rows.append(
                    {
                        "hypothesis": hyp_name,
                        "metric": eff_name,
                        "estimate": mean_val,
                        "ci_low": float(ci[0]),
                        "ci_high": float(ci[1]),
                        "interpretation": interp,
                    }
                )

    df_conf = pd.DataFrame(conf_rows)
    safe_save_csv(df_conf, tables_dir / "table_v2_confirmatory.csv")

    # 全レコードDataFrame化
    df_records = pd.DataFrame(records)
    for col in PAPER_SUMMARY_COLUMNS:
        if col not in df_records.columns:
            df_records[col] = np.nan
    df_records = df_records[PAPER_SUMMARY_COLUMNS].copy()

    # stage summary 出力
    stage_out = out_dir / "stage_summaries" / "v2"
    stage_out.mkdir(parents=True, exist_ok=True)
    safe_save_csv(df_records, stage_out / "v2_paper_results.csv")

    errors = validate_paper_summary_df(df_records, strict=strict)
    if errors:
        qc_info["validation_errors"] = errors
        if strict:
            raise ValueError(f"V2 schema validation errors: {errors}")

    return df_records, qc_info


def main():
    parser = argparse.ArgumentParser(description="Build V2 Paper Summary")
    parser.add_argument("--v2-dir", type=str, default="v2")
    parser.add_argument("--out-dir", type=str, default="results/derived/paper_summary")
    parser.add_argument(
        "--strict", action="store_true", help="Fail on missing artifacts or validation errors"
    )
    args = parser.parse_args()

    v2_dir = Path(args.v2_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    manifest = PaperSummaryManifest()
    df_records, qc_info = build_v2_summary(
        v2_dir=v2_dir,
        out_dir=out_dir,
        manifest=manifest,
        strict=args.strict,
    )
    manifest.save(out_dir / "stage_summaries" / "v2" / "manifest_v2.json")

    print(f"V2 paper summary completed: {len(df_records)} records generated.")


if __name__ == "__main__":
    main()
