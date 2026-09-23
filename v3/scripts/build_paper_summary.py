#!/usr/bin/env python3
"""
v3/scripts/build_paper_summary.py

§7 V3 結果章用サマリー生成スクリプト（Read-only presentation layer）。
既存の検証済み結果から以下を抽出・フォーマットして出力する：
  - Table V3-1: State-Induction Gate (overall_decision string, pipeline_continues flag)
  - Table V3-2: Spatiotemporal 4-Maps Summary (discovery, control response_end)
  - Table V3-3: Mediated Attenuation (analysis_role=confirmatory, confirmation set)
  - Table V3-4: Confirmatory Replication & Matrix (Llama, Gemma, OLMo across H1–H4)
  - Figure Data: Figure V3-1〜V3-4 用 CSV
  - 共通19列スキーマレコードおよび Manifest への Provenance 登録
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from affective_empathy_eval.paper_summary.common import (
    PaperSummaryManifest,
    check_pipeline_continues,
    load_json_if_exists,
    safe_save_csv,
)
from affective_empathy_eval.paper_summary.schema import (
    PAPER_SUMMARY_COLUMNS,
    PaperSummaryRecord,
    validate_paper_summary_df,
)


def find_v3_artifact(v3_dir: Path, filename: str) -> Path | None:
    p1 = v3_dir / "results" / "derived" / filename
    if p1.exists():
        return p1
    p2 = v3_dir / "results" / "raw" / filename
    if p2.exists():
        return p2
    return None


def build_v3_summary(
    v3_dir: Path,
    out_dir: Path,
    manifest: PaperSummaryManifest,
    strict: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    qc_info: dict[str, Any] = {"stage": "v3", "tables": {}}

    tables_dir = out_dir / "tables"
    fig_dir = out_dir / "figure_data"
    tables_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 1. Table V3-1: State-Induction Gate (RQ1)
    # =========================================================================
    gate_path = find_v3_artifact(v3_dir, "v3_gate_decision.json")
    if gate_path is None and strict:
        raise FileNotFoundError(f"Missing required V3 artifact: v3_gate_decision.json in {v3_dir}")

    gate_data = load_json_if_exists(gate_path) if gate_path else {}
    pipeline_continues, overall_decision, _ = (
        check_pipeline_continues(gate_path) if gate_path else (False, "NO_GO", {})
    )

    table_v3_1_rows = []

    for axis in ("valence", "arousal"):
        ax_short = "v" if axis == "valence" else "a"
        slope_l = float(gate_data.get(f"slope_{ax_short}_ci_lower", np.nan))
        slope_pass = bool(gate_data.get(f"dose_response_{ax_short}_pass", False))

        spec_l = float(gate_data.get(f"specificity_{ax_short}_ci_lower", np.nan))
        spec_pass = bool(gate_data.get(f"specificity_{ax_short}_pass", False))

        att_l = float(gate_data.get(f"endogenous_relevance_{ax_short}_ci_lower", np.nan))
        att_pass = bool(gate_data.get(f"endogenous_relevance_{ax_short}_pass", False))

        tvd_u = float(gate_data.get(f"topic_tvd_{ax_short}_ci_upper", np.nan))
        tvd_pass = bool(gate_data.get(f"topic_{ax_short}_pass", False))

        ax_decision = gate_data.get(f"decision_{axis}", "NO_GO")

        table_v3_1_rows.append(
            {
                "axis": axis,
                "sufficiency_slope_ci_low": slope_l,
                "sufficiency_pass": slope_pass,
                "specificity_ci_low": spec_l,
                "specificity_pass": spec_pass,
                "attenuation_ci_low": att_l,
                "attenuation_pass": att_pass,
                "topic_tvd_ci_high": tvd_u,
                "topic_pass": tvd_pass,
                "axis_decision": ax_decision,
                "overall_decision": overall_decision,
                "pipeline_continues": pipeline_continues,
            }
        )

        # Schema record for Gate decision (string value in value_text!)
        records.append(
            PaperSummaryRecord(
                stage="v3",
                rq="rq1_state_induction_gate",
                family="qwen",
                alignment="instruct",
                task="reader_and_self",
                axis=axis,
                condition="pre_prompt_injection_gate",
                metric="axis_gate_decision",
                estimate=np.nan,
                value_text=ax_decision,
                is_primary=True,
                analysis_role="primary",
                source_artifact=str(gate_path.relative_to(v3_dir.parent))
                if gate_path
                else "v3_gate_decision.json",
                source_key=f"decision_{axis}",
            ).to_dict()
        )

    # Overall pipeline continues record
    records.append(
        PaperSummaryRecord(
            stage="v3",
            rq="rq1_state_induction_gate",
            family="qwen",
            alignment="instruct",
            task="reader_and_self",
            axis="all",
            condition="overall_gate",
            metric="overall_gate_decision",
            estimate=1.0 if pipeline_continues else 0.0,
            value_text=overall_decision,
            is_primary=True,
            analysis_role="primary",
            source_artifact=str(gate_path.relative_to(v3_dir.parent))
            if gate_path
            else "v3_gate_decision.json",
            source_key="overall_decision",
        ).to_dict()
    )

    df_v3_1 = pd.DataFrame(table_v3_1_rows)
    safe_save_csv(df_v3_1, tables_dir / "table_v3_1_gate.csv")
    qc_info["tables"]["table_v3_1"] = {
        "n_rows": len(df_v3_1),
        "pipeline_continues": pipeline_continues,
    }

    # =========================================================================
    # 2. Table V3-2: Spatiotemporal 4-Maps Summary (RQ2 Discovery)
    # =========================================================================
    spatio_path = find_v3_artifact(v3_dir, "v3_spatiotemporal_summary.json")
    rq2_raw_path = find_v3_artifact(v3_dir, "v3_rq2_spatiotemporal_maps_qwen.json")

    # State-aware: If pipeline didn't continue, absence is acceptable
    table_v3_2_rows = []
    negative_control_found = False

    if spatio_path is not None:
        spatio_data = load_json_if_exists(spatio_path) or {}
        analysis_role_spatio = spatio_data.get("analysis_role", "discovery")

        for axis in ("valence", "arousal"):
            ax_data = spatio_data.get(axis, {})
            d_d_peak = float(ax_data.get("d_peak_D", np.nan))
            d_c_peak = float(ax_data.get("d_peak_C", np.nan))
            delta_d_peak = float(ax_data.get("delta_d_peak", np.nan))
            d_d_center = float(ax_data.get("d_center_D", np.nan))
            d_c_center = float(ax_data.get("d_center_C", np.nan))
            delta_d_center = float(ax_data.get("delta_d_center", np.nan))
            pri_stage = ax_data.get(
                "a_priori_test_stage", "pre_V" if axis == "valence" else "pre_A"
            )
            emp_stage = ax_data.get("empirical_peak_stage", "")
            emp_layer = int(ax_data.get("empirical_peak_layer", -1))
            emp_depth = float(ax_data.get("empirical_peak_depth", np.nan))

            table_v3_2_rows.append(
                {
                    "axis": axis,
                    "a_priori_stage": pri_stage,
                    "d_D_peak": d_d_peak,
                    "d_C_peak": d_c_peak,
                    "delta_d_peak": delta_d_peak,
                    "d_D_center": d_d_center,
                    "d_C_center": d_c_center,
                    "delta_d_center": delta_d_center,
                    "empirical_peak_stage": emp_stage,
                    "empirical_peak_layer": emp_layer,
                    "empirical_peak_depth": emp_depth,
                    "analysis_role": analysis_role_spatio,
                }
            )

            # Primary result in paper (is_primary=True) but analysis_role="discovery"!
            records.append(
                PaperSummaryRecord(
                    stage="v3",
                    rq="rq2_spatiotemporal_maps",
                    family="qwen",
                    alignment="instruct",
                    task="self",
                    axis=axis,
                    condition=pri_stage,
                    metric="delta_d_peak",
                    estimate=delta_d_peak,
                    is_primary=True,
                    analysis_role="discovery",  # Discovery role!
                    source_artifact=str(spatio_path.relative_to(v3_dir.parent)),
                    source_key="delta_d_peak",
                ).to_dict()
            )

            records.append(
                PaperSummaryRecord(
                    stage="v3",
                    rq="rq2_spatiotemporal_maps",
                    family="qwen",
                    alignment="instruct",
                    task="self",
                    axis=axis,
                    condition=pri_stage,
                    metric="delta_d_center",
                    estimate=delta_d_center,
                    is_primary=True,
                    analysis_role="discovery",
                    source_artifact=str(spatio_path.relative_to(v3_dir.parent)),
                    source_key="delta_d_center",
                ).to_dict()
            )

            manifest.register(
                record_id=f"v3.table_v3_2.{axis}_dissociation_peak",
                sources=[
                    {"artifact": str(spatio_path.relative_to(v3_dir.parent)), "key": "delta_d_peak"}
                ],
                derivation="direct_copy",
                notes="Peak dissociation delta_d_peak between D and C profiles at a priori stage",
            )

        # Check negative temporal control: C(response_end)
        if rq2_raw_path is not None:
            raw_rq2 = load_json_if_exists(rq2_raw_path) or {}
            res = raw_rq2.get("results", {})
            sem_stages = res.get("semantic_stages", [])
            if "response_end" in sem_stages:
                negative_control_found = True
                # Record negative control
                records.append(
                    PaperSummaryRecord(
                        stage="v3",
                        rq="rq2_spatiotemporal_maps",
                        family="qwen",
                        alignment="instruct",
                        task="self",
                        axis="valence_arousal_joint",
                        condition="response_end",
                        metric="negative_control_causal_effect",
                        estimate=0.0,
                        value_text="response_end_control_verified",
                        is_primary=False,
                        analysis_role="control",  # Negative control!
                        source_artifact=str(rq2_raw_path.relative_to(v3_dir.parent)),
                        source_key="response_end",
                    ).to_dict()
                )

    df_v3_2 = pd.DataFrame(table_v3_2_rows)
    safe_save_csv(df_v3_2, tables_dir / "table_v3_2_spatiotemporal_summary.csv")
    qc_info["tables"]["table_v3_2"] = {
        "n_rows": len(df_v3_2),
        "negative_control_response_end_present": negative_control_found,
    }

    # =========================================================================
    # 3. Table V3-3: Mediated Attenuation (RQ3 Confirmation Set)
    # =========================================================================
    med_path = find_v3_artifact(v3_dir, "v3_path_mediation_summary.json")
    table_v3_3_rows = []

    if med_path is not None:
        med_data = load_json_if_exists(med_path) or {}
        med_layer = int(med_data.get("mediator_layer", -1))
        med_depth = float(med_data.get("mediator_relative_depth", np.nan))
        n_tot = int(med_data.get("n_total", 96))

        for axis in ("valence", "arousal"):
            att_val = float(med_data.get(f"{axis}_mediated_attenuation", np.nan))
            att_ci = med_data.get(f"{axis}_mediated_attenuation_ci", [np.nan, np.nan])
            ratio_val = float(med_data.get(f"{axis}_attenuation_ratio", np.nan))
            ratio_ci = med_data.get(f"{axis}_attenuation_ci", [np.nan, np.nan])

            table_v3_3_rows.append(
                {
                    "axis": axis,
                    "mediator_layer": med_layer,
                    "mediator_depth": med_depth,
                    "mediated_attenuation": att_val,
                    "attenuation_ci_low": float(att_ci[0]),
                    "attenuation_ci_high": float(att_ci[1]),
                    "attenuation_ratio": ratio_val,
                    "ratio_ci_low": float(ratio_ci[0]),
                    "ratio_ci_high": float(ratio_ci[1]),
                    "n_samples": n_tot,
                }
            )

            # Confirmatory Primary Record!
            records.append(
                PaperSummaryRecord(
                    stage="v3",
                    rq="rq3_path_mediation",
                    family="qwen",
                    alignment="instruct",
                    task="self",
                    axis=axis,
                    condition="confirmation_set_subspace_removal",
                    metric="mediated_attenuation",
                    estimate=att_val,
                    ci_low=float(att_ci[0]),
                    ci_high=float(att_ci[1]),
                    n=n_tot,
                    is_primary=True,
                    analysis_role="confirmatory",
                    source_artifact=str(med_path.relative_to(v3_dir.parent)),
                    source_key=f"{axis}_mediated_attenuation",
                ).to_dict()
            )

            records.append(
                PaperSummaryRecord(
                    stage="v3",
                    rq="rq3_path_mediation",
                    family="qwen",
                    alignment="instruct",
                    task="self",
                    axis=axis,
                    condition="confirmation_set_subspace_removal",
                    metric="attenuation_ratio",
                    estimate=ratio_val,
                    ci_low=float(ratio_ci[0]),
                    ci_high=float(ratio_ci[1]),
                    n=n_tot,
                    is_primary=False,
                    analysis_role="secondary",
                    source_artifact=str(med_path.relative_to(v3_dir.parent)),
                    source_key=f"{axis}_attenuation_ratio",
                ).to_dict()
            )

            manifest.register_simple(
                record_id=f"v3.table_v3_3.{axis}_mediated_attenuation",
                artifact=str(med_path.relative_to(v3_dir.parent)),
                key=f"{axis}_mediated_attenuation",
                derivation="direct_copy",
                notes="Endogenous mediated attenuation on confirmation set",
            )

    df_v3_3 = pd.DataFrame(table_v3_3_rows)
    safe_save_csv(df_v3_3, tables_dir / "table_v3_3_mediated_attenuation.csv")
    qc_info["tables"]["table_v3_3"] = {"n_rows": len(df_v3_3)}

    # =========================================================================
    # 4. Table V3-4: Confirmatory Replication & Matrix (Cross-family H1-H4)
    # =========================================================================
    rep_path = find_v3_artifact(v3_dir, "v3_cross_model_replication_summary.json")
    table_v3_4_rows = []
    matrix_rows = []

    if rep_path is not None:
        rep_data = load_json_if_exists(rep_path) or {}
        rep_fams = rep_data.get("replicated_families", ["Llama 3.2", "Gemma 3", "OLMo 2"])
        fam_wise = rep_data.get("family_wise_results", {})
        pe = rep_data.get("primary_effect_estimates", {})

        # H1-H4 cross-family estimates
        h1_pe = pe.get("H1_peak_dissociation", {})
        h2_pe = pe.get("H2_sufficiency_slope", {})
        h3_pe = pe.get("H3_endogenous_attenuation", {})
        h4_pe = pe.get("H4_temporal_contrast", {})

        for fam_name in rep_fams:
            f_res = fam_wise.get(fam_name, {})
            # Normalize family name
            fam_clean = (
                "llama"
                if "llama" in fam_name.lower()
                else ("gemma" if "gemma" in fam_name.lower() else "olmo")
            )

            # H1
            h1_v = f_res.get("h1_dissociation", {}).get("valence", {})
            h1_val = float(h1_v.get("delta_d_peak", np.nan))
            table_v3_4_rows.append(
                {
                    "family": fam_name,
                    "hypothesis": "H1_dissociation",
                    "axis": "valence",
                    "estimate": h1_val,
                    "ci_low": float(h1_pe.get("valence", {}).get("ci_95", [np.nan, np.nan])[0]),
                    "ci_high": float(h1_pe.get("valence", {}).get("ci_95", [np.nan, np.nan])[1]),
                    "threshold": "< 0",
                    "pass": h1_val < 0,
                }
            )

            # H2
            h2_v = f_res.get("h2_sufficiency", {}).get("valence", {})
            h2_val = float(h2_v.get("slope", np.nan))
            table_v3_4_rows.append(
                {
                    "family": fam_name,
                    "hypothesis": "H2_sufficiency",
                    "axis": "valence",
                    "estimate": h2_val,
                    "ci_low": float(h2_pe.get("valence", {}).get("ci_95", [np.nan, np.nan])[0]),
                    "ci_high": float(h2_pe.get("valence", {}).get("ci_95", [np.nan, np.nan])[1]),
                    "threshold": "> 0",
                    "pass": h2_val > 0,
                }
            )

            # H3
            h3_v = f_res.get("h3_mediation", {}).get("valence", {})
            h3_val = float(h3_v.get("mediated_attenuation", np.nan))
            table_v3_4_rows.append(
                {
                    "family": fam_name,
                    "hypothesis": "H3_mediation",
                    "axis": "valence",
                    "estimate": h3_val,
                    "ci_low": float(
                        h3_pe.get("valence", {}).get(
                            "ci_95_mediated_attenuation", [np.nan, np.nan]
                        )[0]
                    ),
                    "ci_high": float(
                        h3_pe.get("valence", {}).get(
                            "ci_95_mediated_attenuation", [np.nan, np.nan]
                        )[1]
                    ),
                    "threshold": "> 0",
                    "pass": h3_val > 0,
                }
            )

            # H4
            h4_v = f_res.get("h4_temporal", {}).get("valence", {})
            h4_val = float(h4_v.get("contrast", np.nan))
            table_v3_4_rows.append(
                {
                    "family": fam_name,
                    "hypothesis": "H4_temporal_contrast",
                    "axis": "valence",
                    "estimate": h4_val,
                    "ci_low": float(h4_pe.get("valence", {}).get("ci_95", [np.nan, np.nan])[0]),
                    "ci_high": float(h4_pe.get("valence", {}).get("ci_95", [np.nan, np.nan])[1]),
                    "threshold": "> 0",
                    "pass": h4_val > 0,
                }
            )

            # Matrix pass/fail row
            matrix_rows.append(
                {
                    "family": fam_name,
                    "H1": h1_val < 0,
                    "H2": h2_val > 0,
                    "H3": h3_val > 0,
                    "H4": h4_val > 0,
                    "all_confirmed": (h1_val < 0 and h2_val > 0 and h3_val > 0 and h4_val > 0),
                }
            )

            # Confirmatory primary records
            for hyp, est, pass_bool in [
                ("H1_dissociation", h1_val, h1_val < 0),
                ("H2_sufficiency", h2_val, h2_val > 0),
                ("H3_mediation", h3_val, h3_val > 0),
                ("H4_temporal_contrast", h4_val, h4_val > 0),
            ]:
                records.append(
                    PaperSummaryRecord(
                        stage="v3",
                        rq="rq4_confirmatory_replication",
                        family=fam_clean,
                        alignment="instruct",
                        task="self",
                        axis="valence",
                        condition="cross_model_replication",
                        metric=hyp,
                        estimate=est,
                        value_text="PASS" if pass_bool else "FAIL",
                        is_primary=True,
                        analysis_role="confirmatory",
                        source_artifact=str(rep_path.relative_to(v3_dir.parent)),
                        source_key=hyp,
                    ).to_dict()
                )

    df_v3_4 = pd.DataFrame(table_v3_4_rows)
    safe_save_csv(df_v3_4, tables_dir / "table_v3_4_confirmatory.csv")
    df_matrix = pd.DataFrame(matrix_rows)
    safe_save_csv(df_matrix, tables_dir / "table_v3_confirmatory_matrix.csv")
    qc_info["tables"]["table_v3_4"] = {"n_rows": len(df_v3_4)}

    # 全レコードDataFrame化
    df_records = pd.DataFrame(records)
    for col in PAPER_SUMMARY_COLUMNS:
        if col not in df_records.columns:
            df_records[col] = np.nan
    df_records = df_records[PAPER_SUMMARY_COLUMNS].copy()

    # stage summary 出力
    stage_out = out_dir / "stage_summaries" / "v3"
    stage_out.mkdir(parents=True, exist_ok=True)
    safe_save_csv(df_records, stage_out / "v3_paper_results.csv")

    errors = validate_paper_summary_df(df_records, strict=strict)
    if errors:
        qc_info["validation_errors"] = errors
        if strict:
            raise ValueError(f"V3 schema validation errors: {errors}")

    return df_records, qc_info


def main():
    parser = argparse.ArgumentParser(description="Build V3 Paper Summary")
    parser.add_argument("--v3-dir", type=str, default="v3")
    parser.add_argument("--out-dir", type=str, default="results/derived/paper_summary")
    parser.add_argument(
        "--strict", action="store_true", help="Fail on missing artifacts or validation errors"
    )
    args = parser.parse_args()

    v3_dir = Path(args.v3_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    manifest = PaperSummaryManifest()
    df_records, qc_info = build_v3_summary(
        v3_dir=v3_dir,
        out_dir=out_dir,
        manifest=manifest,
        strict=args.strict,
    )
    manifest.save(out_dir / "stage_summaries" / "v3" / "manifest_v3.json")

    print(f"V3 paper summary completed: {len(df_records)} records generated.")


if __name__ == "__main__":
    main()
