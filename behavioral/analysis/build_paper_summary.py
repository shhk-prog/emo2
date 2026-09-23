#!/usr/bin/env python3
"""
behavioral/analysis/build_paper_summary.py

§4 Behavioral 結果章用サマリー生成スクリプト（Read-only presentation layer）。
既存の検証済み結果から以下を抽出・フォーマットして出力する：
  - Table B1: EmoBank Human Correspondence (Reader/Self/Writer x V/A/D)
  - Table B2: AIPsy Sensitivity (Matched clinical vs neutral)
  - Table B3: AIPsy Dose-Response (Neutral -> Moderate -> Clinical)
  - Table B4: AIPsy Specificity (Clinical vs ComplexNeutral displacement)
  - Table B5: Reader-Self Coupling (Delta correlation as primary, raw as secondary)
  - Table B Summary: 各モデルの代表値一覧（総合点・ランキングなし）
  - Figure Data: Figure B1〜B4 用 CSV
  - 共通19列スキーマレコードおよび Manifest への Provenance 登録
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from affective_empathy_eval.paper_summary.common import (
    PaperSummaryManifest,
    safe_save_csv,
)
from affective_empathy_eval.paper_summary.schema import (
    PAPER_SUMMARY_COLUMNS,
    PaperSummaryRecord,
    validate_paper_summary_df,
)


def parse_model_info(model_str: str) -> tuple[str, str]:
    """model_str から (family, alignment) を抽出"""
    s = model_str.lower()
    if "llama" in s:
        fam = "llama"
    elif "gemma" in s:
        fam = "gemma"
    elif "olmo" in s:
        fam = "olmo"
    elif "qwen" in s:
        fam = "qwen"
    else:
        fam = s.split("_")[0]

    align = "instruct" if any(k in s for k in ("instruct", "chat", "it")) else "base"
    return fam, align


def normalize_axis(dim: str) -> str:
    d = str(dim).strip().upper()
    if d == "V":
        return "valence"
    elif d == "A":
        return "arousal"
    elif d == "D":
        return "dominance"
    return d.lower()


def normalize_task(task: str) -> str:
    t = str(task).strip().lower()
    if t in ("w", "writer"):
        return "writer"
    elif t in ("r", "reader"):
        return "reader"
    elif t in ("s", "self"):
        return "self"
    return t


def build_behavioral_summary(
    behavioral_dir: Path,
    out_dir: Path,
    manifest: PaperSummaryManifest,
    strict: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    qc_info: dict[str, Any] = {"stage": "behavioral", "tables": {}}

    emobank_dir = behavioral_dir / "results" / "derived" / "emobank_3way_summary"
    aipsy_dir = behavioral_dir / "results" / "derived" / "aipsy_4split_summary"

    tables_dir = out_dir / "tables"
    fig_dir = out_dir / "figure_data"
    tables_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 1. Table B1: EmoBank Human Correspondence
    # =========================================================================
    emobank_csv = emobank_dir / "behavioral_emobank_metrics.csv"
    if not emobank_csv.exists() and strict:
        raise FileNotFoundError(f"Required artifact missing: {emobank_csv}")

    df_emo = pd.read_csv(emobank_csv) if emobank_csv.exists() else pd.DataFrame()
    table_b1_rows = []
    fig_b1_rows = []

    if len(df_emo) > 0:
        for _, row in df_emo.iterrows():
            m_str = str(row["model"])
            fam, align = parse_model_info(m_str)
            t_norm = normalize_task(row["task"])
            ax_norm = normalize_axis(row["dimension"])
            n_val = int(row["n"]) if pd.notnull(row["n"]) else 1000

            pr = float(row["r_continuous"]) if pd.notnull(row["r_continuous"]) else np.nan
            pp = float(row["p_continuous"]) if pd.notnull(row["p_continuous"]) else np.nan
            sr = float(row["rho_continuous"]) if pd.notnull(row["rho_continuous"]) else np.nan

            mae_val = float(row["mae"]) if "mae" in row and pd.notnull(row["mae"]) else np.nan
            rmse_val = float(row["rmse"]) if "rmse" in row and pd.notnull(row["rmse"]) else np.nan

            table_b1_rows.append(
                {
                    "model": m_str,
                    "family": fam,
                    "alignment": align,
                    "task": t_norm,
                    "axis": ax_norm,
                    "n": n_val,
                    "pearson_r": pr,
                    "pearson_p": pp,
                    "spearman_rho": sr,
                    "mae": mae_val,
                    "rmse": rmse_val,
                }
            )

            # Primary: Reader V/A and Self V/A pearson r and spearman rho
            is_prim_metric = t_norm in ("reader", "self") and ax_norm in ("valence", "arousal")

            # Record for Pearson r
            records.append(
                PaperSummaryRecord(
                    stage="behavioral",
                    rq="rq1_emobank_correspondence",
                    family=fam,
                    alignment=align,
                    task=t_norm,
                    axis=ax_norm,
                    condition="emobank_human_grounding",
                    metric="pearson_r",
                    estimate=pr,
                    p=pp,
                    n=n_val,
                    is_primary=is_prim_metric,
                    analysis_role="primary" if is_prim_metric else "secondary",
                    source_artifact=str(emobank_csv.relative_to(behavioral_dir.parent)),
                    source_key="r_continuous",
                ).to_dict()
            )

            # Record for Spearman rho
            records.append(
                PaperSummaryRecord(
                    stage="behavioral",
                    rq="rq1_emobank_correspondence",
                    family=fam,
                    alignment=align,
                    task=t_norm,
                    axis=ax_norm,
                    condition="emobank_human_grounding",
                    metric="spearman_rho",
                    estimate=sr,
                    n=n_val,
                    is_primary=is_prim_metric,
                    analysis_role="primary" if is_prim_metric else "secondary",
                    source_artifact=str(emobank_csv.relative_to(behavioral_dir.parent)),
                    source_key="rho_continuous",
                ).to_dict()
            )

            # MAE / RMSE as secondary
            if not np.isnan(mae_val):
                records.append(
                    PaperSummaryRecord(
                        stage="behavioral",
                        rq="rq1_emobank_correspondence",
                        family=fam,
                        alignment=align,
                        task=t_norm,
                        axis=ax_norm,
                        condition="emobank_human_grounding",
                        metric="mae",
                        estimate=mae_val,
                        n=n_val,
                        is_primary=False,
                        analysis_role="secondary",
                        source_artifact=str(emobank_csv.relative_to(behavioral_dir.parent)),
                        source_key="mae",
                    ).to_dict()
                )

            fig_b1_rows.append(
                {
                    "model": m_str,
                    "family": fam,
                    "alignment": align,
                    "task": t_norm,
                    "axis": ax_norm,
                    "r": pr,
                    "rho": sr,
                }
            )

            rec_id = f"behavioral.table_b1.{m_str}_{t_norm}_{ax_norm}_r"
            manifest.register_simple(
                record_id=rec_id,
                artifact=str(emobank_csv.relative_to(behavioral_dir.parent)),
                key="r_continuous",
                derivation="direct_copy",
                notes="EmoBank continuous Pearson r",
            )

    df_b1 = pd.DataFrame(table_b1_rows)
    safe_save_csv(df_b1, tables_dir / "table_b1_emobank_correspondence.csv")
    safe_save_csv(pd.DataFrame(fig_b1_rows), fig_dir / "figure_b1_emobank_corr.csv")
    qc_info["tables"]["table_b1"] = {"n_rows": len(df_b1)}

    # =========================================================================
    # 2. Table B2: Sensitivity (AIPsy Matched Pairs)
    # =========================================================================
    sens_csv = aipsy_dir / "behavioral_aipsy_sensitivity_rq1.csv"
    if not sens_csv.exists() and strict:
        raise FileNotFoundError(f"Required artifact missing: {sens_csv}")

    df_sens = pd.read_csv(sens_csv) if sens_csv.exists() else pd.DataFrame()
    table_b2_rows = []
    fig_b2_rows = []

    if len(df_sens) > 0:
        for _, row in df_sens.iterrows():
            m_str = str(row["model"])
            fam, align = parse_model_info(m_str)
            t_norm = normalize_task(row["task"])
            ax_norm = normalize_axis(row["dimension"])
            n_p = int(row["n_pairs"]) if pd.notnull(row["n_pairs"]) else 192

            m_aligned = float(row["direction_aligned_mean_diff"])
            ci_l = float(row["aligned_ci_low"])
            ci_h = float(row["aligned_ci_high"])
            dz = float(row["aligned_d_z"])
            t_stat = float(row["aligned_t_stat"])
            pval = float(row["aligned_p_value"])
            qval = (
                float(row["p_fdr"])
                if "p_fdr" in row and pd.notnull(row["p_fdr"]) and row["p_fdr"] != ""
                else np.nan
            )
            m_raw = float(row["raw_mean_diff"])

            table_b2_rows.append(
                {
                    "model": m_str,
                    "family": fam,
                    "alignment": align,
                    "task": t_norm,
                    "axis": ax_norm,
                    "n_pairs": n_p,
                    "mean_aligned_delta": m_aligned,
                    "ci_low": ci_l,
                    "ci_high": ci_h,
                    "cohen_dz": dz,
                    "t": t_stat,
                    "p": pval,
                    "q": qval,
                    "mean_raw_delta": m_raw,
                }
            )

            is_prim = t_norm in ("reader", "self") and ax_norm in ("valence", "arousal")

            records.append(
                PaperSummaryRecord(
                    stage="behavioral",
                    rq="rq2_sensitivity",
                    family=fam,
                    alignment=align,
                    task=t_norm,
                    axis=ax_norm,
                    condition="clinical_minus_neutral_matched",
                    metric="mean_aligned_delta",
                    estimate=m_aligned,
                    ci_low=ci_l,
                    ci_high=ci_h,
                    p=pval,
                    q=qval,
                    n=n_p,
                    is_primary=is_prim,
                    analysis_role="primary" if is_prim else "secondary",
                    source_artifact=str(sens_csv.relative_to(behavioral_dir.parent)),
                    source_key="direction_aligned_mean_diff",
                ).to_dict()
            )

            records.append(
                PaperSummaryRecord(
                    stage="behavioral",
                    rq="rq2_sensitivity",
                    family=fam,
                    alignment=align,
                    task=t_norm,
                    axis=ax_norm,
                    condition="clinical_minus_neutral_matched",
                    metric="cohen_dz",
                    estimate=dz,
                    p=pval,
                    q=qval,
                    n=n_p,
                    is_primary=False,
                    analysis_role="secondary",
                    source_artifact=str(sens_csv.relative_to(behavioral_dir.parent)),
                    source_key="aligned_d_z",
                ).to_dict()
            )

            fig_b2_rows.append(
                {
                    "model": m_str,
                    "family": fam,
                    "alignment": align,
                    "task": t_norm,
                    "axis": ax_norm,
                    "mean_aligned_delta": m_aligned,
                    "ci_low": ci_l,
                    "ci_high": ci_h,
                    "cohen_dz": dz,
                }
            )

            rec_id = f"behavioral.table_b2.{m_str}_{t_norm}_{ax_norm}_sensitivity"
            manifest.register_simple(
                record_id=rec_id,
                artifact=str(sens_csv.relative_to(behavioral_dir.parent)),
                key="direction_aligned_mean_diff",
                derivation="direct_copy",
                notes="AIPsy Sensitivity direction-aligned delta",
            )

    df_b2 = pd.DataFrame(table_b2_rows)
    safe_save_csv(df_b2, tables_dir / "table_b2_sensitivity.csv")
    safe_save_csv(pd.DataFrame(fig_b2_rows), fig_dir / "figure_b2_sensitivity.csv")
    qc_info["tables"]["table_b2"] = {"n_rows": len(df_b2)}

    # =========================================================================
    # 3. Table B3: Dose-Response (48 Triplets)
    # =========================================================================
    dose_csv = aipsy_dir / "behavioral_aipsy_dose_response_rq2.csv"
    if not dose_csv.exists() and strict:
        raise FileNotFoundError(f"Required artifact missing: {dose_csv}")

    df_dose = pd.read_csv(dose_csv) if dose_csv.exists() else pd.DataFrame()
    table_b3_rows = []
    fig_b3_rows = []

    if len(df_dose) > 0:
        for _, row in df_dose.iterrows():
            m_str = str(row["model"])
            fam, align = parse_model_info(m_str)
            t_norm = normalize_task(row["task"])
            ax_norm = normalize_axis(row["dimension"])
            n_t = int(row["n_triplets"]) if pd.notnull(row["n_triplets"]) else 48

            s1_m = float(row["mean_step_1_aligned"])
            s1_l = float(row["step_1_ci_low"])
            s1_h = float(row["step_1_ci_high"])
            p_s1 = float(row["step_1_p_value"])

            s2_m = float(row["mean_step_2_aligned"])
            s2_l = float(row["step_2_ci_low"])
            s2_h = float(row["step_2_ci_high"])
            p_s2 = float(row["step_2_p_value"])

            p_iut = float(row["dose_response_p_value"])
            q_iut = (
                float(row["p_fdr"])
                if "p_fdr" in row and pd.notnull(row["p_fdr"]) and row["p_fdr"] != ""
                else np.nan
            )
            mono_rate = float(row["monotonicity_rate"])
            sec_slope = float(row["secondary_mean_slope"])

            table_b3_rows.append(
                {
                    "model": m_str,
                    "family": fam,
                    "alignment": align,
                    "task": t_norm,
                    "axis": ax_norm,
                    "n_triplets": n_t,
                    "step1_mean": s1_m,
                    "step1_ci_low": s1_l,
                    "step1_ci_high": s1_h,
                    "step2_mean": s2_m,
                    "step2_ci_low": s2_l,
                    "step2_ci_high": s2_h,
                    "p_step1": p_s1,
                    "p_step2": p_s2,
                    "p_iut": p_iut,
                    "q_iut": q_iut,
                    "monotonicity_rate": mono_rate,
                    "secondary_slope": sec_slope,
                }
            )

            is_prim = t_norm in ("reader", "self") and ax_norm in ("valence", "arousal")

            records.append(
                PaperSummaryRecord(
                    stage="behavioral",
                    rq="rq3_dose_response",
                    family=fam,
                    alignment=align,
                    task=t_norm,
                    axis=ax_norm,
                    condition="triplet_dose_response",
                    metric="dose_response_p_iut",
                    estimate=p_iut,
                    p=p_iut,
                    q=q_iut,
                    n=n_t,
                    is_primary=is_prim,
                    analysis_role="primary" if is_prim else "secondary",
                    source_artifact=str(dose_csv.relative_to(behavioral_dir.parent)),
                    source_key="dose_response_p_value",
                ).to_dict()
            )

            records.append(
                PaperSummaryRecord(
                    stage="behavioral",
                    rq="rq3_dose_response",
                    family=fam,
                    alignment=align,
                    task=t_norm,
                    axis=ax_norm,
                    condition="triplet_dose_response",
                    metric="monotonicity_rate",
                    estimate=mono_rate,
                    n=n_t,
                    is_primary=False,
                    analysis_role="secondary",
                    source_artifact=str(dose_csv.relative_to(behavioral_dir.parent)),
                    source_key="monotonicity_rate",
                ).to_dict()
            )

            fig_b3_rows.append(
                {
                    "model": m_str,
                    "family": fam,
                    "alignment": align,
                    "task": t_norm,
                    "axis": ax_norm,
                    "mean_neutral": float(row["mean_neutral"]),
                    "mean_moderate": float(row["mean_moderate"]),
                    "mean_clinical": float(row["mean_clinical"]),
                    "step1_mean": s1_m,
                    "step2_mean": s2_m,
                }
            )

            rec_id = f"behavioral.table_b3.{m_str}_{t_norm}_{ax_norm}_dose_response"
            manifest.register_simple(
                record_id=rec_id,
                artifact=str(dose_csv.relative_to(behavioral_dir.parent)),
                key="dose_response_p_value",
                derivation="direct_copy",
                notes="IUT intersection-union p-value",
            )

    df_b3 = pd.DataFrame(table_b3_rows)
    safe_save_csv(df_b3, tables_dir / "table_b3_dose_response.csv")
    safe_save_csv(pd.DataFrame(fig_b3_rows), fig_dir / "figure_b3_dose_response.csv")
    qc_info["tables"]["table_b3"] = {"n_rows": len(df_b3)}

    # =========================================================================
    # 4. Table B4: Specificity (Clinical vs ComplexNeutral)
    # =========================================================================
    spec_csv = aipsy_dir / "behavioral_aipsy_specificity_rq3.csv"
    if not spec_csv.exists() and strict:
        raise FileNotFoundError(f"Required artifact missing: {spec_csv}")

    df_spec = pd.read_csv(spec_csv) if spec_csv.exists() else pd.DataFrame()
    table_b4_rows = []

    if len(df_spec) > 0:
        for _, row in df_spec.iterrows():
            m_str = str(row["model"])
            fam, align = parse_model_info(m_str)
            t_norm = normalize_task(row["task"])
            ax_norm = normalize_axis(row["dimension"])

            c_disp = float(row["mean_clinical_displacement"])
            cn_disp = float(row["mean_complex_neutral_displacement"])
            diff = float(row["displacement_diff"])
            d_val = float(row["displacement_cohen_d"])
            t_val = float(row["displacement_t_stat"])
            p_val = float(row["displacement_p_value"])
            q_val = (
                float(row["p_fdr"])
                if "p_fdr" in row and pd.notnull(row["p_fdr"]) and row["p_fdr"] != ""
                else np.nan
            )

            table_b4_rows.append(
                {
                    "model": m_str,
                    "family": fam,
                    "alignment": align,
                    "task": t_norm,
                    "axis": ax_norm,
                    "clinical_displacement": c_disp,
                    "complex_neutral_displacement": cn_disp,
                    "difference": diff,
                    "cohen_d": d_val,
                    "welch_t": t_val,
                    "p": p_val,
                    "q": q_val,
                }
            )

            is_prim = t_norm in ("reader", "self") and ax_norm in ("valence", "arousal")

            records.append(
                PaperSummaryRecord(
                    stage="behavioral",
                    rq="rq4_specificity",
                    family=fam,
                    alignment=align,
                    task=t_norm,
                    axis=ax_norm,
                    condition="clinical_vs_complex_neutral",
                    metric="displacement_difference",
                    estimate=diff,
                    p=p_val,
                    q=q_val,
                    n=int(row["n_clinical"]) if "n_clinical" in row else 192,
                    is_primary=is_prim,
                    analysis_role="primary" if is_prim else "secondary",
                    source_artifact=str(spec_csv.relative_to(behavioral_dir.parent)),
                    source_key="displacement_diff",
                ).to_dict()
            )

            rec_id = f"behavioral.table_b4.{m_str}_{t_norm}_{ax_norm}_specificity"
            manifest.register_simple(
                record_id=rec_id,
                artifact=str(spec_csv.relative_to(behavioral_dir.parent)),
                key="displacement_diff",
                derivation="direct_copy",
                notes="E[D_clinical] - E[D_complex_neutral]",
            )

    df_b4 = pd.DataFrame(table_b4_rows)
    safe_save_csv(df_b4, tables_dir / "table_b4_specificity.csv")
    qc_info["tables"]["table_b4"] = {"n_rows": len(df_b4)}

    # =========================================================================
    # 5. Table B5: Reader-Self Coupling (最重要結果)
    # =========================================================================
    coup_csv = aipsy_dir / "behavioral_aipsy_coupling_rq4.csv"
    if not coup_csv.exists() and strict:
        raise FileNotFoundError(f"Required artifact missing: {coup_csv}")

    df_coup = pd.read_csv(coup_csv) if coup_csv.exists() else pd.DataFrame()
    table_b5_rows = []
    fig_b4_rows = []

    if len(df_coup) > 0:
        # delta_coupling と raw_coupling を pivot
        models = df_coup["model"].unique()
        dims = df_coup["dimension"].unique()

        for m_str in models:
            fam, align = parse_model_info(m_str)
            for d in dims:
                ax_norm = normalize_axis(d)
                m_sub = df_coup[(df_coup["model"] == m_str) & (df_coup["dimension"] == d)]

                delta_row = m_sub[m_sub["correlation_type"] == "delta_coupling"]
                raw_row = m_sub[m_sub["correlation_type"] == "raw_coupling"]

                if len(delta_row) == 0:
                    continue

                d_r = float(delta_row["r"].iloc[0])
                d_ci_l = float(delta_row["ci_low"].iloc[0])
                d_ci_h = float(delta_row["ci_high"].iloc[0])
                d_p = float(delta_row["p_value"].iloc[0])
                d_q = (
                    float(delta_row["p_fdr"].iloc[0])
                    if "p_fdr" in delta_row
                    and pd.notnull(delta_row["p_fdr"].iloc[0])
                    and delta_row["p_fdr"].iloc[0] != ""
                    else np.nan
                )
                d_rho = (
                    float(delta_row["spearman_rho"].iloc[0])
                    if "spearman_rho" in delta_row
                    else np.nan
                )
                n_pairs = int(delta_row["n_pairs"].iloc[0])

                raw_r = float(raw_row["r"].iloc[0]) if len(raw_row) > 0 else np.nan
                raw_rho = (
                    float(raw_row["spearman_rho"].iloc[0])
                    if len(raw_row) > 0 and "spearman_rho" in raw_row
                    else np.nan
                )

                table_b5_rows.append(
                    {
                        "model": m_str,
                        "family": fam,
                        "alignment": align,
                        "axis": ax_norm,
                        "n_pairs": n_pairs,
                        "delta_pearson_r": d_r,
                        "ci_low": d_ci_l,
                        "ci_high": d_ci_h,
                        "p": d_p,
                        "q": d_q,
                        "delta_spearman_rho": d_rho,
                        "raw_pearson_r": raw_r,
                        "raw_spearman_rho": raw_rho,
                    }
                )

                # Primary: delta coupling r
                records.append(
                    PaperSummaryRecord(
                        stage="behavioral",
                        rq="rq5_coupling",
                        family=fam,
                        alignment=align,
                        task="reader_self",
                        axis=ax_norm,
                        condition="matched_clinical_neutral_pair_delta",
                        metric="delta_pearson_r",
                        estimate=d_r,
                        ci_low=d_ci_l,
                        ci_high=d_ci_h,
                        p=d_p,
                        q=d_q,
                        n=n_pairs,
                        is_primary=True,
                        analysis_role="primary",
                        source_artifact=str(coup_csv.relative_to(behavioral_dir.parent)),
                        source_key="r",
                    ).to_dict()
                )

                # Secondary: raw coupling r
                if not np.isnan(raw_r):
                    records.append(
                        PaperSummaryRecord(
                            stage="behavioral",
                            rq="rq5_coupling",
                            family=fam,
                            alignment=align,
                            task="reader_self",
                            axis=ax_norm,
                            condition="raw_static_coupling",
                            metric="raw_pearson_r",
                            estimate=raw_r,
                            n=int(raw_row["n_pairs"].iloc[0]) if len(raw_row) > 0 else 480,
                            is_primary=False,
                            analysis_role="secondary",
                            source_artifact=str(coup_csv.relative_to(behavioral_dir.parent)),
                            source_key="r",
                        ).to_dict()
                    )

                fig_b4_rows.append(
                    {
                        "model": m_str,
                        "family": fam,
                        "alignment": align,
                        "axis": ax_norm,
                        "delta_r": d_r,
                        "delta_ci_low": d_ci_l,
                        "delta_ci_high": d_ci_h,
                        "raw_r": raw_r,
                    }
                )

                rec_id = f"behavioral.table_b5.{m_str}_{ax_norm}_coupling"
                manifest.register_simple(
                    record_id=rec_id,
                    artifact=str(coup_csv.relative_to(behavioral_dir.parent)),
                    key="r",
                    derivation="direct_copy",
                    notes="Correlation of Delta Reader and Delta Self",
                )

    df_b5 = pd.DataFrame(table_b5_rows)
    safe_save_csv(df_b5, tables_dir / "table_b5_coupling.csv")
    safe_save_csv(pd.DataFrame(fig_b4_rows), fig_dir / "figure_b4_coupling_summary.csv")
    qc_info["tables"]["table_b5"] = {"n_rows": len(df_b5)}

    # =========================================================================
    # 6. Behavioral Summary Table (Section 7)
    # =========================================================================
    # モデルごとの代表実数値を並べる（総合点・ランキングなし）
    summary_rows = []
    models_all = sorted(list(set(df_b1["model"].unique() if len(df_b1) > 0 else [])))
    for m in models_all:
        fam, align = parse_model_info(m)
        # EmoBank reader valence r
        emo_r = np.nan
        if len(df_b1) > 0:
            sub = df_b1[
                (df_b1["model"] == m) & (df_b1["task"] == "reader") & (df_b1["axis"] == "valence")
            ]
            if len(sub) > 0:
                emo_r = sub["pearson_r"].iloc[0]

        # Sensitivity reader valence delta
        sens_delta = np.nan
        if len(df_b2) > 0:
            sub = df_b2[
                (df_b2["model"] == m) & (df_b2["task"] == "reader") & (df_b2["axis"] == "valence")
            ]
            if len(sub) > 0:
                sens_delta = sub["mean_aligned_delta"].iloc[0]

        # Dose-response reader valence p_iut
        dr_p = np.nan
        if len(df_b3) > 0:
            sub = df_b3[
                (df_b3["model"] == m) & (df_b3["task"] == "reader") & (df_b3["axis"] == "valence")
            ]
            if len(sub) > 0:
                dr_p = sub["p_iut"].iloc[0]

        # Specificity reader valence diff
        spec_diff = np.nan
        if len(df_b4) > 0:
            sub = df_b4[
                (df_b4["model"] == m) & (df_b4["task"] == "reader") & (df_b4["axis"] == "valence")
            ]
            if len(sub) > 0:
                spec_diff = sub["difference"].iloc[0]

        # Coupling valence delta r
        coup_r = np.nan
        if len(df_b5) > 0:
            sub = df_b5[(df_b5["model"] == m) & (df_b5["axis"] == "valence")]
            if len(sub) > 0:
                coup_r = sub["delta_pearson_r"].iloc[0]

        summary_rows.append(
            {
                "model": m,
                "family": fam,
                "alignment": align,
                "human_corr_reader_v_r": emo_r,
                "sensitivity_reader_v_delta": sens_delta,
                "dose_response_reader_v_p_iut": dr_p,
                "specificity_reader_v_diff": spec_diff,
                "coupling_v_delta_r": coup_r,
            }
        )

    df_b_sum = pd.DataFrame(summary_rows)
    safe_save_csv(df_b_sum, tables_dir / "table_b_summary.csv")

    # 全レコードをDataFrame化
    df_records = pd.DataFrame(records)
    for col in PAPER_SUMMARY_COLUMNS:
        if col not in df_records.columns:
            df_records[col] = np.nan
    df_records = df_records[PAPER_SUMMARY_COLUMNS].copy()

    # stage summary 出力
    stage_out = out_dir / "stage_summaries" / "behavioral"
    stage_out.mkdir(parents=True, exist_ok=True)
    safe_save_csv(df_records, stage_out / "behavioral_paper_results.csv")

    errors = validate_paper_summary_df(df_records, strict=strict)
    if errors:
        qc_info["validation_errors"] = errors
        if strict:
            raise ValueError(f"Behavioral schema validation errors: {errors}")

    return df_records, qc_info


def main():
    parser = argparse.ArgumentParser(description="Build Behavioral Paper Summary")
    parser.add_argument("--behavioral-dir", type=str, default="behavioral")
    parser.add_argument("--out-dir", type=str, default="results/derived/paper_summary")
    parser.add_argument(
        "--strict", action="store_true", help="Fail on missing artifacts or validation errors"
    )
    args = parser.parse_args()

    beh_dir = Path(args.behavioral_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    manifest = PaperSummaryManifest()
    df_records, qc_info = build_behavioral_summary(
        behavioral_dir=beh_dir,
        out_dir=out_dir,
        manifest=manifest,
        strict=args.strict,
    )
    manifest.save(out_dir / "stage_summaries" / "behavioral" / "manifest_behavioral.json")

    print(f"Behavioral paper summary completed: {len(df_records)} records generated.")


if __name__ == "__main__":
    main()
