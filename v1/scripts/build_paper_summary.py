#!/usr/bin/env python3
"""
v1/scripts/build_paper_summary.py

§5 V1 結果章用サマリー生成スクリプト（Read-only presentation layer）。
既存の検証済み結果から以下を抽出・フォーマットして出力する：
  - Table V1-1: Shared Decodability & Peak Summary + AIPsy Probes
  - Table V1-2: Shared Geometry Profile & Alignment Gain
  - Table V1-3: Semantic / Contextual Controls (Paraphrase, Shuffle, Reversal + nonfallback N)
  - Table V1-4: Shared Causal Map Overlap & Peak Profile
  - Table V1-5: Causal Interchangeability (Primary alpha=1.0 at peak layer, alpha sweep for figures)
  - Table V1-6: Task-Specific Specialization (Distinct site selection & LMM interaction)
  - Figure Data: Figure V1-1〜V1-4 用 CSV
  - 共通19列スキーマレコードおよび Manifest への Provenance 登録
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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


def build_v1_summary(
    v1_dir: Path,
    out_dir: Path,
    manifest: PaperSummaryManifest,
    strict: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    qc_info: dict[str, Any] = {"stage": "v1", "tables": {}}

    phase_a_dir = v1_dir / "results" / "derived" / "v1_phase_a"
    phase_b_dir = v1_dir / "results" / "derived" / "v1_phase_b"
    phase_c_dir = v1_dir / "results" / "derived" / "v1_phase_c_prompt_end"

    tables_dir = out_dir / "tables"
    fig_dir = out_dir / "figure_data"
    tables_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 対象モデル一覧の検出
    available_models = (
        sorted([d.name for d in phase_a_dir.iterdir() if d.is_dir()])
        if phase_a_dir.exists()
        else []
    )
    if not available_models and strict:
        raise FileNotFoundError(f"No model directories found in {phase_a_dir}")

    # =========================================================================
    # 1. Table V1-1: Shared Decodability (EmoBank Peak Decodability)
    # =========================================================================
    table_v1_1_rows = []
    fig_v1_1_rows = []
    aipsy_probe_rows = []

    for m_str in available_models:
        fam, align = parse_model_info(m_str)
        m_dir = phase_a_dir / m_str
        e1_csv = m_dir / "e1_emobank_decodability.csv"

        if e1_csv.exists():
            df_e1 = pd.read_csv(e1_csv)

            for ax, ax_label in [("V", "valence"), ("A", "arousal")]:
                r2_r_col = f"target_{ax}_human_r2_reader"
                r2_s_col = f"target_{ax}_human_r2_self"
                pr_r_col = f"target_{ax}_human_pr_reader"
                pr_s_col = f"target_{ax}_human_pr_self"

                # Reader peak
                r2_r_vals = df_e1[r2_r_col].values
                peak_l_r = int(np.nanargmax(r2_r_vals)) if len(r2_r_vals) > 0 else np.nan
                peak_d_r = (
                    float(df_e1["relative_depth"].iloc[peak_l_r])
                    if not np.isnan(peak_l_r)
                    else np.nan
                )
                peak_r2_r = float(r2_r_vals[peak_l_r]) if not np.isnan(peak_l_r) else np.nan
                pr_r_val = (
                    float(df_e1[pr_r_col].iloc[peak_l_r])
                    if pr_r_col in df_e1 and not np.isnan(peak_l_r)
                    else np.nan
                )

                # Self peak
                r2_s_vals = df_e1[r2_s_col].values
                peak_l_s = int(np.nanargmax(r2_s_vals)) if len(r2_s_vals) > 0 else np.nan
                peak_d_s = (
                    float(df_e1["relative_depth"].iloc[peak_l_s])
                    if not np.isnan(peak_l_s)
                    else np.nan
                )
                peak_r2_s = float(r2_s_vals[peak_l_s]) if not np.isnan(peak_l_s) else np.nan
                pr_s_val = (
                    float(df_e1[pr_s_col].iloc[peak_l_s])
                    if pr_s_col in df_e1 and not np.isnan(peak_l_s)
                    else np.nan
                )

                peak_dist = (
                    abs(peak_d_r - peak_d_s)
                    if (not np.isnan(peak_d_r) and not np.isnan(peak_d_s))
                    else np.nan
                )

                table_v1_1_rows.append(
                    {
                        "model": m_str,
                        "family": fam,
                        "alignment": align,
                        "task": "reader",
                        "axis": ax_label,
                        "peak_layer": peak_l_r,
                        "peak_relative_depth": peak_d_r,
                        "peak_r2": peak_r2_r,
                        "pearson": pr_r_val,
                        "peak_distance_to_self": peak_dist,
                    }
                )
                table_v1_1_rows.append(
                    {
                        "model": m_str,
                        "family": fam,
                        "alignment": align,
                        "task": "self",
                        "axis": ax_label,
                        "peak_layer": peak_l_s,
                        "peak_relative_depth": peak_d_s,
                        "peak_r2": peak_r2_s,
                        "pearson": pr_s_val,
                        "peak_distance_to_reader": peak_dist,
                    }
                )

                # Schema record for Reader peak R2
                records.append(
                    PaperSummaryRecord(
                        stage="v1",
                        rq="rq1_shared_decodability",
                        family=fam,
                        alignment=align,
                        task="reader",
                        axis=ax_label,
                        condition="emobank_linear_probe",
                        metric="peak_r2",
                        estimate=peak_r2_r,
                        n=1000,
                        is_primary=True,
                        analysis_role="primary",
                        source_artifact=str(e1_csv.relative_to(v1_dir.parent)),
                        source_key=r2_r_col,
                    ).to_dict()
                )

                # Schema record for Self peak R2
                records.append(
                    PaperSummaryRecord(
                        stage="v1",
                        rq="rq1_shared_decodability",
                        family=fam,
                        alignment=align,
                        task="self",
                        axis=ax_label,
                        condition="emobank_linear_probe",
                        metric="peak_r2",
                        estimate=peak_r2_s,
                        n=1000,
                        is_primary=True,
                        analysis_role="primary",
                        source_artifact=str(e1_csv.relative_to(v1_dir.parent)),
                        source_key=r2_s_col,
                    ).to_dict()
                )

                # Schema record for peak distance |d_R - d_S|
                records.append(
                    PaperSummaryRecord(
                        stage="v1",
                        rq="rq1_shared_decodability",
                        family=fam,
                        alignment=align,
                        task="reader_self",
                        axis=ax_label,
                        condition="emobank_linear_probe",
                        metric="peak_depth_distance",
                        estimate=peak_dist,
                        is_primary=False,
                        analysis_role="secondary",
                        source_artifact=str(e1_csv.relative_to(v1_dir.parent)),
                        source_key=json.dumps([r2_r_col, r2_s_col]),
                    ).to_dict()
                )

                manifest.register(
                    record_id=f"v1.table_v1_1.{m_str}_{ax_label}_peak_depth_distance",
                    sources=[
                        {"artifact": str(e1_csv.relative_to(v1_dir.parent)), "key": r2_r_col},
                        {"artifact": str(e1_csv.relative_to(v1_dir.parent)), "key": r2_s_col},
                    ],
                    derivation="|argmax(r2_r)/(L-1) - argmax(r2_s)/(L-1)|",
                    notes="Peak relative depth distance between Reader and Self probes",
                )

                # Figure V1-1 data (full curves)
                for _, l_row in df_e1.iterrows():
                    fig_v1_1_rows.append(
                        {
                            "model": m_str,
                            "family": fam,
                            "alignment": align,
                            "axis": ax_label,
                            "layer": int(l_row["layer"]),
                            "relative_depth": float(l_row["relative_depth"]),
                            "r2_reader": float(l_row[r2_r_col]),
                            "r2_self": float(l_row[r2_s_col]),
                        }
                    )

        # AIPsy probes (classification, intensity)
        aipsy_clf_csv = m_dir / "e1_aipsy_classification.csv"
        if aipsy_clf_csv.exists():
            df_clf = pd.read_csv(aipsy_clf_csv)
            for _, c_row in df_clf.iterrows():
                aipsy_probe_rows.append(
                    {
                        "model": m_str,
                        "family": fam,
                        "alignment": align,
                        "probe_type": "clinical_vs_neutral_classification",
                        "layer": int(c_row["layer"]),
                        "relative_depth": float(c_row["relative_depth"]),
                        "roc_auc": float(c_row.get("roc_auc", np.nan)),
                        "balanced_acc": float(c_row.get("balanced_acc", np.nan)),
                        "macro_f1": float(c_row.get("macro_f1", np.nan)),
                    }
                )

    df_v1_1 = pd.DataFrame(table_v1_1_rows)
    safe_save_csv(df_v1_1, tables_dir / "table_v1_1_peak_decodability.csv")
    safe_save_csv(pd.DataFrame(fig_v1_1_rows), fig_dir / "figure_v1_1_decodability.csv")
    if aipsy_probe_rows:
        safe_save_csv(pd.DataFrame(aipsy_probe_rows), tables_dir / "table_v1_1_aipsy_probes.csv")
    qc_info["tables"]["table_v1_1"] = {"n_rows": len(df_v1_1)}

    # =========================================================================
    # 2. Table V1-2: Shared Geometry (Cross-decoding, RSA, Alignment)
    # =========================================================================
    table_v1_2_rows = []
    fig_v1_2_rows = []

    for m_str in available_models:
        fam, align = parse_model_info(m_str)
        m_dir = phase_a_dir / m_str
        e2_csv = m_dir / "e2_emobank_geometry.csv"

        if e2_csv.exists():
            df_e2 = pd.read_csv(e2_csv)
            for _, row in df_e2.iterrows():
                t_raw = str(row["target"])
                ax_label = "valence" if "val" in t_raw.lower() else "arousal"
                l_val = int(row["layer"])
                rel_d = float(row["relative_depth"])

                w_r = float(row["r2_within_r"])
                w_s = float(row["r2_within_s"])
                c_rs = float(row["r2_cross_r_to_s"])
                c_sr = float(row["r2_cross_s_to_r"])
                d_trans = float(
                    row.get(
                        "direct_transfer_score_clipped", row.get("direct_transfer_score", np.nan)
                    )
                )
                rsa_val = float(row["rsa_correlation"])
                align_r2 = float(row["r2_aligned_transfer"])
                align_gain = float(row["alignment_gain"])

                table_v1_2_rows.append(
                    {
                        "model": m_str,
                        "family": fam,
                        "alignment": align,
                        "axis": ax_label,
                        "layer": l_val,
                        "relative_depth": rel_d,
                        "within_reader_r2": w_r,
                        "within_self_r2": w_s,
                        "cross_r_to_s": c_rs,
                        "cross_s_to_r": c_sr,
                        "direct_mean": d_trans,
                        "rsa": rsa_val,
                        "aligned_r2": align_r2,
                        "alignment_gain": align_gain,
                    }
                )

                fig_v1_2_rows.append(
                    {
                        "model": m_str,
                        "family": fam,
                        "alignment": align,
                        "axis": ax_label,
                        "relative_depth": rel_d,
                        "direct_transfer": d_trans,
                        "rsa": rsa_val,
                        "alignment_gain": align_gain,
                    }
                )

            # Peak alignment gain / RSA as primary summaries for Table V1-2
            df_sub_v = df_e2[df_e2["target"].str.contains("Val", case=False, na=False)]
            if len(df_sub_v) > 0:
                max_gain_v = float(np.nanmax(df_sub_v["alignment_gain"]))
                max_rsa_v = float(np.nanmax(df_sub_v["rsa_correlation"]))
                records.append(
                    PaperSummaryRecord(
                        stage="v1",
                        rq="rq2_shared_geometry",
                        family=fam,
                        alignment=align,
                        task="reader_self",
                        axis="valence",
                        condition="emobank_procrustes_alignment",
                        metric="max_alignment_gain",
                        estimate=max_gain_v,
                        is_primary=True,
                        analysis_role="primary",
                        source_artifact=str(e2_csv.relative_to(v1_dir.parent)),
                        source_key="alignment_gain",
                    ).to_dict()
                )

                records.append(
                    PaperSummaryRecord(
                        stage="v1",
                        rq="rq2_shared_geometry",
                        family=fam,
                        alignment=align,
                        task="reader_self",
                        axis="valence",
                        condition="emobank_rsa",
                        metric="max_rsa_correlation",
                        estimate=max_rsa_v,
                        is_primary=True,
                        analysis_role="primary",
                        source_artifact=str(e2_csv.relative_to(v1_dir.parent)),
                        source_key="rsa_correlation",
                    ).to_dict()
                )

    df_v1_2 = pd.DataFrame(table_v1_2_rows)
    safe_save_csv(df_v1_2, tables_dir / "table_v1_2_shared_geometry.csv")
    safe_save_csv(pd.DataFrame(fig_v1_2_rows), fig_dir / "figure_v1_2_geometry.csv")
    qc_info["tables"]["table_v1_2"] = {"n_rows": len(df_v1_2)}

    # =========================================================================
    # 3. Table V1-3: Semantic / Contextual Controls (Paraphrase, Shuffle, Reversal + nonfallback N)
    # =========================================================================
    table_v1_3_rows = []

    if phase_b_dir.exists():
        for task_type in ("reader", "self"):
            t_dir = phase_b_dir / task_type
            if not t_dir.exists():
                continue
            for m_dir in t_dir.iterdir():
                if not m_dir.is_dir():
                    continue
                m_str = m_dir.name
                fam, align = parse_model_info(m_str)
                b_csv = m_dir / "phase_b_semantic_controls.csv"

                if b_csv.exists():
                    df_b = pd.read_csv(b_csv)
                    if len(df_b) > 0:
                        r0 = df_b.iloc[0]
                        orig_acc = float(r0["acc_original_minimal_pair"])
                        para_acc = float(r0["acc_pair_aware_held_out_paraphrase"])
                        n_para_nonfallback = int(r0["n_nonfallback_paraphrase_pairs"])
                        shuf_acc = float(r0["acc_word_shuffle"])
                        shuf_drop = orig_acc - shuf_acc
                        rev_drop = float(r0["outcome_reversal_prob_drop"])
                        n_rev_nonfallback = int(r0["n_nonfallback_reversal_pairs"])

                        table_v1_3_rows.append(
                            {
                                "model": m_str,
                                "family": fam,
                                "alignment": align,
                                "task": task_type,
                                "original_bal_acc": orig_acc,
                                "paraphrase_bal_acc": para_acc,
                                "n_paraphrase_nonfallback": n_para_nonfallback,
                                "shuffle_bal_acc": shuf_acc,
                                "shuffle_drop": shuf_drop,
                                "reversal_delta_probability": rev_drop,
                                "n_reversal_nonfallback": n_rev_nonfallback,
                            }
                        )

                        # Primary: Shuffle drop and outcome reversal drop
                        records.append(
                            PaperSummaryRecord(
                                stage="v1",
                                rq="rq3_semantic_controls",
                                family=fam,
                                alignment=align,
                                task=task_type,
                                axis="valence_arousal_joint",
                                condition="word_shuffle_drop",
                                metric="shuffle_drop_accuracy",
                                estimate=shuf_drop,
                                is_primary=True,
                                analysis_role="primary",
                                source_artifact=str(b_csv.relative_to(v1_dir.parent)),
                                source_key=json.dumps(
                                    ["acc_original_minimal_pair", "acc_word_shuffle"]
                                ),
                            ).to_dict()
                        )

                        # Nonfallback N as diagnostic
                        records.append(
                            PaperSummaryRecord(
                                stage="v1",
                                rq="rq3_semantic_controls",
                                family=fam,
                                alignment=align,
                                task=task_type,
                                axis="valence_arousal_joint",
                                condition="paraphrase_sample_audit",
                                metric="n_paraphrase_nonfallback",
                                estimate=float(n_para_nonfallback),
                                n=n_para_nonfallback,
                                is_primary=False,
                                analysis_role="diagnostic",
                                source_artifact=str(b_csv.relative_to(v1_dir.parent)),
                                source_key="n_nonfallback_paraphrase_pairs",
                            ).to_dict()
                        )

                        manifest.register(
                            record_id=f"v1.table_v1_3.{m_str}_{task_type}_shuffle_drop",
                            sources=[
                                {
                                    "artifact": str(b_csv.relative_to(v1_dir.parent)),
                                    "key": "acc_original_minimal_pair",
                                },
                                {
                                    "artifact": str(b_csv.relative_to(v1_dir.parent)),
                                    "key": "acc_word_shuffle",
                                },
                            ],
                            derivation="acc_original_minimal_pair - acc_word_shuffle",
                            notes="Semantic syntax disruption drop",
                        )

    df_v1_3 = pd.DataFrame(table_v1_3_rows)
    safe_save_csv(df_v1_3, tables_dir / "table_v1_3_semantic_controls.csv")
    qc_info["tables"]["table_v1_3"] = {"n_rows": len(df_v1_3)}

    # =========================================================================
    # 4. Table V1-4: Shared Causal Map (E3 Prompt-End Causal Map)
    # =========================================================================
    table_v1_4_rows = []
    fig_v1_3_rows = []

    for m_str in available_models:
        fam, align = parse_model_info(m_str)
        m_dir = phase_c_dir / m_str
        e3_csv = m_dir / "e3_causal_map.csv"

        if e3_csv.exists():
            df_e3 = pd.read_csv(e3_csv)

            mag_r = df_e3["magnitude_reader"].values
            mag_s = df_e3["magnitude_self"].values

            peak_l_r = int(np.nanargmax(mag_r)) if len(mag_r) > 0 else np.nan
            peak_d_r = (
                float(df_e3["relative_depth"].iloc[peak_l_r]) if not np.isnan(peak_l_r) else np.nan
            )
            peak_mag_r = float(mag_r[peak_l_r]) if not np.isnan(peak_l_r) else np.nan
            mean_mag_r = float(np.nanmean(mag_r))

            peak_l_s = int(np.nanargmax(mag_s)) if len(mag_s) > 0 else np.nan
            peak_d_s = (
                float(df_e3["relative_depth"].iloc[peak_l_s]) if not np.isnan(peak_l_s) else np.nan
            )
            peak_mag_s = float(mag_s[peak_l_s]) if not np.isnan(peak_l_s) else np.nan
            mean_mag_s = float(np.nanmean(mag_s))

            rank_rho = np.nan
            if len(mag_r) >= 3 and np.nanstd(mag_r) > 0 and np.nanstd(mag_s) > 0:
                valid_mask = ~np.isnan(mag_r) & ~np.isnan(mag_s)
                if np.sum(valid_mask) >= 3:
                    rank_rho, _ = spearmanr(mag_r[valid_mask], mag_s[valid_mask])

            mean_cos = (
                float(np.nanmean(df_e3["directional_cosine_similarity"]))
                if "directional_cosine_similarity" in df_e3
                else np.nan
            )

            table_v1_4_rows.append(
                {
                    "model": m_str,
                    "family": fam,
                    "alignment": align,
                    "task": "reader",
                    "peak_layer": peak_l_r,
                    "peak_depth": peak_d_r,
                    "peak_causal_magnitude": peak_mag_r,
                    "mean_causal_magnitude": mean_mag_r,
                    "reader_self_rank_spearman": float(rank_rho)
                    if not np.isnan(rank_rho)
                    else np.nan,
                    "mean_direction_cosine": mean_cos,
                }
            )
            table_v1_4_rows.append(
                {
                    "model": m_str,
                    "family": fam,
                    "alignment": align,
                    "task": "self",
                    "peak_layer": peak_l_s,
                    "peak_depth": peak_d_s,
                    "peak_causal_magnitude": peak_mag_s,
                    "mean_causal_magnitude": mean_mag_s,
                    "reader_self_rank_spearman": float(rank_rho)
                    if not np.isnan(rank_rho)
                    else np.nan,
                    "mean_direction_cosine": mean_cos,
                }
            )

            # Primary: Reader and Self peak causal magnitude
            records.append(
                PaperSummaryRecord(
                    stage="v1",
                    rq="rq4_shared_causal_map",
                    family=fam,
                    alignment=align,
                    task="reader",
                    axis="valence_arousal_joint",
                    condition="prompt_end_causal_trace",
                    metric="peak_causal_magnitude",
                    estimate=peak_mag_r,
                    is_primary=True,
                    analysis_role="primary",
                    source_artifact=str(e3_csv.relative_to(v1_dir.parent)),
                    source_key="magnitude_reader",
                ).to_dict()
            )

            records.append(
                PaperSummaryRecord(
                    stage="v1",
                    rq="rq4_shared_causal_map",
                    family=fam,
                    alignment=align,
                    task="self",
                    axis="valence_arousal_joint",
                    condition="prompt_end_causal_trace",
                    metric="peak_causal_magnitude",
                    estimate=peak_mag_s,
                    is_primary=True,
                    analysis_role="primary",
                    source_artifact=str(e3_csv.relative_to(v1_dir.parent)),
                    source_key="magnitude_self",
                ).to_dict()
            )

            # Secondary: Reader-Self causal rank correlation
            records.append(
                PaperSummaryRecord(
                    stage="v1",
                    rq="rq4_shared_causal_map",
                    family=fam,
                    alignment=align,
                    task="reader_self",
                    axis="valence_arousal_joint",
                    condition="prompt_end_causal_trace",
                    metric="causal_profile_spearman",
                    estimate=float(rank_rho) if not np.isnan(rank_rho) else np.nan,
                    is_primary=False,
                    analysis_role="secondary",
                    source_artifact=str(e3_csv.relative_to(v1_dir.parent)),
                    source_key=json.dumps(["magnitude_reader", "magnitude_self"]),
                ).to_dict()
            )

            for _, l_row in df_e3.iterrows():
                fig_v1_3_rows.append(
                    {
                        "model": m_str,
                        "family": fam,
                        "alignment": align,
                        "layer": int(l_row["layer"]),
                        "relative_depth": float(l_row["relative_depth"]),
                        "magnitude_reader": float(l_row["magnitude_reader"]),
                        "magnitude_self": float(l_row["magnitude_self"]),
                        "direction_cosine": float(
                            l_row.get("directional_cosine_similarity", np.nan)
                        ),
                    }
                )

    df_v1_4 = pd.DataFrame(table_v1_4_rows)
    safe_save_csv(df_v1_4, tables_dir / "table_v1_4_causal_map.csv")
    safe_save_csv(pd.DataFrame(fig_v1_3_rows), fig_dir / "figure_v1_3_causal_map.csv")
    qc_info["tables"]["table_v1_4"] = {"n_rows": len(df_v1_4)}

    # =========================================================================
    # 5. Table V1-5: Causal Interchangeability (E4 Interchangeability)
    # =========================================================================
    table_v1_5_rows = []
    fig_v1_4_rows = []

    for m_str in available_models:
        fam, align = parse_model_info(m_str)
        m_dir = phase_c_dir / m_str
        e4_csv = m_dir / "e4_interchangeability_results.csv"

        if e4_csv.exists():
            df_e4 = pd.read_csv(e4_csv)
            # Figure data for all alpha
            for _, row in df_e4.iterrows():
                fig_v1_4_rows.append(
                    {
                        "model": m_str,
                        "family": fam,
                        "alignment": align,
                        "layer": int(row["layer"]),
                        "relative_depth": float(row["relative_depth"]),
                        "alpha": float(row["alpha"]),
                        "matched_shift_V": float(row["matched_shift_V"]),
                        "random_shift_V": float(row["random_shift_V"]),
                        "specificity_V": float(row["aligned_specificity_V"]),
                        "matched_shift_A": float(row["matched_shift_A"]),
                        "random_shift_A": float(row["random_shift_A"]),
                        "specificity_A": float(row["aligned_specificity_A"]),
                    }
                )

            # Primary: alpha = 1.0 (and confirmatory layer if specified)
            df_a1 = df_e4[df_e4["alpha"] == 1.0]
            if len(df_a1) > 0:
                row_prim = df_a1.iloc[0]  # or peak layer
                l_val = int(row_prim["layer"])

                for ax in ("V", "A"):
                    ax_label = "valence" if ax == "V" else "arousal"
                    spec = float(row_prim[f"aligned_specificity_{ax}"])
                    ci_l = float(row_prim[f"aligned_ci_95_low_{ax}"])
                    ci_h = float(row_prim[f"aligned_ci_95_high_{ax}"])
                    pval = float(row_prim[f"aligned_p_val_{ax}"])
                    perm_p = float(row_prim[f"aligned_permutation_p_{ax}"])
                    qval = float(row_prim.get(f"aligned_p_fdr_{ax}", np.nan))
                    tr = float(row_prim.get(f"transfer_ratio_{ax}", np.nan))

                    table_v1_5_rows.append(
                        {
                            "model": m_str,
                            "family": fam,
                            "alignment": align,
                            "axis": ax_label,
                            "layer": l_val,
                            "alpha": 1.0,
                            "matched_effect": float(row_prim[f"aligned_matched_shift_{ax}"]),
                            "random_mean": float(row_prim[f"aligned_random_shift_{ax}"]),
                            "specificity": spec,
                            "ci_low": ci_l,
                            "ci_high": ci_h,
                            "paired_t": float(row_prim.get(f"aligned_cohen_dz_{ax}", np.nan)),
                            "p": pval,
                            "permutation_p": perm_p,
                            "q": qval,
                            "transfer_ratio": tr,
                        }
                    )

                    records.append(
                        PaperSummaryRecord(
                            stage="v1",
                            rq="rq5_causal_interchangeability",
                            family=fam,
                            alignment=align,
                            task="reader_to_self",
                            axis=ax_label,
                            condition="alpha_1.0_matched_patching",
                            metric="specificity",
                            estimate=spec,
                            ci_low=ci_l,
                            ci_high=ci_h,
                            p=pval,
                            q=qval,
                            is_primary=True,
                            analysis_role="primary",
                            source_artifact=str(e4_csv.relative_to(v1_dir.parent)),
                            source_key=f"aligned_specificity_{ax}",
                        ).to_dict()
                    )

                    manifest.register(
                        record_id=f"v1.table_v1_5.{m_str}_{ax_label}_specificity",
                        sources=[
                            {
                                "artifact": str(e4_csv.relative_to(v1_dir.parent)),
                                "key": f"aligned_specificity_{ax}",
                            }
                        ],
                        derivation="matched_effect - mean(random_derangements)",
                        notes="Reader-to-Self causal specificity at alpha=1.0",
                    )

    df_v1_5 = pd.DataFrame(table_v1_5_rows)
    safe_save_csv(df_v1_5, tables_dir / "table_v1_5_interchangeability.csv")
    safe_save_csv(pd.DataFrame(fig_v1_4_rows), fig_dir / "figure_v1_4_interchangeability.csv")
    qc_info["tables"]["table_v1_5"] = {"n_rows": len(df_v1_5)}

    # =========================================================================
    # 6. Table V1-6: Task-Specific Specialization (E6 Specialization & LMM)
    # =========================================================================
    table_v1_6_rows = []

    for m_str in available_models:
        fam, align = parse_model_info(m_str)
        m_dir = phase_c_dir / m_str
        e6_json = m_dir / "e6_lmm_results.json"

        if e6_json.exists():
            try:
                with open(e6_json) as f:
                    dict_e6 = json.load(f)
            except Exception:
                dict_e6 = None

            if dict_e6 is not None:
                has_crossover = dict_e6.get("has_crossover", False)
                reader_layer = dict_e6.get("reader_layer", None)
                self_layer = dict_e6.get("self_layer", None)

                site_status = (
                    f"Identified (L{reader_layer}, L{self_layer})"
                    if (reader_layer is not None and self_layer is not None)
                    else "No distinct sites identified"
                )

                coef_inter = float(dict_e6.get("coef_interaction", np.nan))
                p_inter = float(dict_e6.get("p_interaction", np.nan))

                means = dict_e6.get("cell_means", {})
                r_rsite = float(means.get("Reader_ReaderSite", np.nan))
                s_rsite = float(means.get("Self_ReaderSite", np.nan))
                r_ssite = float(means.get("Reader_SelfSite", np.nan))
                s_ssite = float(means.get("Self_SelfSite", np.nan))

                table_v1_6_rows.append(
                    {
                        "model": m_str,
                        "family": fam,
                        "alignment": align,
                        "site_selection": site_status,
                        "reader_layer": reader_layer if reader_layer is not None else "None",
                        "self_layer": self_layer if self_layer is not None else "None",
                        "has_crossover": has_crossover,
                        "interaction_beta": coef_inter,
                        "interaction_p": p_inter,
                        "reader_ablation_on_reader": r_rsite,
                        "reader_ablation_on_self": s_rsite,
                        "self_ablation_on_reader": r_ssite,
                        "self_ablation_on_self": s_ssite,
                    }
                )

                # Schema record for Task x SiteType interaction beta
                records.append(
                    PaperSummaryRecord(
                        stage="v1",
                        rq="rq6_specialization",
                        family=fam,
                        alignment=align,
                        task="reader_and_self",
                        axis="valence_arousal_joint",
                        condition="targeted_site_ablation",
                        metric="interaction_beta",
                        estimate=coef_inter,
                        value_text=site_status,
                        p=p_inter,
                        is_primary=True,
                        analysis_role="primary",
                        source_artifact=str(e6_json.relative_to(v1_dir.parent)),
                        source_key="coef_interaction",
                    ).to_dict()
                )

                manifest.register(
                    record_id=f"v1.table_v1_6.{m_str}_interaction",
                    sources=[
                        {
                            "artifact": str(e6_json.relative_to(v1_dir.parent)),
                            "key": "coef_interaction",
                        }
                    ],
                    derivation="direct_copy",
                    notes="LMM task x site_type interaction effect",
                )

    df_v1_6 = pd.DataFrame(table_v1_6_rows)
    safe_save_csv(df_v1_6, tables_dir / "table_v1_6_specialization.csv")
    qc_info["tables"]["table_v1_6"] = {"n_rows": len(df_v1_6)}

    # 全レコードをDataFrame化
    df_records = pd.DataFrame(records)
    for col in PAPER_SUMMARY_COLUMNS:
        if col not in df_records.columns:
            df_records[col] = np.nan
    df_records = df_records[PAPER_SUMMARY_COLUMNS].copy()

    # stage summary 出力
    stage_out = out_dir / "stage_summaries" / "v1"
    stage_out.mkdir(parents=True, exist_ok=True)
    safe_save_csv(df_records, stage_out / "v1_paper_results.csv")

    errors = validate_paper_summary_df(df_records, strict=strict)
    if errors:
        qc_info["validation_errors"] = errors
        if strict:
            raise ValueError(f"V1 schema validation errors: {errors}")

    return df_records, qc_info


def main():
    parser = argparse.ArgumentParser(description="Build V1 Paper Summary")
    parser.add_argument("--v1-dir", type=str, default="v1")
    parser.add_argument("--out-dir", type=str, default="results/derived/paper_summary")
    parser.add_argument(
        "--strict", action="store_true", help="Fail on missing artifacts or validation errors"
    )
    args = parser.parse_args()

    v1_dir = Path(args.v1_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    manifest = PaperSummaryManifest()
    df_records, qc_info = build_v1_summary(
        v1_dir=v1_dir,
        out_dir=out_dir,
        manifest=manifest,
        strict=args.strict,
    )
    manifest.save(out_dir / "stage_summaries" / "v1" / "manifest_v1.json")

    print(f"V1 paper summary completed: {len(df_records)} records generated.")


if __name__ == "__main__":
    main()
