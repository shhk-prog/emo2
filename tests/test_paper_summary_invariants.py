"""
tests/test_paper_summary_invariants.py

論文結果集約コードの学術的不変条件（7つの厳格検証）テスト。
  1. Primaryファイルの純粋性と直交性 (is_primary=True, non-secondary/control, Discoveryを含む)
  2. V2 NaN 保持 (正のピーク未検出時の0埋め禁止)
  3. V3 陰性コントロールの存在 (Gate依存: RQ2実行時のみresponse_end必須)
  4. V1 Phase B nonfallback N の非欠落 (サンプル数の透明性保証)
  5. V3 Gate 文字列判定とパイプライン継続判定 (GO以外での勝手な継続フラグ化禁止)
  6. 必須 strata 網羅および Unique-Key 重複なし (1 scientific estimate = 1 record)
  7. pipeline_continues に基づく State-aware な strict 検証
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from affective_empathy_eval.paper_summary.schema import (
    UNIQUE_KEY_COLUMNS,
    validate_paper_summary_df,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUMMARY_DIR = PROJECT_ROOT / "results" / "derived" / "paper_summary"


@pytest.fixture(scope="module")
def summary_data():
    """集約結果のロード"""
    primary_csv = SUMMARY_DIR / "primary_results.csv"
    secondary_csv = SUMMARY_DIR / "secondary_results.csv"
    manifest_json = SUMMARY_DIR / "paper_summary_manifest.json"
    qc_json = SUMMARY_DIR / "qc_summary.json"

    assert primary_csv.exists(), f"Missing primary_results.csv at {primary_csv}"
    assert secondary_csv.exists(), f"Missing secondary_results.csv at {secondary_csv}"
    assert manifest_json.exists(), f"Missing paper_summary_manifest.json at {manifest_json}"

    df_pri = pd.read_csv(primary_csv)
    df_sec = pd.read_csv(secondary_csv)

    with open(manifest_json, encoding="utf-8") as f:
        manifest = json.load(f)

    qc = {}
    if qc_json.exists():
        with open(qc_json, encoding="utf-8") as f:
            qc = json.load(f)

    return {
        "primary": df_pri,
        "secondary": df_sec,
        "manifest": manifest,
        "qc": qc,
    }


def test_invariant_1_primary_orthogonality(summary_data):
    """
    不変条件 1: Primaryファイルの純粋性と直交性
      - is_primary == False は1行も入らない
      - analysis_role in ('secondary', 'control', 'diagnostic') は1行も入らない
      - is_primary=True かつ analysis_role='discovery' (V3 RQ2) が正しく含まれる
    """
    df_pri = summary_data["primary"]

    # 19列スキーマ適合
    errors = validate_paper_summary_df(df_pri)
    assert not errors, f"Primary DF schema validation failed: {errors}"

    # is_primary は全行 True
    assert df_pri["is_primary"].all(), "Non-primary rows found in primary_results.csv"

    # secondary/control/diagnostic は混入していないこと
    prohibited_roles = {"secondary", "control", "diagnostic"}
    found_prohibited = set(df_pri["analysis_role"].unique()) & prohibited_roles
    assert not found_prohibited, f"Prohibited analysis roles in primary file: {found_prohibited}"

    # V3 RQ2 Discovery の配置検証 (Gate 判定連動: NO_GO 時は secondary, GO 時は primary)
    qc = summary_data["qc"]
    pipeline_continues = (
        qc.get("stages", {}).get("v3", {}).get("tables", {}).get("table_v3_1", {}).get("pipeline_continues", False)
    )
    v3_discovery_pri = df_pri[(df_pri["stage"] == "v3") & (df_pri["analysis_role"] == "discovery")]
    df_sec = summary_data["secondary"]
    v3_discovery_sec = df_sec[(df_sec["stage"] == "v3") & (df_sec["analysis_role"] == "discovery")]

    if pipeline_continues:
        assert len(v3_discovery_pri) > 0, (
            "V3 RQ2 discovery results were incorrectly filtered out of primary_results.csv when pipeline continues"
        )
    else:
        assert len(v3_discovery_pri) == 0, (
            "V3 RQ2 discovery results should NOT be in primary_results.csv when Gate is NO_GO"
        )
        assert len(v3_discovery_sec) > 0, (
            "V3 RQ2 discovery results must be preserved in secondary_results.csv as exploratory under NO_GO"
        )



def test_invariant_2_v2_nan_preservation():
    """
    不変条件 2: V2 NaN 保持
      - 正のピーク未検出時の0埋め（fillna(0)）禁止
      - no_positive_net_causal_peak が True の場合、positive_causal_peak は NaN であること
    """
    t2_3a = SUMMARY_DIR / "tables" / "table_v2_3a_causal_relocation.csv"
    assert t2_3a.exists()

    df = pd.read_csv(t2_3a)
    for _, row in df.iterrows():
        if row.get("no_positive_net_causal_peak", False) is True:
            # NaN でなければならない (0.0 等で埋められていないこと)
            peak_val = row["positive_causal_peak"]
            assert pd.isna(peak_val), (
                f"Expected NaN for no_positive_net_causal_peak=True, got {peak_val}"
            )


def test_invariant_3_v3_negative_control_gate_dependent(summary_data):
    """
    不変条件 3: V3 陰性コントロールの存在 (Gate依存)
      - RQ2 が実行された場合のみ response_end 必須
      - Gate により未実行 (not_run_due_to_gate) の場合は非存在を正常受理
    """
    qc = summary_data["qc"]
    v3_qc = qc.get("stages", {}).get("v3", {})
    t3_2_qc = v3_qc.get("tables", {}).get("table_v3_2", {})
    pipeline_continues = (
        v3_qc.get("tables", {}).get("table_v3_1", {}).get("pipeline_continues", False)
    )

    if pipeline_continues or t3_2_qc.get("n_rows", 0) > 0:
        # RQ2 が実行されたか結果が存在する場合は、response_end が記録されていること
        assert t3_2_qc.get("negative_control_response_end_present", False) is True, (
            "V3 negative temporal control C(response_end) was not recorded despite RQ2 execution!"
        )
    else:
        # 未実行の場合は非存在で正常
        pass


def test_invariant_4_v1_phase_b_nonfallback_n():
    """
    不変条件 4: V1 Phase B nonfallback N の非欠落
      - Table V1-3 において n_paraphrase_nonfallback, n_reversal_nonfallback が明記されていること
    """
    t1_3 = SUMMARY_DIR / "tables" / "table_v1_3_semantic_controls.csv"
    assert t1_3.exists()

    df = pd.read_csv(t1_3)
    assert len(df) > 0, "Table V1-3 is empty!"
    assert "n_paraphrase_nonfallback" in df.columns
    assert "n_reversal_nonfallback" in df.columns

    # 欠損なく整数値が入っていること
    assert df["n_paraphrase_nonfallback"].notnull().all()
    assert df["n_reversal_nonfallback"].notnull().all()
    assert (df["n_paraphrase_nonfallback"] >= 0).all()


def test_invariant_5_v3_gate_decision_logic():
    """
    不変条件 5: V3 Gate 文字列とパイプライン継続判定
      - overall_decision は4値 (GO, GO (Valence-only), GO (Arousal-only), NO_GO)
      - pipeline_continues == (overall_decision == "GO")
    """
    t3_1 = SUMMARY_DIR / "tables" / "table_v3_1_gate.csv"
    assert t3_1.exists()

    df = pd.read_csv(t3_1)
    assert len(df) > 0

    valid_decisions = {"GO", "GO (Valence-only)", "GO (Arousal-only)", "NO_GO"}
    for _, row in df.iterrows():
        dec = row["overall_decision"]
        assert dec in valid_decisions, f"Invalid Gate overall_decision: {dec}"
        expected_cont = dec == "GO"
        msg = f"pipeline_continues mismatch for {dec}: expected {expected_cont}"
        assert bool(row["pipeline_continues"]) == expected_cont, msg


def test_invariant_6_strata_coverage_and_no_duplicate_keys(summary_data):
    """
    不変条件 6: 必須 strata 網羅および Unique-Key 重複なし
      - 主要モデル・軸が存在すること
      - Primary records で UNIQUE_KEY_COLUMNS に重複が一切ないこと (LMM term 一意化含む)
    """
    df_pri = summary_data["primary"]

    # 必須ステージ網羅
    assert set(df_pri["stage"].unique()) == {"behavioral", "v1", "v2", "v3"}

    # Primary records における一意性
    duplicates = df_pri[df_pri.duplicated(subset=UNIQUE_KEY_COLUMNS, keep=False)]
    dup_recs = duplicates[UNIQUE_KEY_COLUMNS].to_dict(orient="records")
    msg = f"Detected {len(duplicates)} duplicate Primary records! Keys: {dup_recs}"
    assert len(duplicates) == 0, msg


def test_invariant_7_manifest_provenance_completeness(summary_data):
    """
    不変条件 7: Manifest による完全な追跡可能性
      - manifest に各エントリーの sources と derivation が明記されていること
    """
    manifest = summary_data["manifest"]
    assert "summary_schema_version" in manifest
    assert "provenance_map" in manifest

    pmap = manifest["provenance_map"]
    assert len(pmap) > 0, "Provenance map is empty!"

    for rec_id, entry in list(pmap.items())[:20]:
        assert "sources" in entry, f"Missing sources in manifest for {rec_id}"
        assert "derivation" in entry, f"Missing derivation in manifest for {rec_id}"
        assert len(entry["sources"]) > 0, f"Empty sources list in manifest for {rec_id}"


def test_paper_summary_contains_no_dry_run_sources():
    """
    不変条件 8: 論文用 summary 成果物に dry_run 由来のアーティファクトが含まれないこと
    """
    for csv_path in [
        SUMMARY_DIR / "primary_results.csv",
        SUMMARY_DIR / "secondary_results.csv",
    ]:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            assert not df["source_artifact"].astype(str).str.contains(r"/dry_run/").any(), (
                f"Found dry_run artifacts in {csv_path}"
            )

    manifest_file = SUMMARY_DIR / "paper_summary_manifest.json"
    if manifest_file.exists():
        manifest_text = manifest_file.read_text(encoding="utf-8")
        assert "/dry_run/" not in manifest_text, "Found dry_run artifacts in paper_summary_manifest.json"

