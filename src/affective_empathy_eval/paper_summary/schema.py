"""
src/affective_empathy_eval/paper_summary/schema.py

共通19列スキーマおよび関連定数・バリデータの定義。
論文用結果集約コードのPresentation層における正本スキーマ。
"""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

# 共通19列スキーマ
PAPER_SUMMARY_COLUMNS: list[str] = [
    "stage",
    "rq",
    "family",
    "alignment",
    "task",
    "axis",
    "condition",
    "metric",
    "estimate",
    "value_text",
    "ci_low",
    "ci_high",
    "p",
    "q",
    "n",
    "is_primary",
    "analysis_role",
    "source_artifact",
    "source_key",
]

# 許容される analysis_role
VALID_ANALYSIS_ROLES = {
    "primary",
    "confirmatory",
    "discovery",
    "exploratory",
    "secondary",
    "control",
    "diagnostic",
}

# 許容される stage
VALID_STAGES = {
    "behavioral",
    "v1",
    "v2",
    "v3",
}

# 許容される alignment
VALID_ALIGNMENTS = {
    "base",
    "instruct",
    "both",
    "all",
    "none",
}

# Unique Key カラム（同一レコード重複検出用）
UNIQUE_KEY_COLUMNS = [
    "stage",
    "rq",
    "family",
    "alignment",
    "task",
    "axis",
    "condition",
    "metric",
]


@dataclass
class PaperSummaryRecord:
    stage: str
    rq: str
    family: str
    alignment: str
    task: str
    axis: str
    condition: str
    metric: str
    estimate: float | None = np.nan
    value_text: str = ""
    ci_low: float | None = np.nan
    ci_high: float | None = np.nan
    p: float | None = np.nan
    q: float | None = np.nan
    n: int | None = None
    is_primary: bool = False
    analysis_role: str = "secondary"
    source_artifact: str = ""
    source_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # NaN / None の整合性
        if d["estimate"] is None:
            d["estimate"] = np.nan
        if d["ci_low"] is None:
            d["ci_low"] = np.nan
        if d["ci_high"] is None:
            d["ci_high"] = np.nan
        if d["p"] is None:
            d["p"] = np.nan
        if d["q"] is None:
            d["q"] = np.nan
        return d


def validate_paper_summary_df(
    df: pd.DataFrame,
    strict: bool = False,
    allow_empty: bool = False,
) -> list[str]:
    """
    データフレームが共通19列スキーマおよび論文不変条件を満たすかを検証する。
    戻り値: エラーメッセージリスト（空なら検証成功）
    """
    errors: list[str] = []

    # 1. カラム存在チェック
    missing_cols = [c for c in PAPER_SUMMARY_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required schema columns: {missing_cols}")
        return errors

    if len(df) == 0:
        if not allow_empty:
            errors.append("DataFrame is unexpectedly empty.")
        return errors

    # 2. analysis_role のチェック
    invalid_roles = df[~df["analysis_role"].isin(VALID_ANALYSIS_ROLES)]["analysis_role"].unique()
    if len(invalid_roles) > 0:
        errors.append(f"Invalid analysis_role values found: {list(invalid_roles)}")

    # 3. is_primary の boolean チェック
    non_bool_primary = df[~df["is_primary"].isin([True, False, 1, 0, "True", "False"])]
    if len(non_bool_primary) > 0:
        errors.append(
            f"Non-boolean values found in is_primary column: {len(non_bool_primary)} rows"
        )

    # 4. stage のチェック
    invalid_stages = df[~df["stage"].isin(VALID_STAGES)]["stage"].unique()
    if len(invalid_stages) > 0:
        errors.append(f"Invalid stage values found: {list(invalid_stages)}")

    # 5. unique key 重複チェック (is_primary=True の行で重複があってはならない)
    primary_df = df[df["is_primary"].eq(True)]
    if len(primary_df) > 0:
        duplicates = primary_df[primary_df.duplicated(subset=UNIQUE_KEY_COLUMNS, keep=False)]
        if len(duplicates) > 0:
            dup_keys = duplicates[UNIQUE_KEY_COLUMNS].drop_duplicates().to_dict(orient="records")
            errors.append(
                f"Duplicate primary records detected for unique key columns "
                f"{UNIQUE_KEY_COLUMNS}: {dup_keys[:5]}"
            )

    return errors


def filter_primary_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    確定最終版の直交フィルター規則に基づき、Primary結果を抽出する。
    is_primary == True かつ analysis_role が secondary/control/diagnostic ではないもの。
    これにより、is_primary=True かつ analysis_role='discovery'（V3 RQ2 4-map等）も正しく残る。
    """
    if len(df) == 0:
        return pd.DataFrame(columns=PAPER_SUMMARY_COLUMNS)
    mask = df["is_primary"].astype(bool) & ~df["analysis_role"].isin(
        ["secondary", "control", "diagnostic", "exploratory"]
    )
    return df[mask].copy().reset_index(drop=True)


def filter_secondary_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Secondary / Control / Diagnostic / Exploratory 結果を抽出する。
    is_primary == False または analysis_role in ('secondary', 'control', 'diagnostic', 'exploratory')。
    """
    if len(df) == 0:
        return pd.DataFrame(columns=PAPER_SUMMARY_COLUMNS)
    mask = (~df["is_primary"].astype(bool)) | df["analysis_role"].isin(
        ["secondary", "control", "diagnostic", "exploratory"]
    )
    return df[mask].copy().reset_index(drop=True)


def format_source_entries(
    artifacts: str | Sequence[str],
    keys: str | Sequence[str],
) -> tuple[str, str]:
    """
    単一または複数のartifact / keyをCSV列用の文字列（単一文字列またはJSON文字列）に変換する。
    """
    if isinstance(artifacts, (list, tuple)):
        art_str = json.dumps(list(artifacts)) if len(artifacts) > 1 else artifacts[0]
    else:
        art_str = str(artifacts)

    if isinstance(keys, (list, tuple)):
        key_str = json.dumps(list(keys)) if len(keys) > 1 else keys[0]
    else:
        key_str = str(keys)

    return art_str, key_str
