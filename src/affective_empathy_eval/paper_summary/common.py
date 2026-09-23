"""
src/affective_empathy_eval/paper_summary/common.py

共通ファイルI/O、Manifestビルダー、QCサマリービルダー、および
State-aware な厳格バリデーションユーティリティ。
（※統計量の新規計算・再解析ロジックは一切含まず、純粋なプレゼンテーション層として動作）
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from affective_empathy_eval.paper_summary.schema import (
    PAPER_SUMMARY_COLUMNS,
)


class PaperSummaryManifest:
    """
    データ追跡可能性（Provenance）を保証するためのManifest管理クラス。
    各metricについて、元artifact、元key配列、導出ルール（derivation）を記録する。
    """

    def __init__(self, schema_version: str = "1.0.0"):
        self.schema_version = schema_version
        self.provenance_map: dict[str, dict[str, Any]] = {}

    def register(
        self,
        record_id: str,
        sources: list[dict[str, str]],
        derivation: str = "direct_copy",
        notes: str = "",
    ):
        """
        record_id: 例 'behavioral.table_b1.reader_valence_pearson_r'
        sources: [{'artifact': 'path/to/file.csv', 'key': 'column_name'}, ...]
        derivation: 導出方法の説明（例: 'direct_copy', 'c_raw - c_rand', 'argmax(r2)/(L-1)'）
        """
        self.provenance_map[record_id] = {
            "sources": sources,
            "derivation": derivation,
            "notes": notes,
        }

    def register_simple(
        self,
        record_id: str,
        artifact: str,
        key: str,
        derivation: str = "direct_copy",
        notes: str = "",
    ):
        self.register(
            record_id=record_id,
            sources=[{"artifact": str(artifact), "key": str(key)}],
            derivation=derivation,
            notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary_schema_version": self.schema_version,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "provenance_map": self.provenance_map,
        }

    def save(self, path: str | Path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class QCSummaryBuilder:
    """各ステージのQCチェック・実行状態・欠損・非fallback Nを記録するビルダー"""

    def __init__(self):
        self.stages: dict[str, dict[str, Any]] = {}
        self.global_notes: list[str] = []

    def set_stage_qc(self, stage: str, data: dict[str, Any]):
        self.stages[stage] = data

    def add_note(self, note: str):
        self.global_notes.append(note)

    def to_dict(self) -> dict[str, Any]:
        return {
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "stages": self.stages,
            "notes": self.global_notes,
        }

    def save(self, path: str | Path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def safe_save_csv(df: pd.DataFrame, path: str | Path, index: bool = False):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=index, encoding="utf-8")


def safe_save_json(obj: Any, path: str | Path, indent: int = 2):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def load_json_if_exists(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {path}: {e}")
        return None


def load_csv_if_exists(path: str | Path) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"Warning: Failed to load {path}: {e}")
        return None


def check_pipeline_continues(gate_decision_path: str | Path) -> tuple[bool, str, dict[str, Any]]:
    """
    V3 Gate決定ファイルから pipeline_continues を判定する。
    Returns:
        (pipeline_continues: bool, overall_decision: str, raw_payload: dict)
    """
    data = load_json_if_exists(gate_decision_path)
    if data is None:
        return False, "UNKNOWN_OR_MISSING", {}

    decision = data.get("overall_decision", data.get("decision", "NO_GO"))
    # pipeline_continues は overall_decision == "GO" の場合のみ True
    pipeline_continues = decision == "GO"
    return pipeline_continues, decision, data


def ensure_df_schema(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrameに必要な19列が揃っているか確認し、不足していれば初期値で埋めて列順を統一する"""
    for col in PAPER_SUMMARY_COLUMNS:
        if col not in df.columns:
            if col in ("estimate", "ci_low", "ci_high", "p", "q"):
                df[col] = np.nan
            elif col in ("is_primary",):
                df[col] = False
            elif col in ("value_text", "source_artifact", "source_key"):
                df[col] = ""
            elif col in ("n",):
                df[col] = None
            else:
                df[col] = "unspecified"
    return df[PAPER_SUMMARY_COLUMNS].copy()
