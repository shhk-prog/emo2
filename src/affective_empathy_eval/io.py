#!/usr/bin/env python3
"""
src/affective_empathy_eval/io.py

実験結果の逐次保存（Incremental Save）、成否判定（execution_success: true/false）、
および途中再開（Resume）判定のための共通 I/O ヘルパー。
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


def json_serializable_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(
        f"Object of type {o.__class__.__name__} is not JSON serializable"
    )


def save_experiment_result(
    output_path: Path | str,
    payload: Dict[str, Any],
    stage: str,
    experiment_id: str,
    status: str = "success",
    success: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
    indent: int = 2,
) -> Path:
    """
    実験結果を指定パスにアトミック（一時ファイル書き込み後置換）かつ逐次に保存する。
    トップレベルに成否フラグ、ステージ、実験ID、タイムスタンプを付与。
    """
    path = Path(output_path)
    if path.exists():
        archive_existing_file(path, stage=stage)
    path.parent.mkdir(parents=True, exist_ok=True)

    envelope = {
        "execution_status": status,
        "execution_success": bool(success),
        "stage": stage,
        "experiment_id": experiment_id,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
        **(metadata or {}),
        "results": payload,
    }

    # 一時ファイルへ書き込み後に置換（書き込み途中クラッシュによる破損防止）
    tmp_dir = path.parent
    with tempfile.NamedTemporaryFile("w", dir=tmp_dir, delete=False, encoding="utf-8") as tf:
        json.dump(envelope, tf, indent=indent, ensure_ascii=False, default=json_serializable_default)
        temp_name = tf.name

    os.replace(temp_name, path)
    logger.info(f"[{stage}:{experiment_id}] Sequentially saved result to {path} (success={success})")
    return path


def is_experiment_completed(
    output_path: Path | str,
    manifest_path: Optional[Path | str] = None,
    force: bool = False,
    expected_model_name: Optional[str] = None,
    expected_config_hash: Optional[str] = None,
) -> bool:
    """
    指定された実験結果が正常に完了しており、スキップ（途中再開）可能かを判定。
    1. force が True ならスキップ不可 (False)
    2. ファイルが存在しないならスキップ不可 (False)
    3. execution_success が True でなければスキップ不可 (False)
    4. manifest が指定されていれば整合性を照合
    """
    if force:
        return False

    path = Path(output_path)
    if not path.exists():
        return False

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 明示的な成否判定フラグの検証
        if not data.get("execution_success", False):
            logger.info(f"Result at {path} exists but execution_success is False. Will re-run.")
            return False

        # manifest の照合
        if manifest_path is not None:
            m_path = Path(manifest_path)
            if m_path.exists():
                from affective_empathy_eval.manifests import is_manifest_matching
                if not is_manifest_matching(
                    str(m_path),
                    expected_model_name=expected_model_name,
                    expected_config_hash=expected_config_hash,
                ):
                    logger.info(f"Manifest mismatch for {path}. Will re-run.")
                    return False

        logger.info(f"Verified valid completed result at {path}. Skipping.")
        return True
    except Exception as e:
        logger.warning(f"Failed to inspect existing result at {path}: {e}. Will re-run.")
        return False


def archive_existing_file(filepath: Path | str, stage: Optional[str] = None) -> Optional[Path]:
    """
    AGENTS.md 1.3 準拠: 既存結果ファイルを上書きせず results/archive/ へ退避する。
    """
    p = Path(filepath)
    if not p.exists():
        return None
    now_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = p.parent.parent / "archive" if p.parent.name in {"raw", "derived", "dry_run"} else p.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived_name = f"{now_str}_{p.name}"
    archived_path = archive_dir / archived_name
    try:
        shutil.copy2(p, archived_path)
        logger.info(f"Archived existing result {p} to {archived_path}")
        return archived_path
    except Exception as e:
        logger.warning(f"Failed to archive existing result {p}: {e}")
        return None


def record_latest_run(stage_dir: Path | str, run_id: str, metadata: Optional[Dict[str, Any]] = None):
    """
    AGENTS.md 5.3 準拠: stage の results/latest.json に最新 run_id を記録する。
    """
    p = Path(stage_dir)
    res_dir = p / "results" if (p / "results").exists() else p
    latest_file = res_dir / "latest.json"
    latest_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }
    try:
        with open(latest_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=json_serializable_default)
        logger.info(f"Recorded latest run {run_id} to {latest_file}")
    except Exception as e:
        logger.warning(f"Failed to record latest run to {latest_file}: {e}")


def resolve_output_dirs(
    config: Optional[Dict[str, Any]] = None,
    model_set: str = "primary_small",
    stage: str = "v2",
    is_dry_run: bool = False,
) -> tuple[Path, Path]:
    """
    model_set および stage に応じた (raw_dir, derived_dir) を決定して作成・返却する。
    - primary_small の場合は config の output 設定（デフォルト: f"{stage}/results/raw", f"{stage}/results/derived"）
    - それ以外の model_set（例: scale_3b, scale_7b, scale_validation）は
      results/ablation/{model_set}/raw, results/ablation/{model_set}/derived
    - is_dry_run の場合は末尾に dry_run を付与
    """
    if model_set != "primary_small":
        raw_dir = Path(f"results/ablation/{model_set}/raw")
        derived_dir = Path(f"results/ablation/{model_set}/derived")
    else:
        out_cfg = config.get("output", {}) if config else {}
        default_raw = f"{stage}/results/raw"
        default_derived = f"{stage}/results/derived"
        raw_dir = Path(out_cfg.get("raw_dir", default_raw))
        derived_dir = Path(out_cfg.get("derived_dir", default_derived))

    if is_dry_run:
        raw_dir = raw_dir / "dry_run"
        derived_dir = derived_dir / "dry_run"

    raw_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, derived_dir


def resolve_log_dir(
    model_set: str = "primary_small",
    stage: str = "v2",
    is_dry_run: bool = False,
) -> Path:
    """
    model_set に応じたログディレクトリを決定して作成・返却する。
    - primary_small: results/logs/
    - その他 (scale_3b, scale_7b, scale_validation 等): results/ablation/{model_set}/logs/
    - is_dry_run の場合は末尾に dry_run を付与
    """
    if model_set != "primary_small":
        log_dir = Path(f"results/ablation/{model_set}/logs")
    else:
        log_dir = Path("results/logs")

    if is_dry_run:
        log_dir = log_dir / "dry_run"

    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir



