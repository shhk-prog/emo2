import json
import tempfile
from pathlib import Path
import pytest
from affective_empathy_eval.io import save_experiment_result, is_experiment_completed

def test_save_and_check_experiment_result():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_json = Path(tmpdir) / "v1_e1_decodability_test.json"
        
        # 1. 未作成状態
        assert not is_experiment_completed(out_json)
        
        # 2. 成功結果の保存
        payload = {"accuracy": 0.85, "metrics": [1, 2, 3]}
        save_experiment_result(
            output_path=out_json,
            payload=payload,
            stage="v1",
            experiment_id="v1_e1_decodability",
            status="success",
            success=True,
            metadata={"model": "test-model"},
        )
        
        assert out_json.exists()
        assert is_experiment_completed(out_json)
        
        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["execution_status"] == "success"
        assert data["execution_success"] is True
        assert data["stage"] == "v1"
        assert data["experiment_id"] == "v1_e1_decodability"
        assert data["results"]["accuracy"] == 0.85
        assert data["metadata"]["model"] == "test-model"
        
        # 3. 失敗状態での保存 (false 判定)
        out_failed = Path(tmpdir) / "v1_e1_decodability_failed.json"
        save_experiment_result(
            output_path=out_failed,
            payload={"error": "OOM"},
            stage="v1",
            experiment_id="v1_e1_decodability",
            status="failed",
            success=False,
        )
        assert out_failed.exists()
        assert not is_experiment_completed(out_failed)
        
        # 4. force オプション
        assert not is_experiment_completed(out_json, force=True)
