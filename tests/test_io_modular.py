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


def test_save_experiment_result_with_numpy_types():
    import numpy as np
    with tempfile.TemporaryDirectory() as tmpdir:
        out_json = Path(tmpdir) / "v2_rq4_numpy_test.json"
        
        # NumPy スカラー、配列、Path、およびネスト構造を含むペイロード
        payload = {
            "sample_idx": np.int64(989),
            "delta_emd": np.float64(0.0123456789),
            "is_valid": np.bool_(True),
            "vector": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "matrix": np.zeros((2, 2), dtype=np.int64),
            "sample_records": [
                {
                    "sample_idx": np.int64(10),
                    "score": np.float64(0.95),
                    "active": np.bool_(False),
                },
                {
                    "sample_idx": np.int64(20),
                    "score": np.float64(0.85),
                    "active": np.bool_(True),
                },
            ],
            "nested_config": {
                "layers": [np.int64(0), np.int64(1), np.int64(2)],
                "weights": [np.float64(0.1), np.float64(0.2)],
            },
        }
        
        save_experiment_result(
            output_path=out_json,
            payload=payload,
            stage="v2",
            experiment_id="v2_rq4_numpy_test",
            status="success",
            success=True,
            metadata={
                "best_layer": np.int64(12),
                "model_path": Path("/tmp/model"),
            },
        )
        
        assert out_json.exists()
        assert is_experiment_completed(out_json)
        
        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        res = data["results"]
        assert isinstance(res["sample_idx"], int)
        assert res["sample_idx"] == 989
        assert isinstance(res["delta_emd"], float)
        assert abs(res["delta_emd"] - 0.0123456789) < 1e-8
        assert isinstance(res["is_valid"], bool)
        assert res["is_valid"] is True
        assert isinstance(res["vector"], list)
        assert res["vector"] == [1.0, 2.0, 3.0]
        assert isinstance(res["matrix"], list)
        assert res["matrix"] == [[0, 0], [0, 0]]
        
        # ネストしたレコード
        assert len(res["sample_records"]) == 2
        assert isinstance(res["sample_records"][0]["sample_idx"], int)
        assert res["sample_records"][0]["sample_idx"] == 10
        assert isinstance(res["sample_records"][0]["score"], float)
        assert res["sample_records"][0]["active"] is False
        
        # ネストした辞書
        assert res["nested_config"]["layers"] == [0, 1, 2]
        assert isinstance(res["nested_config"]["layers"][0], int)
        
        # メタデータ
        assert isinstance(data["metadata"]["best_layer"], int)
        assert data["metadata"]["best_layer"] == 12
        assert data["metadata"]["model_path"] == "/tmp/model"


def test_resolve_output_dirs():
    from affective_empathy_eval.io import resolve_output_dirs

    # 1. primary_small (通常実行)
    raw, derived = resolve_output_dirs(model_set="primary_small", stage="v2", is_dry_run=False)
    assert raw == Path("v2/results/raw")
    assert derived == Path("v2/results/derived")

    # 2. primary_small (dry-run)
    raw, derived = resolve_output_dirs(model_set="primary_small", stage="v2", is_dry_run=True)
    assert raw == Path("v2/results/raw/dry_run")
    assert derived == Path("v2/results/derived/dry_run")

    # 3. scale_3b (通常実行) -> results/ablation/scale_3b/ に分離
    raw, derived = resolve_output_dirs(model_set="scale_3b", stage="v2", is_dry_run=False)
    assert raw == Path("results/ablation/scale_3b/raw")
    assert derived == Path("results/ablation/scale_3b/derived")

    # 4. scale_7b (dry-run) -> results/ablation/scale_7b/raw/dry_run に分離
    raw, derived = resolve_output_dirs(model_set="scale_7b", stage="v2", is_dry_run=True)
    assert raw == Path("results/ablation/scale_7b/raw/dry_run")
    assert derived == Path("results/ablation/scale_7b/derived/dry_run")


def test_resolve_log_dir():
    from affective_empathy_eval.io import resolve_log_dir

    # 1. primary_small -> results/logs/
    log_dir = resolve_log_dir(model_set="primary_small", stage="v2", is_dry_run=False)
    assert log_dir == Path("results/logs")

    # 2. scale_3b -> results/ablation/scale_3b/logs/
    log_dir = resolve_log_dir(model_set="scale_3b", stage="v2", is_dry_run=False)
    assert log_dir == Path("results/ablation/scale_3b/logs")

    # 3. scale_7b (dry-run) -> results/ablation/scale_7b/logs/dry_run
    log_dir = resolve_log_dir(model_set="scale_7b", stage="v2", is_dry_run=True)
    assert log_dir == Path("results/ablation/scale_7b/logs/dry_run")



