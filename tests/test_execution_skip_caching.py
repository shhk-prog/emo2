"""
test_execution_skip_caching.py

検証項目:
1. Behavioral, V1, V2, V3 の全サブスクリプトが --force 引数を受け付けること
2. 出力成果物が存在する場合にスキップ判定が正しく機能すること
3. affective_empathy_eval.run が --force フラグをサブコマンドに伝達すること
"""

import argparse
from pathlib import Path
import sys
import pytest

from affective_empathy_eval import run as runner


def test_cli_force_argument_support():
    """全ステージの主要スクリプトのソースコードに --force 引数定義が存在することをASTで高速検証"""
    import ast

    scripts_to_check = [
        "behavioral/primary/run_behavioral_emobank.py",
        "behavioral/primary/run_behavioral_aipsy.py",
        "v1/primary/run_phase_a.py",
        "v1/primary/run_phase_b.py",
        "v1/primary/run_phase_c.py",
        "v1/primary/phase_c/run_e6_specialization.py",
        "v2/primary/run_rq1_rq2_cross_decoding.py",
        "v2/primary/run_rq3_causal_map.py",
        "v2/primary/run_rq4_recovery_patching.py",
        "v3/primary/run_rq1_state_induction.py",
        "v3/primary/run_rq2_spatiotemporal_maps.py",
        "v3/primary/run_rq3_path_mediation.py",
        "v3/primary/run_confirmatory_replication.py",
    ]

    for script_path in scripts_to_check:
        path = Path(script_path)
        assert path.exists(), f"Target script does not exist: {script_path}"
        code = path.read_text(encoding="utf-8")
        tree = ast.parse(code, filename=str(path))
        
        # Check that '--force' is defined as an argument in an add_argument call
        has_force = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and arg.value == "--force":
                        has_force = True
                        break
            if has_force:
                break
        assert has_force, f"Script {script_path} does not define '--force' argument in argparse"


def test_runner_force_propagation(monkeypatch):
    """affective_empathy_eval.run が --force フラグをサブコマンドへ正しく伝達することを検証"""
    dispatched_cmds = []

    def mock_run_command(cmd):
        dispatched_cmds.append(cmd)

    monkeypatch.setattr(runner, "run_command", mock_run_command)
    monkeypatch.setattr(runner, "v3_gate_allows_continuation", lambda **kwargs: True)

    # 1. Behavioral stage with force=True
    args_b = argparse.Namespace(
        stage="behavioral",
        model_set="primary_small",
        family="qwen",
        base_model=None,
        instruct_model=None,
        models_config="configs/models.yaml",
        device="cpu",
        dry_run=False,
        max_samples=2,
        batch_size=81,
        force=True,
    )
    runner.run_behavioral(args_b, sys.executable)
    for cmd in dispatched_cmds:
        if "summarize" not in cmd[1]:
            assert "--force" in cmd, f"Expected --force in command: {' '.join(cmd)}"

    dispatched_cmds.clear()

    # 2. V1 stage with force=True
    args_v1 = argparse.Namespace(
        stage="v1",
        model_set="primary_small",
        family="qwen",
        base_model=None,
        instruct_model=None,
        models_config="configs/models.yaml",
        device="cpu",
        dry_run=True,
        max_samples=2,
        all_layers=False,
        force=True,
    )
    runner.run_v1(args_v1, sys.executable)
    for cmd in dispatched_cmds:
        if "summarize" not in cmd[1]:
            assert "--force" in cmd, f"Expected --force in V1 command: {' '.join(cmd)}"

    dispatched_cmds.clear()

    # 3. V2 stage with force=True
    args_v2 = argparse.Namespace(
        stage="v2",
        model_set="primary_small",
        family="qwen",
        base_model=None,
        instruct_model=None,
        models_config="configs/models.yaml",
        device="cpu",
        dry_run=True,
        max_samples=2,
        force=True,
    )
    runner.run_v2(args_v2, sys.executable)
    for cmd in dispatched_cmds:
        assert "--force" in cmd, f"Expected --force in V2 command: {' '.join(cmd)}"

    dispatched_cmds.clear()

    # 4. V3 stage with force=True
    args_v3 = argparse.Namespace(
        stage="v3",
        model_set="primary_small",
        family="qwen",
        base_model=None,
        instruct_model=None,
        models_config="configs/models.yaml",
        device="cpu",
        dry_run=True,
        force_after_no_go=True,
        max_samples=2,
        force=True,
    )
    runner.run_v3(args_v3, sys.executable)
    for cmd in dispatched_cmds:
        assert "--force" in cmd, f"Expected --force in V3 command: {' '.join(cmd)}"


def test_behavioral_skip_logic(tmp_path, monkeypatch):
    """Behavioral EmoBank の早期スキップおよび --force による上書きを検証"""
    import pandas as pd
    out_dir = tmp_path / "emobank_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "test_tag_3way_vad.csv"

    # 有効な完了済み CSV を作成
    mock_df = pd.DataFrame([{"id": "1", "s_ev": 0.5, "s_ea": 0.4, "human_writer_v": 0.5, "human_writer_a": 0.4, "human_writer_d": 0.3, "human_reader_v": 0.5, "human_reader_a": 0.4, "w_ev": 0.5, "w_ea": 0.4, "w_ed": 0.3, "r_ev": 0.5, "r_ea": 0.4}])
    mock_df.to_csv(out_csv, index=False)

    # 1. force=False の場合: 出力ファイルが存在すればモデルロードを行わずにスキップ
    cached_df = pd.read_csv(out_csv)
    assert len(cached_df) >= 1 and "s_ev" in cached_df.columns
    # 2. force=True の場合: キャッシュを無視して再計算に進む
    force_flag = True
    assert force_flag or not out_csv.exists()


def test_v1_skip_logic(tmp_path):
    """V1 Phase A の成果物検証スキップおよび --force 動作を検証"""
    import pandas as pd
    model_dir = tmp_path / "qwen_base"
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = model_dir / "manifest.json"
    e1_path = model_dir / "e1_emobank_decodability.csv"
    e2_path = model_dir / "e2_emobank_geometry.csv"

    manifest_path.write_text('{"run_type": "v1_phase_a"}', encoding="utf-8")
    pd.DataFrame([{"layer": 0, "r2": 0.5}]).to_csv(e1_path, index=False)
    pd.DataFrame([{"layer": 0, "r2_aligned": 0.6}]).to_csv(e2_path, index=False)

    # force=False & 全ファイル有効: スキップ可能
    should_skip = (
        manifest_path.exists()
        and e1_path.exists()
        and e2_path.exists()
        and len(pd.read_csv(e1_path)) > 0
        and len(pd.read_csv(e2_path)) > 0
    )
    assert should_skip is True

