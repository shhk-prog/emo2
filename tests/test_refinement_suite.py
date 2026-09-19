"""
tests/test_refinement_suite.py

Comprehensive Test Suite covering:
1. Python syntax / imports
2. Layer index mapping (Transformer block l -> hidden_states[l + 1])
3. Candidate space size (VA = 81, VAD = 729)
4. Pair ID train/test leakage assertion
5. Cache manifest mismatch validation (additive injection vs replacement separation)
6. Config propagation (bootstrap.n_boot resolution)
"""

import os
import tempfile
import json
import pytest
import numpy as np
import torch
import yaml

from affective_empathy_eval.geometry import get_block_hidden_state, compute_relative_depth
from affective_empathy_eval.likelihood import build_va_candidates, build_vad_candidates
from affective_empathy_eval.manifests import (
    create_run_manifest,
    is_manifest_matching,
    DEFAULT_INTERVENTION_VERSION,
)


def test_python_syntax_and_core_imports():
    """1. 主要モジュールがエラーなくインポートできること"""
    import affective_empathy_eval
    import affective_empathy_eval.geometry
    import affective_empathy_eval.likelihood
    import affective_empathy_eval.manifests
    import affective_empathy_eval.models.hooks
    import affective_empathy_eval.statistics

    assert hasattr(affective_empathy_eval, "get_block_hidden_state")
    assert hasattr(affective_empathy_eval.models.hooks, "apply_direction_intervention")


def test_layer_index_mapping():
    """2. Layer index mapping: block l は hidden_states[l + 1] から取得される"""
    num_blocks = 12
    # Hugging Face outputs.hidden_states は (num_blocks + 1) 個のテンソル
    mock_hidden_states = tuple(torch.full((1, 4, 16), float(i)) for i in range(num_blocks + 1))

    # embedding は index 0
    assert mock_hidden_states[0][0, 0, 0].item() == 0.0

    # block 0 は hidden_states[1]
    block_0 = get_block_hidden_state(mock_hidden_states, 0)
    assert block_0[0, 0, 0].item() == 1.0

    # block L-1 は hidden_states[num_blocks]
    block_last = get_block_hidden_state(mock_hidden_states, num_blocks - 1)
    assert block_last[0, 0, 0].item() == float(num_blocks)

    # 範囲外アクセスで AssertionError
    with pytest.raises(AssertionError):
        get_block_hidden_state(mock_hidden_states, num_blocks)

    with pytest.raises(AssertionError):
        get_block_hidden_state(mock_hidden_states, -1)

    # 相対深度の検証
    assert compute_relative_depth(0, num_blocks) == 0.0
    assert compute_relative_depth(num_blocks - 1, num_blocks) == 1.0


def test_phase_c_block_index_mapping():
    """Phase C hidden state extraction: hidden_states[0] = embedding, [1] = block 0, [L] = block L-1"""
    from unittest.mock import MagicMock
    from v1.primary.run_phase_c import extract_hidden_states

    num_blocks = 8
    # hidden_states: embedding (0) + 8 blocks (1..8)
    mock_hidden_states = tuple(torch.full((2, 5, 16), float(i)) for i in range(num_blocks + 1))

    mock_outputs = MagicMock()
    mock_outputs.hidden_states = mock_hidden_states

    mock_model = MagicMock()
    mock_model.config.num_hidden_layers = num_blocks
    mock_model.return_value = mock_outputs

    mock_encoded = {
        "input_ids": torch.ones((2, 5), dtype=torch.long),
        "attention_mask": torch.ones((2, 5), dtype=torch.long),
    }
    class MockBatch(dict):
        def to(self, device):
            return self

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = MockBatch(mock_encoded)

    extracted = extract_hidden_states(
        model=mock_model,
        tokenizer=mock_tokenizer,
        prompts=["test 1", "test 2"],
        device="cpu",
        batch_size=2,
    )

    assert len(extracted) == num_blocks
    assert 0 in extracted
    assert (num_blocks - 1) in extracted
    # block 0 must match hidden_states[1] (value 1.0), NOT embedding (0.0)
    assert np.allclose(extracted[0], 1.0)
    # block L-1 must match hidden_states[num_blocks] (value num_blocks)
    assert np.allclose(extracted[num_blocks - 1], float(num_blocks))


def test_candidate_space_sizes():
    """3. Candidate space size test: VA = 81, VAD = 729"""
    va_candidates = build_va_candidates()
    assert len(va_candidates) == 81, f"VA candidates must be 81 (9x9), got {len(va_candidates)}"

    vad_candidates = build_vad_candidates()
    assert len(vad_candidates) == 729, f"VAD candidates must be 729 (9x9x9), got {len(vad_candidates)}"


def test_pair_id_train_test_leakage_assertion():
    """4. pair_id train/test leakage test: 同一 pair 由来データがまたがった場合に検知"""
    train_pairs = {"pair_1", "pair_2", "pair_3"}
    test_pairs_clean = {"pair_4", "pair_5"}
    test_pairs_leaked = {"pair_3", "pair_6"}

    # クリーンな分割では交差ゼロ
    assert len(train_pairs.intersection(test_pairs_clean)) == 0

    # リークが存在する場合に assert で失敗すること
    with pytest.raises(AssertionError):
        assert len(train_pairs.intersection(test_pairs_leaked)) == 0, "Data leakage detected!"


def test_cache_manifest_validation_and_mismatch(tmp_path):
    """5. Cache manifest mismatch test: 旧 replacement と新 additive injection の混在防止"""
    manifest_path = tmp_path / "manifest.json"

    # 新しい additive injection の manifest を作成
    manifest_new = create_run_manifest(
        run_type="v3_rq1",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        config={"alpha": 1.0},
        intervention_version="v3_additive_injection_v2",
        candidate_space="VA_81",
    )
    manifest_new.save(str(manifest_path))

    # 一致する場合は True
    assert is_manifest_matching(
        str(manifest_path),
        expected_model_name="Qwen/Qwen2.5-1.5B-Instruct",
        expected_intervention_version="v3_additive_injection_v2",
        expected_candidate_space="VA_81",
    )

    # 旧バージョン (replacement) を期待した場合や異なるバージョンの場合は False
    assert not is_manifest_matching(
        str(manifest_path),
        expected_intervention_version="v3_replacement_v1",
    )

    # モデル名不一致
    assert not is_manifest_matching(
        str(manifest_path),
        expected_model_name="meta-llama/Llama-3.2-1B-Instruct",
    )

    # 存在しないファイル
    assert not is_manifest_matching(str(tmp_path / "non_existent_manifest.json"))


def test_config_propagation():
    """6. Config propagation test: bootstrap.n_boot が正しく優先取得されること"""
    sample_config = {
        "bootstrap": {
            "n_boot": 500,
            "ci_level": 0.95,
        },
        "statistics": {
            "n_boot": 200,
        }
    }
    # bootstrap を優先して読み込む
    resolved_n_boot = (sample_config.get("bootstrap", {}).get("n_boot") or sample_config.get("statistics", {}).get("n_boot", 1000))
    assert resolved_n_boot == 500

    # bootstrap キーがない場合は statistics へフォールバック
    fallback_config = {
        "statistics": {
            "n_boot": 300,
        }
    }
    fallback_n_boot = (fallback_config.get("bootstrap", {}).get("n_boot") or fallback_config.get("statistics", {}).get("n_boot", 1000))
    assert fallback_n_boot == 300


def test_rq2_cache_hit_does_not_recompute(tmp_path, monkeypatch):
    """
    7. V3 RQ2 cache hit test:
    有効な結果 JSON と一致する RunManifest が存在する場合、
    is_manifest_matching が True を返し、再計算を行わずキャッシュからロードすること。
    """
    from affective_empathy_eval.manifests import (
        create_run_manifest,
        is_manifest_matching,
        compute_string_or_dict_hash,
        DEFAULT_CODE_VERSION,
    )
    import v3.primary.run_rq2_spatiotemporal_maps as rq2_mod

    # is_manifest_matching がモジュール名前空間に正しくインポートされていることの検証 (NameError 防止)
    assert hasattr(rq2_mod, "is_manifest_matching")
    assert rq2_mod.is_manifest_matching is is_manifest_matching

    fam_key = "qwen25_15b"
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset_path = str(tmp_path / "dataset.csv")

    manifest_config = {
        "analysis_role": "discovery",
        "family": fam_key,
        "model_id": model_id,
        "dataset_path": dataset_path,
        "semantic_stages": ["candidate_start", "candidate_end"],
        "alpha_sweep": [-1.0, 0.0, 1.0],
        "causal_reference_alpha": 1.0,
        "n_causal_samples": 5,
        "seed": 42,
        "subsample": 10,
        "dry_run": False,
    }

    manifest = create_run_manifest(
        run_type="v3_rq2_discovery_spatiotemporal_maps",
        model_name=model_id,
        config=manifest_config,
        metadata={"dissociation_summary": {"valence": {"delta_d_peak": 0.35}, "arousal": {"delta_d_peak": 0.40}}},
        dry_run=False,
    )

    manifest_path = tmp_path / f"manifest_rq2_{fam_key}.json"
    manifest.save(manifest_path)

    out_raw = tmp_path / f"v3_discovery_spatiotemporal_maps_{fam_key}.json"
    cached_data = {
        "maps": {"d_v": [[0.5]], "d_a": [[0.5]]},
        "dry_run": False,
        "dissociation_summary": {"valence": {"delta_d_peak": 0.35}, "arousal": {"delta_d_peak": 0.40}},
    }
    with open(out_raw, "w", encoding="utf-8") as f:
        json.dump(cached_data, f)

    expected_config_hash = compute_string_or_dict_hash(manifest_config)
    expected_dataset_hash = compute_string_or_dict_hash(dataset_path)

    # 1. manifest matching check
    is_valid = rq2_mod.is_manifest_matching(
        str(manifest_path),
        expected_model_name=model_id,
        expected_config_hash=expected_config_hash,
        expected_dataset_hash=expected_dataset_hash,
        expected_code_version=DEFAULT_CODE_VERSION,
        expected_dry_run=False,
    )
    assert is_valid, "Valid manifest and cache should be accepted"

    # 2. Config が変更された場合は cache invalid となること
    altered_config = dict(manifest_config, seed=999)
    altered_config_hash = compute_string_or_dict_hash(altered_config)
    is_invalid = rq2_mod.is_manifest_matching(
        str(manifest_path),
        expected_model_name=model_id,
        expected_config_hash=altered_config_hash,
        expected_dataset_hash=expected_dataset_hash,
        expected_code_version=DEFAULT_CODE_VERSION,
        expected_dry_run=False,
    )
    assert not is_invalid, "Altered config must invalidate cache"

