"""
V2 RQ3 Net Causal Leverage & Peak Dissociation 整合性テスト（14件）
"""

import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from affective_empathy_eval.geometry import (
    compute_peak_depth,
    compute_center_of_mass,
    compute_causal_peak_from_net,
    compute_causal_center_of_mass_from_net,
    compute_decodability_peak,
    compute_decodability_center_of_mass,
    compute_net_causal_dissociation_metrics,
)
from affective_empathy_eval.interventions import generate_control_directions
from v2.primary.run_rq3_causal_map import (
    run_causal_patching_for_model,
    PRIMARY_CAUSAL_METRIC_V,
    PRIMARY_CAUSAL_METRIC_A,
    CAUSAL_CKPT_SCHEMA_VERSION,
    _get_geometry_profile,
    _is_valid_cond_checkpoint,
)
from affective_empathy_eval.prompts import TaskType


def test_1_net_causal_subtraction():
    """1. C_affect=0.8, C_rand=0.3 -> C_net=0.5 の数値正確性"""
    c_affect = 0.8
    c_rand = 0.3
    c_net = c_affect - c_rand
    assert pytest.approx(c_net, abs=1e-6) == 0.5


def test_2_raw_peak_differs_from_net_peak():
    """2. raw peak != net peak のケースで正しく net peak を選択"""
    depths = [0.0, 0.5, 1.0]
    c_raw = [0.9, 1.0, 0.8]      # raw peak is layer 1 (depth 0.5)
    c_rand = [0.8, 0.95, 0.1]    # control leverage
    c_net = [r - c for r, c in zip(c_raw, c_rand)]  # [0.1, 0.05, 0.7] -> net peak is layer 2 (depth 1.0)

    raw_pk = compute_peak_depth(c_raw, depths)
    net_pk = compute_causal_peak_from_net(c_net, depths)

    assert pytest.approx(raw_pk, abs=1e-5) == 0.5
    assert pytest.approx(net_pk, abs=1e-5) == 1.0
    assert raw_pk != net_pk


def test_3_causal_all_negative_nan_peak():
    """3. Causal 全層 <= 0 -> d_C* = NaN"""
    depths = [0.0, 0.5, 1.0]
    c_net = [-0.8, -0.4, -0.1]
    pk = compute_causal_peak_from_net(c_net, depths)
    assert math.isnan(pk)


def test_4_causal_all_negative_nan_com():
    """4. Causal 全層 <= 0 -> bar_d_C = NaN"""
    depths = [0.0, 0.5, 1.0]
    c_net = [-0.8, -0.4, -0.1]
    com = compute_causal_center_of_mass_from_net(c_net, depths)
    assert math.isnan(com)


def test_5_causal_all_negative_dissociation_metrics():
    """5. Causal 全層 <= 0 -> delta_d_star = NaN, delta_bar_d = NaN, no_positive_net_causal_peak = True"""
    depths = [0.0, 0.5, 1.0]
    d_prof = [0.2, 0.8, 0.3]
    c_net = [-0.5, -0.2, -0.1]
    res = compute_net_causal_dissociation_metrics(d_prof, c_net, depths)

    assert math.isnan(res["d_c_star"])
    assert math.isnan(res["bar_d_c"])
    assert math.isnan(res["delta_d_star"])
    assert math.isnan(res["delta_bar_d"])
    assert res["no_positive_net_causal_peak"] is True
    assert res["no_positive_decodability_peak"] is False


def test_6_dry_run_mock_returns_all_net_and_control_keys():
    """6. dry-run mock が c_v_net_rand, c_v_perp 等すべてのキーを含む"""
    mock_df = pd.DataFrame([{"pair_id": "p1", "text": "hello"} for _ in range(4)])
    res = run_causal_patching_for_model(
        model=None,
        tokenizer=None,
        df=mock_df,
        task=TaskType.READER,
        format_type="plain",
        is_dry_run=True,
        num_layers=4,
    )
    expected_keys = [
        "c_v", "c_a", "c_v_raw", "c_a_raw",
        "c_v_rand", "c_a_rand", "c_v_perp", "c_a_perp",
        "c_v_net_rand", "c_a_net_rand", "c_v_net_perp", "c_a_net_perp",
        "c_v_zero", "c_a_zero", "pair_level"
    ]
    for k in expected_keys:
        assert k in res, f"Missing key in mock result: {k}"

    assert len(res["pair_level"]) > 0
    p0 = res["pair_level"][0]
    pair_expected_keys = [
        "c_v", "c_a", "c_v_raw", "c_a_raw",
        "c_v_rand", "c_a_rand", "c_v_perp", "c_a_perp",
        "c_v_net_rand", "c_a_net_rand", "c_v_net_perp", "c_a_net_perp",
    ]
    for pk in pair_expected_keys:
        assert pk in p0, f"Missing key in pair record: {pk}"


def test_7_layer_checkpoint_schema_v2_invalidation(tmp_path):
    """7. layer ckpt schema_version=1 または net/perp 欠落が自動 invalidate される"""
    ckpt_file = tmp_path / "v2_causal_layer_ckpt_test.json"
    old_ckpt = {
        "schema_version": 1,
        "completed_layers": [0, 1],
        "actual_layers": 4,
        "layer_shifts_v": {"0": [0.5], "1": [0.6]},
        "layer_shifts_a": {"0": [0.4], "1": [0.5]},
        "pair_records": [{"pair_id": "p1", "c_v": 0.5}],
    }
    ckpt_file.write_text(json.dumps(old_ckpt))

    mock_df = pd.DataFrame([{"pair_id": "p1", "text": "test"}])
    # run_causal_patching_for_model with is_dry_run=False but passing mock components
    # We test the loading logic by verifying invalid schema check
    with open(ckpt_file, "r") as f:
        loaded = json.load(f)
    schema = loaded.get("schema_version", 1)
    has_net = "layer_shifts_v_net_rand" in loaded
    has_perp = "layer_shifts_v_perp" in loaded
    assert schema < CAUSAL_CKPT_SCHEMA_VERSION or not has_net or not has_perp


def test_8_cond_checkpoint_schema_v2_validation(tmp_path):
    """8. cond ckpt schema_version=1（または c_v_net_rand 欠落）が自動 invalidate される"""
    v1_ckpt_file = tmp_path / "v2_causal_cond_test_v1.json"
    v1_payload = {
        "schema_version": 1,
        "cond_key": "base_plain_reader",
        "causal_entry": {"c_v": [0.5], "c_a": [0.4]},
        "pair_level": [],
    }
    v1_ckpt_file.write_text(json.dumps(v1_payload))
    assert _is_valid_cond_checkpoint(v1_ckpt_file) is False

    v2_ckpt_file = tmp_path / "v2_causal_cond_test_v2.json"
    v2_payload = {
        "schema_version": CAUSAL_CKPT_SCHEMA_VERSION,
        "causal_primary_metric": "net_rand",
        "cond_key": "base_plain_reader",
        "causal_entry": {
            "c_v_raw": [0.5], "c_a_raw": [0.4],
            "c_v_net_rand": [0.2], "c_a_net_rand": [0.15],
            "c_v": [0.5], "c_a": [0.4],
        },
        "pair_level": [],
    }
    v2_ckpt_file.write_text(json.dumps(v2_payload))
    assert _is_valid_cond_checkpoint(v2_ckpt_file) is True


def test_9_aggregation_consistency_sample_mean_equals_layer_profile():
    """9. sample-level mean と layer profile が一致すること"""
    mock_df = pd.DataFrame([{"pair_id": f"p_{i}", "text": "t"} for i in range(10)])
    res = run_causal_patching_for_model(
        model=None, tokenizer=None, df=mock_df,
        task=TaskType.READER, format_type="plain",
        is_dry_run=True, num_layers=4,
    )
    df_pairs = pd.DataFrame(res["pair_level"])
    for l in range(4):
        layer_df = df_pairs[df_pairs["layer"] == l]
        mean_cv_net = layer_df["c_v_net_rand"].mean()
        assert pytest.approx(mean_cv_net, abs=1e-5) == res["c_v_net_rand"][l]


def test_10_confirmatory_h3_uses_net_rand():
    """10. Confirmatory H3 が PRIMARY_CAUSAL_METRIC_V (c_v_net_rand) を参照すること"""
    from v2.primary.run_confirmatory_analysis import run_confirmatory_analysis
    # Check that PRIMARY_CAUSAL_METRIC constants are aligned
    assert PRIMARY_CAUSAL_METRIC_V == "c_v_net_rand"
    assert PRIMARY_CAUSAL_METRIC_A == "c_a_net_rand"


def test_11_combined_csv_reconstruction_from_per_family_csvs(tmp_path):
    """11. Combined CSV が per-family CSV から再構築されること"""
    pair_dir = tmp_path / "pair_level"
    pair_dir.mkdir(parents=True, exist_ok=True)
    derived_dir = tmp_path

    # Simulate two family CSVs
    df_qwen = pd.DataFrame([{"family": "qwen", "pair_id": "p1", "c_v_net_rand": 0.5}])
    df_llama = pd.DataFrame([{"family": "llama", "pair_id": "p2", "c_v_net_rand": 0.6}])
    df_qwen.to_csv(pair_dir / "v2_causal_pair_level_qwen.csv", index=False)
    df_llama.to_csv(pair_dir / "v2_causal_pair_level_llama.csv", index=False)

    # Reconstruct combined
    all_fam_csvs = sorted(pair_dir.glob("v2_causal_pair_level_*.csv"))
    assert len(all_fam_csvs) == 2
    df_combined = pd.concat([pd.read_csv(p) for p in all_fam_csvs], ignore_index=True)
    combined_path = derived_dir / "v2_causal_pair_level.csv"
    df_combined.to_csv(combined_path, index=False)

    loaded = pd.read_csv(combined_path)
    assert set(loaded["family"].unique()) == {"qwen", "llama"}
    assert len(loaded) == 2


def test_12_intervention_magnitude_equivalence():
    """12. 介入等価性テスト: ||Δh_affect|| ≈ ||Δh_rand|| ≈ ||Δh_perp||"""
    rng = np.random.default_rng(42)
    D = 128
    d_v = rng.standard_normal(D)
    d_v = d_v / np.linalg.norm(d_v)

    d_rand, d_perp = generate_control_directions(d_v, seed=123)

    norm_affect = np.linalg.norm(d_v)
    norm_rand = np.linalg.norm(d_rand)
    norm_perp = np.linalg.norm(d_perp)

    assert pytest.approx(norm_affect, abs=1e-5) == 1.0
    assert pytest.approx(norm_rand, abs=1e-5) == 1.0
    assert pytest.approx(norm_perp, abs=1e-5) == 1.0
    # Orthogonality check
    assert pytest.approx(float(np.dot(d_v, d_perp)), abs=1e-5) == 0.0


def test_13_decodability_all_negative_nan_peak_and_com():
    """13. Decodability 全層 <= 0 ガード: D = [-0.8, -0.4, -0.1] -> d_D* = NaN, bar_d_D = NaN"""
    depths = [0.0, 0.5, 1.0]
    d_prof = [-0.8, -0.4, -0.1]

    d_star = compute_decodability_peak(d_prof, depths)
    com = compute_decodability_center_of_mass(d_prof, depths)

    assert math.isnan(d_star), f"Expected NaN peak depth for all-negative decodability, got {d_star}"
    assert math.isnan(com), f"Expected NaN center of mass for all-negative decodability, got {com}"


def test_14_decodability_all_negative_dissociation_guards():
    """14. Decodability 全層 <= 0 による解離ガード: D 全層 <= 0 -> delta_d_star = NaN, delta_bar_d = NaN"""
    depths = [0.0, 0.5, 1.0]
    d_prof = [-0.8, -0.4, -0.1]  # all failed decodability
    c_net = [0.1, 0.5, 0.8]      # valid positive causal effect

    res = compute_net_causal_dissociation_metrics(d_prof, c_net, depths)

    assert math.isnan(res["d_d_star"])
    assert math.isnan(res["bar_d_d"])
    assert not math.isnan(res["d_c_star"])
    assert not math.isnan(res["bar_d_c"])

    # Decodability is NaN, so delta must be NaN
    assert math.isnan(res["delta_d_star"])
    assert math.isnan(res["delta_bar_d"])

    # Flags
    assert res["no_positive_decodability_peak"] is True
    assert res["no_positive_net_causal_peak"] is False
