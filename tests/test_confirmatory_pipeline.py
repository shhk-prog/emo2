import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from v3.primary.run_confirmatory_replication import validate_stage_index_invariance
from affective_empathy_eval.prompts import TaskType, build_prompt
from affective_empathy_eval.likelihood import build_va_candidates


def test_validate_stage_index_invariance():
    """81候補のトークナイズ長およびステージインデックスが完全に不変であることを検証"""
    class DummyTokenizer:
        def encode(self, text, add_special_tokens=False):
            # 1文字=1トークン
            return [ord(c) for c in text]

        def decode(self, tokens, **kwargs):
            return "".join(chr(t) for t in tokens)

    tok = DummyTokenizer()
    candidates = build_va_candidates()
    stage_names = ["candidate_start", "pre_V", "V_value", "response_end"]
    # 正常系: すべて同一
    stage_indices = validate_stage_index_invariance(
        tok, "Test prefix prompt:", candidates, stage_names
    )
    assert len(stage_indices) == 4
    assert "response_end" in stage_indices

    # 異常系: トークン長が一致しない場合 AssertionError
    class BrokenTokenizer:
        def encode(self, text, add_special_tokens=False):
            if "1" in text:
                return [ord(c) for c in text]
            return [ord(c) for c in text] + [999]

        def decode(self, tokens, **kwargs):
            return "".join(chr(t % 128) for t in tokens)

    with pytest.raises(AssertionError):
        validate_stage_index_invariance(BrokenTokenizer(), "Test prompt:", candidates, stage_names)


def test_behavioral_aipsy_expected_direction_alignment():
    """AIPsy 分析で期待方向符号整列が正しく機能することを検証"""
    from behavioral.analysis.summarize_behavioral_aipsy import EXPECTED_DIRECTION

    assert "grief" in EXPECTED_DIRECTION
    assert EXPECTED_DIRECTION["grief"]["V"] == -1
    assert EXPECTED_DIRECTION["terror"]["V"] == -1
    assert EXPECTED_DIRECTION["terror"]["A"] == +1
    assert EXPECTED_DIRECTION["ecstasy"]["V"] == +1
    assert EXPECTED_DIRECTION["ecstasy"]["A"] == +1


def test_v2_confirmatory_analysis_dry_run(tmp_path):
    """V2 確証的統合解析が合成データで正常終了することを検証"""
    from v2.primary.run_confirmatory_analysis import run_confirmatory_analysis

    raw_dir = tmp_path / "raw"
    derived_dir = tmp_path / "derived"
    report = run_confirmatory_analysis(raw_dir=raw_dir, derived_dir=derived_dir, is_dry_run=True)

    assert report["status"] == "success"
    assert "H3_causal_profile_reorganization_lmm" in report["hypotheses"]
    assert "H3_causal_dissociation_lmm" in report["hypotheses"]
    assert "H4_recovery_asymmetry" in report["hypotheses"]
    assert (derived_dir / "dry_run" / "v2_lmm_confirmatory.json").exists() or (derived_dir / "v2_lmm_confirmatory.json").exists()


def test_v3_confirmatory_h1_joint_bootstrap():
    """H1 の joint bootstrap が D(l) と C(l) の同時標本変動を考慮して CI を生成することを検証"""
    from sklearn.metrics import r2_score
    from affective_empathy_eval.geometry import compute_layer_dissociation

    num_layers = 10
    relative_depths = [l / (num_layers - 1) for l in range(num_layers)]
    n_h1 = 20
    rng = np.random.default_rng(42)

    y_v_h1 = rng.standard_normal(n_h1)
    oof_preds_v_by_layer = {
        l: y_v_h1 * (0.8 - abs(relative_depths[l] - 0.4)) + rng.standard_normal(n_h1) * 0.1
        for l in range(num_layers)
    }
    test_c_shifts_v = {
        l: [float(np.exp(-((relative_depths[l] - 0.7) ** 2) / 0.05) + rng.normal(0, 0.05)) for _ in range(n_h1)]
        for l in range(num_layers)
    }

    d_prof_h1 = [float(r2_score(y_v_h1, oof_preds_v_by_layer[l])) for l in range(num_layers)]
    c_prof = [float(np.mean(test_c_shifts_v[l])) for l in range(num_layers)]
    pt_dissoc = compute_layer_dissociation(relative_depths, d_prof_h1, c_prof)

    peak_boots = []
    center_boots = []
    for _ in range(100):
        pos = rng.integers(0, n_h1, size=n_h1)
        sampled_y = y_v_h1[pos]
        d_b = [float(r2_score(sampled_y, oof_preds_v_by_layer[l][pos])) for l in range(num_layers)]
        c_b = [float(np.mean([test_c_shifts_v[l][p] for p in pos])) for l in range(num_layers)]
        res_b = compute_layer_dissociation(relative_depths, d_b, c_b)
        peak_boots.append(res_b["delta_d_peak"])
        center_boots.append(res_b["delta_d_center"])

    ci_peak = [float(np.percentile(peak_boots, 2.5)), float(np.percentile(peak_boots, 97.5))]
    ci_center = [float(np.percentile(center_boots, 2.5)), float(np.percentile(center_boots, 97.5))]

    assert ci_peak[0] <= pt_dissoc["delta_d_peak"] <= ci_peak[1]
    assert ci_center[0] <= pt_dissoc["delta_d_center"] <= ci_center[1]
    assert ci_peak[1] - ci_peak[0] > 0
    assert ci_center[1] - ci_center[0] > 0
