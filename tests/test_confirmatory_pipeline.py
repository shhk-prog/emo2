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
    assert "H3_causal_dissociation_lmm" in report["hypotheses"]
    assert "H4_recovery_asymmetry" in report["hypotheses"]
    assert (derived_dir / "v2_lmm_confirmatory.json").exists()
