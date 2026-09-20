import json
from pathlib import Path

import pytest

from affective_empathy_eval.data import (
    load_v3_matched_pair_table,
    resolve_matched_neutral_text,
)
from affective_empathy_eval.likelihood import resolve_joint_stage_index
from affective_empathy_eval.run import v3_gate_allows_continuation


def test_v3_matched_pair_table_from_aipsy():
    df = load_v3_matched_pair_table("v1/data/processed/aipsy_4split_all.csv")
    assert len(df) == 192
    assert df["pair_id"].nunique() == 192
    assert "neutral_text" in df.columns
    assert df["neutral_text"].map(lambda x: isinstance(x, str) and len(x.strip()) > 0).all()
    first = df.iloc[0]
    neu = resolve_matched_neutral_text(first, df)
    assert neu == first["neutral_text"]
    assert neu != first["text"]


def test_v3_rejects_emobank_without_pairs():
    with pytest.raises(ValueError, match="pair_id"):
        load_v3_matched_pair_table("v1/data/processed/stimuli_vad_3way_test1k.csv")


def test_joint_stage_index_is_not_clamped_to_prompt_end():
    offsets = {
        "candidate_start": 0,
        "pre_V": 2,
        "V_value": 3,
        "pre_A": 6,
        "A_value": 7,
        "response_end": 10,
    }
    cand_start = 20
    seq_len = 40
    assert resolve_joint_stage_index(cand_start, "pre_V", offsets, seq_len) == 22
    assert resolve_joint_stage_index(cand_start, "candidate_start", offsets, seq_len) == 20
    assert resolve_joint_stage_index(cand_start, "response_start", offsets, seq_len) == 19
    with pytest.raises(ValueError, match="outside joint sequence"):
        resolve_joint_stage_index(cand_start, "response_end", offsets, seq_len=25)


def test_v3_gate_blocks_no_go(tmp_path, monkeypatch):
    gate = tmp_path / "v3_gate_decision.json"
    gate.write_text(json.dumps({"decision": "NO_GO"}), encoding="utf-8")
    assert v3_gate_allows_continuation(gate, force=False) is False
    assert v3_gate_allows_continuation(gate, force=True) is True
    gate.write_text(json.dumps({"decision": "GO"}), encoding="utf-8")
    assert v3_gate_allows_continuation(gate, force=False) is True
    gate.write_text(json.dumps({"decision": "GO (Valence-only)"}), encoding="utf-8")
    assert v3_gate_allows_continuation(gate, force=False) is False
    assert v3_gate_allows_continuation(gate, force=True) is True
    missing = tmp_path / "missing.json"
    assert v3_gate_allows_continuation(missing, force=False) is False
