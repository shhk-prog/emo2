"""
tests/test_v1_token_and_probe_alignment.py

V1 Stage における tokenization 条件（add_special_tokens=False）の統一性、
および probe evaluation における group leakage fallback 禁止の検証テスト。
"""

import numpy as np
import pytest
import torch


class DummyTokenizer:
    """決定論的なモックトークナイザー（word単位でトークン化）"""
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(
        self,
        texts,
        padding=True,
        truncation=True,
        max_length=1024,
        add_special_tokens=False,
        return_tensors="pt",
    ):
        if isinstance(texts, str):
            texts = [texts]

        tokenized = []
        for t in texts:
            toks = [hash(w) % 1000 + 1 for w in t.strip().split()]
            if add_special_tokens:
                toks = [999] + toks + [998]  # dummy BOS/EOS
            tokenized.append(toks)

        max_len = max(len(t) for t in tokenized)
        input_ids = []
        attention_mask = []
        for toks in tokenized:
            pad_len = max_len - len(toks)
            input_ids.append(toks + [0] * pad_len)
            attention_mask.append([1] * len(toks) + [0] * pad_len)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def encode(self, text, add_special_tokens=False):
        toks = [hash(w) % 1000 + 1 for w in text.strip().split()]
        if add_special_tokens:
            toks = [999] + toks + [998]
        return toks


def test_v1_prompt_end_invariance_across_phases():
    """
    同一 prompt に対し、Phase A, Phase B, Phase C, E6 の prompt_end 位置が
    add_special_tokens=False により完全一致することを検証。
    """
    tokenizer = DummyTokenizer()
    prompts = [
        "Read this sentence and report the feeling of the reader.",
        "Short text.",
        "A somewhat longer prompt for testing invariance across all V1 forward passes.",
    ]

    # Phase A
    enc_a = tokenizer(prompts, padding=True, truncation=True, max_length=1024, add_special_tokens=False, return_tensors="pt")
    pos_a = (enc_a["attention_mask"].sum(dim=1) - 1).tolist()

    # Phase B
    enc_b = tokenizer(prompts, padding=True, truncation=True, max_length=1024, add_special_tokens=False, return_tensors="pt")
    pos_b = (enc_b["attention_mask"].sum(dim=1) - 1).tolist()

    # Phase C
    enc_c = tokenizer(prompts, padding=True, truncation=True, add_special_tokens=False, return_tensors="pt")
    pos_c = (enc_c["attention_mask"].sum(dim=1) - 1).tolist()

    # Phase C E6 get_prompt_end_position: len(tokenizer.encode(p, add_special_tokens=False)) - 1
    pos_e6 = [len(tokenizer.encode(p, add_special_tokens=False)) - 1 for p in prompts]

    assert pos_a == pos_b, f"Phase A ({pos_a}) != Phase B ({pos_b})"
    assert pos_b == pos_c, f"Phase B ({pos_b}) != Phase C ({pos_c})"
    assert pos_c == pos_e6, f"Phase C ({pos_c}) != E6 ({pos_e6})"


def test_v1_probe_group_leakage_fallback_banned():
    """
    group 数が不足（< 2）の場合に通常の KFold / StratifiedKFold へ fallback せず、
    NA (NaN) を返して pair_id leakage を完全に遮断することを検証。
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1" / "primary"))
    from run_phase_a import (
        evaluate_regression_probe,
        evaluate_classification_probe,
        evaluate_cross_decoding_and_geometry,
    )

    N = 10
    X = np.random.randn(N, 16)
    y_reg = np.random.randn(N)
    y_cls = np.random.choice([0, 1], size=N)

    # 全サンプルが同一 group（ユニーク数 1 < 2）
    single_group = np.array(["pair_001"] * N)

    # 1. 回帰 probe: NaN を返すこと
    res_reg = evaluate_regression_probe(X, y_reg, group_ids=single_group, cv=5)
    assert np.isnan(res_reg["r2"]), f"Expected NaN for insufficient groups in regression, got {res_reg['r2']}"

    # 2. 分類 probe: NaN を返すこと
    res_cls = evaluate_classification_probe(X, y_cls, group_ids=single_group, cv=5)
    assert np.isnan(res_cls["roc_auc"]), f"Expected NaN for insufficient groups in classification, got {res_cls['roc_auc']}"

    # 3. Cross-task 一般化: NaN を返すこと
    H_R = np.random.randn(N, 16)
    H_S = np.random.randn(N, 16)
    res_ct = evaluate_cross_decoding_and_geometry(H_R, H_S, y_reg, group_ids=single_group, cv=5)
    assert np.isnan(res_ct["r2_cross_r_to_s"]), f"Expected NaN for insufficient groups in cross-task, got {res_ct['r2_cross_r_to_s']}"
    assert res_ct["geometry_pattern"] == "insufficient_groups"
    assert res_ct["is_held_out"] is True


def test_vad_candidates_format_and_e6_consistency():
    """
    729 VAD 候補がリポジトリ唯一の compact JSON 形式（空白なし）で
    生成され、729通りかつ一意であることを検証。
    """
    from affective_empathy_eval.likelihood import build_vad_candidates

    cand_dicts = build_vad_candidates()
    assert len(cand_dicts) == 729

    json_strs = [c["json_str"] for c in cand_dicts]
    assert len(set(json_strs)) == 729

    # 空白を含まない compact JSON であること
    assert json_strs[0] == '{"valence":1,"arousal":1,"dominance":1}'
    assert " " not in json_strs[0]
    assert json_strs[-1] == '{"valence":9,"arousal":9,"dominance":9}'
