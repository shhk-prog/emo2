import numpy as np
import pytest

from affective_empathy_eval.likelihood import (
    build_va_candidates,
    build_vad_candidates,
    compute_distribution_metrics,
    compute_emd_va,
    compute_expected_va,
    compute_marginal_distributions,
    get_euclidean_ground_cost_matrix,
)


def test_build_candidates():
    va_cands = build_va_candidates()
    assert len(va_cands) == 81
    assert va_cands[0]["valence"] == 1 and va_cands[0]["arousal"] == 1
    assert va_cands[-1]["valence"] == 9 and va_cands[-1]["arousal"] == 9

    vad_cands = build_vad_candidates()
    assert len(vad_cands) == 729


def test_compute_expected_va():
    # 均等分布の場合、期待値は 5.0
    uniform_log_probs = np.zeros(81)
    ev, ea = compute_expected_va(uniform_log_probs)
    assert pytest.approx(ev, abs=1e-3) == 5.0
    assert pytest.approx(ea, abs=1e-3) == 5.0

    # 特定の状態 (V=7, A=3) に確率1が集中している場合
    one_hot_log_probs = np.full(81, -100.0)
    # V=7, A=3 のインデックス: (7-1)*9 + (3-1) = 6*9 + 2 = 56
    one_hot_log_probs[56] = 0.0
    ev, ea = compute_expected_va(one_hot_log_probs)
    assert pytest.approx(ev, abs=1e-3) == 7.0
    assert pytest.approx(ea, abs=1e-3) == 3.0


def test_marginal_distributions():
    probs = np.ones(81) / 81.0
    pv, pa = compute_marginal_distributions(probs)
    assert len(pv) == 9 and len(pa) == 9
    assert np.allclose(pv, 1.0 / 9.0)
    assert np.allclose(pa, 1.0 / 9.0)


def test_ground_cost_and_emd():
    cost_mat = get_euclidean_ground_cost_matrix()
    assert cost_mat.shape == (81, 81)
    assert cost_mat[0, 0] == 0.0
    # (1,1) と (1,2) の距離は 1.0
    assert pytest.approx(cost_mat[0, 1], abs=1e-4) == 1.0
    # (1,1) と (9,9) の距離は sqrt(8^2 + 8^2) = sqrt(128) ≈ 11.3137
    assert pytest.approx(cost_mat[0, 80], abs=1e-4) == np.sqrt(128)

    # 同一分布間の EMD は 0.0
    p = np.ones(81) / 81.0
    emd_self = compute_emd_va(p, p, cost_mat)
    assert pytest.approx(emd_self, abs=1e-4) == 0.0

    # (V=1, A=1) から (V=2, A=1) への完全シフト: EMD は 1.0
    p_shift = np.zeros(81)
    p_shift[0] = 1.0  # (1,1)
    q_shift = np.zeros(81)
    q_shift[9] = 1.0  # (2,1) -> インデックス (2-1)*9 + 0 = 9
    emd_shift = compute_emd_va(p_shift, q_shift, cost_mat)
    assert pytest.approx(emd_shift, abs=1e-4) == 1.0


def test_compute_distribution_metrics():
    p = np.ones(81) / 81.0
    q = np.ones(81) / 81.0
    metrics = compute_distribution_metrics(p, q)
    assert "emd_va" in metrics
    assert "w1_v" in metrics
    assert "w1_a" in metrics
    assert "jsd" in metrics
    assert pytest.approx(metrics["emd_va"], abs=1e-4) == 0.0
    assert pytest.approx(metrics["jsd"], abs=1e-4) == 0.0


def test_compute_sequence_likelihoods_mock():
    from affective_empathy_eval.likelihood import compute_sequence_likelihoods_for_candidates
    import torch

    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 1
        def encode(self, text, add_special_tokens=False):
            return [hash(c) % 50 + 2 for c in text.split() if c] or [2]

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy_param = torch.nn.Parameter(torch.zeros(1))
        def forward(self, input_ids, attention_mask=None):
            bsz, seq_len = input_ids.shape
            logits = torch.randn(bsz, seq_len, 100)
            from collections import namedtuple
            Outputs = namedtuple("Outputs", ["logits"])
            return Outputs(logits=logits)

    model = MockModel()
    tok = MockTokenizer()
    cands = build_va_candidates()

    ll, probs = compute_sequence_likelihoods_for_candidates(
        model=model, tokenizer=tok, prompt="Test prompt", candidates=cands, device="cpu", batch_size=32
    )

    assert len(ll) == 81
    assert len(probs) == 81
    assert pytest.approx(float(np.sum(probs)), abs=1e-5) == 1.0
    assert np.all(probs >= 0.0)


def test_encode_prompt_canonical_and_anchors():
    from affective_empathy_eval.prompts import (
        build_prompt,
        encode_prompt_canonical,
        find_semantic_anchors,
        TaskType,
    )

    class MockTok:
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]
        def __call__(self, text, return_tensors="pt", add_special_tokens=False):
            import torch
            ids = torch.tensor([[ord(c) for c in text]])
            return {"input_ids": ids}

    tok = MockTok()
    prompt = "Text: Hello world\n\nResponse: "
    enc = encode_prompt_canonical(tok, prompt, return_tensors="pt")
    assert "input_ids" in enc
    assert enc["input_ids"].shape[1] == len(prompt)

    # アンカー検出で prompt_end, pre_response が正しく取得できること
    anchors = find_semantic_anchors(enc["input_ids"][0].tolist(), tok, "Hello world")
    assert "prompt_end" in anchors
    assert "pre_response" in anchors
    assert "response_start" in anchors
    assert anchors["prompt_end"] == len(prompt) - 1
    assert anchors["stimulus_end"] >= 0


def test_joint_tokenization_boundary():
    from affective_empathy_eval.likelihood import compute_sequence_likelihoods_for_candidates
    import torch

    class CharTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 1
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

    class IdentityModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))
        def forward(self, input_ids, attention_mask=None):
            bsz, seq_len = input_ids.shape
            logits = torch.zeros(bsz, seq_len, 256)
            from collections import namedtuple
            return namedtuple("Outputs", ["logits"])(logits=logits)

    model = IdentityModel()
    tok = CharTokenizer()
    candidates = ['{"v": 1}', '{"v": 2}']

    ll, probs = compute_sequence_likelihoods_for_candidates(
        model=model,
        tokenizer=tok,
        prompt="Prompt: ",
        candidates=candidates,
        device="cpu",
    )
    assert len(ll) == 2
    assert len(probs) == 2
    assert pytest.approx(float(np.sum(probs)), abs=1e-5) == 1.0


def test_prepare_joint_sequence_boundary_strict():
    from affective_empathy_eval.likelihood import prepare_joint_sequence_with_boundary

    class MockMergingTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 1
        def encode(self, text, add_special_tokens=False):
            # 単純なトークナイズ（文字単位だが、": {" は 1 トークンにマージされる模擬挙動）
            tokens = []
            i = 0
            while i < len(text):
                if text[i:i+3] == ": {":
                    tokens.append(999)  # マージトークン
                    i += 3
                else:
                    tokens.append(ord(text[i]))
                    i += 1
            return tokens

    tok = MockMergingTokenizer()
    cand = '{"valence": 5, "arousal": 5}'

    # 1. 境界マージが起きない安全な delimiter（改行）を使用した場合
    prompt_safe = "Text: Hello\n\nResponse:\n"
    full_ids, c_start = prepare_joint_sequence_with_boundary(
        prompt=prompt_safe,
        candidate=cand,
        tokenizer=tok,
        delimiter="",
        require_strict_prefix=True,
    )
    p_ids = tok.encode(prompt_safe, add_special_tokens=False)
    assert full_ids[:len(p_ids)] == p_ids
    assert c_start == len(p_ids)

    # 2. 境界マージが起きるケース（末尾が ": " で cand が "{" から始まる場合）
    prompt_merge = "Text: Hello\n\nResponse"
    # delimiter=": " だと ": {" にマージ
    full_ids_m, c_start_m = prepare_joint_sequence_with_boundary(
        prompt=prompt_merge,
        candidate=cand,
        tokenizer=tok,
        delimiter=": ",
        require_strict_prefix=False,
    )
    assert c_start_m >= 0
    assert len(full_ids_m) > 0


def test_offset_based_generation_stage_tokens():
    from affective_empathy_eval.prompts import get_generation_stage_tokens

    class MockOffsetTokenizer:
        def decode(self, tokens):
            return "".join([chr(t) for t in tokens])
        def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False):
            # 文字単位の offset mapping
            offsets = [(i, i + 1) for i in range(len(text))]
            return {"offset_mapping": offsets}

    tok = MockOffsetTokenizer()
    cand_str = '{"valence": 7, "arousal": 3}'
    cand_tokens = [ord(c) for c in cand_str]

    stages = get_generation_stage_tokens(cand_tokens, tok, candidate_str=cand_str)
    assert stages["candidate_start"] == 0
    assert stages["response_start"] == 0
    assert stages["response_end"] == len(cand_tokens) - 1
    # valence 数値 '7' の文字位置は 12
    assert cand_tokens[stages["V_value"]] == ord('7')
    # arousal 数値 '3' の文字位置は 26
    assert cand_tokens[stages["A_value"]] == ord('3')
    assert stages["pre_V"] < stages["V_value"]
    assert stages["pre_A"] < stages["A_value"]



