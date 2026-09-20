import numpy as np
import pandas as pd
import pytest

from affective_empathy_eval.likelihood import (
    build_va_candidates,
    build_vad_candidates,
    compute_distribution_metrics,
    compute_emd_va,
    compute_expected_va,
    compute_expected_va_from_probs,
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


def test_compute_expected_va_uniform():
    """Test 1: 均等分布の場合、期待値は E[V]=5.0, E[A]=5.0"""
    candidates = build_va_candidates()
    uniform_log_probs = np.zeros(81)
    ev, ea = compute_expected_va(uniform_log_probs, candidates)
    assert np.isclose(ev, 5.0, atol=1e-5)
    assert np.isclose(ea, 5.0, atol=1e-5)


def test_compute_expected_va_extreme_concentration():
    """Test 2: 特定の状態 (V=9, A=9) に確率が極端に集中している場合"""
    candidates = build_va_candidates()
    # V=9, A=9 のインデックス: (9-1)*9 + (9-1) = 80
    idx_9_9 = 80
    assert candidates[idx_9_9]["valence"] == 9 and candidates[idx_9_9]["arousal"] == 9

    log_scores = np.full(81, -100.0)
    log_scores[idx_9_9] = 0.0
    ev, ea = compute_expected_va(log_scores, candidates)
    assert ev > 8.9
    assert ea > 8.9


def test_expected_va_log_score_vs_probability_equivalence():
    """Test 3: 未正規化 log score 経由と正規化 probability 経由の期待値が完全に一致すること"""
    candidates = build_va_candidates()
    rng = np.random.default_rng(42)
    # 任意の非一様 log scores
    log_scores = rng.normal(loc=0.0, scale=2.0, size=81)

    # 手計算 Softmax による確率分布
    exp_scores = np.exp(log_scores - np.max(log_scores))
    probs = exp_scores / np.sum(exp_scores)

    ev_log, ea_log = compute_expected_va(log_scores, candidates)
    ev_prob, ea_prob = compute_expected_va_from_probs(probs, candidates)

    assert np.isclose(ev_log, ev_prob, atol=1e-6)
    assert np.isclose(ea_log, ea_prob, atol=1e-6)


def test_regression_double_softmax_distorts_expected_va():
    """
    Test 4 (回帰テスト): 二重 Softmax による期待値平滑化・歪みの検出
    注意: compute_expected_va に正規化済み probs を渡すことは正しい API 利用法ではない。
    過去に probs を誤って渡したことで期待値が 5.0 付近に潰れた事故を再発防止するための fixture 回帰テスト。
    """
    candidates = build_va_candidates()
    # V=9, A=9 に強く偏った分布を作成
    idx_9_9 = 80
    log_scores = np.full(81, -10.0)
    log_scores[idx_9_9] = 5.0
    exp_scores = np.exp(log_scores - np.max(log_scores))
    probs = exp_scores / np.sum(exp_scores)

    # 正しい期待値（> 8.0）
    correct_ev, correct_ea = compute_expected_va(log_scores, candidates)
    assert correct_ev > 8.0
    assert correct_ea > 8.0

    # 誤って probs を再度 Softmax に通した場合（二重 Softmax）
    # probs の最大値が高々 1.0 であるため、exp(probs - max) は全要素が ~exp(0) になり一様分布 (~5.0) に潰れる
    wrong_ev, wrong_ea = compute_expected_va(probs, candidates)

    # 正しい期待値と誤った二重 Softmax の期待値が大きく乖離することを確認
    assert abs(correct_ev - wrong_ev) > 2.0
    assert abs(correct_ea - wrong_ea) > 2.0


def test_compute_expected_va_from_probs_validation():
    """確率和が 1.0 でない不正な入力に対して ValueError を送出することの検証"""
    candidates = build_va_candidates()
    invalid_probs = np.ones(81)  # 和が 81.0
    with pytest.raises(ValueError, match="probs must sum to 1.0"):
        compute_expected_va_from_probs(invalid_probs, candidates)


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


def _one_hot_va(valence: int, arousal: int) -> np.ndarray:
    p = np.zeros(81, dtype=np.float64)
    p[(valence - 1) * 9 + (arousal - 1)] = 1.0
    return p


def test_emd_recovery_ratio_matches_readme_formula():
    from affective_empathy_eval.likelihood import (
        RECOVERY_RATIO_EPS,
        compute_emd_recovery_ratio,
        compute_distribution_metrics,
    )

    p_target = _one_hot_va(5, 5)
    p_clean = _one_hot_va(1, 1)
    p_patch_mid = _one_hot_va(3, 3)
    p_patch_full = _one_hot_va(5, 5)

    d_clean = compute_distribution_metrics(p_clean, p_target)["emd_va"]
    d_mid = compute_distribution_metrics(p_patch_mid, p_target)["emd_va"]
    d_full = compute_distribution_metrics(p_patch_full, p_target)["emd_va"]

    ratio_mid = compute_emd_recovery_ratio(d_clean, d_mid)
    expected_mid = (d_clean - d_mid) / (d_clean + RECOVERY_RATIO_EPS)
    assert pytest.approx(ratio_mid, abs=1e-12) == expected_mid
    assert 0.0 < ratio_mid < 1.0

    ratio_none = compute_emd_recovery_ratio(d_clean, d_clean)
    assert pytest.approx(ratio_none, abs=1e-12) == 0.0

    ratio_full = compute_emd_recovery_ratio(d_clean, d_full)
    assert pytest.approx(ratio_full, abs=1e-9) == d_clean / (d_clean + RECOVERY_RATIO_EPS)
    assert ratio_full > 0.999

    # より遠い patch は負の recovery（clean を target 近傍にする）
    p_clean_near = _one_hot_va(4, 5)
    d_clean_near = compute_distribution_metrics(p_clean_near, p_target)["emd_va"]
    p_farther = _one_hot_va(1, 1)
    d_far = compute_distribution_metrics(p_farther, p_target)["emd_va"]
    assert d_far > d_clean_near
    assert compute_emd_recovery_ratio(d_clean_near, d_far) < 0.0


def test_dry_run_va_label_vector_is_deterministic_fixture():
    from affective_empathy_eval.data import dry_run_va_label_vector

    df_missing = pd.DataFrame({"text": ["a", "b", "c", "d"]})
    v1 = dry_run_va_label_vector(df_missing, "reader_V", 4)
    v2 = dry_run_va_label_vector(df_missing, "reader_V", 4)
    assert np.allclose(v1, np.linspace(1.0, 9.0, 4))
    assert np.allclose(v1, v2)

    df_present = pd.DataFrame({"reader_V": [2.0, 4.0, 6.0]})
    got = dry_run_va_label_vector(df_present, "reader_V", 3)
    assert np.allclose(got, [2.0, 4.0, 6.0])


def test_compute_sequence_likelihoods_sliced_equivalence():
    """スライス計算による対数尤度が、全系列に対して log_softmax を計算した場合と完全に一致することを検証"""
    import torch
    import torch.nn.functional as F
    from affective_empathy_eval.likelihood import compute_sequence_likelihoods_for_candidates

    class DeterministicModel(torch.nn.Module):
        def __init__(self, vocab_size=50):
            super().__init__()
            self.vocab_size = vocab_size
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, attention_mask=None):
            bsz, seq_len = input_ids.shape
            # 決定論的な logits を生成
            steps = torch.arange(seq_len).unsqueeze(0).unsqueeze(-1).repeat(bsz, 1, self.vocab_size)
            vocab_idx = torch.arange(self.vocab_size).unsqueeze(0).unsqueeze(0).repeat(bsz, seq_len, 1)
            logits = (steps * 0.1 + vocab_idx * 0.05).float()
            from collections import namedtuple
            Outputs = namedtuple("Outputs", ["logits"])
            return Outputs(logits=logits)

    class CharTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 1

        def encode(self, text, add_special_tokens=False):
            return [ord(c) % 40 + 2 for c in text]

    model = DeterministicModel()
    tokenizer = CharTokenizer()
    candidates = [{"json_str": '{"v": 1}'}, {"json_str": '{"v": 5}'}, {"json_str": '{"v": 9}'}]
    prompt = "This is a prompt of reasonable length to test slicing."

    ll, probs = compute_sequence_likelihoods_for_candidates(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        candidates=candidates,
        device="cpu",
        batch_size=2,
    )

    # 全系列 log_softmax による参照値を手動計算して比較
    from affective_empathy_eval.likelihood import prepare_joint_sequence_with_boundary
    seq_list = []
    c_starts = []
    for c in [x["json_str"] for x in candidates]:
        f_ids, c_st = prepare_joint_sequence_with_boundary(prompt, c, tokenizer)
        seq_list.append(f_ids)
        c_starts.append(c_st)

    max_len = max(len(s) for s in seq_list)
    padded = [s + [0] * (max_len - len(s)) for s in seq_list]
    with torch.no_grad():
        full_logits = model(torch.tensor(padded, dtype=torch.long)).logits
        full_log_probs = F.log_softmax(full_logits[:, :-1, :], dim=-1)

    expected_ll = []
    for i, (f_ids, c_st) in enumerate(zip(seq_list, c_starts)):
        cand_toks = f_ids[c_st:]
        cur_ll = sum(full_log_probs[i, c_st - 1 + j, t].item() for j, t in enumerate(cand_toks))
        expected_ll.append(cur_ll)

    assert np.allclose(ll, expected_ll, atol=1e-5)
    exp_expected_ll = np.exp(expected_ll - np.max(expected_ll))
    expected_probs = exp_expected_ll / np.sum(exp_expected_ll)
    assert np.allclose(probs, expected_probs, atol=1e-5)




