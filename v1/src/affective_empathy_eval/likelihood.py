"""
尤度算出・期待値・分布間距離（EMD_VA, W1, JSD）計算モジュール
"""

import json
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance


def build_va_candidates() -> list[dict[str, Any]]:
    """
    81通りの VA 候補（Valence, Arousal in 1..9）を生成
    """
    candidates = []
    for v in range(1, 10):
        for a in range(1, 10):
            json_str = json.dumps({"valence": v, "arousal": a}, separators=(",", ":"))
            candidates.append({
                "valence": v,
                "arousal": a,
                "json_str": json_str,
            })
    return candidates


def build_vad_candidates() -> list[dict[str, Any]]:
    """
    729通りの VAD 候補（Valence, Arousal, Dominance in 1..9）を生成
    """
    candidates = []
    for v in range(1, 10):
        for a in range(1, 10):
            for d in range(1, 10):
                json_str = json.dumps({"valence": v, "arousal": a, "dominance": d}, separators=(",", ":"))
                candidates.append({
                    "valence": v,
                    "arousal": a,
                    "dominance": d,
                    "json_str": json_str,
                })
    return candidates


def compute_expected_va(
    log_probs: torch.Tensor | np.ndarray,
    candidates: list[dict[str, Any]] | None = None,
) -> tuple[float, float]:
    """
    81候補の対数確率から Softmax 確率分布を求め、Valence および Arousal の期待値を算出
    """
    if candidates is None:
        candidates = build_va_candidates()

    if isinstance(log_probs, torch.Tensor):
        probs = F.softmax(log_probs, dim=-1).cpu().numpy()
    else:
        probs = np.exp(log_probs - np.max(log_probs))
        probs = probs / np.sum(probs)

    v_values = np.array([c["valence"] for c in candidates], dtype=np.float64)
    a_values = np.array([c["arousal"] for c in candidates], dtype=np.float64)

    expected_v = float(np.sum(probs * v_values))
    expected_a = float(np.sum(probs * a_values))
    return expected_v, expected_a


def compute_marginal_distributions(
    probs: np.ndarray,
    candidates: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    81状態の同時確率分布から Valence (1..9) および Arousal (1..9) の周辺確率分布を算出
    """
    if candidates is None:
        candidates = build_va_candidates()

    p_v = np.zeros(9, dtype=np.float64)
    p_a = np.zeros(9, dtype=np.float64)

    for i, c in enumerate(candidates):
        p_v[c["valence"] - 1] += probs[i]
        p_a[c["arousal"] - 1] += probs[i]

    # 正規化の微調整
    p_v /= np.sum(p_v)
    p_a /= np.sum(p_a)
    return p_v, p_a


def get_euclidean_ground_cost_matrix(candidates: list[dict[str, Any]] | None = None) -> np.ndarray:
    """
    81格子上のユークリッド距離行列 C を事前生成: c((v,a), (v',a')) = sqrt((v-v')^2 + (a-a')^2)
    """
    if candidates is None:
        candidates = build_va_candidates()

    coords = np.array([[c["valence"], c["arousal"]] for c in candidates], dtype=np.float64)
    return cdist(coords, coords, metric="euclidean")


_PRECOMPUTED_GROUND_COST: np.ndarray | None = None


def compute_emd_va(
    probs_p: np.ndarray,
    probs_q: np.ndarray,
    ground_cost: np.ndarray | None = None,
) -> float:
    """
    Joint 81状態の 2D Earth Mover's Distance (EMD_VA) を算出
    POT (ot) パッケージが利用可能な場合は ot.emd2、それ以外は scipy.optimize.linprog によるフォールバック
    """
    global _PRECOMPUTED_GROUND_COST
    if ground_cost is None:
        if _PRECOMPUTED_GROUND_COST is None:
            _PRECOMPUTED_GROUND_COST = get_euclidean_ground_cost_matrix()
        ground_cost = _PRECOMPUTED_GROUND_COST

    p = np.asarray(probs_p, dtype=np.float64).ravel()
    q = np.asarray(probs_q, dtype=np.float64).ravel()

    # 生ロジットが渡された場合、または負値を含む場合は softmax を適用
    if np.any(p < 0) or np.any(q < 0):
        from scipy.special import softmax
        p = softmax(p)
        q = softmax(q)
    else:
        p = np.clip(p, 0.0, None)
        q = np.clip(q, 0.0, None)
        sum_p = np.sum(p)
        sum_q = np.sum(q)
        p = p / sum_p if sum_p > 0 else np.ones_like(p) / len(p)
        q = q / sum_q if sum_q > 0 else np.ones_like(q) / len(q)

    # 厳密なシンプレックス正規化（浮動小数点誤差対策）
    p = p / np.sum(p)
    q = q / np.sum(q)

    try:
        import ot
        val = ot.emd2(p, q, ground_cost)
        if not np.isnan(val) and not np.isinf(val):
            return float(val)
    except Exception:
        pass

    # scipy linprog による最適輸送解法（81×81 = 6561変数）
    from scipy.optimize import linprog

    n = len(p)
    c = ground_cost.flatten()

    # A_eq @ x = b_eq
    # 行和制約: sum_j x_ij = p_i (i = 1..n)
    # 列和制約: sum_i x_ij = q_j (j = 1..n-1) (冗長排除のため 1つ減らす)
    A_row = np.zeros((n, n * n))
    for i in range(n):
        A_row[i, i * n : (i + 1) * n] = 1.0

    A_col = np.zeros((n - 1, n * n))
    for j in range(n - 1):
        A_col[j, j::n] = 1.0

    A_eq = np.vstack([A_row, A_col])
    b_eq = np.concatenate([p, q[:-1]])

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if res.success:
        return float(res.fun)
    else:
        # 簡易近似: 周辺分布の平均距離
        p_v, p_a = compute_marginal_distributions(p)
        q_v, q_a = compute_marginal_distributions(q)
        return float(np.sqrt(wasserstein_distance(range(1, 10), range(1, 10), p_v, q_v)**2 +
                             wasserstein_distance(range(1, 10), range(1, 10), p_a, q_a)**2))


def compute_distribution_metrics(
    probs_p: np.ndarray,
    probs_q: np.ndarray,
) -> dict[str, float]:
    """
    Primary (EMD_VA) および Secondary (W1_V, W1_A, JSD) 指標を一括算出
    """
    p = np.asarray(probs_p, dtype=np.float64).ravel()
    q = np.asarray(probs_q, dtype=np.float64).ravel()

    # 生ロジットが渡された場合、または負値を含む場合は softmax を適用
    if np.any(p < 0) or np.any(q < 0):
        from scipy.special import softmax
        p = softmax(p)
        q = softmax(q)
    else:
        p = np.clip(p, 0.0, None)
        q = np.clip(q, 0.0, None)
        sum_p = np.sum(p)
        sum_q = np.sum(q)
        p = p / sum_p if sum_p > 0 else np.ones_like(p) / len(p)
        q = q / sum_q if sum_q > 0 else np.ones_like(q) / len(q)

    # 厳密なシンプレックス正規化（浮動小数点誤差対策）
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    p = p / np.sum(p)
    q = q / np.sum(q)

    # 1. Primary: EMD_VA
    emd_va = compute_emd_va(p, q)

    # 2. Secondary: W1_V, W1_A
    p_v, p_a = compute_marginal_distributions(p)
    q_v, q_a = compute_marginal_distributions(q)
    # 周辺分布も厳密に非負・正規化
    p_v = np.clip(p_v, 0.0, None)
    q_v = np.clip(q_v, 0.0, None)
    p_a = np.clip(p_a, 0.0, None)
    q_a = np.clip(q_a, 0.0, None)
    p_v = p_v / np.sum(p_v)
    q_v = q_v / np.sum(q_v)
    p_a = p_a / np.sum(p_a)
    q_a = q_a / np.sum(q_a)

    coords_1d = np.arange(1, 10, dtype=np.float64)
    w1_v = float(wasserstein_distance(coords_1d, coords_1d, p_v, q_v))
    w1_a = float(wasserstein_distance(coords_1d, coords_1d, p_a, q_a))

    # 3. Secondary: JSD
    m = 0.5 * (p + q)
    # KL(p || m)
    kl_pm = np.sum(np.where(p > 0, p * np.log((p + 1e-12) / (m + 1e-12)), 0.0))
    kl_qm = np.sum(np.where(q > 0, q * np.log((q + 1e-12) / (m + 1e-12)), 0.0))
    jsd = float(0.5 * (kl_pm + kl_qm))

    return {
        "emd_va": emd_va,
        "w1_v": w1_v,
        "w1_a": w1_a,
        "jsd": jsd,
    }


def compute_sequence_likelihoods_for_candidates(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidates: list[dict[str, Any]] | list[str] | None = None,
    device: str | torch.device = "cuda",
    batch_size: int = 81,
    normalize_length: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    プロンプトに続く各候補文字列（81通りのJSON等）について、
    完全な条件付き対数尤度 sum_t log P(token_t | prompt + cand_{<t}) をバッチ計算する。
    BPE境界の不一致を防ぐため、各候補について full_text = prompt + candidate を結合した
    Joint Tokenization を行い、prompt 終端以降の候補トークンのみを厳密にスコアリングする。

    戻り値:
        log_likelihoods: (num_candidates,) - 各候補の対数尤度配列
        probs: (num_candidates,) - Softmax により正規化された確率分布
    """
    if candidates is None:
        cand_dicts = build_va_candidates()
        cand_strings = [c["json_str"] for c in cand_dicts]
    elif isinstance(candidates[0], dict) and "json_str" in candidates[0]:
        cand_strings = [c["json_str"] for c in candidates]
    elif isinstance(candidates[0], str):
        cand_strings = list(candidates)
    else:
        raise ValueError("candidates must be a list of dicts with 'json_str' or a list of strings")

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    num_cands = len(cand_strings)

    # 候補ごとに joint tokenization: full_text = prompt + c
    # prompt との境界（candidate 開始トークンインデックス）を厳密に特定
    joint_token_ids_list = []
    cand_start_indices = []

    for c in cand_strings:
        full_text = prompt + c
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        # 境界検出:
        # offset_mapping を試行
        start_idx = -1
        try:
            enc_with_offsets = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
            offsets = enc_with_offsets.get("offset_mapping", None)
            if offsets is not None:
                p_char_len = len(prompt)
                for idx, (ch_start, ch_end) in enumerate(offsets):
                    if ch_start >= p_char_len:
                        start_idx = idx
                        break
        except Exception:
            start_idx = -1

        if start_idx == -1:
            # 最長共通プレフィックスによるフォールバック
            match_len = 0
            while match_len < len(prompt_ids) and match_len < len(full_ids) and prompt_ids[match_len] == full_ids[match_len]:
                match_len += 1
            start_idx = match_len

        joint_token_ids_list.append(full_ids)
        cand_start_indices.append(start_idx)

    log_likelihoods = np.zeros(num_cands, dtype=np.float64)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)

    model_device = next(model.parameters()).device if hasattr(model, "parameters") else device

    for b_start in range(0, num_cands, batch_size):
        b_end = min(b_start + batch_size, num_cands)
        b_full_ids = joint_token_ids_list[b_start:b_end]
        b_start_idxs = cand_start_indices[b_start:b_end]

        max_seq_len = max(len(seq) for seq in b_full_ids)

        batch_input_ids = []
        batch_attention_mask = []
        for seq in b_full_ids:
            pad_len = max_seq_len - len(seq)
            batch_input_ids.append(seq + [pad_token_id] * pad_len)
            batch_attention_mask.append([1] * len(seq) + [0] * pad_len)

        inp_tensor = torch.tensor(batch_input_ids, dtype=torch.long, device=model_device)
        attn_tensor = torch.tensor(batch_attention_mask, dtype=torch.long, device=model_device)

        with torch.no_grad():
            outputs = model(input_ids=inp_tensor, attention_mask=attn_tensor)
            logits = outputs.logits  # (batch_size, max_seq_len, vocab_size)
            log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)

        for i, (full_ids, c_start) in enumerate(zip(b_full_ids, b_start_idxs)):
            # c_start 以降の各トークン t について、位置 t-1 のロジットから full_ids[t] の対数確率を取得
            cand_tokens = full_ids[c_start:]
            c_len = len(cand_tokens)
            eval_log_probs = []
            for j, token_id in enumerate(cand_tokens):
                step_idx = c_start - 1 + j
                if 0 <= step_idx < log_probs.shape[1]:
                    eval_log_probs.append(log_probs[i, step_idx, token_id].item())

            seq_ll = sum(eval_log_probs) if len(eval_log_probs) > 0 else -100.0
            if normalize_length and c_len > 0:
                seq_ll /= c_len
            log_likelihoods[b_start + i] = seq_ll

    # Softmax により正規化された確率分布を計算
    l_max = np.max(log_likelihoods)
    exp_ll = np.exp(log_likelihoods - l_max)
    probs = exp_ll / np.sum(exp_ll)

    return log_likelihoods, probs


