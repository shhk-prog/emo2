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


RECOVERY_RATIO_EPS = 1e-12


def compute_emd_recovery_ratio(
    d_clean_to_target: float,
    d_patch_to_target: float,
    eps: float = RECOVERY_RATIO_EPS,
) -> float:
    """V2 RQ4 Recovery 比。

    Recovery = (W1(P_clean, P_target) - W1(P_patch, P_target)) / W1(P_clean, P_target)

    正の値は、未介入分布よりパッチ後分布の方が target に近いことを意味する。
    分母はゼロ除算回避のため eps を加える。
    """
    return (float(d_clean_to_target) - float(d_patch_to_target)) / (
        float(d_clean_to_target) + float(eps)
    )


def prepare_joint_sequence_with_boundary(
    prompt: str,
    candidate: str,
    tokenizer: Any,
    delimiter: str = "",
    require_strict_prefix: bool = False,
) -> tuple[list[int], int]:
    """
    プロンプトと候補文字列の結合（Joint Sequence）をトークナイズし、
    candidate 開始トークンインデックス（cand_start）を同定する。

    prompt末尾とcandidateの間に安定したdelimiter（または空白/改行）がある場合、
    tokens(prompt + delimiter) は tokens(prompt + delimiter + candidate) の厳密なプレフィックスとなる。
    境界マージが発生した場合は最長共通プレフィックス（LCP）と文字オフセットから安全に特定する。
    """
    prompt_full = prompt + delimiter
    full_text = prompt_full + candidate

    prompt_ids = tokenizer.encode(prompt_full, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    p_len = len(prompt_ids)
    if len(full_ids) >= p_len and full_ids[:p_len] == prompt_ids:
        # 厳密な prefix 一致が成立
        return full_ids, p_len

    if require_strict_prefix:
        raise ValueError(
            f"Strict prefix property violated across boundary for candidate: {candidate[:20]}... "
            f"Prompt ids ({p_len}): {prompt_ids[-3:]}, Full ids prefix: {full_ids[:p_len][-3:]}"
        )

    # 境界跨ぎトークン（BPEマージ）の安全な同定
    # 1. offset_mapping を試行
    try:
        enc = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc.get("offset_mapping", None)
        if offsets is not None:
            prompt_char_len = len(prompt_full)
            for idx, (ch_start, ch_end) in enumerate(offsets):
                # 候補文字列領域に含まれるトークン
                if ch_start >= prompt_char_len:
                    return full_ids, idx
    except Exception:
        pass

    # 2. 最長共通プレフィックス (LCP) によるフォールバック
    match_len = 0
    while match_len < len(prompt_ids) and match_len < len(full_ids) and prompt_ids[match_len] == full_ids[match_len]:
        match_len += 1

    return full_ids, match_len


def resolve_joint_stage_index(
    cand_start: int,
    stage_name: str,
    stage_offsets: dict[str, int],
    seq_len: int,
) -> int:
    """生成段階の絶対 token 位置。prompt_end への丸めはしない。"""
    key = "candidate_start" if stage_name == "response_start" else stage_name
    if key not in stage_offsets:
        raise KeyError(f"Unknown generation stage {stage_name!r}; available={sorted(stage_offsets)}")
    t_idx = int(cand_start) + int(stage_offsets[key])
    if t_idx < 0 or t_idx >= int(seq_len):
        raise ValueError(
            f"Generation-stage index {t_idx} is outside joint sequence [0, {seq_len}). "
            "Do not clamp to prompt_end."
        )
    return t_idx


def compute_sequence_likelihoods_for_candidates(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidates: list[dict[str, Any]] | list[str] | None = None,
    device: str | torch.device = "cuda",
    batch_size: int = 81,
    normalize_length: bool = False,
    delimiter: str = "",
    generation_patch: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    プロンプトに続く各候補文字列（81通りのVA JSON、729通りのVAD JSON等）について、
    完全な条件付き対数尤度 sum_t log P(token_t | prompt + cand_{<t}) をバッチ計算する。
    BPE境界の不一致を防ぐため、full_text = prompt + delimiter + candidate を結合した
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

    num_cands = len(cand_strings)

    joint_token_ids_list = []
    cand_start_indices = []

    for c in cand_strings:
        full_ids, c_start = prepare_joint_sequence_with_boundary(
            prompt=prompt,
            candidate=c,
            tokenizer=tokenizer,
            delimiter=delimiter,
            require_strict_prefix=False,
        )
        joint_token_ids_list.append(full_ids)
        cand_start_indices.append(c_start)

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
            if generation_patch is None:
                outputs = model(input_ids=inp_tensor, attention_mask=attn_tensor)
            else:
                from affective_empathy_eval.models.hooks import ActivationHookManager, HookPoint

                token_index = int(generation_patch["token_index"])
                for seq in b_full_ids:
                    if token_index >= len(seq):
                        raise ValueError(
                            f"Generation-stage patch index {token_index} exceeds joint length {len(seq)}."
                        )
                with ActivationHookManager(generation_patch["adapter"]) as hook_mgr:
                    mode = str(generation_patch.get("mode", "inject")).lower()
                    hook_point = generation_patch.get("hook_point", HookPoint.POST_MLP_RESID)
                    layer_idx = int(generation_patch["layer_idx"])

                    if "direction" in generation_patch:
                        dir_val = generation_patch["direction"]
                        alpha_val = float(generation_patch.get("alpha", 1.0))
                        h_std_val = float(generation_patch.get("hidden_std", 1.0))
                        hook_mgr.register_direction_intervention_hook(
                            layer_idx=layer_idx,
                            direction=dir_val,
                            alpha=alpha_val,
                            hidden_std=h_std_val,
                            token_indices=token_index,
                            hook_point=hook_point,
                            mode=mode,
                        )
                    elif "patch_tensor" in generation_patch:
                        patch_t = generation_patch["patch_tensor"]
                        if mode == "inject":
                            # additive injection with direct patch tensor
                            hook_mgr.register_direction_intervention_hook(
                                layer_idx=layer_idx,
                                direction=patch_t,
                                alpha=float(generation_patch.get("alpha", 1.0)),
                                hidden_std=float(generation_patch.get("hidden_std", 1.0)),
                                token_indices=token_index,
                                hook_point=hook_point,
                                mode="inject",
                            )
                        else:
                            # replace mode
                            hook_mgr.register_patch_hook(
                                layer_idx=layer_idx,
                                patch_tensor=patch_t,
                                token_indices=token_index,
                                hook_point=hook_point,
                            )
                    else:
                        raise KeyError("generation_patch requires either 'direction' or 'patch_tensor'.")

                    outputs = model(input_ids=inp_tensor, attention_mask=attn_tensor)

            logits = outputs.logits  # (batch_size, max_seq_len, vocab_size)
            # Memory optimization: Only compute float cast & log_softmax on steps covering candidates
            min_step = max(0, min(b_start_idxs) - 1)
            if min_step < logits.shape[1] - 1:
                sliced_logits = logits[:, min_step:-1, :].float()
                log_probs_slice = F.log_softmax(sliced_logits, dim=-1)
            else:
                log_probs_slice = None

        for i, (full_ids, c_start) in enumerate(zip(b_full_ids, b_start_idxs)):
            cand_tokens = full_ids[c_start:]
            c_len = len(cand_tokens)
            eval_log_probs = []
            for j, token_id in enumerate(cand_tokens):
                step_idx = c_start - 1 + j
                rel_idx = step_idx - min_step
                if log_probs_slice is not None and 0 <= rel_idx < log_probs_slice.shape[1]:
                    eval_log_probs.append(log_probs_slice[i, rel_idx, token_id].item())

            seq_ll = sum(eval_log_probs) if len(eval_log_probs) > 0 else -100.0
            if normalize_length and c_len > 0:
                seq_ll /= c_len
            log_likelihoods[b_start + i] = seq_ll

        # Explicitly clean up batch tensors to prevent CUDA memory fragmentation
        del outputs, logits
        if log_probs_slice is not None:
            del sliced_logits, log_probs_slice

    # Softmax により正規化された確率分布を計算
    l_max = np.max(log_likelihoods)
    exp_ll = np.exp(log_likelihoods - l_max)
    probs = exp_ll / np.sum(exp_ll)

    return log_likelihoods, probs


def evaluate_expected_va_from_prompt(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidates: list[dict[str, Any]] | None = None,
    device: str | torch.device = "cuda",
    batch_size: int = 81,
    delimiter: str = "",
) -> tuple[float, float, np.ndarray]:
    """
    単一プロンプトに対し、81通りのVA候補についての条件付き確率分布を求め、
    期待値 (E[V], E[A]) および確率分布 (81,) を算出する。
    """
    if candidates is None:
        candidates = build_va_candidates()

    log_ll, probs = compute_sequence_likelihoods_for_candidates(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        candidates=candidates,
        device=device,
        batch_size=batch_size,
        delimiter=delimiter,
    )
    ev, ea = compute_expected_va(log_ll, candidates=candidates)
    return ev, ea, probs


def evaluate_expected_vad_from_prompt(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidates: list[dict[str, Any]] | None = None,
    device: str | torch.device = "cuda",
    batch_size: int = 81,
    delimiter: str = "",
) -> tuple[float, float, float, np.ndarray]:
    """
    単一プロンプトに対し、729通りのVAD候補についての条件付き確率分布を求め、
    期待値 (E[V], E[A], E[D]) および確率分布 (729,) を算出する。
    """
    if candidates is None:
        candidates = build_vad_candidates()

    log_ll, probs = compute_sequence_likelihoods_for_candidates(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        candidates=candidates,
        device=device,
        batch_size=batch_size,
        delimiter=delimiter,
    )

    v_values = np.array([c["valence"] for c in candidates], dtype=np.float64)
    a_values = np.array([c["arousal"] for c in candidates], dtype=np.float64)
    d_values = np.array([c["dominance"] for c in candidates], dtype=np.float64)

    ev = float(np.sum(probs * v_values))
    ea = float(np.sum(probs * a_values))
    ed = float(np.sum(probs * d_values))
    return ev, ea, ed, probs



