#!/usr/bin/env python3
"""
V3-RQ1: 内部情動状態から自己報告への因果的依存性の検証および Go/No-Go ゲート判定
外部刺激ラベル由来の d_V, d_A による介入、Centered projection removal、Topic control
実モデルロード・介入・共通Sequence LikelihoodおよびBootstrap CIによる厳密ゲート判定
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import yaml
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # --dry-run は transformers 未導入環境でも起動できるようにする
    AutoModelForCausalLM = None  # type: ignore[misc, assignment]
    AutoTokenizer = None  # type: ignore[misc, assignment]

from affective_empathy_eval.interventions import (
    compute_orthonormal_subspace,
    estimate_interventional_slope,
    extract_conditional_directions,
    generate_control_directions,
)
from affective_empathy_eval.likelihood import (
    build_va_candidates,
    compute_expected_va,
    compute_sequence_likelihoods_for_candidates,
)
from affective_empathy_eval.models.adapters import get_model_adapter
from affective_empathy_eval.models.hooks import ActivationHookManager, HookPoint
from affective_empathy_eval.data import (
    describe_loaded_frame,
    dry_run_va_label_vector,
    load_v3_matched_pair_table,
    resolve_matched_neutral_text,
)
from affective_empathy_eval.models.registry import (
    add_model_selection_args,
    get_registry,
    resolve_architecture_dims,
    resolve_instruct_target_from_args,
)
from affective_empathy_eval.prompts import (
    TaskType,
    build_prompt,
    encode_prompt_canonical,
    find_semantic_anchors,
)
from affective_empathy_eval.statistics import compute_bootstrap_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run V3-RQ1 State Induction and Go/No-Go Gate")
    parser.add_argument("--config", type=str, default="configs/v3_experiments.yaml", help="Path to V3 config")
    parser.add_argument("--models-config", type=str, default="configs/models.yaml", help="Path to models config")
    parser.add_argument("--pilot", action="store_true", help="Run in pilot mode (50 pairs smoke test)")
    parser.add_argument("--dry-run", action="store_true", help="Run in mock/dry-run mode")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Target layer for state injection. If omitted, computed from --relative-depth.",
    )
    parser.add_argument(
        "--relative-depth",
        type=float,
        default=0.5,
        help="A priori mid-depth used when --layer is omitted: l = round(d * (L - 1))",
    )
    add_model_selection_args(parser)
    return parser.parse_args()


def simulate_mock_intervention_responses(
    df: pd.DataFrame,
    alpha_grid: List[float],
) -> Dict[str, Any]:
    """
    dry-run用のモック介入応答シミュレーション
    5 Criteria (Sufficiency, Necessity, Specificity, Dose-response, Selectivity) を模擬
    各指標についてサンプル値および Bootstrap 95% 信頼区間を生成
    """
    rng = np.random.default_rng(42)
    N = len(df)
    v_clean = dry_run_va_label_vector(df, "reader_V", N)
    a_clean = dry_run_va_label_vector(df, "reader_A", N)

    # 1. Dose-response: alpha に応じて報告値が線形シフト
    dose_responses_v = []
    dose_responses_a = []
    sample_slopes_v = []
    sample_slopes_a = []

    for alpha in alpha_grid:
        shift_v = alpha * 0.85 + rng.normal(0, 0.05, N)
        shift_a = alpha * 0.70 + rng.normal(0, 0.05, N)
        dose_responses_v.append(float(np.mean(shift_v)))
        dose_responses_a.append(float(np.mean(shift_a)))

    slope_v = estimate_interventional_slope(alpha_grid, dose_responses_v)
    slope_a = estimate_interventional_slope(alpha_grid, dose_responses_a)

    for i in range(N):
        s_v = [alpha * 0.85 + rng.normal(0, 0.05) for alpha in alpha_grid]
        s_a = [alpha * 0.70 + rng.normal(0, 0.05) for alpha in alpha_grid]
        sample_slopes_v.append(estimate_interventional_slope(alpha_grid, s_v))
        sample_slopes_a.append(estimate_interventional_slope(alpha_grid, s_a))

    pt_sv, sv_low, sv_high = compute_bootstrap_ci(sample_slopes_v)
    pt_sa, sa_low, sa_high = compute_bootstrap_ci(sample_slopes_a)

    # 2. Specificity: 情動方向 vs ランダム方向 vs 直交方向 (VA両軸独立)
    sample_spec_diff_v = []
    sample_spec_rand_v = []
    sample_spec_perp_v = []
    sample_spec_diff_a = []
    sample_spec_rand_a = []
    sample_spec_perp_a = []
    for i in range(N):
        eff_aff_v = float(0.85 + rng.normal(0, 0.04))
        eff_rand_v = float(0.10 + rng.normal(0, 0.03))
        eff_perp_v = float(0.12 + rng.normal(0, 0.03))
        sample_spec_diff_v.append(eff_aff_v - max(eff_rand_v, eff_perp_v))
        sample_spec_rand_v.append(eff_aff_v - eff_rand_v)
        sample_spec_perp_v.append(eff_aff_v - eff_perp_v)

        eff_aff_a = float(0.70 + rng.normal(0, 0.04))
        eff_rand_a = float(0.09 + rng.normal(0, 0.03))
        eff_perp_a = float(0.11 + rng.normal(0, 0.03))
        sample_spec_diff_a.append(eff_aff_a - max(eff_rand_a, eff_perp_a))
        sample_spec_rand_a.append(eff_aff_a - eff_rand_a)
        sample_spec_perp_a.append(eff_aff_a - eff_perp_a)

    pt_spec_v, spec_v_low, spec_v_high = compute_bootstrap_ci(sample_spec_diff_v)
    pt_spec_rand_v, spec_rand_v_low, spec_rand_v_high = compute_bootstrap_ci(sample_spec_rand_v)
    pt_spec_perp_v, spec_perp_v_low, spec_perp_v_high = compute_bootstrap_ci(sample_spec_perp_v)

    pt_spec_a, spec_a_low, spec_a_high = compute_bootstrap_ci(sample_spec_diff_a)
    pt_spec_rand_a, spec_rand_a_low, spec_rand_a_high = compute_bootstrap_ci(sample_spec_rand_a)
    pt_spec_perp_a, spec_perp_a_low, spec_perp_a_high = compute_bootstrap_ci(sample_spec_perp_a)

    # 3. Necessity: Centered projection removal による自然変位の減衰 (VA両軸独立)
    natural_shift_v = float(np.std(v_clean))
    natural_shift_a = float(np.std(a_clean))
    sample_att_v = [float(0.42 + rng.normal(0, 0.05)) for _ in range(N)]
    sample_att_a = [float(0.38 + rng.normal(0, 0.05)) for _ in range(N)]
    pt_att_v, att_v_low, att_v_high = compute_bootstrap_ci(sample_att_v)
    pt_att_a, att_a_low, att_a_high = compute_bootstrap_ci(sample_att_a)
    attenuated_shift_v = natural_shift_v * (1.0 - pt_att_v)
    attenuated_shift_a = natural_shift_a * (1.0 - pt_att_a)

    # 4. Topic control: Topic 課題への非特異的摂動 (TVD) が小さいことを確認する統制 (VA両軸独立)
    sample_topic_tvd_v = [float(np.clip(0.05 + rng.normal(0, 0.01), 0.0, 1.0)) for _ in range(N)]
    sample_topic_tvd_a = [float(np.clip(0.04 + rng.normal(0, 0.01), 0.0, 1.0)) for _ in range(N)]
    pt_topic_v, topic_v_low, topic_v_high = compute_bootstrap_ci(sample_topic_tvd_v)
    pt_topic_a, topic_a_low, topic_a_high = compute_bootstrap_ci(sample_topic_tvd_a)
    sample_sel_diff_v = [float(0.73 / 4.0 - t) for t in sample_topic_tvd_v]
    sample_sel_diff_a = [float(0.65 / 4.0 - t) for t in sample_topic_tvd_a]
    pt_sel_v, sel_v_low, sel_v_high = compute_bootstrap_ci(sample_sel_diff_v)
    pt_sel_a, sel_a_low, sel_a_high = compute_bootstrap_ci(sample_sel_diff_a)

    return {
        "affect_direction_grounding": "reader_prediction (simulated)",
        "alignment_reader_vs_self_directions": {
            "valence": 0.78,
            "arousal": 0.72,
        },
        "alpha_grid": alpha_grid,
        "dose_response_v": dose_responses_v,
        "dose_response_a": dose_responses_a,
        "slope_v": slope_v,
        "slope_a": slope_a,
        "slope_v_ci": {"point": pt_sv, "ci_lower": sv_low, "ci_upper": sv_high},
        "slope_a_ci": {"point": pt_sa, "ci_lower": sa_low, "ci_upper": sa_high},
        # Specificity
        "specificity_v": float(pt_spec_v),
        "specificity_a": float(pt_spec_a),
        "specificity_v_ci": {"point": pt_spec_v, "ci_lower": spec_v_low, "ci_upper": spec_v_high},
        "specificity_a_ci": {"point": pt_spec_a, "ci_lower": spec_a_low, "ci_upper": spec_a_high},
        "specificity_diff": float(pt_spec_v),  # 互換用
        "specificity_diff_ci": {"point": pt_spec_v, "ci_lower": spec_v_low, "ci_upper": spec_v_high},
        "specificity_vs_random_ci": {"point": pt_spec_rand_v, "ci_lower": spec_rand_v_low, "ci_upper": spec_rand_v_high},
        "specificity_vs_orthogonal_ci": {"point": pt_spec_perp_v, "ci_lower": spec_perp_v_low, "ci_upper": spec_perp_v_high},
        # Necessity
        "natural_shift": float(natural_shift_v),
        "attenuated_shift": float(attenuated_shift_v),
        "natural_shift_v": float(natural_shift_v),
        "natural_shift_a": float(natural_shift_a),
        "attenuation_ratio_v": float(pt_att_v),
        "attenuation_ratio_a": float(pt_att_a),
        "attenuation_ratio_v_ci": {"point": pt_att_v, "ci_lower": att_v_low, "ci_upper": att_v_high},
        "attenuation_ratio_a_ci": {"point": pt_att_a, "ci_lower": att_a_low, "ci_upper": att_a_high},
        "attenuation_ratio": float(pt_att_v),  # 互換用
        "attenuation_ratio_ci": {"point": pt_att_v, "ci_lower": att_v_low, "ci_upper": att_v_high},
        # Topic control / Task selectivity
        "topic_tvd_v": float(pt_topic_v),
        "topic_tvd_a": float(pt_topic_a),
        "topic_tvd_v_ci": {"point": pt_topic_v, "ci_lower": topic_v_low, "ci_upper": topic_v_high},
        "topic_tvd_a_ci": {"point": pt_topic_a, "ci_lower": topic_a_low, "ci_upper": topic_a_high},
        "task_selectivity": {
            "effect_self": 0.85 / 4.0,
            "effect_reader": 0.72 / 4.0,
            "effect_control": float(pt_topic_v),
            "topic_tvd": float(pt_topic_v),
            "topic_tvd_ci": {"point": pt_topic_v, "ci_lower": topic_v_low, "ci_upper": topic_v_high},
            "self_minus_control_ci": {"point": pt_sel_v, "ci_lower": sel_v_low, "ci_upper": sel_v_high},
            "note": (
                "Topic TVD is a non-specific perturbation control, not a same-construct "
                "effect size comparable to Self VA shift. Primary check: topic_tvd remains small."
            ),
            "pattern": "nonspecific_perturbation_small",
        },
    }



def run_real_state_induction(
    df: pd.DataFrame,
    alpha_grid: List[float],
    model_id: str,
    target_layer: int,
    device: str = "cpu",
    batch_size: int = 81,
) -> Dict[str, Any]:
    """
    実モデルを用いた V3-RQ1 State Induction パイプライン
    1. Train split から外部正解ラベル (reader_V, reader_A) に対する情動方向 d_V, d_A を推定
    2. Test split に対し、活性化介入 (alpha-sweep, projection removal, random/perp controls, topic control) を実行
    3. 共通 Sequence Likelihood により各条件下の期待値変位を実測
    4. サンプル単位の変位から Bootstrap 95% CI を算出
    """
    logger.info(f"Loading tokenizer and model: {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" and torch.cuda.is_available() else torch.float32,
        device_map=device if device != "cpu" and torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()

    registry = get_registry()
    fam_cfg = registry.get_family_by_model_id(model_id)
    adapter = get_model_adapter(model, fam_cfg)

    # 1. データを 50/50 Train / Test split に厳格分割 (pair_id に基づく Group split)
    seed = 42
    rng = np.random.RandomState(seed)
    if "pair_id" in df.columns and df["pair_id"].nunique() > 1:
        unique_pairs = df["pair_id"].unique()
        rng.shuffle(unique_pairs)
        half_pairs = len(unique_pairs) // 2
        train_pairs = set(unique_pairs[:half_pairs])
        train_mask = df["pair_id"].isin(train_pairs)
        train_df = df[train_mask].copy().reset_index(drop=True)
        test_df = df[~train_mask].copy().reset_index(drop=True)
        logger.info(f"Group split on pair_id: {len(train_pairs)} pairs train ({len(train_df)} rows), {len(unique_pairs) - half_pairs} pairs test ({len(test_df)} rows)")
    else:
        indices = rng.permutation(len(df))
        half = len(df) // 2
        train_idx, test_idx = indices[:half], indices[half:]
        train_df = df.iloc[train_idx].copy().reset_index(drop=True)
        test_df = df.iloc[test_idx].copy().reset_index(drop=True)
        logger.info(f"Index split: {len(train_df)} train, {len(test_df)} test")
    logger.info(f"Split dataset: {len(train_df)} train pairs, {len(test_df)} held-out test pairs")

    # 2. Train split による情動方向 d_V, d_A の推定 (Reader-grounded Primary / Self-derived Secondary)
    logger.info(f"Extracting target layer {target_layer} representations and Reader/Self predictions on train split...")
    train_hiddens = []
    train_self_ev: List[float] = []
    train_self_ea: List[float] = []
    train_reader_ev: List[float] = []
    train_reader_ea: List[float] = []
    train_candidates = build_va_candidates()
    with torch.no_grad():
        for _, row in train_df.iterrows():
            text = str(row["text"])
            # a. Self condition: capture target_layer representation and evaluate clean self-report
            prompt_self = build_prompt(text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
            enc_self = encode_prompt_canonical(tokenizer, prompt_self, device=device)
            anchors_self = find_semantic_anchors(enc_self["input_ids"][0].tolist(), tokenizer, text)
            patch_pos = anchors_self["prompt_end"]

            with ActivationHookManager(adapter) as hook_mgr:
                hook_mgr.register_capture_hook(
                    layer_idx=target_layer,
                    hook_point=HookPoint.POST_MLP_RESID,
                    token_indices=patch_pos,
                    key="train_h",
                )
                _ = model(**enc_self)
                train_hiddens.append(hook_mgr.captured_activations["train_h"].cpu().float().numpy().ravel())
            _, tr_self_probs = compute_sequence_likelihoods_for_candidates(
                model=model, tokenizer=tokenizer, prompt=prompt_self, candidates=train_candidates, device=device, batch_size=batch_size
            )
            ev_s, ea_s = compute_expected_va(tr_self_probs, train_candidates)
            train_self_ev.append(float(ev_s))
            train_self_ea.append(float(ea_s))

            # b. Reader condition: evaluate model's objective perception of reader affect (Reader Prediction)
            prompt_reader = build_prompt(text, task=TaskType.READER, format_type="chat", tokenizer=tokenizer)
            _, tr_reader_probs = compute_sequence_likelihoods_for_candidates(
                model=model, tokenizer=tokenizer, prompt=prompt_reader, candidates=train_candidates, device=device, batch_size=batch_size
            )
            ev_r, ea_r = compute_expected_va(tr_reader_probs, train_candidates)
            train_reader_ev.append(float(ev_r))
            train_reader_ea.append(float(ea_r))

    H_train = np.array(train_hiddens)  # (N_train, D)

    # Primary: Reader-grounded direction targets (human reader if available, otherwise model Reader prediction)
    if "reader_V" in train_df.columns and "reader_A" in train_df.columns:
        logger.info("Using human reader_V/A labels for Primary affect direction targets.")
        y_v_train_primary = train_df["reader_V"].to_numpy()
        y_a_train_primary = train_df["reader_A"].to_numpy()
    else:
        logger.info("Using train-set model Reader Predictions (TaskType.READER) as Primary affect direction targets.")
        y_v_train_primary = np.array(train_reader_ev, dtype=np.float64)
        y_a_train_primary = np.array(train_reader_ea, dtype=np.float64)

    directions_primary = extract_conditional_directions(H_train, y_v_train_primary, y_a_train_primary, method="ridge", alpha=1.0)
    d_v = directions_primary["direction_v"]  # (D,) Primary Reader-grounded
    d_a = directions_primary["direction_a"]

    # Secondary: Self-derived direction targets (model self-report)
    y_v_train_secondary = np.array(train_self_ev, dtype=np.float64)
    y_a_train_secondary = np.array(train_self_ea, dtype=np.float64)
    directions_secondary = extract_conditional_directions(H_train, y_v_train_secondary, y_a_train_secondary, method="ridge", alpha=1.0)
    d_v_self = directions_secondary["direction_v"]
    d_a_self = directions_secondary["direction_a"]

    norm_v_p = np.linalg.norm(d_v)
    norm_v_s = np.linalg.norm(d_v_self)
    norm_a_p = np.linalg.norm(d_a)
    norm_a_s = np.linalg.norm(d_a_self)
    alignment_v = float(np.dot(d_v, d_v_self) / (norm_v_p * norm_v_s + 1e-6)) if norm_v_p > 0 and norm_v_s > 0 else 0.0
    alignment_a = float(np.dot(d_a, d_a_self) / (norm_a_p * norm_a_s + 1e-6)) if norm_a_p > 0 and norm_a_s > 0 else 0.0
    logger.info(f"Reader-Grounded vs Self-Derived Direction Alignment: cos_V={alignment_v:.3f}, cos_A={alignment_a:.3f}")

    Q_sub, _ = compute_orthonormal_subspace(d_v, d_a)  # (D, 2)
    d_rand_v, d_perp_v = generate_control_directions(d_v, seed=42)
    d_rand_a, d_perp_a = generate_control_directions(d_a, seed=43)

    h_std_v = float(np.std(H_train @ d_v)) or 1.0
    h_std_a = float(np.std(H_train @ d_a)) or 1.0

    # train split から matched-neutral 表現を抽出し mu_neu を算出
    logger.info(f"Extracting target layer {target_layer} matched-neutral representations on train split...")
    train_neutral_hiddens = []
    with torch.no_grad():
        for _, row in train_df.iterrows():
            neu_text = resolve_matched_neutral_text(row, df)
            if neu_text is not None and len(neu_text.strip()) > 0:
                p_neu = build_prompt(neu_text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
                enc_neu = encode_prompt_canonical(tokenizer, p_neu, device=device)
                anchors_neu = find_semantic_anchors(enc_neu["input_ids"][0].tolist(), tokenizer, neu_text)
                with ActivationHookManager(adapter) as hook_mgr:
                    hook_mgr.register_capture_hook(
                        layer_idx=target_layer,
                        hook_point=HookPoint.POST_MLP_RESID,
                        token_indices=anchors_neu["prompt_end"],
                        key="train_h_neu",
                    )
                    _ = model(**enc_neu)
                    train_neutral_hiddens.append(hook_mgr.captured_activations["train_h_neu"].cpu().float().numpy().ravel())

    if len(train_neutral_hiddens) == 0:
        raise ValueError(
            "No matched-neutral stimuli found in the V3 train split. "
            "Generic neutral prompts are forbidden in Primary."
        )
    mu_neu = np.mean(train_neutral_hiddens, axis=0)  # (D,)
    logger.info(f"Computed mu_neu from {len(train_neutral_hiddens)} matched-neutral train samples")

    # 3. Held-out test split における実介入実験
    logger.info(f"Running causal state induction interventions on {len(test_df)} test samples...")
    candidates = build_va_candidates()
    from affective_empathy_eval.prompts import TOPIC_OPTIONS
    topic_candidates = [f'{{"topic": "{opt}"}}' for opt in TOPIC_OPTIONS]

    sample_slopes_v = []
    sample_slopes_a = []
    sample_spec_diff_v = []
    sample_spec_rand_v = []
    sample_spec_perp_v = []
    sample_spec_diff_a = []
    sample_spec_rand_a = []
    sample_spec_perp_a = []
    sample_att_ratios_v = []
    sample_att_ratios_a = []
    sample_self_eff_v = []
    sample_self_eff_a = []
    sample_ctrl_eff_v = []
    sample_ctrl_eff_a = []

    dose_curves_v = {alpha: [] for alpha in alpha_grid}
    dose_curves_a = {alpha: [] for alpha in alpha_grid}

    with torch.no_grad():
        for _, row in test_df.iterrows():
            aff_text = str(row["text"])
            neu_text = resolve_matched_neutral_text(row, df)
            if not neu_text or not neu_text.strip():
                raise ValueError(f"Missing matched-neutral for pair_id={row.get('pair_id', 'unknown')}")

            prompt_aff_self = build_prompt(aff_text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
            prompt_neu_self = build_prompt(neu_text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
            prompt_ctrl = build_prompt(neu_text, task=TaskType.CONTROL_TOPIC, format_type="chat", tokenizer=tokenizer)

            enc_neu = encode_prompt_canonical(tokenizer, prompt_neu_self, device=device)
            anchors_neu = find_semantic_anchors(enc_neu["input_ids"][0].tolist(), tokenizer, neu_text)
            patch_pos_neu = anchors_neu["prompt_end"]

            enc_aff = encode_prompt_canonical(tokenizer, prompt_aff_self, device=device)
            anchors_aff = find_semantic_anchors(enc_aff["input_ids"][0].tolist(), tokenizer, aff_text)
            patch_pos_aff = anchors_aff["prompt_end"]

            # 1. Clean Baselines
            # Neutral baseline: clean_neu_ev, clean_neu_ea (Sufficiency の基準点)
            _, probs_neu = compute_sequence_likelihoods_for_candidates(
                model=model, tokenizer=tokenizer, prompt=prompt_neu_self, candidates=candidates, device=device, batch_size=batch_size
            )
            clean_neu_ev, clean_neu_ea = compute_expected_va(probs_neu, candidates)

            # Affective baseline: clean_aff_ev, clean_aff_ea (Necessity / Natural shift の基準点)
            _, probs_aff = compute_sequence_likelihoods_for_candidates(
                model=model, tokenizer=tokenizer, prompt=prompt_aff_self, candidates=candidates, device=device, batch_size=batch_size
            )
            clean_aff_ev, clean_aff_ea = compute_expected_va(probs_aff, candidates)

            # 2. Sufficiency / Dose-response: neutral 側へ d_V / d_A を注入 (shift = patched_neu - clean_neu)
            alpha_shifts_v = []
            alpha_shifts_a = []
            for alpha in alpha_grid:
                for axis_name, direction, h_std, clean_val, collect, curves in (
                    ("v", d_v, h_std_v, clean_neu_ev, alpha_shifts_v, dose_curves_v),
                    ("a", d_a, h_std_a, clean_neu_ea, alpha_shifts_a, dose_curves_a),
                ):
                    with ActivationHookManager(adapter) as hook_mgr:
                        hook_mgr.register_direction_intervention_hook(
                            layer_idx=target_layer,
                            direction=direction,
                            alpha=alpha,
                            hidden_std=h_std,
                            token_indices=patch_pos_neu,
                            hook_point=HookPoint.POST_MLP_RESID,
                            mode="inject",
                        )
                        _, probs_patch = compute_sequence_likelihoods_for_candidates(
                            model=model, tokenizer=tokenizer, prompt=prompt_neu_self, candidates=candidates, device=device, batch_size=batch_size
                        )
                    ev_p, ea_p = compute_expected_va(probs_patch, candidates)
                    shift = (ev_p if axis_name == "v" else ea_p) - clean_val
                    collect.append(shift)
                    curves[alpha].append(shift)

            sample_slopes_v.append(estimate_interventional_slope(alpha_grid, alpha_shifts_v))
            sample_slopes_a.append(estimate_interventional_slope(alpha_grid, alpha_shifts_a))

            # 3. Specificity: neutral 側へランダム方向・直交方向を注入して比較 (alpha = 1.0)
            # Valence: d_V vs d_rand_v vs d_perp_v
            with ActivationHookManager(adapter) as hook_mgr:
                hook_mgr.register_direction_intervention_hook(
                    layer_idx=target_layer,
                    direction=d_rand_v,
                    alpha=1.0,
                    hidden_std=h_std_v,
                    token_indices=patch_pos_neu,
                    hook_point=HookPoint.POST_MLP_RESID,
                    mode="inject",
                )
                _, probs_rand_v = compute_sequence_likelihoods_for_candidates(
                    model=model, tokenizer=tokenizer, prompt=prompt_neu_self, candidates=candidates, device=device, batch_size=batch_size
                )
            ev_rand_v, _ = compute_expected_va(probs_rand_v, candidates)

            with ActivationHookManager(adapter) as hook_mgr:
                hook_mgr.register_direction_intervention_hook(
                    layer_idx=target_layer,
                    direction=d_perp_v,
                    alpha=1.0,
                    hidden_std=h_std_v,
                    token_indices=patch_pos_neu,
                    hook_point=HookPoint.POST_MLP_RESID,
                    mode="inject",
                )
                _, probs_perp_v = compute_sequence_likelihoods_for_candidates(
                    model=model, tokenizer=tokenizer, prompt=prompt_neu_self, candidates=candidates, device=device, batch_size=batch_size
                )
            ev_perp_v, _ = compute_expected_va(probs_perp_v, candidates)

            eff_affect_v = abs(alpha_shifts_v[-1])  # alpha = 1.0 (relative to clean_neu_ev)
            eff_rand_v = abs(ev_rand_v - clean_neu_ev)
            eff_perp_v = abs(ev_perp_v - clean_neu_ev)
            sample_spec_diff_v.append(eff_affect_v - max(eff_rand_v, eff_perp_v))
            sample_spec_rand_v.append(eff_affect_v - eff_rand_v)
            sample_spec_perp_v.append(eff_affect_v - eff_perp_v)

            # Arousal: d_A vs d_rand_a vs d_perp_a
            with ActivationHookManager(adapter) as hook_mgr:
                hook_mgr.register_direction_intervention_hook(
                    layer_idx=target_layer,
                    direction=d_rand_a,
                    alpha=1.0,
                    hidden_std=h_std_a,
                    token_indices=patch_pos_neu,
                    hook_point=HookPoint.POST_MLP_RESID,
                    mode="inject",
                )
                _, probs_rand_a = compute_sequence_likelihoods_for_candidates(
                    model=model, tokenizer=tokenizer, prompt=prompt_neu_self, candidates=candidates, device=device, batch_size=batch_size
                )
            _, ea_rand_a = compute_expected_va(probs_rand_a, candidates)

            with ActivationHookManager(adapter) as hook_mgr:
                hook_mgr.register_direction_intervention_hook(
                    layer_idx=target_layer,
                    direction=d_perp_a,
                    alpha=1.0,
                    hidden_std=h_std_a,
                    token_indices=patch_pos_neu,
                    hook_point=HookPoint.POST_MLP_RESID,
                    mode="inject",
                )
                _, probs_perp_a = compute_sequence_likelihoods_for_candidates(
                    model=model, tokenizer=tokenizer, prompt=prompt_neu_self, candidates=candidates, device=device, batch_size=batch_size
                )
            _, ea_perp_a = compute_expected_va(probs_perp_a, candidates)

            eff_affect_a = abs(alpha_shifts_a[-1])  # alpha = 1.0
            eff_rand_a = abs(ea_rand_a - clean_neu_ea)
            eff_perp_a = abs(ea_perp_a - clean_neu_ea)
            sample_spec_diff_a.append(eff_affect_a - max(eff_rand_a, eff_perp_a))
            sample_spec_rand_a.append(eff_affect_a - eff_rand_a)
            sample_spec_perp_a.append(eff_affect_a - eff_perp_a)

            # 4. Endogenous relevance / Necessity: affective 側から Q 部分空間を除去
            with ActivationHookManager(adapter) as hook_mgr:
                hook_mgr.register_capture_hook(
                    layer_idx=target_layer,
                    hook_point=HookPoint.POST_MLP_RESID,
                    token_indices=patch_pos_aff,
                    key="h_aff",
                )
                _ = model(**enc_aff)
                h_aff = hook_mgr.captured_activations["h_aff"].cpu().float().numpy().ravel()

            h_centered = h_aff - mu_neu
            proj = (h_centered @ Q_sub) @ Q_sub.T
            h_ablated = h_aff - proj
            patch_abl = torch.tensor(h_ablated, dtype=torch.float32, device=device)

            with ActivationHookManager(adapter) as hook_mgr:
                hook_mgr.register_patch_hook(
                    layer_idx=target_layer,
                    patch_tensor=patch_abl,
                    token_indices=patch_pos_aff,
                    hook_point=HookPoint.POST_MLP_RESID,
                )
                _, probs_abl = compute_sequence_likelihoods_for_candidates(
                    model=model, tokenizer=tokenizer, prompt=prompt_aff_self, candidates=candidates, device=device, batch_size=batch_size
                )
            abl_aff_ev, abl_aff_ea = compute_expected_va(probs_abl, candidates)

            # 自然変位と残余変位 (clean_aff - clean_neu vs abl_aff - clean_neu)
            natural_shift_v = abs(clean_aff_ev - clean_neu_ev)
            residual_shift_v = abs(abl_aff_ev - clean_neu_ev)
            attenuation_v = natural_shift_v - residual_shift_v
            if natural_shift_v > 0.05:
                sample_att_ratios_v.append(float(attenuation_v / natural_shift_v))

            natural_shift_a = abs(clean_aff_ea - clean_neu_ea)
            residual_shift_a = abs(abl_aff_ea - clean_neu_ea)
            attenuation_a = natural_shift_a - residual_shift_a
            if natural_shift_a > 0.05:
                sample_att_ratios_a.append(float(attenuation_a / natural_shift_a))

            # 5. Topic control (非特異的摂動の確認統制: VA両軸)
            self_norm_eff_v = eff_affect_v / 4.0
            self_norm_eff_a = eff_affect_a / 4.0
            sample_self_eff_v.append(self_norm_eff_v)
            sample_self_eff_a.append(self_norm_eff_a)

            enc_ctrl = encode_prompt_canonical(tokenizer, prompt_ctrl, device=device)
            anchors_ctrl = find_semantic_anchors(enc_ctrl["input_ids"][0].tolist(), tokenizer, neu_text)
            patch_pos_ctrl = anchors_ctrl["prompt_end"]
            _, probs_ctrl_clean = compute_sequence_likelihoods_for_candidates(
                model=model, tokenizer=tokenizer, prompt=prompt_ctrl, candidates=topic_candidates, device=device, batch_size=batch_size
            )

            # Topic control with d_V
            with ActivationHookManager(adapter) as hook_mgr:
                hook_mgr.register_direction_intervention_hook(
                    layer_idx=target_layer,
                    direction=d_v,
                    alpha=1.0,
                    hidden_std=h_std_v,
                    token_indices=patch_pos_ctrl,
                    hook_point=HookPoint.POST_MLP_RESID,
                    mode="inject",
                )
                _, probs_ctrl_patch_v = compute_sequence_likelihoods_for_candidates(
                    model=model, tokenizer=tokenizer, prompt=prompt_ctrl, candidates=topic_candidates, device=device, batch_size=batch_size
                )
            topic_tvd_v = 0.5 * float(np.sum(np.abs(np.array(probs_ctrl_patch_v) - np.array(probs_ctrl_clean))))
            sample_ctrl_eff_v.append(topic_tvd_v)

            # Topic control with d_A
            with ActivationHookManager(adapter) as hook_mgr:
                hook_mgr.register_direction_intervention_hook(
                    layer_idx=target_layer,
                    direction=d_a,
                    alpha=1.0,
                    hidden_std=h_std_a,
                    token_indices=patch_pos_ctrl,
                    hook_point=HookPoint.POST_MLP_RESID,
                    mode="inject",
                )
                _, probs_ctrl_patch_a = compute_sequence_likelihoods_for_candidates(
                    model=model, tokenizer=tokenizer, prompt=prompt_ctrl, candidates=topic_candidates, device=device, batch_size=batch_size
                )
            topic_tvd_a = 0.5 * float(np.sum(np.abs(np.array(probs_ctrl_patch_a) - np.array(probs_ctrl_clean))))
            sample_ctrl_eff_a.append(topic_tvd_a)

    # 4. Bootstrap CI の算出 (VA両軸独立)
    pt_sv, sv_low, sv_high = compute_bootstrap_ci(sample_slopes_v)
    pt_sa, sa_low, sa_high = compute_bootstrap_ci(sample_slopes_a)

    pt_spec_v, spec_v_low, spec_v_high = compute_bootstrap_ci(sample_spec_diff_v)
    pt_spec_rand_v, spec_rand_v_low, spec_rand_v_high = compute_bootstrap_ci(sample_spec_rand_v)
    pt_spec_perp_v, spec_perp_v_low, spec_perp_v_high = compute_bootstrap_ci(sample_spec_perp_v)

    pt_spec_a, spec_a_low, spec_a_high = compute_bootstrap_ci(sample_spec_diff_a)
    pt_spec_rand_a, spec_rand_a_low, spec_rand_a_high = compute_bootstrap_ci(sample_spec_rand_a)
    pt_spec_perp_a, spec_perp_a_low, spec_perp_a_high = compute_bootstrap_ci(sample_spec_perp_a)

    if len(sample_att_ratios_v) >= 2:
        pt_att_v, att_v_low, att_v_high = compute_bootstrap_ci(sample_att_ratios_v)
    elif len(sample_att_ratios_v) == 1:
        pt_att_v, att_v_low, att_v_high = sample_att_ratios_v[0], sample_att_ratios_v[0], sample_att_ratios_v[0]
    else:
        pt_att_v, att_v_low, att_v_high = np.nan, np.nan, np.nan

    if len(sample_att_ratios_a) >= 2:
        pt_att_a, att_a_low, att_a_high = compute_bootstrap_ci(sample_att_ratios_a)
    elif len(sample_att_ratios_a) == 1:
        pt_att_a, att_a_low, att_a_high = sample_att_ratios_a[0], sample_att_ratios_a[0], sample_att_ratios_a[0]
    else:
        pt_att_a, att_a_low, att_a_high = np.nan, np.nan, np.nan

    pt_topic_v, topic_v_low, topic_v_high = compute_bootstrap_ci(sample_ctrl_eff_v)
    pt_topic_a, topic_a_low, topic_a_high = compute_bootstrap_ci(sample_ctrl_eff_a)

    self_minus_ctrl_v = [s - c for s, c in zip(sample_self_eff_v, sample_ctrl_eff_v)]
    pt_sel_v, sel_v_low, sel_v_high = compute_bootstrap_ci(self_minus_ctrl_v)

    mean_dose_v = [float(np.mean(dose_curves_v[a])) for a in alpha_grid]
    mean_dose_a = [float(np.mean(dose_curves_a[a])) for a in alpha_grid]

    return {
        "affect_direction_grounding": "reader_prediction",
        "alignment_reader_vs_self_directions": {
            "valence": alignment_v,
            "arousal": alignment_a,
        },
        "alpha_grid": alpha_grid,
        "dose_response_v": mean_dose_v,
        "dose_response_a": mean_dose_a,
        "slope_v": float(pt_sv),
        "slope_a": float(pt_sa),
        "slope_v_ci": {"point": pt_sv, "ci_lower": sv_low, "ci_upper": sv_high},
        "slope_a_ci": {"point": pt_sa, "ci_lower": sa_low, "ci_upper": sa_high},
        # Specificity
        "specificity_v": float(pt_spec_v),
        "specificity_a": float(pt_spec_a),
        "specificity_v_ci": {"point": pt_spec_v, "ci_lower": spec_v_low, "ci_upper": spec_v_high},
        "specificity_a_ci": {"point": pt_spec_a, "ci_lower": spec_a_low, "ci_upper": spec_a_high},
        "specificity_diff": float(pt_spec_v),  # 互換用
        "specificity_diff_ci": {"point": pt_spec_v, "ci_lower": spec_v_low, "ci_upper": spec_v_high},
        "specificity_vs_random_ci": {"point": pt_spec_rand_v, "ci_lower": spec_rand_v_low, "ci_upper": spec_rand_v_high},
        "specificity_vs_orthogonal_ci": {"point": pt_spec_perp_v, "ci_lower": spec_perp_v_low, "ci_upper": spec_perp_v_high},
        # Necessity
        "attenuation_ratio_v": float(pt_att_v),
        "attenuation_ratio_a": float(pt_att_a),
        "attenuation_ratio_v_ci": {"point": pt_att_v, "ci_lower": att_v_low, "ci_upper": att_v_high},
        "attenuation_ratio_a_ci": {"point": pt_att_a, "ci_lower": att_a_low, "ci_upper": att_a_high},
        "attenuation_ratio": float(pt_att_v),  # 互換用
        "attenuation_ratio_ci": {"point": pt_att_v, "ci_lower": att_v_low, "ci_upper": att_v_high},
        # Topic control / Task selectivity
        "topic_tvd_v": float(pt_topic_v),
        "topic_tvd_a": float(pt_topic_a),
        "topic_tvd_v_ci": {"point": pt_topic_v, "ci_lower": topic_v_low, "ci_upper": topic_v_high},
        "topic_tvd_a_ci": {"point": pt_topic_a, "ci_lower": topic_a_low, "ci_upper": topic_a_high},
        "task_selectivity": {
            "effect_self": float(np.mean(sample_self_eff_v)),
            "effect_control": float(np.mean(sample_ctrl_eff_v)),
            "topic_tvd": float(pt_topic_v),
            "topic_tvd_ci": {"point": pt_topic_v, "ci_lower": topic_v_low, "ci_upper": topic_v_high},
            "self_minus_control_ci": {"point": pt_sel_v, "ci_lower": sel_v_low, "ci_upper": sel_v_high},
            "note": (
                "Topic TVD is a non-specific perturbation control, not a same-construct "
                "effect size comparable to Self VA shift."
            ),
            "pattern": "nonspecific_perturbation_small" if float(np.mean(sample_ctrl_eff_v)) < 0.15 else "nonspecific_perturbation_large",
        },
    }


def evaluate_go_no_go_gate(
    results: Dict[str, Any],
    gate_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Bootstrap 95% 信頼区間 (CI) の下限に基づく厳密な Go/No-Go 判定ゲート (Valence / Arousal 完全軸別化)
    """
    min_spec = gate_cfg.get("min_specificity_diff", 0.05)
    min_nec = gate_cfg.get("min_necessity_attenuation", 0.05)
    max_topic_tvd = gate_cfg.get("max_topic_tvd", 0.15)

    # 1. Dose-response: 傾き slope の 95% CI 下限が 0.1 超
    slope_v_ci = results.get("slope_v_ci", {})
    slope_a_ci = results.get("slope_a_ci", {})
    sv_lower = slope_v_ci.get("ci_lower", results.get("slope_v", 0.0))
    sa_lower = slope_a_ci.get("ci_lower", results.get("slope_a", 0.0))
    dose_v_pass = bool(sv_lower > 0.1)
    dose_a_pass = bool(sa_lower > 0.1)

    # 2. Specificity: 95% CI 下限が閾値超 (VA独立)
    spec_v_ci = results.get("specificity_v_ci", results.get("specificity_diff_ci", {}))
    spec_a_ci = results.get("specificity_a_ci", {})
    spec_v_lower = spec_v_ci.get("ci_lower", results.get("specificity_v", results.get("specificity_diff", 0.0)))
    spec_a_lower = spec_a_ci.get("ci_lower", results.get("specificity_a", 0.0))
    specificity_v_pass = bool(spec_v_lower > min_spec)
    specificity_a_pass = bool(spec_a_lower > min_spec)

    # 3. Necessity: 射影除去減衰率 95% CI 下限が閾値超 (VA独立)
    nec_v_ci = results.get("attenuation_ratio_v_ci", results.get("attenuation_ratio_ci", {}))
    nec_a_ci = results.get("attenuation_ratio_a_ci", {})
    nec_v_lower = nec_v_ci.get("ci_lower", results.get("attenuation_ratio_v", results.get("attenuation_ratio", 0.0)))
    nec_a_lower = nec_a_ci.get("ci_lower", results.get("attenuation_ratio_a", 0.0))
    necessity_v_pass = bool(nec_v_lower > min_nec)
    necessity_a_pass = bool(nec_a_lower > min_nec)

    # 4. Topic control: Topic 課題 TVD の 95% CI 上限が小さいこと (VA独立)
    topic_v_ci = results.get("topic_tvd_v_ci", results.get("task_selectivity", {}).get("topic_tvd_ci", {}))
    topic_a_ci = results.get("topic_tvd_a_ci", {})
    topic_v_upper = topic_v_ci.get("ci_upper", results.get("topic_tvd_v", 0.0))
    topic_a_upper = topic_a_ci.get("ci_upper", results.get("topic_tvd_a", 0.0))
    topic_v_pass = bool(topic_v_upper < max_topic_tvd)
    topic_a_pass = bool(topic_a_upper < max_topic_tvd)

    # 軸別 Gate 判定
    decision_v = "GO" if (dose_v_pass and specificity_v_pass and necessity_v_pass and topic_v_pass) else "NO_GO"
    decision_a = "GO" if (dose_a_pass and specificity_a_pass and necessity_a_pass and topic_a_pass) else "NO_GO"

    if decision_v == "GO" and decision_a == "GO":
        decision = "GO"
    elif decision_v == "GO":
        decision = "GO (Valence-only)"
    elif decision_a == "GO":
        decision = "GO (Arousal-only)"
    else:
        decision = "NO_GO"

    checklist = {
        "dose_response_v_pass": dose_v_pass,
        "dose_response_a_pass": dose_a_pass,
        "dose_response_pass": bool(dose_v_pass and dose_a_pass),
        "slope_v_ci_lower": float(sv_lower),
        "slope_a_ci_lower": float(sa_lower),
        "specificity_v_pass": specificity_v_pass,
        "specificity_a_pass": specificity_a_pass,
        "specificity_pass": bool(specificity_v_pass and specificity_a_pass),
        "specificity_v_ci_lower": float(spec_v_lower),
        "specificity_a_ci_lower": float(spec_a_lower),
        "necessity_v_pass": necessity_v_pass,
        "necessity_a_pass": necessity_a_pass,
        "necessity_pass": bool(necessity_v_pass and necessity_a_pass),
        "necessity_v_ci_lower": float(nec_v_lower),
        "necessity_a_ci_lower": float(nec_a_lower),
        "topic_v_pass": topic_v_pass,
        "topic_a_pass": topic_a_pass,
        "topic_control_pass": bool(topic_v_pass and topic_a_pass),
        "topic_tvd_v_ci_upper": float(topic_v_upper),
        "topic_tvd_a_ci_upper": float(topic_a_upper),
        "decision_valence": decision_v,
        "decision_arousal": decision_a,
        "decision": decision,
    }
    return checklist



def main():
    args = parse_args()
    logger.info(f"Starting V3-RQ1 State Induction (pilot={args.pilot}, dry_run={args.dry_run})")

    with open(args.config, "r", encoding="utf-8") as f:
        v3_cfg = yaml.safe_load(f)

    raw_dir = Path(v3_cfg["output"]["raw_dir"])
    derived_dir = Path(v3_cfg["output"]["derived_dir"])
    if args.dry_run:
        raw_dir = raw_dir / "dry_run"
        derived_dir = derived_dir / "dry_run"
    elif args.pilot:
        raw_dir = raw_dir / "pilot"
        derived_dir = derived_dir / "pilot"
    raw_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    df = load_v3_matched_pair_table(v3_cfg["dataset"]["path"])
    logger.info(describe_loaded_frame(df, "V3-RQ1 matched-pair table", v3_cfg["dataset"]["path"]))
    if args.pilot:
        df = df.iloc[: v3_cfg["dataset"].get("pilot_size", 50)].copy()
        logger.info(f"Running Pilot mode with {len(df)} rows")

    fam_key, target_model_id = resolve_instruct_target_from_args(
        args,
        Path(args.models_config),
        fallback_family=v3_cfg.get("target_family"),
    )
    num_layers = resolve_architecture_dims(target_model_id)[0]
    if args.layer is None:
        args.layer = int(round(args.relative_depth * (num_layers - 1)))
        logger.info(
            f"Resolved V3-RQ1 layer={args.layer} from relative_depth={args.relative_depth} "
            f"(L={num_layers}, family={fam_key})"
        )

    alpha_grid = v3_cfg["interventions"]["alpha_grid"]

    if args.dry_run:
        logger.info("Executing mock state induction and gate simulation (--dry-run specified)...")
        results = simulate_mock_intervention_responses(df, alpha_grid)
    else:
        logger.info(f"Executing REAL state induction pipeline for {target_model_id}...")
        results = run_real_state_induction(
            df=df,
            alpha_grid=alpha_grid,
            model_id=target_model_id,
            target_layer=args.layer,
            device=args.device,
            batch_size=v3_cfg.get("inference", {}).get("batch_size", 81),
        )

    # Go/No-Go ゲート判定の評価 (CI ベース)
    gate_decision = evaluate_go_no_go_gate(results, v3_cfg["gate_criteria"])
    logger.info(f"Gate Evaluation Completed: DECISION = {gate_decision['decision']}")
    logger.info(f"  Checklist: {gate_decision}")

    # 結果の保存
    out_results = {
        "pilot": args.pilot,
        "dry_run": args.dry_run,
        "sample_size": len(df),
        "target_layer": args.layer,
        "results": results,
        "gate_decision": gate_decision,
    }

    def _json_serial(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    out_raw = raw_dir / ("v3_pilot_results.json" if args.pilot else "v3_rq1_results.json")
    with open(out_raw, "w", encoding="utf-8") as f:
        json.dump(out_results, f, indent=2, default=_json_serial)
    logger.info(f"Saved RQ1 results to {out_raw}")

    out_gate = derived_dir / "v3_gate_decision.json"
    with open(out_gate, "w", encoding="utf-8") as f:
        json.dump(gate_decision, f, indent=2, default=_json_serial)
    logger.info(f"Saved gate decision to {out_gate}")

    # RunManifest 保存
    from affective_empathy_eval.manifests import create_run_manifest
    manifest = create_run_manifest(
        run_type="v3_rq1",
        model_name=target_model_id,
        config=v3_cfg,
        dataset_path=v3_cfg["dataset"]["path"],
        metadata={
            "target_family": fam_key,
            "layer": args.layer,
            "gate_decision": gate_decision["decision"],
        },
        dry_run=bool(args.dry_run),
    )
    manifest.save(str(raw_dir / f"manifest_rq1_{fam_key}.json"))
    logger.info(f"Saved gate decision to {out_gate}")

    if gate_decision["decision"] == "GO":
        logger.info(">>> GATE STATUS: GO. Production pipeline may continue to RQ2/RQ3/Confirmatory.")
    else:
        logger.warning(
            f">>> GATE STATUS: {gate_decision['decision']}. "
            "Production pipeline stops unless --force-after-no-go is set. "
            "Only exact GO continues; Valence-only / Arousal-only are not GO."
        )


if __name__ == "__main__":
    main()
