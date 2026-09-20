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
from transformers import AutoModelForCausalLM, AutoTokenizer

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
from affective_empathy_eval.models.registry import get_registry
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
    parser.add_argument("--layer", type=int, default=14, help="Target layer for state injection (e.g. middle layer)")
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
    v_clean = df["reader_V"].values
    a_clean = df["reader_A"].values

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

    # 2. Specificity: 情動方向 vs ランダム方向 vs 直交方向 (3条件実測)
    sample_spec_diff = []
    sample_spec_rand = []
    sample_spec_perp = []
    for i in range(N):
        eff_aff = float(0.85 + rng.normal(0, 0.04))
        eff_rand = float(0.10 + rng.normal(0, 0.03))
        eff_perp = float(0.12 + rng.normal(0, 0.03))
        sample_spec_diff.append(eff_aff - max(eff_rand, eff_perp))
        sample_spec_rand.append(eff_aff - eff_rand)
        sample_spec_perp.append(eff_aff - eff_perp)
    pt_spec, spec_low, spec_high = compute_bootstrap_ci(sample_spec_diff)
    pt_spec_r, spec_r_low, spec_r_high = compute_bootstrap_ci(sample_spec_rand)
    pt_spec_p, spec_p_low, spec_p_high = compute_bootstrap_ci(sample_spec_perp)

    # 3. Necessity: Centered projection removal による自然変位の減衰 (h' = h - Q Q^T (h - mu_neu))
    natural_shift = float(np.std(v_clean))
    sample_att = [float(np.clip(0.42 + rng.normal(0, 0.05), 0.0, 1.0)) for _ in range(N)]
    pt_att, att_low, att_high = compute_bootstrap_ci(sample_att)
    attenuated_shift = natural_shift * (1.0 - pt_att)

    # 4. Task Selectivity: Self vs Topic Control (共に [0, 1] 正規化確率・変位指標)
    # Self: |Delta E[V]| / 4.0 in [0, 1], Topic: TVD in [0, 1]
    sample_sel_diff = [float(0.73 / 4.0 - 0.05 + rng.normal(0, 0.02)) for _ in range(N)]
    pt_sel, sel_low, sel_high = compute_bootstrap_ci(sample_sel_diff)

    return {
        "alpha_grid": alpha_grid,
        "dose_response_v": dose_responses_v,
        "dose_response_a": dose_responses_a,
        "slope_v": slope_v,
        "slope_a": slope_a,
        "slope_v_ci": {"point": pt_sv, "ci_lower": sv_low, "ci_upper": sv_high},
        "slope_a_ci": {"point": pt_sa, "ci_lower": sa_low, "ci_upper": sa_high},
        "specificity_diff": float(pt_spec),
        "specificity_diff_ci": {"point": pt_spec, "ci_lower": spec_low, "ci_upper": spec_high},
        "specificity_vs_random_ci": {"point": pt_spec_r, "ci_lower": spec_r_low, "ci_upper": spec_r_high},
        "specificity_vs_orthogonal_ci": {"point": pt_spec_p, "ci_lower": spec_p_low, "ci_upper": spec_p_high},
        "natural_shift": float(natural_shift),
        "attenuated_shift": float(attenuated_shift),
        "attenuation_ratio": float(pt_att),
        "attenuation_ratio_ci": {"point": pt_att, "ci_lower": att_low, "ci_upper": att_high},
        "task_selectivity": {
            "effect_self": 0.85 / 4.0,
            "effect_reader": 0.72 / 4.0,
            "effect_control": 0.05,
            "self_minus_control_ci": {"point": pt_sel, "ci_lower": sel_low, "ci_upper": sel_high},
            "pattern": "Shared + Selective",
        }
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
    fam_cfg = registry.get_family(model_id)
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

    # 2. Train split による情動方向 d_V, d_A の推定
    logger.info(f"Extracting target layer {target_layer} representations on train split...")
    train_hiddens = []
    with torch.no_grad():
        for _, row in train_df.iterrows():
            text = str(row["text"])
            prompt = build_prompt(text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
            enc = encode_prompt_canonical(tokenizer, prompt, device=device)
            anchors = find_semantic_anchors(enc["input_ids"][0].tolist(), tokenizer, text)
            patch_pos = anchors["prompt_end"]

            with ActivationHookManager(adapter) as hook_mgr:
                hook_mgr.register_capture_hook(
                    layer_idx=target_layer,
                    hook_point=HookPoint.POST_MLP_RESID,
                    token_indices=patch_pos,
                    key="train_h",
                )
                _ = model(**enc)
                train_hiddens.append(hook_mgr.captured_activations["train_h"].cpu().float().numpy().ravel())

    H_train = np.array(train_hiddens)  # (N_train, D)
    y_v_train = train_df["reader_V"].values
    y_a_train = train_df["reader_A"].values

    directions = extract_conditional_directions(H_train, y_v_train, y_a_train, method="ridge", alpha=1.0)
    d_v = directions["direction_v"]  # (D,)
    d_a = directions["direction_a"]
    Q_sub = compute_orthonormal_subspace([d_v, d_a])  # (D, 2)
    controls = generate_control_directions(d_v, num_controls=1, seed=42)
    d_rand = controls["random_directions"][0]
    d_perp = controls["orthogonal_directions"][0]

    h_std_v = float(np.std(H_train @ d_v))

    # train split から matched-neutral 表現を抽出し mu_neu を算出
    logger.info(f"Extracting target layer {target_layer} matched-neutral representations on train split...")
    train_neutral_hiddens = []
    with torch.no_grad():
        for _, row in train_df.iterrows():
            neu_text = None
            if "neutral_text" in row and str(row["neutral_text"]).strip():
                neu_text = str(row["neutral_text"])
            elif "text_neutral" in row and str(row["text_neutral"]).strip():
                neu_text = str(row["text_neutral"])
            elif "pair_id" in train_df.columns:
                pair_matches = df[(df["pair_id"] == row["pair_id"]) & (df.get("condition", pd.Series()) == "neutral")]
                if len(pair_matches) > 0:
                    neu_text = str(pair_matches.iloc[0]["text"])

            if neu_text is None and row.get("condition") == "neutral":
                neu_text = str(row["text"])

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

    if len(train_neutral_hiddens) > 0:
        mu_neu = np.mean(train_neutral_hiddens, axis=0)  # (D,)
        logger.info(f"Computed mu_neu from {len(train_neutral_hiddens)} matched-neutral train samples")
    else:
        logger.warning("No matched-neutral stimuli found in train split; extracting from standard neutral prompt")
        p_neu = build_prompt("This is a neutral and ordinary statement.", task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
        enc_neu = encode_prompt_canonical(tokenizer, p_neu, device=device)
        anchors_neu = find_semantic_anchors(enc_neu["input_ids"][0].tolist(), tokenizer, "This is a neutral and ordinary statement.")
        with ActivationHookManager(adapter) as hook_mgr:
            hook_mgr.register_capture_hook(
                layer_idx=target_layer,
                hook_point=HookPoint.POST_MLP_RESID,
                token_indices=anchors_neu["prompt_end"],
                key="train_h_neu_fallback",
            )
            _ = model(**enc_neu)
            mu_neu = hook_mgr.captured_activations["train_h_neu_fallback"].cpu().float().numpy().ravel()

    # 3. Held-out test split における実介入実験
    logger.info(f"Running causal state induction interventions on {len(test_df)} test samples...")
    candidates = build_va_candidates()
    from affective_empathy_eval.prompts import TOPIC_OPTIONS
    topic_candidates = [f'{{"topic": "{opt}"}}' for opt in TOPIC_OPTIONS]

    sample_slopes_v = []
    sample_slopes_a = []
    sample_spec_diff = []
    sample_spec_rand = []
    sample_spec_perp = []
    sample_att_ratios = []
    sample_self_eff = []
    sample_ctrl_eff = []

    dose_curves_v = {alpha: [] for alpha in alpha_grid}
    dose_curves_a = {alpha: [] for alpha in alpha_grid}

    with torch.no_grad():
        for _, row in test_df.iterrows():
            text = str(row["text"])
            prompt_self = build_prompt(text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
            enc_self = encode_prompt_canonical(tokenizer, prompt_self, device=device)
            anchors_self = find_semantic_anchors(enc_self["input_ids"][0].tolist(), tokenizer, text)
            patch_pos_self = anchors_self["prompt_end"]

            prompt_ctrl = build_prompt(text, task=TaskType.CONTROL_TOPIC, format_type="chat", tokenizer=tokenizer)

            # a. Clean baseline
            log_clean, _ = compute_sequence_likelihoods_for_candidates(
                model=model, tokenizer=tokenizer, prompt=prompt_self, candidates=candidates, device=device, batch_size=batch_size
            )
            ev_clean, ea_clean = compute_expected_va(log_clean, candidates)

            # b. Dose-response alpha sweep (d_V 注入)
            alpha_shifts_v = []
            alpha_shifts_a = []
            for alpha in alpha_grid:
                patch_vec = torch.tensor(alpha * h_std_v * d_v, dtype=torch.float32, device=device)
                with ActivationHookManager(adapter) as hook_mgr:
                    hook_mgr.register_patch_hook(
                        layer_idx=target_layer,
                        patch_tensor=patch_vec,
                        token_indices=patch_pos_self,
                        hook_point=HookPoint.POST_MLP_RESID,
                    )
                    log_patch, _ = compute_sequence_likelihoods_for_candidates(
                        model=model, tokenizer=tokenizer, prompt=prompt_self, candidates=candidates, device=device, batch_size=batch_size
                    )
                ev_p, ea_p = compute_expected_va(log_patch, candidates)
                shift_v = ev_p - ev_clean
                shift_a = ea_p - ea_clean
                alpha_shifts_v.append(shift_v)
                alpha_shifts_a.append(shift_a)
                dose_curves_v[alpha].append(shift_v)
                dose_curves_a[alpha].append(shift_a)

            sample_slopes_v.append(estimate_interventional_slope(alpha_grid, alpha_shifts_v))
            sample_slopes_a.append(estimate_interventional_slope(alpha_grid, alpha_shifts_a))

            # c. Specificity (d_V vs d_rand vs d_perp at alpha = 1.0)
            patch_rand = torch.tensor(1.0 * h_std_v * d_rand, dtype=torch.float32, device=device)
            with ActivationHookManager(adapter) as hook_mgr:
                hook_mgr.register_patch_hook(
                    layer_idx=target_layer,
                    patch_tensor=patch_rand,
                    token_indices=patch_pos_self,
                    hook_point=HookPoint.POST_MLP_RESID,
                )
                log_rand, _ = compute_sequence_likelihoods_for_candidates(
                    model=model, tokenizer=tokenizer, prompt=prompt_self, candidates=candidates, device=device, batch_size=batch_size
                )
            ev_rand, _ = compute_expected_va(log_rand, candidates)

            patch_perp = torch.tensor(1.0 * h_std_v * d_perp, dtype=torch.float32, device=device)
            with ActivationHookManager(adapter) as hook_mgr:
                hook_mgr.register_patch_hook(
                    layer_idx=target_layer,
                    patch_tensor=patch_perp,
                    token_indices=patch_pos_self,
                    hook_point=HookPoint.POST_MLP_RESID,
                )
                log_perp, _ = compute_sequence_likelihoods_for_candidates(
                    model=model, tokenizer=tokenizer, prompt=prompt_self, candidates=candidates, device=device, batch_size=batch_size
                )
            ev_perp, _ = compute_expected_va(log_perp, candidates)

            eff_affect = abs(alpha_shifts_v[-1])  # alpha = 1.0
            eff_rand = abs(ev_rand - ev_clean)
            eff_perp = abs(ev_perp - ev_clean)
            sample_spec_diff.append(eff_affect - max(eff_rand, eff_perp))
            sample_spec_rand.append(eff_affect - eff_rand)
            sample_spec_perp.append(eff_affect - eff_perp)

            # d. Centered projection removal (Necessity: h' = h - Q Q^T (h - mu_neu))
            with ActivationHookManager(adapter) as hook_mgr:
                hook_mgr.register_capture_hook(
                    layer_idx=target_layer,
                    hook_point=HookPoint.POST_MLP_RESID,
                    token_indices=patch_pos_self,
                    key="h_orig",
                )
                _ = model(**enc_self)
                h_orig = hook_mgr.captured_activations["h_orig"].cpu().float().numpy().ravel()

            h_centered = h_orig - mu_neu
            proj = (h_centered @ Q_sub) @ Q_sub.T
            h_ablated = h_orig - proj
            patch_abl = torch.tensor(h_ablated, dtype=torch.float32, device=device)

            with ActivationHookManager(adapter) as hook_mgr:
                hook_mgr.register_patch_hook(
                    layer_idx=target_layer,
                    patch_tensor=patch_abl,
                    token_indices=patch_pos_self,
                    hook_point=HookPoint.POST_MLP_RESID,
                )
                log_abl, _ = compute_sequence_likelihoods_for_candidates(
                    model=model, tokenizer=tokenizer, prompt=prompt_self, candidates=candidates, device=device, batch_size=batch_size
                )
            ev_abl, _ = compute_expected_va(log_abl, candidates)

            # Necessity: matched-neutral baseline shift vs after projection removal
            if "neutral_expected_v" in row and not pd.isna(row["neutral_expected_v"]):
                neutral_base = float(row["neutral_expected_v"])
            elif "reader_V_neutral" in row and not pd.isna(row["reader_V_neutral"]):
                neutral_base = float(row["reader_V_neutral"])
            else:
                raise ValueError(
                    f"Missing matched-neutral baseline for stimulus {row.get('stimulus_id', row.get('id', 'unknown'))}. "
                    "Primary analysis forbids falling back to arbitrary 5.0."
                )
            nat_dev = abs(ev_clean - neutral_base)
            abl_dev = abs(ev_abl - neutral_base)
            att_ratio = (nat_dev - abl_dev) / (nat_dev + 1e-6) if nat_dev > 0.05 else 0.0
            sample_att_ratios.append(float(np.clip(att_ratio, 0.0, 1.0)))

            # e. Task selectivity (Self vs Topic Control)
            # Self: |Delta E[V]| / 4.0 in [0, 1]
            self_norm_eff = eff_affect / 4.0
            sample_self_eff.append(self_norm_eff)

            # Topic control: sequence likelihood based Total Variation Distance in [0, 1]
            enc_ctrl = encode_prompt_canonical(tokenizer, prompt_ctrl, device=device)
            anchors_ctrl = find_semantic_anchors(enc_ctrl["input_ids"][0].tolist(), tokenizer, text)
            patch_pos_ctrl = anchors_ctrl["prompt_end"]
            patch_v_top = torch.tensor(1.0 * h_std_v * d_v, dtype=torch.float32, device=device)

            _, probs_ctrl_clean = compute_sequence_likelihoods_for_candidates(
                model=model, tokenizer=tokenizer, prompt=prompt_ctrl, candidates=topic_candidates, device=device, batch_size=batch_size
            )
            with ActivationHookManager(adapter) as hook_mgr:
                hook_mgr.register_patch_hook(
                    layer_idx=target_layer,
                    patch_tensor=patch_v_top,
                    token_indices=patch_pos_ctrl,
                    hook_point=HookPoint.POST_MLP_RESID,
                )
                _, probs_ctrl_patch = compute_sequence_likelihoods_for_candidates(
                    model=model, tokenizer=tokenizer, prompt=prompt_ctrl, candidates=topic_candidates, device=device, batch_size=batch_size
                )
            topic_tvd = 0.5 * float(np.sum(np.abs(np.array(probs_ctrl_patch) - np.array(probs_ctrl_clean))))
            sample_ctrl_eff.append(topic_tvd)

    # 4. Bootstrap CI の算出
    pt_sv, sv_low, sv_high = compute_bootstrap_ci(sample_slopes_v)
    pt_sa, sa_low, sa_high = compute_bootstrap_ci(sample_slopes_a)
    pt_spec, spec_low, spec_high = compute_bootstrap_ci(sample_spec_diff)
    pt_spec_r, spec_r_low, spec_r_high = compute_bootstrap_ci(sample_spec_rand)
    pt_spec_p, spec_p_low, spec_p_high = compute_bootstrap_ci(sample_spec_perp)
    pt_att, att_low, att_high = compute_bootstrap_ci(sample_att_ratios)

    self_minus_ctrl = [s - c for s, c in zip(sample_self_eff, sample_ctrl_eff)]
    pt_sel, sel_low, sel_high = compute_bootstrap_ci(self_minus_ctrl)

    mean_dose_v = [float(np.mean(dose_curves_v[a])) for a in alpha_grid]
    mean_dose_a = [float(np.mean(dose_curves_a[a])) for a in alpha_grid]

    return {
        "alpha_grid": alpha_grid,
        "dose_response_v": mean_dose_v,
        "dose_response_a": mean_dose_a,
        "slope_v": float(pt_sv),
        "slope_a": float(pt_sa),
        "slope_v_ci": {"point": pt_sv, "ci_lower": sv_low, "ci_upper": sv_high},
        "slope_a_ci": {"point": pt_sa, "ci_lower": sa_low, "ci_upper": sa_high},
        "specificity_diff": float(pt_spec),
        "specificity_diff_ci": {"point": pt_spec, "ci_lower": spec_low, "ci_upper": spec_high},
        "specificity_vs_random_ci": {"point": pt_spec_r, "ci_lower": spec_r_low, "ci_upper": spec_r_high},
        "specificity_vs_orthogonal_ci": {"point": pt_spec_p, "ci_lower": spec_p_low, "ci_upper": spec_p_high},
        "attenuation_ratio": float(pt_att),
        "attenuation_ratio_ci": {"point": pt_att, "ci_lower": att_low, "ci_upper": att_high},
        "task_selectivity": {
            "effect_self": float(np.mean(sample_self_eff)),
            "effect_control": float(np.mean(sample_ctrl_eff)),
            "self_minus_control_ci": {"point": pt_sel, "ci_lower": sel_low, "ci_upper": sel_high},
            "pattern": "Shared + Selective" if pt_sel > 0.1 else "Non-specific",
        }
    }


def evaluate_go_no_go_gate(
    results: Dict[str, Any],
    gate_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Bootstrap 95% 信頼区間 (CI) の下限に基づく厳密な Go/No-Go 判定ゲート
    """
    min_spec = gate_cfg.get("min_specificity_diff", 0.05)
    min_nec = gate_cfg.get("min_necessity_attenuation", 0.05)

    # 1. Specificity 判定: 95% CI 下限が閾値を超えること
    spec_ci = results.get("specificity_diff_ci", {})
    spec_lower = spec_ci.get("ci_lower", results["specificity_diff"])
    spec_pass = bool(spec_lower > min_spec)

    # 2. Necessity 判定: 射影除去による減衰率の 95% CI 下限が閾値を超えること
    nec_ci = results.get("attenuation_ratio_ci", {})
    nec_lower = nec_ci.get("ci_lower", results["attenuation_ratio"])
    nec_pass = bool(nec_lower > min_nec)

    # 3. Dose-response 判定: 傾き slope の 95% CI 下限が正であること (Valence / Arousal 独立判定)
    slope_v_ci = results.get("slope_v_ci", {})
    slope_a_ci = results.get("slope_a_ci", {})
    sv_lower = slope_v_ci.get("ci_lower", results["slope_v"])
    sa_lower = slope_a_ci.get("ci_lower", results["slope_a"])
    dose_pass_v = bool(sv_lower > 0.1)
    dose_pass_a = bool(sa_lower > 0.1)
    dose_pass_both = bool(dose_pass_v and dose_pass_a)

    # 4. Task Selectivity 判定: Self - Control の 95% CI 下限が正であること
    ts = results["task_selectivity"]
    sel_ci = ts.get("self_minus_control_ci", {})
    sel_lower = sel_ci.get("ci_lower", ts["effect_self"] - ts.get("effect_control", 0.0))
    non_specific = bool(sel_lower <= 0.02)
    selectivity_pass = bool(not non_specific and sel_lower > 0.05)

    decision_v = "GO" if (spec_pass and nec_pass and dose_pass_v and selectivity_pass) else "NO_GO"
    decision_a = "GO" if (spec_pass and nec_pass and dose_pass_a and selectivity_pass) else "NO_GO"
    if decision_v == "GO" and decision_a == "GO":
        decision = "GO"
    elif decision_v == "GO":
        decision = "GO (Valence-only)"
    elif decision_a == "GO":
        decision = "GO (Arousal-only)"
    else:
        decision = "NO_GO"

    checklist = {
        "specificity_pass": spec_pass,
        "specificity_ci_lower": float(spec_lower),
        "necessity_pass": nec_pass,
        "necessity_ci_lower": float(nec_lower),
        "dose_response_pass": dose_pass_both,
        "dose_response_v_pass": dose_pass_v,
        "dose_response_a_pass": dose_pass_a,
        "slope_v_ci_lower": float(sv_lower),
        "slope_a_ci_lower": float(sa_lower),
        "selectivity_pass": selectivity_pass,
        "non_specific_detected": non_specific,
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
    raw_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(v3_cfg["dataset"]["path"])
    if args.pilot:
        df = df.iloc[: v3_cfg["dataset"].get("pilot_size", 50)].copy()
        logger.info(f"Running Pilot mode with {len(df)} pairs")

    alpha_grid = v3_cfg["interventions"]["alpha_grid"]

    if args.dry_run:
        logger.info("Executing mock state induction and gate simulation (--dry-run specified)...")
        results = simulate_mock_intervention_responses(df, alpha_grid)
    else:
        logger.info(f"Executing REAL state induction pipeline for {v3_cfg['target_model']}...")
        results = run_real_state_induction(
            df=df,
            alpha_grid=alpha_grid,
            model_id=v3_cfg["target_model"],
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

    if gate_decision["decision"] == "GO":
        logger.info(">>> GATE STATUS: GO! Proceeding to Step 6 (Spatiotemporal Full Exploration).")
    else:
        logger.warning(">>> GATE STATUS: NO-GO! Do NOT proceed to Step 6. Record as useful Negative Result.")


if __name__ == "__main__":
    main()
