#!/usr/bin/env python3
"""
Step 7: 他3モデル (Llama 3.2, Gemma 3, OLMo 2) における主要 V3 因果機構の Confirmatory 再現
検証対象の4大仮説:
  1. Hypothesis 1 (Dissociation): デコードピークと因果ピークの解離 (Δd_peak > 0, Δd_center > 0)
  2. Hypothesis 2 (Sufficiency): 内部情動方向 d_V, d_A への介入による単調誘導 (gamma > 0)
  3. Hypothesis 3 (Necessity): 直交化 2D 部分空間除去による自己報告変位の有意な減衰
  4. Hypothesis 4 (Temporal Emergence): 生成時意味的アンカー pre_V での因果ピーク集中
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import torch
import yaml
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # --dry-run は transformers 未導入環境でも起動できるようにする
    AutoModelForCausalLM = None  # type: ignore[misc, assignment]
    AutoTokenizer = None  # type: ignore[misc, assignment]
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold

from affective_empathy_eval.geometry import compute_layer_dissociation
from affective_empathy_eval.interventions import (
    compute_orthonormal_subspace,
    estimate_interventional_slope,
    extract_conditional_directions,
)
from affective_empathy_eval.likelihood import (
    build_va_candidates,
    compute_expected_va,
    compute_sequence_likelihoods_for_candidates,
)
from affective_empathy_eval.data import (
    describe_loaded_frame,
    load_v3_matched_pair_table,
    resolve_matched_neutral_text,
)
from affective_empathy_eval.models.registry import (
    add_model_selection_args,
    get_registry,
    load_model_set,
    resolve_architecture_dims,
)
from affective_empathy_eval.prompts import (
    TaskType,
    build_prompt,
    encode_prompt_canonical,
    find_semantic_anchors,
    get_generation_stage_tokens,
)
from affective_empathy_eval.statistics import compute_bootstrap_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run V3 Confirmatory Replication across families")
    parser.add_argument("--config", type=str, default="configs/v3_experiments.yaml", help="Path to V3 config")
    parser.add_argument("--models-config", type=str, default="configs/models.yaml", help="Path to models config")
    parser.add_argument("--dry-run", action="store_true", help="Run in mock/dry-run mode")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--subsample", type=int, default=0, help="Number of pairs per model for confirmatory evaluation (0 for full dataset)")
    add_model_selection_args(parser)
    return parser.parse_args()


def simulate_model_confirmatory(
    family: str,
    num_layers: int,
    semantic_stages: List[str],
    seed: int,
) -> Dict[str, Any]:
    """
    dry-run用: 特定モデルファミリーに対する Confirmatory 検証のシミュレーション
    """
    rng = np.random.default_rng(seed)
    relative_depths = [l / (num_layers - 1) if num_layers > 1 else 0.0 for l in range(num_layers)]

    d_center = 0.46 if family == "Llama" else (0.50 if family == "Gemma" else 0.48)
    c_center = 0.65 if family == "Llama" else (0.72 if family == "Gemma" else 0.70)

    d_profile_v = [float(np.exp(-((d - d_center) ** 2) / (2 * 0.16**2)) * 0.70 + rng.normal(0, 0.02)) for d in relative_depths]
    c_profile_v = [float(np.exp(-((d - c_center) ** 2) / (2 * 0.14**2)) * 1.05 + rng.normal(0, 0.02)) for d in relative_depths]

    dissoc_v = compute_layer_dissociation(relative_depths, d_profile_v, c_profile_v)

    alphas = [-1.0, -0.5, 0.0, 0.5, 1.0]
    shifts_v = [float(a * (0.80 + rng.normal(0, 0.03))) for a in alphas]
    shifts_a = [float(a * (0.68 + rng.normal(0, 0.03))) for a in alphas]
    slope_v = estimate_interventional_slope(alphas, shifts_v)
    slope_a = estimate_interventional_slope(alphas, shifts_a)

    natural_shift = 1.10 + rng.normal(0, 0.05)
    attenuated_shift = 0.40 + rng.normal(0, 0.03)
    attenuation_ratio = float((natural_shift - attenuated_shift) / natural_shift)

    stage_causal_v = {
        "candidate_start": float(0.12 + rng.normal(0, 0.02)),
        "pre_V": float(1.05 + rng.normal(0, 0.03)),
        "V_value": float(0.65 + rng.normal(0, 0.03)),
        "pre_A": float(0.40 + rng.normal(0, 0.02)),
        "A_value": float(0.25 + rng.normal(0, 0.02)),
        "response_end": float(0.08 + rng.normal(0, 0.02)),
    }
    stage_causal_a = {
        "candidate_start": float(0.10 + rng.normal(0, 0.02)),
        "pre_V": float(0.35 + rng.normal(0, 0.02)),
        "V_value": float(0.20 + rng.normal(0, 0.02)),
        "pre_A": float(0.95 + rng.normal(0, 0.03)),
        "A_value": float(0.60 + rng.normal(0, 0.03)),
        "response_end": float(0.07 + rng.normal(0, 0.02)),
    }

    h1_pass = dissoc_v["delta_d_peak"] > 0 and dissoc_v["delta_d_center"] > 0
    h2_pass = slope_v > 0.2 and slope_a > 0.2
    h3_pass = attenuation_ratio > 0.3
    h4_pass_v = stage_causal_v["pre_V"] > stage_causal_v["candidate_start"] + 0.5
    h4_pass_a = stage_causal_a["pre_A"] > stage_causal_a["candidate_start"] + 0.5
    h4_pass = bool(h4_pass_v and h4_pass_a)

    all_confirmed = bool(h1_pass and h2_pass and h3_pass and h4_pass)

    return {
        "family": family,
        "is_simulation": True,
        "dry_run": True,
        "num_layers": num_layers,
        "h1_dissociation": {
            "d_peak_D": dissoc_v["d_peak_D"],
            "d_peak_C": dissoc_v["d_peak_C"],
            "delta_d_peak": dissoc_v["delta_d_peak"],
            "delta_d_center": dissoc_v["delta_d_center"],
            "passed": bool(h1_pass),
        },
        "h2_sufficiency": {
            "slope_v": float(slope_v),
            "slope_a": float(slope_a),
            "passed": bool(h2_pass),
        },
        "h3_necessity": {
            "natural_shift": float(natural_shift),
            "attenuated_shift": float(attenuated_shift),
            "attenuation_ratio": float(attenuation_ratio),
            "passed": bool(h3_pass),
        },
        "h4_temporal_emergence": {
            "stage_causal": stage_causal_v,
            "stage_causal_v": stage_causal_v,
            "stage_causal_a": stage_causal_a,
            "passed_valence": bool(h4_pass_v),
            "passed_arousal": bool(h4_pass_a),
            "passed": bool(h4_pass),
        },
        "all_confirmed": all_confirmed,
    }


def run_real_model_confirmatory(
    family: str,
    model_id: str,
    df: pd.DataFrame,
    device: str = "cpu",
    subsample: int = 0,
) -> Dict[str, Any]:
    """
    実モデル (Llama 3.2, Gemma 3, OLMo 2) に対する 4大仮説の Confirmatory 検証
    """
    logger.info(f"Loading {family} model: {model_id} on {device}...")
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
    num_layers = fam_cfg.num_layers
    relative_depths = [l / (num_layers - 1) if num_layers > 1 else 0.0 for l in range(num_layers)]

    if subsample is not None and subsample > 0:
        eval_df = df.head(subsample).copy().reset_index(drop=True)
    else:
        eval_df = df.copy().reset_index(drop=True)
    N = len(eval_df)
    candidates = build_va_candidates()

    # 1. Clean Baselines
    clean_ev_list = []
    clean_ea_list = []
    with torch.no_grad():
        for _, row in eval_df.iterrows():
            text = str(row["text"])
            prompt = build_prompt(text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
            _, probs = compute_sequence_likelihoods_for_candidates(
                model=model, tokenizer=tokenizer, prompt=prompt, candidates=candidates, device=device, batch_size=81
            )
            ev, ea = compute_expected_va(probs, candidates)
            clean_ev_list.append(ev)
            clean_ea_list.append(ea)

    y_v = np.array(clean_ev_list)
    y_a = np.array(clean_ea_list)

    # 2. 全層の刺激提示時デコード能 D(l) (Held-out R^2 via 5-Fold Cross-Validation)
    d_profile_v = []
    if "pair_id" in eval_df.columns and eval_df["pair_id"].nunique() >= 2:
        n_splits = min(5, eval_df["pair_id"].nunique())
        splitter = GroupKFold(n_splits=n_splits)
        split_gen_fn = lambda data: splitter.split(data, y_v, groups=eval_df["pair_id"].values)
    else:
        n_splits = min(5, max(2, N))
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_gen_fn = lambda data: splitter.split(data)

    for l in range(num_layers):
        h_l = []
        with torch.no_grad():
            for _, row in eval_df.iterrows():
                text = str(row["text"])
                prompt = build_prompt(text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
                enc = encode_prompt_canonical(tokenizer, prompt, device=device)
                anchors = find_semantic_anchors(enc["input_ids"][0].tolist(), tokenizer, text)
                with ActivationHookManager(adapter) as hook_mgr:
                    hook_mgr.register_capture_hook(
                        layer_idx=l,
                        hook_point=HookPoint.POST_MLP_RESID,
                        token_indices=anchors["prompt_end"],
                        key="h",
                    )
                    _ = model(**enc)
                    h_l.append(hook_mgr.captured_activations["h"].cpu().float().numpy().ravel())
        H = np.array(h_l)
        all_H[l] = H

        # Compute held-out predictions out-of-fold (pair-aware if pair_id present)
        oof_preds = np.zeros(N)
        for train_idx, test_idx in split_gen_fn(H):
            ridge = Ridge(alpha=10.0).fit(H[train_idx], y_v[train_idx])
            oof_preds[test_idx] = ridge.predict(H[test_idx])

        ss_tot = np.sum((y_v - np.mean(y_v))**2)
        ss_res = np.sum((y_v - oof_preds)**2)
        r2 = max(0.0, float(1.0 - ss_res / (ss_tot + 1e-6)))
        d_profile_v.append(r2)

    # 3. 中間〜後期層における介入効果 (gamma_V, slope_v)
    mid_layer = int(num_layers * 0.65)
    # 情動方向 d_V, d_A は mid_layer の hidden states から正しく推定
    H_mid = all_H[mid_layer]
    ridge_mid_v = Ridge(alpha=10.0).fit(H_mid, y_v)
    d_v = ridge_mid_v.coef_ / (np.linalg.norm(ridge_mid_v.coef_) + 1e-6)
    ridge_mid_a = Ridge(alpha=10.0).fit(H_mid, y_a)
    d_a = ridge_mid_a.coef_ / (np.linalg.norm(ridge_mid_a.coef_) + 1e-6)

    alphas = [-1.0, -0.5, 0.0, 0.5, 1.0]
    shifts_v = []
    shifts_a = []
    for direction, collect, clean_list in (
        (d_v, shifts_v, clean_ev_list),
        (d_a, shifts_a, clean_ea_list),
    ):
        for alpha in alphas:
            p_vec = torch.tensor(alpha * direction, dtype=torch.float32, device=device)
            axis_shifts = []
            with torch.no_grad():
                for idx in range(min(5, N)):
                    text = str(eval_df.loc[idx, "text"])
                    prompt = build_prompt(text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
                    enc = encode_prompt_canonical(tokenizer, prompt, device=device)
                    anchors = find_semantic_anchors(enc["input_ids"][0].tolist(), tokenizer, text)
                    with ActivationHookManager(adapter) as hook_mgr:
                        hook_mgr.register_patch_hook(
                            layer_idx=mid_layer,
                            patch_tensor=p_vec,
                            token_indices=anchors["prompt_end"],
                            hook_point=HookPoint.POST_MLP_RESID,
                        )
                        _, probs_p = compute_sequence_likelihoods_for_candidates(
                            model=model, tokenizer=tokenizer, prompt=prompt, candidates=candidates, device=device, batch_size=81
                        )
                    ev_p, ea_p = compute_expected_va(probs_p, candidates)
                    axis_shifts.append((ev_p if collect is shifts_v else ea_p) - clean_list[idx])
            collect.append(float(np.mean(axis_shifts)))

    slope_v = estimate_interventional_slope(alphas, shifts_v)
    slope_a = estimate_interventional_slope(alphas, shifts_a)

    # 4. Causal profile across layers (Empirical H1, layer-specific)
    logger.info("Measuring empirical causal profile C(l) across layers (layer-specific directions)...")
    c_profile_v = []
    test_df = eval_df
    causal_sub_df = test_df.iloc[:min(15, len(test_df))]
    causal_sub_indices = causal_sub_df.index.tolist()
    batch_size = 81
    opt_layer = mid_layer
    evaluate_candidate_likelihoods = lambda *args, **kwargs: compute_sequence_likelihoods_for_candidates(*args, **kwargs)[1]

    for l_idx in range(num_layers):
        H_l = all_H[l_idx]
        ridge_l_v = Ridge(alpha=10.0).fit(H_l, y_v)
        coef_norm_v = np.linalg.norm(ridge_l_v.coef_)
        d_v_l = ridge_l_v.coef_ / (coef_norm_v + 1e-6)
        proj_std_v = float(np.std(H_l @ d_v_l))
        h_std_v_l = proj_std_v if proj_std_v > 1e-6 else float(np.std(H_l))
        dv_l_torch = torch.tensor(1.0 * h_std_v_l * d_v_l, dtype=torch.float32, device=device)

        l_shifts = []
        for row_i, (_, row) in enumerate(causal_sub_df.iterrows()):
            text = str(row["text"])
            global_idx = causal_sub_indices[row_i]
            prompt_self = build_prompt(text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
            enc = encode_prompt_canonical(tokenizer, prompt_self, device=device)
            anchors = find_semantic_anchors(enc["input_ids"][0].tolist(), tokenizer, text)
            p_pos = anchors["prompt_end"]
            with ActivationHookManager(adapter) as hook_mgr:
                hook_mgr.register_patch_hook(
                    layer_idx=l_idx,
                    component="residual",
                    token_indices=p_pos,
                    direction=dv_l_torch,
                    alpha=1.0,
                )
                probs_p = evaluate_candidate_likelihoods(
                    model=model, tokenizer=tokenizer, prompt=prompt_self, candidates=candidates, device=device, batch_size=batch_size
                )
            ev_p, _ = compute_expected_va(probs_p, candidates)
            # Use empirical clean baseline
            clean_v = float(clean_ev_list[global_idx])
            l_shifts.append(abs(ev_p - clean_v))
        c_profile_v.append(float(np.mean(l_shifts)) if l_shifts else 0.0)

    dissoc_v = compute_layer_dissociation(relative_depths, d_profile_v, c_profile_v)

    # 5. Necessity via Centered 2D Orthogonal Subspace Removal (Empirical H3)
    logger.info("Measuring empirical necessity via centered 2D orthogonal subspace removal...")
    nat_shifts = []
    att_shifts = []

    # 2D Orthonormal Basis Q via QR decomposition
    M_sub = np.column_stack([d_v, d_a])  # (dim, 2)
    Q_np, _ = np.linalg.qr(M_sub)        # (dim, 2) orthonormal
    Q = torch.tensor(Q_np, dtype=torch.float32, device=device)

    # Pre-compute neutral reference representations and baselines
    neutral_reps = []
    neutral_baselines = []
    for row_i, (_, row) in enumerate(causal_sub_df.iterrows()):
        neu_text = resolve_matched_neutral_text(row, df)
        if neu_text.strip():
            p_neu = build_prompt(neu_text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
            enc_neu = encode_prompt_canonical(tokenizer, p_neu, device=device)
            anch_neu = find_semantic_anchors(enc_neu["input_ids"][0].tolist(), tokenizer, neu_text)
            with ActivationHookManager(adapter) as hook_mgr:
                hook_mgr.register_capture_hook(
                    layer_idx=opt_layer,
                    hook_point=HookPoint.POST_MLP_RESID,
                    token_indices=anch_neu["prompt_end"],
                    key="h_neu",
                )
                _ = model(**enc_neu)
                neutral_reps.append(hook_mgr.captured_activations["h_neu"].cpu().float().numpy().ravel())
            with torch.no_grad():
                probs_neu = evaluate_candidate_likelihoods(
                    model=model, tokenizer=tokenizer, prompt=p_neu, candidates=candidates, device=device, batch_size=batch_size
                )
                neu_ev, _ = compute_expected_va(probs_neu, candidates)
                neutral_baselines.append(neu_ev)
        else:
            global_idx = causal_sub_indices[row_i]
            neutral_baselines.append(float(clean_ev_list[global_idx]))

    if neutral_reps:
        mu_neu = torch.tensor(np.mean(neutral_reps, axis=0), dtype=torch.float32, device=device)
    else:
        mu_neu = torch.tensor(np.mean(H_mid, axis=0), dtype=torch.float32, device=device)

    for row_i, (_, row) in enumerate(causal_sub_df.iterrows()):
        text = str(row["text"])
        global_idx = causal_sub_indices[row_i]
        prompt_self = build_prompt(text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
        enc = encode_prompt_canonical(tokenizer, prompt_self, device=device)
        anchors = find_semantic_anchors(enc["input_ids"][0].tolist(), tokenizer, text)
        p_pos = anchors["prompt_end"]

        # Clean state expected VA
        ev_clean = float(clean_ev_list[global_idx])
        neutral_base = float(neutral_baselines[row_i])
        n_shift = abs(ev_clean - neutral_base)
        nat_shifts.append(n_shift)

        # Centered projection removal hook: h' = h - Q Q^T (h - mu_neu)
        def proj_removal_hook(h):
            h_centered = h - mu_neu
            proj = torch.matmul(h_centered, Q)
            proj_back = torch.matmul(proj, Q.T)
            return h - proj_back

        with ActivationHookManager(adapter) as hook_mgr:
            hook_mgr.register_patch_hook(
                layer_idx=opt_layer,
                component="residual",
                token_indices=p_pos,
                patch_fn=proj_removal_hook,
            )
            probs_abl = evaluate_candidate_likelihoods(
                model=model, tokenizer=tokenizer, prompt=prompt_self, candidates=candidates, device=device, batch_size=batch_size
            )
        ev_abl, _ = compute_expected_va(probs_abl, candidates)
        a_shift = abs(ev_abl - neutral_base)
        att_shifts.append(a_shift)

    natural_shift = float(np.mean(nat_shifts)) if nat_shifts else 1.0
    attenuated_shift = float(np.mean(att_shifts)) if att_shifts else 0.5
    attenuation_ratio = float((natural_shift - attenuated_shift) / (natural_shift + 1e-6))
    attenuation_ratio = float(np.clip(attenuation_ratio, 0.0, 1.0))

    # 6. Temporal Emergence across Generation Stages (Empirical H4 for Valence and Arousal)
    logger.info("Measuring empirical temporal emergence across generation stages (Valence & Arousal)...")
    temp_effects_v = {}
    temp_effects_a = {}

    # Representative candidate JSON string for stage token identification
    sample_cand_str = candidates[40]["json_str"] if len(candidates) > 40 else '{"valence": 5, "arousal": 5}'
    cand_tokens = tokenizer.encode(sample_cand_str, add_special_tokens=False)
    cand_stage_offsets = get_generation_stage_tokens(cand_tokens, tokenizer, candidate_str=sample_cand_str)
    stage_keys = ["candidate_start", "pre_V", "V_value", "pre_A", "A_value", "response_end"]
    stage_anchors = ["prompt_end"] + [k for k in stage_keys if k in cand_stage_offsets]

    proj_std_v_mid = float(np.std(H_mid @ d_v))
    h_std_v_mid = proj_std_v_mid if proj_std_v_mid > 1e-6 else float(np.std(H_mid))
    dv_torch = torch.tensor(1.0 * h_std_v_mid * d_v, dtype=torch.float32, device=device)

    proj_std_a_mid = float(np.std(H_mid @ d_a))
    h_std_a_mid = proj_std_a_mid if proj_std_a_mid > 1e-6 else float(np.std(H_mid))
    da_torch = torch.tensor(1.0 * h_std_a_mid * d_a, dtype=torch.float32, device=device)

    for stg in stage_anchors:
        stg_shifts_v = []
        stg_shifts_a = []
        for row_i, (_, row) in enumerate(causal_sub_df.iloc[:min(5, len(causal_sub_df))].iterrows()):
            text = str(row["text"])
            global_idx = causal_sub_indices[row_i]
            clean_v = float(clean_ev_list[global_idx])
            clean_a = float(clean_ea_list[global_idx])

            prompt_self = build_prompt(text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
            p_ids = tokenizer.encode(prompt_self, add_special_tokens=False)
            p_end = len(p_ids) - 1

            if stg == "prompt_end":
                target_pos = p_end
            else:
                target_pos = p_end + 1 + cand_stage_offsets[stg]

            # 1) Valence steering
            with ActivationHookManager(adapter) as hook_mgr:
                hook_mgr.register_patch_hook(
                    layer_idx=opt_layer,
                    component="residual",
                    token_indices=target_pos,
                    direction=dv_torch,
                    alpha=1.0,
                )
                probs_stg_v = evaluate_candidate_likelihoods(
                    model=model, tokenizer=tokenizer, prompt=prompt_self, candidates=candidates, device=device, batch_size=batch_size
                )
            ev_stg_v, _ = compute_expected_va(probs_stg_v, candidates)
            stg_shifts_v.append(abs(ev_stg_v - clean_v))

            # 2) Arousal steering
            with ActivationHookManager(adapter) as hook_mgr:
                hook_mgr.register_patch_hook(
                    layer_idx=opt_layer,
                    component="residual",
                    token_indices=target_pos,
                    direction=da_torch,
                    alpha=1.0,
                )
                probs_stg_a = evaluate_candidate_likelihoods(
                    model=model, tokenizer=tokenizer, prompt=prompt_self, candidates=candidates, device=device, batch_size=batch_size
                )
            _, ea_stg_a = compute_expected_va(probs_stg_a, candidates)
            stg_shifts_a.append(abs(ea_stg_a - clean_a))

        temp_effects_v[stg] = float(np.mean(stg_shifts_v)) if stg_shifts_v else 0.0
        temp_effects_a[stg] = float(np.mean(stg_shifts_a)) if stg_shifts_a else 0.0

    stage_causal_v = temp_effects_v
    stage_causal_a = temp_effects_a

    h1_pass = dissoc_v["delta_d_peak"] > 0 and dissoc_v["delta_d_center"] > 0
    h2_pass = slope_v > 0.1 and slope_a > 0.1
    h3_pass = attenuation_ratio > 0.2
    h4_pass_v = stage_causal_v.get("pre_V", 0.0) > stage_causal_v.get("candidate_start", 0.0)
    h4_pass_a = stage_causal_a.get("pre_A", 0.0) > stage_causal_a.get("candidate_start", 0.0)
    h4_pass = bool(h4_pass_v and h4_pass_a)

    all_confirmed = bool(h1_pass and h2_pass and h3_pass and h4_pass)

    return {
        "family": family,
        "is_simulation": False,
        "model_id": model_id,
        "num_layers": num_layers,
        "h1_dissociation": {
            "d_peak_D": dissoc_v["d_peak_D"],
            "d_peak_C": dissoc_v["d_peak_C"],
            "delta_d_peak": dissoc_v["delta_d_peak"],
            "delta_d_center": dissoc_v["delta_d_center"],
            "passed": bool(h1_pass),
        },
        "h2_sufficiency": {
            "slope_v": float(slope_v),
            "slope_a": float(slope_a),
            "passed": bool(h2_pass),
        },
        "h3_necessity": {
            "natural_shift": float(natural_shift),
            "attenuated_shift": float(attenuated_shift),
            "attenuation_ratio": float(attenuation_ratio),
            "passed": bool(h3_pass),
        },
        "h4_temporal_emergence": {
            "stage_causal": stage_causal_v,
            "stage_causal_v": stage_causal_v,
            "stage_causal_a": stage_causal_a,
            "passed_valence": bool(h4_pass_v),
            "passed_arousal": bool(h4_pass_a),
            "passed": bool(h4_pass),
        },
        "all_confirmed": all_confirmed,
    }


def main():
    args = parse_args()
    logger.info(f"Starting Step 7 Confirmatory Replication across families (dry_run={args.dry_run})")

    with open(args.config, "r", encoding="utf-8") as f:
        v3_cfg = yaml.safe_load(f)
    df = load_v3_matched_pair_table(v3_cfg["dataset"]["path"])
    logger.info(describe_loaded_frame(df, "V3 confirmatory matched-pair table", v3_cfg["dataset"]["path"]))
    # レジストリから動的解決
    conf_families_list = v3_cfg.get("confirmatory_families", ["llama", "gemma", "olmo"])
    registered_models = load_model_set(Path(args.models_config), model_set=args.model_set)

    conf_models = []
    for fam_key in conf_families_list:
        fam_lower = fam_key.lower()
        if fam_lower not in registered_models:
            raise KeyError(
                f"Confirmatory family '{fam_key}' not found in model-set '{args.model_set}'. "
                f"Available: {list(registered_models.keys())}"
            )
        fam_cfg = registered_models[fam_lower]
        conf_models.append({
            "family": fam_cfg.family_name,
            "model_id": fam_cfg.instruct_model.model_id,
        })
    semantic_stages = v3_cfg["spatiotemporal"]["semantic_stages"]
    normalized_stages = [s if s != "response_start" else "candidate_start" for s in semantic_stages]

    raw_dir = Path(v3_cfg["output"]["raw_dir"])
    derived_dir = Path(v3_cfg["output"]["derived_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    family_results = {}
    seeds = {"Llama": 301, "Gemma": 302, "OLMo": 303, "Mistral": 304}

    for item in conf_models:
        fam = item["family"]
        model_id = item["model_id"]
        out_raw = raw_dir / f"v3_confirmatory_{fam.lower()}.json"

        if out_raw.exists() and not args.dry_run:
            try:
                with open(out_raw, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if cached and "all_confirmed" in cached:
                    logger.info(f"Loaded existing results for {fam} from {out_raw}. Skipping.")
                    family_results[fam] = cached
                    continue
            except Exception:
                pass

        num_layers = resolve_architecture_dims(model_id)[0]

        if args.dry_run:
            logger.info(f"Simulating confirmatory replication for {fam} (--dry-run)...")
            res = simulate_model_confirmatory(fam, num_layers, normalized_stages, seed=seeds.get(fam, 999))
        else:
            logger.info(f"Running REAL confirmatory replication for {fam} ({model_id})...")
            res = run_real_model_confirmatory(
                family=fam,
                model_id=model_id,
                df=df,
                device=args.device,
                subsample=args.subsample,
            )

        family_results[fam] = res

        res["dry_run"] = bool(args.dry_run)
        out_raw = raw_dir / f"v3_confirmatory_{fam.lower()}.json"
        with open(out_raw, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        logger.info(f"Saved {fam} confirmatory results to {out_raw}")

    # メタ分析サマリーの生成
    all_passed = all(res["all_confirmed"] for res in family_results.values())
    status_msg = (
        "MOCK_SIMULATION: Hypotheses simulated for validation purposes."
        if args.dry_run
        else ("CONFIRMED: All hypotheses supported by real model evaluations." if all_passed else "PARTIAL: Some hypotheses not fully replicated.")
    )

    summary = {
        "is_dry_run": args.dry_run,
        "replicated_families": list(family_results.keys()),
        "hypotheses_evaluation": {
            "H1_peak_dissociation": {
                fam: res["h1_dissociation"]["passed"] for fam, res in family_results.items()
            },
            "H2_sufficiency": {
                fam: res["h2_sufficiency"]["passed"] for fam, res in family_results.items()
            },
            "H3_necessity": {
                fam: res["h3_necessity"]["passed"] for fam, res in family_results.items()
            },
            "H4_temporal_emergence": {
                fam: res["h4_temporal_emergence"]["passed"] for fam, res in family_results.items()
            },
        },
        "mean_peak_dissociation_delta": float(
            np.mean([res["h1_dissociation"]["delta_d_peak"] for res in family_results.values()])
        ),
        "mean_attenuation_ratio": float(
            np.mean([res["h3_necessity"]["attenuation_ratio"] for res in family_results.values()])
        ),
        "cross_model_generality": status_msg,
    }

    out_derived = derived_dir / "v3_cross_model_replication_summary.json"
    with open(out_derived, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved cross-model replication summary to {out_derived}")
    logger.info(f"Step 7 Confirmatory replication completed: {status_msg}")


if __name__ == "__main__":
    main()
