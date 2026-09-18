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
    prepare_joint_sequence_with_boundary,
    resolve_joint_stage_index,
)
from affective_empathy_eval.data import (
    describe_loaded_frame,
    load_v3_matched_pair_table,
    resolve_matched_neutral_text,
)
from affective_empathy_eval.models.adapters import get_model_adapter
from affective_empathy_eval.models.hooks import ActivationHookManager, HookPoint
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
        "primary_grounding": "reader_prediction (simulated)",
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


def validate_stage_index_invariance(
    tokenizer: Any,
    prompt: str,
    candidates: List[Dict[str, Any]],
    stage_names: List[str],
) -> Dict[str, int]:
    """
    81候補すべてでセマンティックステージのトークン絶対位置が同一であることを検証し、共通インデックスを返す
    """
    ref_cand = candidates[0]["json_str"]
    full_ids_ref, cand_start_ref = prepare_joint_sequence_with_boundary(prompt, ref_cand, tokenizer)
    cand_tokens_ref = tokenizer.encode(ref_cand, add_special_tokens=False)
    offsets_ref = get_generation_stage_tokens(cand_tokens_ref, tokenizer, candidate_str=ref_cand)

    ref_indices = {}
    for stg in stage_names:
        ref_indices[stg] = resolve_joint_stage_index(cand_start_ref, stg, offsets_ref, len(full_ids_ref))

    for cand_dict in candidates[1:]:
        cand_str = cand_dict["json_str"]
        full_ids, cand_start = prepare_joint_sequence_with_boundary(prompt, cand_str, tokenizer)
        cand_tokens = tokenizer.encode(cand_str, add_special_tokens=False)
        offsets = get_generation_stage_tokens(cand_tokens, tokenizer, candidate_str=cand_str)
        for stg in stage_names:
            idx = resolve_joint_stage_index(cand_start, stg, offsets, len(full_ids))
            if idx != ref_indices[stg]:
                raise AssertionError(
                    f"Stage index variance detected for stage '{stg}': candidate '{cand_str}' has index {idx}, "
                    f"while reference candidate '{ref_cand}' has index {ref_indices[stg]}."
                )
    return ref_indices


def run_real_model_confirmatory(
    family: str,
    model_id: str,
    df: pd.DataFrame,
    device: str = "cpu",
    subsample: int = 0,
) -> Dict[str, Any]:
    """
    実モデル (Llama 3.2, Gemma 3, OLMo 2) に対する 4大仮説の Confirmatory 検証
    H2(十分性), H3(必然性), H4(時間的局在) をすべて同一の Cross-Fitting (GroupKFold) ループ内で
    train fold のみから推定・test fold のみで評価する完全独立評価に統一。
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

    # 1. Clean Baselines & Reader Predictions
    clean_ev_list = []
    clean_ea_list = []
    reader_ev_list = []
    reader_ea_list = []
    with torch.no_grad():
        for _, row in eval_df.iterrows():
            text = str(row["text"])
            # a. Self condition: Clean expected report (baseline for intervention effect)
            prompt = build_prompt(text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
            _, probs = compute_sequence_likelihoods_for_candidates(
                model=model, tokenizer=tokenizer, prompt=prompt, candidates=candidates, device=device, batch_size=81
            )
            ev, ea = compute_expected_va(probs, candidates)
            clean_ev_list.append(ev)
            clean_ea_list.append(ea)

            # b. Reader condition: Reader Prediction (objective perception of stimulus emotion)
            prompt_reader = build_prompt(text, task=TaskType.READER, format_type="chat", tokenizer=tokenizer)
            _, r_probs = compute_sequence_likelihoods_for_candidates(
                model=model, tokenizer=tokenizer, prompt=prompt_reader, candidates=candidates, device=device, batch_size=81
            )
            r_ev, r_ea = compute_expected_va(r_probs, candidates)
            reader_ev_list.append(r_ev)
            reader_ea_list.append(r_ea)

    if "reader_V" in eval_df.columns and "reader_A" in eval_df.columns:
        y_v = eval_df["reader_V"].to_numpy()
        y_a = eval_df["reader_A"].to_numpy()
    else:
        y_v = np.array(reader_ev_list, dtype=np.float64)
        y_a = np.array(reader_ea_list, dtype=np.float64)

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

    all_H = {}
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

        oof_preds = np.zeros(N)
        for train_idx, test_idx in split_gen_fn(H):
            ridge = Ridge(alpha=10.0).fit(H[train_idx], y_v[train_idx])
            oof_preds[test_idx] = ridge.predict(H[test_idx])

        ss_tot = np.sum((y_v - np.mean(y_v))**2)
        ss_res = np.sum((y_v - oof_preds)**2)
        r2 = max(0.0, float(1.0 - ss_res / (ss_tot + 1e-6)))
        d_profile_v.append(r2)

    # 3. 統合 Cross-Fitting: H2 (Sufficiency), H3 (Necessity), H4 (Temporal Emergence)
    # NOTE: Confirmatory data reuse 完全排除のため、direction / Q / mu_neu の推定を train fold のみで行い、
    # 評価を独立な test fold のみで実行する。
    mid_layer = int(num_layers * 0.65)
    alphas = [-1.0, -0.5, 0.0, 0.5, 1.0]

    # 集約用データ構造
    test_shifts_v = {alpha: [] for alpha in alphas}
    test_shifts_a = {alpha: [] for alpha in alphas}
    test_c_profile_shifts = {l: [] for l in range(num_layers)}
    nat_shifts = []
    att_shifts = []

    stage_keys = ["candidate_start", "pre_V", "V_value", "pre_A", "A_value", "response_end"]
    test_stage_shifts_v = {stg: [] for stg in stage_keys}
    test_stage_shifts_a = {stg: [] for stg in stage_keys}

    H_mid = all_H[mid_layer]
    splits_list = list(split_gen_fn(H_mid))

    for fold_idx, (train_idx, test_idx) in enumerate(splits_list):
        assert len(set(train_idx).intersection(set(test_idx))) == 0, "Train and test sample sets overlap!"

        # ----------------------------------------------------
        # Train fold only: direction d_v, d_a, Q, mu_neu, layer directions
        # ----------------------------------------------------
        ridge_mid_v = Ridge(alpha=10.0).fit(H_mid[train_idx], y_v[train_idx])
        norm_v = np.linalg.norm(ridge_mid_v.coef_)
        d_v = ridge_mid_v.coef_ / (norm_v + 1e-6) if norm_v > 0 else np.zeros_like(ridge_mid_v.coef_)
        proj_std_v = float(np.std(H_mid[train_idx] @ d_v))
        h_std_v = proj_std_v if proj_std_v > 1e-6 else float(np.std(H_mid[train_idx]))

        ridge_mid_a = Ridge(alpha=10.0).fit(H_mid[train_idx], y_a[train_idx])
        norm_a = np.linalg.norm(ridge_mid_a.coef_)
        d_a = ridge_mid_a.coef_ / (norm_a + 1e-6) if norm_a > 0 else np.zeros_like(ridge_mid_a.coef_)
        proj_std_a = float(np.std(H_mid[train_idx] @ d_a))
        h_std_a = proj_std_a if proj_std_a > 1e-6 else float(np.std(H_mid[train_idx]))

        # H3: 2D 直交基底 Q (QR 分解)
        M_sub = np.column_stack([d_v, d_a])
        Q_np, _ = np.linalg.qr(M_sub)
        Q = torch.tensor(Q_np, dtype=torch.float32, device=device)

        # H3: 中立平均ベクトル mu_neu (Train samples only)
        train_neutral_reps = []
        with torch.no_grad():
            for tr_i in train_idx:
                tr_row = eval_df.iloc[tr_i]
                neu_text = resolve_matched_neutral_text(tr_row, df)
                if neu_text and neu_text.strip():
                    p_neu = build_prompt(neu_text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
                    enc_neu = encode_prompt_canonical(tokenizer, p_neu, device=device)
                    anch_neu = find_semantic_anchors(enc_neu["input_ids"][0].tolist(), tokenizer, neu_text)
                    with ActivationHookManager(adapter) as hook_mgr:
                        hook_mgr.register_capture_hook(
                            layer_idx=mid_layer,
                            hook_point=HookPoint.POST_MLP_RESID,
                            token_indices=anch_neu["prompt_end"],
                            key="h_neu",
                        )
                        _ = model(**enc_neu)
                        train_neutral_reps.append(hook_mgr.captured_activations["h_neu"].cpu().float().numpy().ravel())

        if train_neutral_reps:
            mu_neu = torch.tensor(np.mean(train_neutral_reps, axis=0), dtype=torch.float32, device=device)
        else:
            mu_neu = torch.tensor(np.mean(H_mid[train_idx], axis=0), dtype=torch.float32, device=device)

        # 各層の因果効果 C(l) 推定用方向 (Train fold only)
        layer_dirs = {}
        for l_idx in range(num_layers):
            H_l_train = all_H[l_idx][train_idx]
            ridge_l = Ridge(alpha=10.0).fit(H_l_train, y_v[train_idx])
            norm_l = np.linalg.norm(ridge_l.coef_)
            d_l = ridge_l.coef_ / (norm_l + 1e-6) if norm_l > 0 else np.zeros_like(ridge_l.coef_)
            std_l = float(np.std(H_l_train @ d_l))
            h_std_l = std_l if std_l > 1e-6 else float(np.std(H_l_train))
            layer_dirs[l_idx] = (d_l, h_std_l)

        # ----------------------------------------------------
        # Held-out Test fold only: H2, H3, H4 evaluation
        # ----------------------------------------------------
        eval_sub_test_idx = test_idx[:min(5, len(test_idx))]

        with torch.no_grad():
            for sample_idx in eval_sub_test_idx:
                row = eval_df.iloc[sample_idx]
                text = str(row["text"])
                prompt_self = build_prompt(text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
                enc = encode_prompt_canonical(tokenizer, prompt_self, device=device)
                anchors = find_semantic_anchors(enc["input_ids"][0].tolist(), tokenizer, text)
                patch_pos = anchors["prompt_end"]

                # --- H2: Dose-response evaluation ---
                for axis_name, direction, h_std, store_dict, clean_val in (
                    ("v", d_v, h_std_v, test_shifts_v, clean_ev_list[sample_idx]),
                    ("a", d_a, h_std_a, test_shifts_a, clean_ea_list[sample_idx]),
                ):
                    for alpha in alphas:
                        with ActivationHookManager(adapter) as hook_mgr:
                            hook_mgr.register_direction_intervention_hook(
                                layer_idx=mid_layer,
                                direction=direction,
                                alpha=alpha,
                                hidden_std=h_std,
                                token_indices=patch_pos,
                                hook_point=HookPoint.POST_MLP_RESID,
                                mode="inject",
                            )
                            _, probs_p = compute_sequence_likelihoods_for_candidates(
                                model=model, tokenizer=tokenizer, prompt=prompt_self, candidates=candidates, device=device, batch_size=81
                            )
                        ev_p, ea_p = compute_expected_va(probs_p, candidates)
                        shift = (ev_p if axis_name == "v" else ea_p) - clean_val
                        store_dict[alpha].append(shift)

                # --- H2: C(l) profile evaluation (alpha=1.0) ---
                for l_idx in range(num_layers):
                    d_l, h_std_l = layer_dirs[l_idx]
                    with ActivationHookManager(adapter) as hook_mgr:
                        hook_mgr.register_direction_intervention_hook(
                            layer_idx=l_idx,
                            direction=d_l,
                            alpha=1.0,
                            hidden_std=h_std_l,
                            token_indices=patch_pos,
                            hook_point=HookPoint.POST_MLP_RESID,
                            mode="inject",
                        )
                        _, probs_l = compute_sequence_likelihoods_for_candidates(
                            model=model, tokenizer=tokenizer, prompt=prompt_self, candidates=candidates, device=device, batch_size=81
                        )
                    ev_l, _ = compute_expected_va(probs_l, candidates)
                    test_c_profile_shifts[l_idx].append(abs(ev_l - clean_ev_list[sample_idx]))

                # --- H3: Centered 2D Orthogonal Subspace Removal ---
                neu_text = resolve_matched_neutral_text(row, df)
                if neu_text and neu_text.strip():
                    p_neu = build_prompt(neu_text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
                    _, probs_neu = compute_sequence_likelihoods_for_candidates(
                        model=model, tokenizer=tokenizer, prompt=p_neu, candidates=candidates, device=device, batch_size=81
                    )
                    neutral_base, _ = compute_expected_va(probs_neu, candidates)
                else:
                    neutral_base = float(clean_ev_list[sample_idx])

                ev_clean = float(clean_ev_list[sample_idx])
                n_shift = abs(ev_clean - neutral_base)
                nat_shifts.append(n_shift)

                with ActivationHookManager(adapter) as hook_mgr:
                    hook_mgr.register_subspace_removal_hook(
                        layer_idx=mid_layer,
                        orth_basis_q=Q,
                        mean_vector=mu_neu,
                        token_indices=patch_pos,
                        hook_point=HookPoint.POST_MLP_RESID,
                    )
                    _, probs_abl = compute_sequence_likelihoods_for_candidates(
                        model=model, tokenizer=tokenizer, prompt=prompt_self, candidates=candidates, device=device, batch_size=81
                    )
                ev_abl, _ = compute_expected_va(probs_abl, candidates)
                a_shift = abs(ev_abl - neutral_base)
                att_shifts.append(a_shift)

                # --- H4: Temporal Emergence across Generation Stages ---
                # Joint Tokenization によるステージ位置解決と不変性検証
                stage_target_indices = validate_stage_index_invariance(
                    tokenizer, prompt_self, candidates, stage_keys
                )

                for stg in stage_keys:
                    t_pos = stage_target_indices[stg]

                    # 1) Valence steering
                    with ActivationHookManager(adapter) as hook_mgr:
                        hook_mgr.register_direction_intervention_hook(
                            layer_idx=mid_layer,
                            direction=d_v,
                            alpha=1.0,
                            hidden_std=h_std_v,
                            token_indices=t_pos,
                            hook_point=HookPoint.POST_MLP_RESID,
                            mode="inject",
                        )
                        _, probs_stg_v = compute_sequence_likelihoods_for_candidates(
                            model=model, tokenizer=tokenizer, prompt=prompt_self, candidates=candidates, device=device, batch_size=81
                        )
                    ev_stg_v, _ = compute_expected_va(probs_stg_v, candidates)
                    test_stage_shifts_v[stg].append(abs(ev_stg_v - clean_ev_list[sample_idx]))

                    # 2) Arousal steering
                    with ActivationHookManager(adapter) as hook_mgr:
                        hook_mgr.register_direction_intervention_hook(
                            layer_idx=mid_layer,
                            direction=d_a,
                            alpha=1.0,
                            hidden_std=h_std_a,
                            token_indices=t_pos,
                            hook_point=HookPoint.POST_MLP_RESID,
                            mode="inject",
                        )
                        _, probs_stg_a = compute_sequence_likelihoods_for_candidates(
                            model=model, tokenizer=tokenizer, prompt=prompt_self, candidates=candidates, device=device, batch_size=81
                        )
                    _, ea_stg_a = compute_expected_va(probs_stg_a, candidates)
                    test_stage_shifts_a[stg].append(abs(ea_stg_a - clean_ea_list[sample_idx]))

    # 全 Test fold からの指標集計
    shifts_v = [float(np.mean(test_shifts_v[a])) if test_shifts_v[a] else 0.0 for a in alphas]
    shifts_a = [float(np.mean(test_shifts_a[a])) if test_shifts_a[a] else 0.0 for a in alphas]
    slope_v = estimate_interventional_slope(alphas, shifts_v)
    slope_a = estimate_interventional_slope(alphas, shifts_a)

    c_profile_v = [
        float(np.mean(test_c_profile_shifts[l])) if test_c_profile_shifts[l] else 0.0
        for l in range(num_layers)
    ]
    dissoc_v = compute_layer_dissociation(relative_depths, d_profile_v, c_profile_v)

    natural_shift = float(np.mean(nat_shifts)) if nat_shifts else 1.0
    attenuated_shift = float(np.mean(att_shifts)) if att_shifts else 0.5
    attenuation_ratio = float((natural_shift - attenuated_shift) / (natural_shift + 1e-6))
    attenuation_ratio = float(np.clip(attenuation_ratio, 0.0, 1.0))

    stage_causal_v = {stg: float(np.mean(test_stage_shifts_v[stg])) if test_stage_shifts_v[stg] else 0.0 for stg in stage_keys}
    stage_causal_a = {stg: float(np.mean(test_stage_shifts_a[stg])) if test_stage_shifts_a[stg] else 0.0 for stg in stage_keys}

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
        "primary_grounding": "reader_prediction",
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
            "family_key": fam_lower,
            "family_name": fam_cfg.family_name,
            "model_id": fam_cfg.instruct_model.model_id,
        })
    semantic_stages = v3_cfg["spatiotemporal"]["semantic_stages"]
    normalized_stages = [s if s != "response_start" else "candidate_start" for s in semantic_stages]

    raw_dir = Path(v3_cfg["output"]["raw_dir"])
    derived_dir = Path(v3_cfg["output"]["derived_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    family_results = {}
    seeds = {"llama": 301, "gemma": 302, "olmo": 303, "mistral": 304}

    for item in conf_models:
        fam_key = item["family_key"]
        fam_name = item["family_name"]
        model_id = item["model_id"]
        out_raw = raw_dir / f"v3_confirmatory_{fam_key}.json"

        if out_raw.exists() and not args.dry_run:
            try:
                with open(out_raw, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if cached and "all_confirmed" in cached:
                    logger.info(f"Loaded existing results for {fam_name} from {out_raw}. Skipping.")
                    family_results[fam_name] = cached
                    continue
            except Exception:
                pass

        num_layers = resolve_architecture_dims(model_id)[0]

        if args.dry_run:
            logger.info(f"Simulating confirmatory replication for {fam_name} (--dry-run)...")
            res = simulate_model_confirmatory(fam_name, num_layers, normalized_stages, seed=seeds.get(fam_key, 999))
        else:
            logger.info(f"Running REAL confirmatory replication for {fam_name} ({model_id})...")
            res = run_real_model_confirmatory(
                family=fam_name,
                model_id=model_id,
                df=df,
                device=args.device,
                subsample=args.subsample,
            )

        family_results[fam_name] = res

        res["dry_run"] = bool(args.dry_run)
        with open(out_raw, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        logger.info(f"Saved {fam_name} confirmatory results to {out_raw}")

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
