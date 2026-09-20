#!/usr/bin/env python3
"""
Step 7: 他3モデル (Llama 3.2, Gemma 3, OLMo 2) における主要 V3 因果機構の Confirmatory 再現
検証対象の4大仮説:
  1. Hypothesis 1 (Dissociation): デコードピークと因果ピークの解離 (Δd_peak > 0, Δd_center > 0)
  2. Hypothesis 2 (Sufficiency): 内部情動方向 d_V, d_A への介入による単調誘導 (gamma > 0)
  3. Hypothesis 3 (Necessity): 直交化 2D 部分空間除去による自己報告変位の有意な減衰
  4. Hypothesis 4 (Temporal Emergence): Discovery で同定された frozen site (Valence: pre_V, Arousal: pre_A) での因果ピーク集中を検証
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
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
from affective_empathy_eval.manifests import (
    create_run_manifest,
    is_manifest_matching,
    compute_string_or_dict_hash,
    compute_file_hash,
    DEFAULT_CODE_VERSION,
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
    validate_stage_index_invariance,
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
    parser.add_argument("--force", action="store_true", help="Force recomputation even if valid cached results exist")
    add_model_selection_args(parser)
    return parser.parse_args()


def simulate_model_confirmatory(
    family: str,
    num_layers: int,
    semantic_stages: List[str],
    seed: int,
) -> Dict[str, Any]:
    """
    dry-run用: 特定モデルファミリーに対する Confirmatory 検証のシミュレーション (VA両軸完全対応)
    """
    rng = np.random.default_rng(seed)
    relative_depths = [l / (num_layers - 1) if num_layers > 1 else 0.0 for l in range(num_layers)]

    d_center = 0.46 if family == "Llama" else (0.50 if family == "Gemma" else 0.48)
    c_center = 0.65 if family == "Llama" else (0.72 if family == "Gemma" else 0.70)

    d_profile_v = [float(np.exp(-((d - d_center) ** 2) / (2 * 0.16**2)) * 0.70 + rng.normal(0, 0.02)) for d in relative_depths]
    c_profile_v = [float(np.exp(-((d - c_center) ** 2) / (2 * 0.14**2)) * 1.05 + rng.normal(0, 0.02)) for d in relative_depths]
    dissoc_v = compute_layer_dissociation(relative_depths, d_profile_v, c_profile_v)

    d_profile_a = [float(np.exp(-((d - d_center) ** 2) / (2 * 0.18**2)) * 0.65 + rng.normal(0, 0.02)) for d in relative_depths]
    c_profile_a = [float(np.exp(-((d - c_center) ** 2) / (2 * 0.15**2)) * 0.95 + rng.normal(0, 0.02)) for d in relative_depths]
    dissoc_a = compute_layer_dissociation(relative_depths, d_profile_a, c_profile_a)

    alphas = [-1.0, -0.5, 0.0, 0.5, 1.0]
    shifts_v = [float(a * (0.80 + rng.normal(0, 0.03))) for a in alphas]
    shifts_a = [float(a * (0.68 + rng.normal(0, 0.03))) for a in alphas]
    slope_v = estimate_interventional_slope(alphas, shifts_v)
    slope_a = estimate_interventional_slope(alphas, shifts_a)

    natural_shift_v = 1.10 + rng.normal(0, 0.05)
    attenuated_shift_v = 0.40 + rng.normal(0, 0.03)
    attenuation_ratio_v = float((natural_shift_v - attenuated_shift_v) / natural_shift_v)

    natural_shift_a = 0.95 + rng.normal(0, 0.05)
    attenuated_shift_a = 0.42 + rng.normal(0, 0.03)
    attenuation_ratio_a = float((natural_shift_a - attenuated_shift_a) / natural_shift_a)

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

    contrast_v = float(stage_causal_v["pre_V"] - stage_causal_v["candidate_start"])
    contrast_a = float(stage_causal_a["pre_A"] - stage_causal_a["candidate_start"])

    h1_pass_v = dissoc_v["delta_d_peak"] > 0 and dissoc_v["delta_d_center"] > 0
    h1_pass_a = dissoc_a["delta_d_peak"] > 0 and dissoc_a["delta_d_center"] > 0
    h1_pass = bool(h1_pass_v and h1_pass_a)

    h2_pass = bool(slope_v > 0.1 and slope_a > 0.1)
    # Primary: Absolute mediated attenuation > 0.0
    h3_pass_v = bool((natural_shift_v - attenuated_shift_v) > 0.0)
    h3_pass_a = bool((natural_shift_a - attenuated_shift_a) > 0.0)
    h3_pass = bool(h3_pass_v and h3_pass_a)

    h4_pass_v = bool(stage_causal_v["pre_V"] > stage_causal_v["candidate_start"])
    h4_pass_a = bool(stage_causal_a["pre_A"] > stage_causal_a["candidate_start"])
    h4_pass = bool(h4_pass_v and h4_pass_a)

    auxiliary_qc_all_pass = bool(h1_pass and h2_pass and h3_pass and h4_pass)

    return {
        "family": family,
        "is_simulation": True,
        "primary_grounding": "reader_prediction (simulated)",
        "num_layers": num_layers,
        "h1_dissociation": {
            "valence": dissoc_v,
            "arousal": dissoc_a,
            "d_peak_D": dissoc_v["d_peak_D"],  # 互換用
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
        "h3_endogenous_relevance": {
            "valence": {
                "natural_shift": float(natural_shift_v),
                "attenuated_shift": float(attenuated_shift_v),
                "mediated_attenuation": float(natural_shift_v - attenuated_shift_v),
                "attenuation_ratio": float(attenuation_ratio_v),
            },
            "arousal": {
                "natural_shift": float(natural_shift_a),
                "attenuated_shift": float(attenuated_shift_a),
                "mediated_attenuation": float(natural_shift_a - attenuated_shift_a),
                "attenuation_ratio": float(attenuation_ratio_a),
            },
            "natural_shift": float(natural_shift_v),  # 互換用
            "attenuated_shift": float(attenuated_shift_v),
            "mediated_attenuation": float(natural_shift_v - attenuated_shift_v),
            "attenuation_ratio": float(attenuation_ratio_v),
            "passed": bool(h3_pass),
        },
        "h3_necessity": {  # 互換用キー
            "natural_shift": float(natural_shift_v),
            "attenuated_shift": float(attenuated_shift_v),
            "attenuation_ratio": float(attenuation_ratio_v),
            "passed": bool(h3_pass),
        },
        "h4_temporal_emergence": {
            "stage_causal": stage_causal_v,
            "stage_causal_v": stage_causal_v,
            "stage_causal_a": stage_causal_a,
            "contrast_v": contrast_v,
            "contrast_a": contrast_a,
            "non_uniform_leverage": True,
            "passed_valence": bool(h4_pass_v),
            "passed_arousal": bool(h4_pass_a),
            "passed": bool(h4_pass),
        },
        "auxiliary_qc_all_pass": auxiliary_qc_all_pass,
        "all_confirmed": auxiliary_qc_all_pass,  # 後方互換性用エイリアス
    }


def run_real_model_confirmatory(
    family: str,
    model_id: str,
    df: pd.DataFrame,
    device: str = "cpu",
    subsample: int = 0,
    v3_cfg: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    実モデル (Llama 3.2, Gemma 3, OLMo 2) に対する 4大仮説の Confirmatory 検証
    H2(十分性), H3(必然性), H4(時間的局在) をすべて同一の Cross-Fitting (GroupKFold) ループ内で
    train fold のみから推定・test fold のみで評価する完全独立評価に統一。
    """
    registry = get_registry()
    fam_cfg = registry.get_family_by_model_id(model_id)
    model_spec = None
    if fam_cfg:
        for role in ("instruct", "base"):
            spec = fam_cfg.get_model_spec(role)
            if spec.model_id == model_id:
                model_spec = spec
                break
    revision = model_spec.revision if model_spec else None

    logger.info(f"Loading {family} model: {model_id} (revision={revision}) on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_cuda = device != "cpu" and torch.cuda.is_available()
    actual_torch_dtype = (
        torch.bfloat16
        if is_cuda and torch.cuda.is_bf16_supported()
        else (torch.float16 if is_cuda else torch.float32)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=actual_torch_dtype,
        device_map=device if is_cuda else None,
        trust_remote_code=True,
    )
    model.eval()

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
            log_liks, probs = compute_sequence_likelihoods_for_candidates(
                model=model, tokenizer=tokenizer, prompt=prompt, candidates=candidates, device=device, batch_size=81
            )
            ev, ea = compute_expected_va(log_liks, candidates)
            clean_ev_list.append(ev)
            clean_ea_list.append(ea)

            # b. Reader condition: Reader Prediction (objective perception of stimulus emotion)
            prompt_reader = build_prompt(text, task=TaskType.READER, format_type="chat", tokenizer=tokenizer)
            r_log_liks, r_probs = compute_sequence_likelihoods_for_candidates(
                model=model, tokenizer=tokenizer, prompt=prompt_reader, candidates=candidates, device=device, batch_size=81
            )
            r_ev, r_ea = compute_expected_va(r_log_liks, candidates)
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
    d_profile_a = []
    if "pair_id" in eval_df.columns and eval_df["pair_id"].nunique() >= 2:
        n_splits = min(5, eval_df["pair_id"].nunique())
        splitter = GroupKFold(n_splits=n_splits)
        split_gen_fn = lambda data: splitter.split(data, y_v, groups=eval_df["pair_id"].values)
    else:
        if not dry_run:
            raise ValueError(
                "Confirmatory analysis requires >=2 independent pair groups ('pair_id') for GroupKFold cross-fitting in production."
            )
        n_splits = min(5, max(2, N))
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=v3_cfg.get("seed", 42) if v3_cfg else 42)
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
        oof_preds_v = np.zeros(N)
        oof_preds_a = np.zeros(N)
        for train_idx, test_idx in split_gen_fn(H):
            ridge_v = Ridge(alpha=10.0).fit(H[train_idx], y_v[train_idx])
            oof_preds_v[test_idx] = ridge_v.predict(H[test_idx])

            ridge_a = Ridge(alpha=10.0).fit(H[train_idx], y_a[train_idx])
            oof_preds_a[test_idx] = ridge_a.predict(H[test_idx])

        ss_tot_v = np.sum((y_v - np.mean(y_v))**2)
        ss_res_v = np.sum((y_v - oof_preds_v)**2)
        r2_v = float(1.0 - ss_res_v / (ss_tot_v + 1e-6))
        d_profile_v.append(r2_v)

        ss_tot_a = np.sum((y_a - np.mean(y_a))**2)
        ss_res_a = np.sum((y_a - oof_preds_a)**2)
        r2_a = float(1.0 - ss_res_a / (ss_tot_a + 1e-6))
        d_profile_a.append(r2_a)

    assert len(d_profile_v) == num_layers, f"d_profile_v length {len(d_profile_v)} != num_layers {num_layers}"
    assert len(d_profile_a) == num_layers, f"d_profile_a length {len(d_profile_a)} != num_layers {num_layers}"

    # 3. 統合 Cross-Fitting: H2 (Sufficiency), H3 (Necessity), H4 (Temporal Emergence)
    # NOTE: Confirmatory data reuse 完全排除のため、direction / Q / mu_neu の推定を train fold のみで行い、
    # 評価を独立な test fold のみで実行する。
    base_seed = v3_cfg.get("seed", 42) if v3_cfg else 42
    frozen_sites = v3_cfg.get("frozen_sites") if v3_cfg else None
    stage_v = "pre_V"
    stage_a = "pre_A"
    if frozen_sites:
        suff_rel_depth = float(frozen_sites["sufficiency_relative_depth"])
        temp_rel_depth_v = float(frozen_sites["temporal_relative_depth_v"])
        temp_rel_depth_a = float(frozen_sites["temporal_relative_depth_a"])
        med_rel_depth = float(frozen_sites["mediation_relative_depth"])
        stage_v = str(frozen_sites.get("temporal_stage_v", "pre_V"))
        stage_a = str(frozen_sites.get("temporal_stage_a", "pre_A"))
    else:
        suff_rel_depth = float(v3_cfg.get("confirmatory", {}).get("sufficiency_relative_depth", 0.5)) if v3_cfg else 0.5
        temp_rel_depth_v = float(v3_cfg.get("confirmatory", {}).get("temporal_relative_depth_v", 0.65)) if v3_cfg else 0.65
        temp_rel_depth_a = float(v3_cfg.get("confirmatory", {}).get("temporal_relative_depth_a", 0.65)) if v3_cfg else 0.65
        med_rel_depth = float(v3_cfg.get("confirmatory", {}).get("mediation_relative_depth", 0.65)) if v3_cfg else 0.65
    sufficiency_layer = round(suff_rel_depth * (num_layers - 1))
    temporal_layer_v = round(temp_rel_depth_v * (num_layers - 1))
    temporal_layer_a = round(temp_rel_depth_a * (num_layers - 1))
    mediation_layer = round(med_rel_depth * (num_layers - 1))
    alphas = [-1.0, -0.5, 0.0, 0.5, 1.0]

    # 集約用データ構造
    test_shifts_v = {alpha: [] for alpha in alphas}
    test_shifts_a = {alpha: [] for alpha in alphas}
    test_c_profile_shifts_v = {l: [] for l in range(num_layers)}
    test_c_profile_shifts_a = {l: [] for l in range(num_layers)}
    nat_shifts_v = []
    nat_shifts_a = []
    att_shifts_v = []
    att_shifts_a = []

    stage_keys = ["candidate_start", "pre_V", "V_value", "pre_A", "A_value", "response_end"]
    test_stage_shifts_v = {stg: [] for stg in stage_keys}
    test_stage_shifts_a = {stg: [] for stg in stage_keys}

    # H4: Discovery RQ2 準拠の stage-local direction 学習のため、
    # temporal_layer_v, temporal_layer_a それぞれにおける全サンプルの各 generation stage 表現を抽出
    template_cand = candidates[40]["json_str"]  # {"valence": 5, "arousal": 5}
    cand_tokens = tokenizer.encode(template_cand, add_special_tokens=False)
    stage_offsets = get_generation_stage_tokens(cand_tokens, tokenizer, candidate_str=template_cand)

    all_H_stage_v = {stg: [] for stg in stage_keys}
    all_H_stage_a = {stg: [] for stg in stage_keys}
    with torch.no_grad():
        for _, row in eval_df.iterrows():
            text = str(row["text"])
            prompt = build_prompt(text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
            full_ids, cand_start = prepare_joint_sequence_with_boundary(
                prompt=prompt, candidate=template_cand, tokenizer=tokenizer
            )
            seq_len = len(full_ids)
            enc_full = {
                "input_ids": torch.tensor([full_ids], device=device),
                "attention_mask": torch.ones(1, seq_len, dtype=torch.long, device=device),
            }
            for stg in stage_keys:
                t_idx = resolve_joint_stage_index(cand_start, stg, stage_offsets, seq_len)
                with ActivationHookManager(adapter) as hook_mgr:
                    hook_mgr.register_capture_hook(
                        layer_idx=temporal_layer_v,
                        hook_point=HookPoint.POST_MLP_RESID,
                        token_indices=t_idx,
                        key="h_stg_v",
                    )
                    hook_mgr.register_capture_hook(
                        layer_idx=temporal_layer_a,
                        hook_point=HookPoint.POST_MLP_RESID,
                        token_indices=t_idx,
                        key="h_stg_a",
                    )
                    _ = model(**enc_full)
                    h_vec_v = hook_mgr.captured_activations["h_stg_v"].cpu().float().numpy().ravel()
                    h_vec_a = hook_mgr.captured_activations["h_stg_a"].cpu().float().numpy().ravel()
                    all_H_stage_v[stg].append(h_vec_v)
                    all_H_stage_a[stg].append(h_vec_a)
    for stg in stage_keys:
        all_H_stage_v[stg] = np.array(all_H_stage_v[stg])
        all_H_stage_a[stg] = np.array(all_H_stage_a[stg])

    H_suff = all_H[sufficiency_layer]
    H_med = all_H[mediation_layer]
    splits_list = list(split_gen_fn(H_suff))

    for fold_idx, (train_idx, test_idx) in enumerate(splits_list):
        assert len(set(train_idx).intersection(set(test_idx))) == 0, "Train and test sample sets overlap!"

        # ----------------------------------------------------
        # Train fold only: H2 direction d_v, d_a (at sufficiency_layer), H3 Q, mu_neu (at mediation_layer)
        # ----------------------------------------------------
        ridge_suff_v = Ridge(alpha=10.0).fit(H_suff[train_idx], y_v[train_idx])
        norm_v = np.linalg.norm(ridge_suff_v.coef_)
        d_v = ridge_suff_v.coef_ / (norm_v + 1e-6) if norm_v > 0 else np.zeros_like(ridge_suff_v.coef_)
        proj_std_v = float(np.std(H_suff[train_idx] @ d_v))
        h_std_v = proj_std_v if proj_std_v > 1e-6 else float(np.std(H_suff[train_idx]))

        ridge_suff_a = Ridge(alpha=10.0).fit(H_suff[train_idx], y_a[train_idx])
        norm_a = np.linalg.norm(ridge_suff_a.coef_)
        d_a = ridge_suff_a.coef_ / (norm_a + 1e-6) if norm_a > 0 else np.zeros_like(ridge_suff_a.coef_)
        proj_std_a = float(np.std(H_suff[train_idx] @ d_a))
        h_std_a = proj_std_a if proj_std_a > 1e-6 else float(np.std(H_suff[train_idx]))

        # H3: mediation_layer における 2D 直交基底 Q
        ridge_med_v = Ridge(alpha=10.0).fit(H_med[train_idx], y_v[train_idx])
        norm_mv = np.linalg.norm(ridge_med_v.coef_)
        d_med_v = ridge_med_v.coef_ / (norm_mv + 1e-6) if norm_mv > 0 else np.zeros_like(ridge_med_v.coef_)

        ridge_med_a = Ridge(alpha=10.0).fit(H_med[train_idx], y_a[train_idx])
        norm_ma = np.linalg.norm(ridge_med_a.coef_)
        d_med_a = ridge_med_a.coef_ / (norm_ma + 1e-6) if norm_ma > 0 else np.zeros_like(ridge_med_a.coef_)

        Q_sub, _ = compute_orthonormal_subspace(d_med_v, d_med_a)
        Q = torch.tensor(Q_sub, dtype=torch.float32, device=device)

        # H3: 中立平均ベクトル mu_neu (mediation_layer, Train samples only)
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
                            layer_idx=mediation_layer,
                            hook_point=HookPoint.POST_MLP_RESID,
                            token_indices=anch_neu["prompt_end"],
                            key="h_neu",
                        )
                        _ = model(**enc_neu)
                        train_neutral_reps.append(hook_mgr.captured_activations["h_neu"].cpu().float().numpy().ravel())

        if not train_neutral_reps:
            raise ValueError(f"No matched-neutral representations available for training fold in {family}")
        mu_neu = torch.tensor(np.mean(train_neutral_reps, axis=0), dtype=torch.float32, device=device)

        # 各層の因果効果 C(l) 推定用方向 (Train fold only: VA両軸)
        layer_dirs_v = {}
        layer_dirs_a = {}
        for l_idx in range(num_layers):
            H_l_train = all_H[l_idx][train_idx]
            ridge_l_v = Ridge(alpha=10.0).fit(H_l_train, y_v[train_idx])
            norm_l_v = np.linalg.norm(ridge_l_v.coef_)
            d_l_v = ridge_l_v.coef_ / (norm_l_v + 1e-6) if norm_l_v > 0 else np.zeros_like(ridge_l_v.coef_)
            std_l_v = float(np.std(H_l_train @ d_l_v))
            h_std_l_v = std_l_v if std_l_v > 1e-6 else float(np.std(H_l_train))
            layer_dirs_v[l_idx] = (d_l_v, h_std_l_v)

            ridge_l_a = Ridge(alpha=10.0).fit(H_l_train, y_a[train_idx])
            norm_l_a = np.linalg.norm(ridge_l_a.coef_)
            d_l_a = ridge_l_a.coef_ / (norm_l_a + 1e-6) if norm_l_a > 0 else np.zeros_like(ridge_l_a.coef_)
            std_l_a = float(np.std(H_l_train @ d_l_a))
            h_std_l_a = std_l_a if std_l_a > 1e-6 else float(np.std(H_l_train))
            layer_dirs_a[l_idx] = (d_l_a, h_std_l_a)

        # H4: Discovery RQ2 準拠の局所方向推定 (Train fold only: 各 stage 自身の表現から学習)
        stage_dirs_v = {}
        stage_dirs_a = {}
        for stg in stage_keys:
            H_stg_tr_v = all_H_stage_v[stg][train_idx]
            ridge_stg_v = Ridge(alpha=10.0).fit(H_stg_tr_v, y_v[train_idx])
            norm_sv = np.linalg.norm(ridge_stg_v.coef_)
            d_sv = ridge_stg_v.coef_ / (norm_sv + 1e-6) if norm_sv > 0 else np.zeros_like(ridge_stg_v.coef_)
            std_sv = float(np.std(H_stg_tr_v @ d_sv))
            h_std_sv = std_sv if std_sv > 1e-6 else float(np.std(H_stg_tr_v))
            stage_dirs_v[stg] = (d_sv, h_std_sv)

            H_stg_tr_a = all_H_stage_a[stg][train_idx]
            ridge_stg_a = Ridge(alpha=10.0).fit(H_stg_tr_a, y_a[train_idx])
            norm_sa = np.linalg.norm(ridge_stg_a.coef_)
            d_sa = ridge_stg_a.coef_ / (norm_sa + 1e-6) if norm_sa > 0 else np.zeros_like(ridge_stg_a.coef_)
            std_sa = float(np.std(H_stg_tr_a @ d_sa))
            h_std_sa = std_sa if std_sa > 1e-6 else float(np.std(H_stg_tr_a))
            stage_dirs_a[stg] = (d_sa, h_std_sa)

        # ----------------------------------------------------
        # Held-out Test fold only: H2, H3, H4 evaluation
        # 指示18: 乱数シードに基づく再現可能なサブサンプリング (pair順バイアス排除)
        # ----------------------------------------------------
        rng_fold = np.random.default_rng(base_seed + fold_idx)
        n_per_fold = min(5, len(test_idx))
        eval_sub_test_idx = rng_fold.choice(test_idx, size=n_per_fold, replace=False)

        with torch.no_grad():
            for sample_idx in eval_sub_test_idx:
                row = eval_df.iloc[sample_idx]
                text = str(row["text"])
                prompt_self = build_prompt(text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
                enc = encode_prompt_canonical(tokenizer, prompt_self, device=device)
                anchors = find_semantic_anchors(enc["input_ids"][0].tolist(), tokenizer, text)
                patch_pos = anchors["prompt_end"]

                # --- H2: Dose-response evaluation (Neutral injection at sufficiency_layer) ---
                neu_text = resolve_matched_neutral_text(row, df)
                if not neu_text or not neu_text.strip():
                    raise ValueError(f"Missing matched-neutral for pair_id={row.get('pair_id', 'unknown')}")
                prompt_neu_self = build_prompt(neu_text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
                enc_neu = encode_prompt_canonical(tokenizer, prompt_neu_self, device=device)
                anchors_neu = find_semantic_anchors(enc_neu["input_ids"][0].tolist(), tokenizer, neu_text)
                patch_pos_neu = anchors_neu["prompt_end"]

                log_neu, probs_neu = compute_sequence_likelihoods_for_candidates(
                    model=model, tokenizer=tokenizer, prompt=prompt_neu_self, candidates=candidates, device=device, batch_size=81
                )
                clean_neu_v, clean_neu_a = compute_expected_va(log_neu, candidates)

                for axis_name, direction, h_std, store_dict, clean_neu_val in (
                    ("v", d_v, h_std_v, test_shifts_v, clean_neu_v),
                    ("a", d_a, h_std_a, test_shifts_a, clean_neu_a),
                ):
                    for alpha in alphas:
                        with ActivationHookManager(adapter) as hook_mgr:
                            hook_mgr.register_direction_intervention_hook(
                                layer_idx=sufficiency_layer,
                                direction=direction,
                                alpha=alpha,
                                hidden_std=h_std,
                                token_indices=patch_pos_neu,
                                hook_point=HookPoint.POST_MLP_RESID,
                                mode="inject",
                            )
                            log_p, probs_p = compute_sequence_likelihoods_for_candidates(
                                model=model, tokenizer=tokenizer, prompt=prompt_neu_self, candidates=candidates, device=device, batch_size=81
                            )
                        ev_p, ea_p = compute_expected_va(log_p, candidates)
                        shift = (ev_p if axis_name == "v" else ea_p) - clean_neu_val
                        store_dict[alpha].append(shift)

                # --- H1: C(l) profile evaluation (alpha=1.0, VA両軸) ---
                for l_idx in range(num_layers):
                    d_l_v, h_std_l_v = layer_dirs_v[l_idx]
                    with ActivationHookManager(adapter) as hook_mgr:
                        hook_mgr.register_direction_intervention_hook(
                            layer_idx=l_idx,
                            direction=d_l_v,
                            alpha=1.0,
                            hidden_std=h_std_l_v,
                            token_indices=patch_pos,
                            hook_point=HookPoint.POST_MLP_RESID,
                            mode="inject",
                        )
                        log_l_v, probs_l_v = compute_sequence_likelihoods_for_candidates(
                            model=model, tokenizer=tokenizer, prompt=prompt_self, candidates=candidates, device=device, batch_size=81
                        )
                    ev_l, _ = compute_expected_va(log_l_v, candidates)
                    test_c_profile_shifts_v[l_idx].append(abs(ev_l - clean_ev_list[sample_idx]))

                    d_l_a, h_std_l_a = layer_dirs_a[l_idx]
                    with ActivationHookManager(adapter) as hook_mgr:
                        hook_mgr.register_direction_intervention_hook(
                            layer_idx=l_idx,
                            direction=d_l_a,
                            alpha=1.0,
                            hidden_std=h_std_l_a,
                            token_indices=patch_pos,
                            hook_point=HookPoint.POST_MLP_RESID,
                            mode="inject",
                        )
                        log_l_a, probs_l_a = compute_sequence_likelihoods_for_candidates(
                            model=model, tokenizer=tokenizer, prompt=prompt_self, candidates=candidates, device=device, batch_size=81
                        )
                    _, ea_l = compute_expected_va(log_l_a, candidates)
                    test_c_profile_shifts_a[l_idx].append(abs(ea_l - clean_ea_list[sample_idx]))

                # --- H3: Centered 2D Orthogonal Subspace Removal (Affective stimulus at mediation_layer) ---
                ev_clean = float(clean_ev_list[sample_idx])
                ea_clean = float(clean_ea_list[sample_idx])
                nat_shifts_v.append(abs(ev_clean - clean_neu_v))
                nat_shifts_a.append(abs(ea_clean - clean_neu_a))

                with ActivationHookManager(adapter) as hook_mgr:
                    hook_mgr.register_subspace_removal_hook(
                        layer_idx=mediation_layer,
                        orth_basis_q=Q,
                        mean_vector=mu_neu,
                        token_indices=patch_pos,
                        hook_point=HookPoint.POST_MLP_RESID,
                    )
                    log_abl, probs_abl = compute_sequence_likelihoods_for_candidates(
                        model=model, tokenizer=tokenizer, prompt=prompt_self, candidates=candidates, device=device, batch_size=81
                    )
                ev_abl, ea_abl = compute_expected_va(log_abl, candidates)
                att_shifts_v.append(abs(ev_abl - clean_neu_v))
                att_shifts_a.append(abs(ea_abl - clean_neu_a))

                # --- H4: Temporal Emergence across Generation Stages (at temporal_layer_v and temporal_layer_a) ---
                # Discovery RQ2 準拠: 各 generation stage 固有の表現から推定した局所方向を用いて介入
                stage_target_indices = validate_stage_index_invariance(
                    tokenizer, prompt_self, candidates, stage_keys
                )

                for stg in stage_keys:
                    t_pos = stage_target_indices[stg]
                    d_stg_v, h_std_stg_v = stage_dirs_v[stg]
                    d_stg_a, h_std_stg_a = stage_dirs_a[stg]

                    # 1) Valence steering (stage-local direction at temporal_layer_v)
                    with ActivationHookManager(adapter) as hook_mgr:
                        hook_mgr.register_direction_intervention_hook(
                            layer_idx=temporal_layer_v,
                            direction=d_stg_v,
                            alpha=1.0,
                            hidden_std=h_std_stg_v,
                            token_indices=t_pos,
                            hook_point=HookPoint.POST_MLP_RESID,
                            mode="inject",
                        )
                        log_stg_v, probs_stg_v = compute_sequence_likelihoods_for_candidates(
                            model=model, tokenizer=tokenizer, prompt=prompt_self, candidates=candidates, device=device, batch_size=81
                        )
                    ev_stg_v, _ = compute_expected_va(log_stg_v, candidates)
                    test_stage_shifts_v[stg].append(abs(ev_stg_v - clean_ev_list[sample_idx]))

                    # 2) Arousal steering (stage-local direction at temporal_layer_a)
                    with ActivationHookManager(adapter) as hook_mgr:
                        hook_mgr.register_direction_intervention_hook(
                            layer_idx=temporal_layer_a,
                            direction=d_stg_a,
                            alpha=1.0,
                            hidden_std=h_std_stg_a,
                            token_indices=t_pos,
                            hook_point=HookPoint.POST_MLP_RESID,
                            mode="inject",
                        )
                        log_stg_a, probs_stg_a = compute_sequence_likelihoods_for_candidates(
                            model=model, tokenizer=tokenizer, prompt=prompt_self, candidates=candidates, device=device, batch_size=81
                        )
                    _, ea_stg_a = compute_expected_va(log_stg_a, candidates)
                    test_stage_shifts_a[stg].append(abs(ea_stg_a - clean_ea_list[sample_idx]))

    # 全 Test fold からの指標集計
    shifts_v = [float(np.mean(test_shifts_v[a])) if test_shifts_v[a] else 0.0 for a in alphas]
    shifts_a = [float(np.mean(test_shifts_a[a])) if test_shifts_a[a] else 0.0 for a in alphas]
    slope_v = estimate_interventional_slope(alphas, shifts_v)
    slope_a = estimate_interventional_slope(alphas, shifts_a)

    c_profile_v = [
        float(np.mean(test_c_profile_shifts_v[l])) if test_c_profile_shifts_v[l] else 0.0
        for l in range(num_layers)
    ]
    c_profile_a = [
        float(np.mean(test_c_profile_shifts_a[l])) if test_c_profile_shifts_a[l] else 0.0
        for l in range(num_layers)
    ]
    dissoc_v = compute_layer_dissociation(relative_depths, d_profile_v, c_profile_v)
    dissoc_a = compute_layer_dissociation(relative_depths, d_profile_a, c_profile_a)

    natural_shift_v = float(np.mean(nat_shifts_v)) if nat_shifts_v else 1.0
    attenuated_shift_v = float(np.mean(att_shifts_v)) if att_shifts_v else 0.5
    mediated_attenuation_v = natural_shift_v - attenuated_shift_v

    natural_shift_a = float(np.mean(nat_shifts_a)) if nat_shifts_a else 1.0
    attenuated_shift_a = float(np.mean(att_shifts_a)) if att_shifts_a else 0.5
    mediated_attenuation_a = natural_shift_a - attenuated_shift_a

    # Primary: Absolute mediated attenuation samples & CI
    sample_atten_v = [n - a for n, a in zip(nat_shifts_v, att_shifts_v)]
    sample_atten_a = [n - a for n, a in zip(nat_shifts_a, att_shifts_a)]
    pt_atten_v, atten_v_low, atten_v_high = (
        compute_bootstrap_ci(sample_atten_v)
        if len(sample_atten_v) >= 2
        else (mediated_attenuation_v, mediated_attenuation_v, mediated_attenuation_v)
    )
    pt_atten_a, atten_a_low, atten_a_high = (
        compute_bootstrap_ci(sample_atten_a)
        if len(sample_atten_a) >= 2
        else (mediated_attenuation_a, mediated_attenuation_a, mediated_attenuation_a)
    )

    # Secondary: Attenuation ratio (filtered to natural_shift > MIN_NATURAL_SHIFT=0.05 to avoid division by near-zero)
    MIN_NATURAL_SHIFT = 0.05
    valid_ratios_v = [(n - a) / n for n, a in zip(nat_shifts_v, att_shifts_v) if n > MIN_NATURAL_SHIFT]
    valid_ratios_a = [(n - a) / n for n, a in zip(nat_shifts_a, att_shifts_a) if n > MIN_NATURAL_SHIFT]

    pt_att_v, att_v_low, att_v_high = (
        compute_bootstrap_ci(valid_ratios_v)
        if len(valid_ratios_v) >= 2
        else (float(np.mean(valid_ratios_v)) if valid_ratios_v else np.nan, np.nan, np.nan)
    )
    pt_att_a, att_a_low, att_a_high = (
        compute_bootstrap_ci(valid_ratios_a)
        if len(valid_ratios_a) >= 2
        else (float(np.mean(valid_ratios_a)) if valid_ratios_a else np.nan, np.nan, np.nan)
    )
    attenuation_ratio_v = float(pt_att_v) if not np.isnan(pt_att_v) else np.nan
    attenuation_ratio_a = float(pt_att_a) if not np.isnan(pt_att_a) else np.nan

    stage_causal_v = {stg: float(np.mean(test_stage_shifts_v[stg])) if test_stage_shifts_v[stg] else 0.0 for stg in stage_keys}
    stage_causal_a = {stg: float(np.mean(test_stage_shifts_a[stg])) if test_stage_shifts_a[stg] else 0.0 for stg in stage_keys}

    contrast_v = float(stage_causal_v.get(stage_v, 0.0) - stage_causal_v.get("candidate_start", 0.0))
    contrast_a = float(stage_causal_a.get(stage_a, 0.0) - stage_causal_a.get("candidate_start", 0.0))

    qc_cfg = v3_cfg.get("confirmatory", {}).get("qc", {})
    min_slope = float(qc_cfg.get("min_sufficiency_slope", 0.1))
    min_atten_ci_low = float(qc_cfg.get("min_mediated_attenuation_ci_lower", 0.0))
    min_contrast = float(qc_cfg.get("min_temporal_contrast", 0.0))

    # 判定は補助QCとし、効果量とCIを主出力とする
    h1_pass_v = dissoc_v["delta_d_peak"] > 0 and dissoc_v["delta_d_center"] > 0
    h1_pass_a = dissoc_a["delta_d_peak"] > 0 and dissoc_a["delta_d_center"] > 0
    h1_pass = bool(h1_pass_v and h1_pass_a)

    h2_pass = bool(slope_v > min_slope and slope_a > min_slope)
    # Primary: Absolute mediated attenuation CI lower > min_atten_ci_low
    h3_pass_v = bool(atten_v_low > min_atten_ci_low)
    h3_pass_a = bool(atten_a_low > min_atten_ci_low)
    h3_pass = bool(h3_pass_v and h3_pass_a)

    # Temporal emergence = decodability != uniform causal leverage
    h4_pass_v = bool(contrast_v > min_contrast)
    h4_pass_a = bool(contrast_a > min_contrast)
    h4_pass = bool(h4_pass_v and h4_pass_a)

    auxiliary_qc_all_pass = bool(h1_pass and h2_pass and h3_pass and h4_pass)

    return {
        "family": family,
        "is_simulation": False,
        "primary_grounding": "reader_prediction",
        "model_id": model_id,
        "num_layers": num_layers,
        "h1_dissociation": {
            "valence": dissoc_v,
            "arousal": dissoc_a,
            "d_peak_D": dissoc_v["d_peak_D"],  # 互換用
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
        "h3_endogenous_relevance": {
            "valence": {
                "natural_shift": float(natural_shift_v),
                "attenuated_shift": float(attenuated_shift_v),
                # Primary: Absolute mediated attenuation
                "mediated_attenuation": float(mediated_attenuation_v),
                "mediated_attenuation_ci": [float(atten_v_low), float(atten_v_high)],
                # Secondary: Attenuation ratio
                "attenuation_ratio": float(attenuation_ratio_v),
                "attenuation_ratio_ci": [float(att_v_low), float(att_v_high)],
                "attenuation_ratio_display": float(np.clip(attenuation_ratio_v, 0.0, 1.0)),
                "n_valid_ratio": len(valid_ratios_v),
            },
            "arousal": {
                "natural_shift": float(natural_shift_a),
                "attenuated_shift": float(attenuated_shift_a),
                # Primary: Absolute mediated attenuation
                "mediated_attenuation": float(mediated_attenuation_a),
                "mediated_attenuation_ci": [float(atten_a_low), float(atten_a_high)],
                # Secondary: Attenuation ratio
                "attenuation_ratio": float(attenuation_ratio_a),
                "attenuation_ratio_ci": [float(att_a_low), float(att_a_high)],
                "attenuation_ratio_display": float(np.clip(attenuation_ratio_a, 0.0, 1.0)),
                "n_valid_ratio": len(valid_ratios_a),
            },
            "natural_shift": float(natural_shift_v),  # 互換用
            "attenuated_shift": float(attenuated_shift_v),
            "mediated_attenuation": float(mediated_attenuation_v),
            "attenuation_ratio": float(attenuation_ratio_v),
            "passed": bool(h3_pass),
        },
        "h3_necessity": {  # 互換用キー
            "natural_shift": float(natural_shift_v),
            "attenuated_shift": float(attenuated_shift_v),
            "mediated_attenuation": float(mediated_attenuation_v),
            "attenuation_ratio": float(attenuation_ratio_v),
            "passed": bool(h3_pass),
        },
        "h4_temporal_emergence": {
            "stage_causal": stage_causal_v,
            "stage_causal_v": stage_causal_v,
            "stage_causal_a": stage_causal_a,
            "contrast_v": contrast_v,
            "contrast_a": contrast_a,
            "non_uniform_leverage": True,
            "passed_valence": bool(h4_pass_v),
            "passed_arousal": bool(h4_pass_a),
            "passed": bool(h4_pass),
        },
        "auxiliary_qc_all_pass": auxiliary_qc_all_pass,
        "all_confirmed": auxiliary_qc_all_pass,  # 後方互換用エイリアス
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
    if args.dry_run:
        raw_dir = raw_dir / "dry_run"
        derived_dir = derived_dir / "dry_run"
    raw_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    family_results = {}
    base_seed = v3_cfg.get("seed", 42)
    seeds = {
        "llama": base_seed + 10,
        "gemma": base_seed + 20,
        "olmo": base_seed + 30,
        "mistral": base_seed + 40,
    }

    conf_cfg = v3_cfg.get("confirmatory", {})
    selection_source = conf_cfg.get("selection_source", "qwen_discovery_frozen")

    # Load frozen confirmatory sites artifact
    frozen_sites_path = derived_dir / "frozen_confirmatory_sites.json"
    if not frozen_sites_path.exists():
        frozen_sites_path = Path(v3_cfg["output"]["derived_dir"]) / "frozen_confirmatory_sites.json"

    frozen_sites = None
    if frozen_sites_path.exists():
        try:
            with open(frozen_sites_path, "r", encoding="utf-8") as f:
                frozen_sites = json.load(f)
            logger.info(f"Successfully loaded frozen confirmatory sites from {frozen_sites_path}")
            v3_cfg["frozen_sites"] = frozen_sites
            selection_source = f"frozen_confirmatory_sites_from_{frozen_sites.get('discovery_model', 'discovery')}"
        except Exception as e:
            logger.error(f"Failed to read frozen confirmatory sites from {frozen_sites_path}: {e}")
            if not args.dry_run:
                raise

    if frozen_sites is None:
        if args.dry_run:
            logger.warning("[DRY-RUN] frozen_confirmatory_sites.json not found. Using fallback values for mock dry-run only.")
            suff_depth_val = float(conf_cfg.get("sufficiency_relative_depth", 0.5))
            temp_depth_v_val = float(conf_cfg.get("temporal_relative_depth_v", 0.65))
            temp_depth_a_val = float(conf_cfg.get("temporal_relative_depth_a", 0.65))
            stage_v_val = str(conf_cfg.get("temporal_stage_v", "pre_V"))
            stage_a_val = str(conf_cfg.get("temporal_stage_a", "pre_A"))
            med_depth_val = float(conf_cfg.get("mediation_relative_depth", 0.65))
            frozen_sites_hash = "mock_hash"
        else:
            raise FileNotFoundError(
                f"Missing required artifact: {frozen_sites_path}. "
                "Confirmatory replication in production requires frozen_confirmatory_sites.json "
                "generated from Discovery RQ1-RQ3. Config fallback is strictly disabled for paper results."
            )
    else:
        suff_depth_val = float(frozen_sites["sufficiency_relative_depth"])
        temp_depth_v_val = float(frozen_sites["temporal_relative_depth_v"])
        temp_depth_a_val = float(frozen_sites["temporal_relative_depth_a"])
        stage_v_val = str(frozen_sites.get("temporal_stage_v", "pre_V"))
        stage_a_val = str(frozen_sites.get("temporal_stage_a", "pre_A"))
        med_depth_val = float(frozen_sites["mediation_relative_depth"])
        frozen_sites_hash = compute_string_or_dict_hash(frozen_sites)

    for item in conf_models:
        fam_key = item["family_key"]
        fam_name = item["family_name"]
        model_id = item["model_id"]
        out_raw = raw_dir / f"v3_confirmatory_{fam_key}.json"
        manifest_path = raw_dir / f"manifest_confirmatory_{fam_key}.json"

        registry = get_registry()
        fam_cfg = registry.get_family_by_model_id(model_id) or registry.get_family(fam_key)
        base_rev = fam_cfg.base_model.revision if fam_cfg else "main"
        inst_rev = fam_cfg.instruct_model.revision if fam_cfg else "main"
        inf_dtype = getattr(fam_cfg, "inference_dtype", "bfloat16") if fam_cfg else "bfloat16"

        manifest_config = {
            "analysis_role": "confirmatory",
            "family_key": fam_key,
            "family_name": fam_name,
            "model_id": model_id,
            "base_revision": base_rev,
            "instruct_revision": inst_rev,
            "inference_dtype": inf_dtype,
            "dataset_path": str(v3_cfg["dataset"]["path"]),
            "confirmatory_site_selection_source": selection_source,
            "sufficiency_relative_depth": suff_depth_val,
            "temporal_relative_depth_v": temp_depth_v_val,
            "temporal_relative_depth_a": temp_depth_a_val,
            "temporal_stage_v": stage_v_val,
            "temporal_stage_a": stage_a_val,
            "mediation_relative_depth": med_depth_val,
            "frozen_sites_hash": frozen_sites_hash,
            "seed": seeds.get(fam_key, base_seed + 99),
            "subsample": args.subsample,
            "dry_run": bool(args.dry_run),
        }

        from affective_empathy_eval.io import save_experiment_result, is_experiment_completed

        modular_conf_path = raw_dir / f"v3_confirmatory_replication_{fam_key}.json"

        if not args.force and not args.dry_run:
            ds_path_p = Path(v3_cfg["dataset"]["path"])
            exp_ds_hash = compute_file_hash(ds_path_p) if ds_path_p.exists() else compute_string_or_dict_hash(str(ds_path_p))
            exp_cfg_hash = compute_string_or_dict_hash(manifest_config)

            if manifest_path.exists():
                manifest_valid = is_manifest_matching(
                    manifest_path=str(manifest_path),
                    expected_model_name=model_id,
                    expected_config_hash=exp_cfg_hash,
                    expected_dataset_hash=exp_ds_hash,
                    expected_dry_run=False,
                )
                if manifest_valid:
                    target_check = modular_conf_path if modular_conf_path.exists() else out_raw
                    if is_experiment_completed(str(target_check), manifest_path=str(manifest_path)):
                        try:
                            read_p = modular_conf_path if modular_conf_path.exists() else out_raw
                            with open(read_p, "r", encoding="utf-8") as f:
                                cached = json.load(f)
                            if cached and ("auxiliary_qc_all_pass" in cached or "all_confirmed" in cached or "h1_dissociation" in cached):
                                logger.info(f"Loaded existing valid confirmatory results matching manifest for {fam_name} from {read_p}. Skipping.")
                                family_results[fam_name] = cached
                                continue
                        except Exception as e:
                            logger.warning(f"Cache check failed for {fam_name}: {e}")

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
                v3_cfg=v3_cfg,
                dry_run=bool(args.dry_run),
            )

        family_results[fam_name] = res

        with open(out_raw, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)

        save_experiment_result(
            output_path=str(modular_conf_path),
            payload=res,
            stage="v3",
            experiment_id="v3_confirmatory_replication",
            status="success",
            success=True,
            metadata={"family_id": fam_key, "model_id": model_id, "dry_run": bool(args.dry_run)},
        )
        logger.info(f"Saved {fam_name} confirmatory results to {out_raw} and {modular_conf_path}")

        # Manifest 保存
        registry = get_registry()
        fam_cfg = registry.get_family_by_model_id(model_id) or registry.get_family(fam_key)
        model_revision = fam_cfg.get_model_spec("instruct").revision if fam_cfg else "main"
        manifest = create_run_manifest(
            run_type="v3_confirmatory",
            model_name=model_id,
            model_revision=model_revision,
            config=manifest_config,
            metadata={
                "family": fam_name,
                "confirmatory_site_selection_source": selection_source,
                "n_input_pairs": int(df.attrs.get("n_input_pairs", len(df))),
                "n_matched_pairs": int(df.attrs.get("n_matched_pairs", len(df))),
                "n_excluded_pairs": int(df.attrs.get("n_excluded_pairs", 0)),
                "slope_v": res["h2_sufficiency"]["slope_v"],
                "slope_a": res["h2_sufficiency"]["slope_a"],
                "auxiliary_qc_all_pass": res.get("auxiliary_qc_all_pass", res.get("all_confirmed", False)),
                "all_confirmed": res.get("all_confirmed", False),
            },
            candidate_space="VA_81",
            measurement_space="VA_81",
            actual_dtype="bfloat16" if (args.device != "cpu" and torch.cuda.is_available()) else "float32",
            dry_run=bool(args.dry_run),
        )
        manifest.save(str(manifest_path))

    # メタ分析サマリーの生成 (効果量推定値とCI中心の出力構成)
    all_passed = all(res.get("auxiliary_qc_all_pass", res.get("all_confirmed", False)) for res in family_results.values())
    status_msg = (
        "MOCK_SIMULATION: Hypotheses simulated for validation purposes."
        if args.dry_run
        else "EFFECT_ESTIMATES_AVAILABLE"
    )

    h1_dp_v = [res["h1_dissociation"]["valence"]["delta_d_peak"] for res in family_results.values()]
    h1_dp_a = [res["h1_dissociation"]["arousal"]["delta_d_peak"] for res in family_results.values()]
    h1_dc_v = [res["h1_dissociation"]["valence"]["delta_d_center"] for res in family_results.values()]
    h1_dc_a = [res["h1_dissociation"]["arousal"]["delta_d_center"] for res in family_results.values()]

    h2_sv = [res["h2_sufficiency"]["slope_v"] for res in family_results.values()]
    h2_sa = [res["h2_sufficiency"]["slope_a"] for res in family_results.values()]

    h3_med_v = [res["h3_endogenous_relevance"]["valence"]["mediated_attenuation"] for res in family_results.values()]
    h3_med_a = [res["h3_endogenous_relevance"]["arousal"]["mediated_attenuation"] for res in family_results.values()]
    h3_av = [res["h3_endogenous_relevance"]["valence"]["attenuation_ratio"] for res in family_results.values()]
    h3_aa = [res["h3_endogenous_relevance"]["arousal"]["attenuation_ratio"] for res in family_results.values()]

    h4_cv = [res["h4_temporal_emergence"]["contrast_v"] for res in family_results.values()]
    h4_ca = [res["h4_temporal_emergence"]["contrast_a"] for res in family_results.values()]

    pt_h1_v, h1_v_low, h1_v_high = compute_bootstrap_ci(h1_dp_v, n_boot=1000) if len(h1_dp_v) > 1 else (np.mean(h1_dp_v), np.nan, np.nan)
    pt_h1_a, h1_a_low, h1_a_high = compute_bootstrap_ci(h1_dp_a, n_boot=1000) if len(h1_dp_a) > 1 else (np.mean(h1_dp_a), np.nan, np.nan)
    pt_h2_v, h2_v_low, h2_v_high = compute_bootstrap_ci(h2_sv, n_boot=1000) if len(h2_sv) > 1 else (np.mean(h2_sv), np.nan, np.nan)
    pt_h2_a, h2_a_low, h2_a_high = compute_bootstrap_ci(h2_sa, n_boot=1000) if len(h2_sa) > 1 else (np.mean(h2_sa), np.nan, np.nan)
    pt_h3_mv, h3_mv_low, h3_mv_high = compute_bootstrap_ci(h3_med_v, n_boot=1000) if len(h3_med_v) > 1 else (np.mean(h3_med_v), np.nan, np.nan)
    pt_h3_ma, h3_ma_low, h3_ma_high = compute_bootstrap_ci(h3_med_a, n_boot=1000) if len(h3_med_a) > 1 else (np.mean(h3_med_a), np.nan, np.nan)
    pt_h3_v, h3_v_low, h3_v_high = compute_bootstrap_ci(h3_av, n_boot=1000) if len(h3_av) > 1 else (np.mean(h3_av), np.nan, np.nan)
    pt_h3_a, h3_a_low, h3_a_high = compute_bootstrap_ci(h3_aa, n_boot=1000) if len(h3_aa) > 1 else (np.mean(h3_aa), np.nan, np.nan)
    pt_h4_v, h4_v_low, h4_v_high = compute_bootstrap_ci(h4_cv, n_boot=1000) if len(h4_cv) > 1 else (np.mean(h4_cv), np.nan, np.nan)
    pt_h4_a, h4_a_low, h4_a_high = compute_bootstrap_ci(h4_ca, n_boot=1000) if len(h4_ca) > 1 else (np.mean(h4_ca), np.nan, np.nan)

    summary = {
        "is_dry_run": args.dry_run,
        "replicated_families": list(family_results.keys()),
        "primary_effect_estimates": {
            "H1_peak_dissociation": {
                "valence": {"mean_delta_d_peak": float(pt_h1_v), "ci_95": [float(h1_v_low), float(h1_v_high)]},
                "arousal": {"mean_delta_d_peak": float(pt_h1_a), "ci_95": [float(h1_a_low), float(h1_a_high)]},
            },
            "H2_sufficiency_slope": {
                "valence": {"mean_slope_v": float(pt_h2_v), "ci_95": [float(h2_v_low), float(h2_v_high)]},
                "arousal": {"mean_slope_a": float(pt_h2_a), "ci_95": [float(h2_a_low), float(h2_a_high)]},
            },
            "H3_endogenous_attenuation": {
                "valence": {
                    "primary_mean_mediated_attenuation": float(pt_h3_mv),
                    "ci_95_mediated_attenuation": [float(h3_mv_low), float(h3_mv_high)],
                    "secondary_mean_attenuation_ratio": float(pt_h3_v),
                    "ci_95_attenuation_ratio": [float(h3_v_low), float(h3_v_high)],
                },
                "arousal": {
                    "primary_mean_mediated_attenuation": float(pt_h3_ma),
                    "ci_95_mediated_attenuation": [float(h3_ma_low), float(h3_ma_high)],
                    "secondary_mean_attenuation_ratio": float(pt_h3_a),
                    "ci_95_attenuation_ratio": [float(h3_a_low), float(h3_a_high)],
                },
            },
            "H4_temporal_contrast": {
                "valence": {"mean_contrast_v": float(pt_h4_v), "ci_95": [float(h4_v_low), float(h4_v_high)]},
                "arousal": {"mean_contrast_a": float(pt_h4_a), "ci_95": [float(h4_a_low), float(h4_a_high)]},
            },
        },
        "family_wise_results": {
            fam: {
                "h1_dissociation": res["h1_dissociation"],
                "h2_sufficiency": res["h2_sufficiency"],
                "h3_endogenous_relevance": res["h3_endogenous_relevance"],
                "h4_temporal_emergence": res["h4_temporal_emergence"],
            }
            for fam, res in family_results.items()
        },
        "auxiliary_qc_checklist": {
            "H1_peak_dissociation": {
                fam: res["h1_dissociation"]["passed"] for fam, res in family_results.items()
            },
            "H2_sufficiency": {
                fam: res["h2_sufficiency"]["passed"] for fam, res in family_results.items()
            },
            "H3_endogenous_relevance": {
                fam: res["h3_endogenous_relevance"]["passed"] for fam, res in family_results.items()
            },
            "H3_necessity": {  # 互換用キー
                fam: res["h3_necessity"]["passed"] for fam, res in family_results.items()
            },
            "H4_temporal_emergence": {
                fam: res["h4_temporal_emergence"]["passed"] for fam, res in family_results.items()
            },
        },
        "auxiliary_qc_all_pass": bool(all_passed),
        "status": status_msg,
        "cross_model_generality": status_msg,
    }

    out_derived = derived_dir / "v3_cross_model_replication_summary.json"
    with open(out_derived, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved cross-model replication summary to {out_derived}")
    logger.info(f"Step 7 Confirmatory replication completed: {status_msg}")


if __name__ == "__main__":
    main()

