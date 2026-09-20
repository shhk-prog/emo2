#!/usr/bin/env python3
"""
V3-RQ3: 刺激提示時情動表現から自己報告ロジットへの Path Mediation 解析
モデル: Qwen 2.5 (1.5B) Instruct
プロトコル:
  - 刺激データを Discovery (50%) と Confirmation (50%) に厳格分割 (Data-splitting)
  - Discovery セットで Mediator 層 (l_med*) を自動選定
  - Confirmation セットで固定した Mediator 層を 2D 部分空間除去 (P_A = Q Q^T) で遮断
  - Total Affective Shift, Residual Shift after Blocking, Mediated Attenuation, Attenuation Ratio を算出
  - Bootstrap 95% 信頼区間による統計的検証
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
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

from affective_empathy_eval.interventions import (
    compute_orthonormal_subspace,
    extract_conditional_directions,
)
from affective_empathy_eval.likelihood import (
    build_va_candidates,
    compute_expected_va,
    compute_sequence_likelihoods_for_candidates,
)
from affective_empathy_eval.manifests import (
    create_run_manifest,
    is_manifest_matching,
    compute_string_or_dict_hash,
    DEFAULT_CODE_VERSION,
)
from affective_empathy_eval.models.adapters import get_model_adapter
from affective_empathy_eval.data import (
    describe_loaded_frame,
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
    get_generation_stage_tokens,
)
from affective_empathy_eval.statistics import compute_bootstrap_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run V3 Path Mediation Analysis")
    parser.add_argument("--config", type=str, default="configs/v3_experiments.yaml", help="Path to V3 config")
    parser.add_argument("--models-config", type=str, default="configs/models.yaml", help="Path to models config")
    parser.add_argument("--dry-run", action="store_true", help="Run in mock/dry-run mode")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--subsample", type=int, default=0, help="Number of pairs per split for evaluation (0 for full split)")
    parser.add_argument("--force", action="store_true", help="Force recomputation even if valid cached results exist")
    add_model_selection_args(parser)
    return parser.parse_args()


def simulate_path_mediation_discovery(
    discovery_df: pd.DataFrame,
    num_layers: int,
    seed: int = 101,
) -> Dict[str, Any]:
    """
    dry-run用: Discovery セットを用いた刺激提示時ピークおよび Mediator 層候補の模擬選定
    """
    rng = np.random.default_rng(seed)
    relative_depths = [l / (num_layers - 1) if num_layers > 1 else 0.0 for l in range(num_layers)]

    d_stim = [np.exp(-((d - 0.48) ** 2) / (2 * 0.16**2)) * 0.70 + rng.normal(0, 0.02) for d in relative_depths]
    c_gen = [np.exp(-((d - 0.68) ** 2) / (2 * 0.14**2)) * 1.10 + rng.normal(0, 0.02) for d in relative_depths]

    stim_peak_layer = int(np.argmax(d_stim))
    gen_peak_layer = int(np.argmax(c_gen))

    return {
        "stim_peak_layer": stim_peak_layer,
        "stim_peak_depth": relative_depths[stim_peak_layer],
        "mediator_layer": gen_peak_layer,
        "mediator_depth": relative_depths[gen_peak_layer],
        "d_stim_profile": [float(x) for x in d_stim],
        "c_gen_profile": [float(x) for x in c_gen],
        "dry_run": True,
    }


def simulate_path_mediation_confirmation(
    confirmation_df: pd.DataFrame,
    mediator_layer: int,
    num_layers: int = 28,
    bootstrap_n: int = 1000,
    seed: int = 202,
) -> Dict[str, Any]:
    """
    dry-run用: Confirmation セットにおいて固定された Mediator 層を遮断し、媒介効果・減衰を模擬推定
    """
    rng = np.random.default_rng(seed)
    n = len(confirmation_df)

    te_samples_v = rng.normal(1.25, 0.18, n)
    te_samples_a = rng.normal(1.05, 0.16, n)

    residual_samples_v = rng.normal(0.32, 0.12, n)
    residual_samples_a = rng.normal(0.28, 0.11, n)

    atten_samples_v = te_samples_v - residual_samples_v
    atten_samples_a = te_samples_a - residual_samples_a

    MIN_NATURAL_SHIFT = 0.05
    valid_v = te_samples_v > MIN_NATURAL_SHIFT
    valid_a = te_samples_a > MIN_NATURAL_SHIFT

    n_valid_v = int(np.sum(valid_v))
    n_valid_a = int(np.sum(valid_a))

    ratio_samples_v = (
        atten_samples_v[valid_v] / te_samples_v[valid_v]
        if n_valid_v > 0
        else np.array([], dtype=float)
    )
    ratio_samples_a = (
        atten_samples_a[valid_a] / te_samples_a[valid_a]
        if n_valid_a > 0
        else np.array([], dtype=float)
    )

    te_v_mean, te_v_low, te_v_high = compute_bootstrap_ci(te_samples_v, n_boot=bootstrap_n)
    res_v_mean, res_v_low, res_v_high = compute_bootstrap_ci(residual_samples_v, n_boot=bootstrap_n)
    atten_v_mean, atten_v_low, atten_v_high = compute_bootstrap_ci(atten_samples_v, n_boot=bootstrap_n)
    if n_valid_v >= 2:
        ratio_v_mean, ratio_v_low, ratio_v_high = compute_bootstrap_ci(ratio_samples_v, n_boot=bootstrap_n)
    elif n_valid_v == 1:
        ratio_v_mean, ratio_v_low, ratio_v_high = float(ratio_samples_v[0]), float(ratio_samples_v[0]), float(ratio_samples_v[0])
    else:
        ratio_v_mean, ratio_v_low, ratio_v_high = np.nan, np.nan, np.nan

    te_a_mean, te_a_low, te_a_high = compute_bootstrap_ci(te_samples_a, n_boot=bootstrap_n)
    res_a_mean, res_a_low, res_a_high = compute_bootstrap_ci(residual_samples_a, n_boot=bootstrap_n)
    atten_a_mean, atten_a_low, atten_a_high = compute_bootstrap_ci(atten_samples_a, n_boot=bootstrap_n)
    if n_valid_a >= 2:
        ratio_a_mean, ratio_a_low, ratio_a_high = compute_bootstrap_ci(ratio_samples_a, n_boot=bootstrap_n)
    elif n_valid_a == 1:
        ratio_a_mean, ratio_a_low, ratio_a_high = float(ratio_samples_a[0]), float(ratio_samples_a[0]), float(ratio_samples_a[0])
    else:
        ratio_a_mean, ratio_a_low, ratio_a_high = np.nan, np.nan, np.nan

    med_depth = float(mediator_layer / (num_layers - 1)) if num_layers > 1 else 0.68

    return {
        "primary_grounding": "reader_prediction",
        "mediator_layer": mediator_layer,
        "mediator_relative_depth": med_depth,
        "n_total": n,
        "n_valid_ratio_v": n,
        "n_valid_ratio_a": n,
        "min_natural_shift_threshold": 0.05,
        "valence": {
            "total_affective_shift": {"mean": te_v_mean, "ci_lower": te_v_low, "ci_upper": te_v_high},
            "residual_shift_after_blocking": {"mean": res_v_mean, "ci_lower": res_v_low, "ci_upper": res_v_high},
            "mediated_attenuation": {"mean": atten_v_mean, "ci_lower": atten_v_low, "ci_upper": atten_v_high},
            "attenuation_ratio": {"mean": ratio_v_mean, "ci_lower": ratio_v_low, "ci_upper": ratio_v_high},
        },
        "arousal": {
            "total_affective_shift": {"mean": te_a_mean, "ci_lower": te_a_low, "ci_upper": te_a_high},
            "residual_shift_after_blocking": {"mean": res_a_mean, "ci_lower": res_a_low, "ci_upper": res_a_high},
            "mediated_attenuation": {"mean": atten_a_mean, "ci_lower": atten_a_low, "ci_upper": atten_a_high},
            "attenuation_ratio": {"mean": ratio_a_mean, "ci_lower": ratio_a_low, "ci_upper": ratio_a_high},
        },
        "dry_run": True,
    }


def run_real_path_mediation(
    df: pd.DataFrame,
    model_id: str,
    device: str = "cpu",
    subsample: int = 0,
    bootstrap_n: int = 1000,
    v3_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    実モデルを用いた V3-RQ3 Path Mediation 解析
    1. Discovery split で全層のデコード・因果変位から Mediator 層 l_med* を自動選定
    2. Confirmation split で固定した l_med* の情動部分空間を除去し、Total Shift, Residual Shift, Attenuation, Attenuation Ratio を測定
    """
    logger.info(f"Loading model {model_id} for Path Mediation on {device}...")
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

    # 1. Config に基づく Data Splitting (Item 11: seed & discovery_ratio from config, strict disjoint)
    seed = int(v3_cfg.get("seed", 42)) if v3_cfg else 42
    discovery_ratio = float(v3_cfg.get("path_mediation", {}).get("discovery_ratio", 0.5)) if v3_cfg else 0.5
    rng = np.random.RandomState(seed)
    if "pair_id" in df.columns and df["pair_id"].nunique() > 1:
        unique_pairs = df["pair_id"].unique()
        rng.shuffle(unique_pairs)
        n_disc = int(round(len(unique_pairs) * discovery_ratio))
        disc_pairs = set(unique_pairs[:n_disc])
        conf_pairs = set(unique_pairs[n_disc:])
        assert disc_pairs.isdisjoint(conf_pairs), "Data leakage! Discovery and Confirmation pair sets must be strictly disjoint."
        disc_df = df[df["pair_id"].isin(disc_pairs)].copy().reset_index(drop=True)
        conf_df = df[df["pair_id"].isin(conf_pairs)].copy().reset_index(drop=True)
        if subsample is not None and subsample > 0:
            disc_df = disc_df.head(subsample)
            conf_df = conf_df.head(subsample)
        logger.info(f"Group split on pair_id: {len(disc_pairs)} pairs Discovery ({len(disc_df)} samples), {len(conf_pairs)} pairs Confirmation ({len(conf_df)} samples)")
    else:
        indices = rng.permutation(len(df))
        n_disc = int(round(len(df) * discovery_ratio))
        disc_df = df.iloc[indices[:n_disc]].copy().reset_index(drop=True)
        conf_df = df.iloc[indices[n_disc:]].copy().reset_index(drop=True)
        if subsample is not None and subsample > 0:
            disc_df = disc_df.head(subsample)
            conf_df = conf_df.head(subsample)
        logger.info(f"Index split: {len(disc_df)} samples Discovery, {len(conf_df)} samples Confirmation")

    candidates = build_va_candidates()

    # 2. Discovery: ピーク層および Mediator 層の自動同定（探索的 site selection）
    # 注: Discovery サブセット内での方向推定と評価は候補層スクリーニングであり、
    # 最終的な媒介推論は完全に独立した Confirmation サブセットで実行されます。
    logger.info(f"Running Discovery stage (exploratory site selection) on {len(disc_df)} samples across {num_layers} layers...")
    d_stim_v_profile = []
    d_stim_a_profile = []
    d_stim_joint_profile = []
    c_gen_profile = []

    disc_texts = [str(t) for t in disc_df["text"]]
    if "reader_V" in disc_df.columns:
        y_v_disc = disc_df["reader_V"].to_numpy()
        y_a_disc = disc_df["reader_A"].to_numpy() if "reader_A" in disc_df.columns else y_v_disc
    else:
        logger.info("No human reader_V/A on discovery; using model Reader Predictions (TaskType.READER) as primary affect targets.")
        y_v_list, y_a_list = [], []
        with torch.no_grad():
            for text in disc_texts:
                p_reader = build_prompt(text, task=TaskType.READER, format_type="chat", tokenizer=tokenizer)
                log_liks, probs = compute_sequence_likelihoods_for_candidates(
                    model=model, tokenizer=tokenizer, prompt=p_reader, candidates=candidates, device=device, batch_size=81
                )
                ev, ea = compute_expected_va(log_liks, candidates)
                y_v_list.append(ev)
                y_a_list.append(ea)
        y_v_disc = np.array(y_v_list, dtype=np.float64)
        y_a_disc = np.array(y_a_list, dtype=np.float64)
    disc_groups = disc_df["pair_id"].values if "pair_id" in disc_df.columns else disc_df["id"].values

    for l in range(num_layers):
        h_stim = []
        with torch.no_grad():
            for text in disc_texts:
                p = build_prompt(text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
                enc = encode_prompt_canonical(tokenizer, p, device=device)
                anchors = find_semantic_anchors(enc["input_ids"][0].tolist(), tokenizer, text)
                with ActivationHookManager(adapter) as hook_mgr:
                    hook_mgr.register_capture_hook(
                        layer_idx=l,
                        hook_point=HookPoint.POST_MLP_RESID,
                        token_indices=anchors["prompt_end"],
                        key="h_stim",
                    )
                    _ = model(**enc)
                    h_stim.append(hook_mgr.captured_activations["h_stim"].cpu().float().numpy().ravel())

        H_s = np.array(h_stim)
        # Held-out 5-fold cross-validation R^2 with leakage-free GroupKFold (Valence and Arousal both)
        from sklearn.model_selection import GroupKFold
        n_unique_groups = len(np.unique(disc_groups))
        n_splits = min(5, n_unique_groups)
        if n_splits > 1:
            gkf = GroupKFold(n_splits=n_splits)
            preds_v = np.zeros_like(y_v_disc)
            preds_a = np.zeros_like(y_a_disc)
            for train_idx, val_idx in gkf.split(H_s, y_v_disc, groups=disc_groups):
                ridge_v = Ridge(alpha=10.0).fit(H_s[train_idx], y_v_disc[train_idx])
                preds_v[val_idx] = ridge_v.predict(H_s[val_idx])

                ridge_a = Ridge(alpha=10.0).fit(H_s[train_idx], y_a_disc[train_idx])
                preds_a[val_idx] = ridge_a.predict(H_s[val_idx])

            ss_res_v = np.sum((y_v_disc - preds_v)**2)
            ss_tot_v = np.sum((y_v_disc - np.mean(y_v_disc))**2) + 1e-6
            r2_v = float(1.0 - ss_res_v / ss_tot_v)

            ss_res_a = np.sum((y_a_disc - preds_a)**2)
            ss_tot_a = np.sum((y_a_disc - np.mean(y_a_disc))**2) + 1e-6
            r2_a = float(1.0 - ss_res_a / ss_tot_a)

            r2_joint = float(0.5 * (r2_v + r2_a))
        else:
            r2_v, r2_a, r2_joint = 0.0, 0.0, 0.0

        d_stim_v_profile.append(r2_v)
        d_stim_a_profile.append(r2_a)
        d_stim_joint_profile.append(r2_joint)

        # 実 activation intervention による因果的変位 C_joint(l) = (C_V(l) + C_A(l)) / 2 の実測 (探索的スクリーニング)
        from affective_empathy_eval.data import stratified_causal_subset
        selected_disc_idx = stratified_causal_subset(disc_df, n_samples=16, seed=42, stratify_col="target_emotion")

        # Layer l の内部表現から方向 d_v_l, d_a_l を推定
        ridge_dir_v = Ridge(alpha=10.0).fit(H_s, y_v_disc)
        norm_v = np.linalg.norm(ridge_dir_v.coef_)
        d_v_l = ridge_dir_v.coef_ / (norm_v + 1e-6) if norm_v > 0 else np.zeros_like(ridge_dir_v.coef_)
        h_std_v = float(np.std(H_s @ d_v_l)) or 1.0

        ridge_dir_a = Ridge(alpha=10.0).fit(H_s, y_a_disc)
        norm_a = np.linalg.norm(ridge_dir_a.coef_)
        d_a_l = ridge_dir_a.coef_ / (norm_a + 1e-6) if norm_a > 0 else np.zeros_like(ridge_dir_a.coef_)
        h_std_a = float(np.std(H_s @ d_a_l)) or 1.0

        sample_c_v = []
        sample_c_a = []

        with torch.no_grad():
            for k_idx in selected_disc_idx:
                text_k = disc_texts[k_idx]
                p_k = build_prompt(text_k, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
                enc_k = encode_prompt_canonical(tokenizer, p_k, device=device)
                anch_k = find_semantic_anchors(enc_k["input_ids"][0].tolist(), tokenizer, text_k)
                patch_pos_k = anch_k["prompt_end"]

                # a. Clean expected report
                log_clean_k, probs_clean_k = compute_sequence_likelihoods_for_candidates(
                    model=model, tokenizer=tokenizer, prompt=p_k, candidates=candidates, device=device, batch_size=81
                )
                ev_c, ea_c = compute_expected_va(log_clean_k, candidates)

                # b. Valence intervention (inject d_v_l) -> |Delta V|
                with ActivationHookManager(adapter) as hook_mgr:
                    hook_mgr.register_direction_intervention_hook(
                        layer_idx=l,
                        direction=d_v_l,
                        alpha=1.0,
                        hidden_std=h_std_v,
                        token_indices=patch_pos_k,
                        hook_point=HookPoint.POST_MLP_RESID,
                        mode="inject",
                    )
                    log_int_v, probs_int_v = compute_sequence_likelihoods_for_candidates(
                        model=model, tokenizer=tokenizer, prompt=p_k, candidates=candidates, device=device, batch_size=81
                    )
                ev_i_v, _ = compute_expected_va(log_int_v, candidates)
                sample_c_v.append(abs(ev_i_v - ev_c))

                # c. Arousal intervention (inject d_a_l) -> |Delta A|
                with ActivationHookManager(adapter) as hook_mgr:
                    hook_mgr.register_direction_intervention_hook(
                        layer_idx=l,
                        direction=d_a_l,
                        alpha=1.0,
                        hidden_std=h_std_a,
                        token_indices=patch_pos_k,
                        hook_point=HookPoint.POST_MLP_RESID,
                        mode="inject",
                    )
                    log_int_a, probs_int_a = compute_sequence_likelihoods_for_candidates(
                        model=model, tokenizer=tokenizer, prompt=p_k, candidates=candidates, device=device, batch_size=81
                    )
                _, ea_i_a = compute_expected_va(log_int_a, candidates)
                sample_c_a.append(abs(ea_i_a - ea_c))

        c_v_score = float(np.mean(sample_c_v)) if sample_c_v else 0.0
        c_a_score = float(np.mean(sample_c_a)) if sample_c_a else 0.0
        c_joint_score = float((c_v_score + c_a_score) / 2.0)
        c_gen_profile.append(c_joint_score)

    stim_peak_layer = int(np.argmax(d_stim_joint_profile))
    mediator_layer = int(np.argmax(c_gen_profile))
    logger.info(f"Discovery Result: stim_peak_layer={stim_peak_layer}, mediator_layer={mediator_layer}")

    discovery_summary = {
        "primary_grounding": "reader_prediction",
        "stimulus_peak_layer": stim_peak_layer,
        "stim_peak_depth": relative_depths[stim_peak_layer],
        "mediator_layer": mediator_layer,
        "mediator_depth": relative_depths[mediator_layer],
        "d_stim_v_profile": [float(x) for x in d_stim_v_profile],
        "d_stim_a_profile": [float(x) for x in d_stim_a_profile],
        "d_stim_joint_profile": [float(x) for x in d_stim_joint_profile],
        "d_stim_profile": [float(x) for x in d_stim_joint_profile],
        "c_joint_profile": [float(x) for x in c_gen_profile],
    }

    # 3. Confirmation: Mediator 層の情動部分空間除去による因果媒介効果の検定
    logger.info(f"Running Confirmation stage on {len(conf_df)} samples at Mediator Layer {mediator_layer}...")

    # Discovery データから Mediator 層における情動方向 d_V, d_A を推定
    h_med_disc = []
    with torch.no_grad():
        for text in disc_texts:
            p = build_prompt(text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
            enc = encode_prompt_canonical(tokenizer, p, device=device)
            anchors = find_semantic_anchors(enc["input_ids"][0].tolist(), tokenizer, text)
            with ActivationHookManager(adapter) as hook_mgr:
                hook_mgr.register_capture_hook(
                    layer_idx=mediator_layer,
                    hook_point=HookPoint.POST_MLP_RESID,
                    token_indices=anchors["prompt_end"],
                    key="h_med",
                )
                _ = model(**enc)
                h_med_disc.append(hook_mgr.captured_activations["h_med"].cpu().float().numpy().ravel())

    H_med = np.array(h_med_disc)
    dirs = extract_conditional_directions(H_med, y_v_disc, y_a_disc, alpha=1.0)
    Q_sub, _ = compute_orthonormal_subspace(dirs["direction_v"], dirs["direction_a"])  # (D, 2)


    # matched-neutral 表現の抽出 (Discovery split)
    disc_neu_hiddens = []
    with torch.no_grad():
        for _, row in disc_df.iterrows():
            neu_text = resolve_matched_neutral_text(row, df)
            if neu_text is not None and len(neu_text.strip()) > 0:
                p_neu = build_prompt(neu_text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
                enc_neu = encode_prompt_canonical(tokenizer, p_neu, device=device)
                anchors_neu = find_semantic_anchors(enc_neu["input_ids"][0].tolist(), tokenizer, neu_text)
                with ActivationHookManager(adapter) as hook_mgr:
                    hook_mgr.register_capture_hook(
                        layer_idx=mediator_layer,
                        hook_point=HookPoint.POST_MLP_RESID,
                        token_indices=anchors_neu["prompt_end"],
                        key="h_med_neu",
                    )
                    _ = model(**enc_neu)
                    disc_neu_hiddens.append(hook_mgr.captured_activations["h_med_neu"].cpu().float().numpy().ravel())
    if len(disc_neu_hiddens) == 0:
        raise ValueError("No matched-neutral representations available in Discovery split.")
    mu_neu = np.mean(disc_neu_hiddens, axis=0)

    # Confirmation セットで自然な情動変位 (Total affective shift) と Mediator 遮断後の残差変位 (Residual shift) を実測
    te_v_list, te_a_list = [], []
    residual_v_list, residual_a_list = [], []

    with torch.no_grad():
        for _, row in conf_df.iterrows():
            text = str(row["text"])
            prompt = build_prompt(text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
            enc = encode_prompt_canonical(tokenizer, prompt, device=device)
            anchors = find_semantic_anchors(enc["input_ids"][0].tolist(), tokenizer, text)
            patch_pos = anchors["prompt_end"]

            # matched-neutral の特定と baseline 自己報告の計測
            neutral_text = resolve_matched_neutral_text(row, df)
            if neutral_text is not None and len(neutral_text.strip()) > 0:
                p_neu = build_prompt(neutral_text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
                log_neu, probs_neu = compute_sequence_likelihoods_for_candidates(
                    model=model, tokenizer=tokenizer, prompt=p_neu, candidates=candidates, device=device, batch_size=81
                )
                ev_neu, ea_neu = compute_expected_va(log_neu, candidates)
            else:
                # neutral baseline がない場合はエラーを送出（Primaryでは固定5.0へのサイレントfallbackは禁止）
                raise ValueError(f"Missing matched-neutral baseline for stimulus: {row.get('stimulus_id', row.get('id', 'unknown'))}")

            # a. Clean baseline (Total affective shift: 自己報告の情動変位 |ev_clean - ev_neu|)
            log_clean, probs_clean = compute_sequence_likelihoods_for_candidates(
                model=model, tokenizer=tokenizer, prompt=prompt, candidates=candidates, device=device, batch_size=81
            )
            ev_clean, ea_clean = compute_expected_va(log_clean, candidates)
            te_v = abs(ev_clean - ev_neu)
            te_a = abs(ea_clean - ea_neu)
            te_v_list.append(te_v)
            te_a_list.append(te_a)

            # b. Mediator 遮断 (Residual shift after mediator blocking: 情動部分空間除去下の変位 |ev_abl - ev_neu|)
            # h' = h - Q Q^T (h - mu_neu)
            with ActivationHookManager(adapter) as hook_mgr:
                hook_mgr.register_capture_hook(
                    layer_idx=mediator_layer,
                    hook_point=HookPoint.POST_MLP_RESID,
                    token_indices=patch_pos,
                    key="h_conf",
                )
                _ = model(**enc)
                h_conf = hook_mgr.captured_activations["h_conf"].cpu().float().numpy().ravel()

            h_centered = h_conf - mu_neu
            proj = (h_centered @ Q_sub) @ Q_sub.T
            h_abl = h_conf - proj
            patch_tensor = torch.tensor(h_abl, dtype=torch.float32, device=device)

            with ActivationHookManager(adapter) as hook_mgr:
                hook_mgr.register_patch_hook(
                    layer_idx=mediator_layer,
                    patch_tensor=patch_tensor,
                    token_indices=patch_pos,
                    hook_point=HookPoint.POST_MLP_RESID,
                )
                log_abl, probs_abl = compute_sequence_likelihoods_for_candidates(
                    model=model, tokenizer=tokenizer, prompt=prompt, candidates=candidates, device=device, batch_size=81
                )
            ev_abl, ea_abl = compute_expected_va(log_abl, candidates)
            res_v = abs(ev_abl - ev_neu)
            res_a = abs(ea_abl - ea_neu)
            residual_v_list.append(res_v)
            residual_a_list.append(res_a)

    te_samples_v = np.array(te_v_list)
    te_samples_a = np.array(te_a_list)
    res_samples_v = np.array(residual_v_list)
    res_samples_a = np.array(residual_a_list)

    atten_samples_v = te_samples_v - res_samples_v
    atten_samples_a = te_samples_a - res_samples_a

    MIN_NATURAL_SHIFT = 0.05
    valid_v = te_samples_v > MIN_NATURAL_SHIFT
    valid_a = te_samples_a > MIN_NATURAL_SHIFT

    n_total = len(te_samples_v)
    n_valid_v = int(np.sum(valid_v))
    n_valid_a = int(np.sum(valid_a))

    ratio_samples_v = (
        atten_samples_v[valid_v] / te_samples_v[valid_v]
        if n_valid_v > 0
        else np.array([], dtype=float)
    )
    ratio_samples_a = (
        atten_samples_a[valid_a] / te_samples_a[valid_a]
        if n_valid_a > 0
        else np.array([], dtype=float)
    )

    te_v_mean, te_v_low, te_v_high = compute_bootstrap_ci(te_samples_v, n_boot=bootstrap_n)
    res_v_mean, res_v_low, res_v_high = compute_bootstrap_ci(res_samples_v, n_boot=bootstrap_n)
    atten_v_mean, atten_v_low, atten_v_high = compute_bootstrap_ci(atten_samples_v, n_boot=bootstrap_n)
    if n_valid_v >= 2:
        ratio_v_mean, ratio_v_low, ratio_v_high = compute_bootstrap_ci(ratio_samples_v, n_boot=bootstrap_n)
    elif n_valid_v == 1:
        ratio_v_mean, ratio_v_low, ratio_v_high = float(ratio_samples_v[0]), float(ratio_samples_v[0]), float(ratio_samples_v[0])
    else:
        ratio_v_mean, ratio_v_low, ratio_v_high = np.nan, np.nan, np.nan

    te_a_mean, te_a_low, te_a_high = compute_bootstrap_ci(te_samples_a, n_boot=bootstrap_n)
    res_a_mean, res_a_low, res_a_high = compute_bootstrap_ci(res_samples_a, n_boot=bootstrap_n)
    atten_a_mean, atten_a_low, atten_a_high = compute_bootstrap_ci(atten_samples_a, n_boot=bootstrap_n)
    if n_valid_a >= 2:
        ratio_a_mean, ratio_a_low, ratio_a_high = compute_bootstrap_ci(ratio_samples_a, n_boot=bootstrap_n)
    elif n_valid_a == 1:
        ratio_a_mean, ratio_a_low, ratio_a_high = float(ratio_samples_a[0]), float(ratio_samples_a[0]), float(ratio_samples_a[0])
    else:
        ratio_a_mean, ratio_a_low, ratio_a_high = np.nan, np.nan, np.nan

    confirmation_res = {
        "primary_grounding": "reader_prediction",
        "mediator_layer": mediator_layer,
        "mediator_relative_depth": float(mediator_layer / (num_layers - 1)) if num_layers > 1 else 0.0,
        "n_total": n_total,
        "n_valid_ratio_v": n_valid_v,
        "n_valid_ratio_a": n_valid_a,
        "min_natural_shift_threshold": MIN_NATURAL_SHIFT,
        "valence": {
            "total_affective_shift": {"mean": te_v_mean, "ci_lower": te_v_low, "ci_upper": te_v_high},
            "residual_shift_after_blocking": {"mean": res_v_mean, "ci_lower": res_v_low, "ci_upper": res_v_high},
            "mediated_attenuation": {"mean": atten_v_mean, "ci_lower": atten_v_low, "ci_upper": atten_v_high},
            "attenuation_ratio": {"mean": ratio_v_mean, "ci_lower": ratio_v_low, "ci_upper": ratio_v_high},
        },
        "arousal": {
            "total_affective_shift": {"mean": te_a_mean, "ci_lower": te_a_low, "ci_upper": te_a_high},
            "residual_shift_after_blocking": {"mean": res_a_mean, "ci_lower": res_a_low, "ci_upper": res_a_high},
            "mediated_attenuation": {"mean": atten_a_mean, "ci_lower": atten_a_low, "ci_upper": atten_a_high},
            "attenuation_ratio": {"mean": ratio_a_mean, "ci_lower": ratio_a_low, "ci_upper": ratio_a_high},
        },
    }

    return discovery_res, confirmation_res


def main():
    args = parse_args()
    logger.info(f"Starting V3 Path Mediation Analysis (dry_run={args.dry_run})")

    with open(args.config, "r", encoding="utf-8") as f:
        v3_cfg = yaml.safe_load(f)

    raw_dir = Path(v3_cfg["output"]["raw_dir"])
    derived_dir = Path(v3_cfg["output"]["derived_dir"])
    if args.dry_run:
        raw_dir = raw_dir / "dry_run"
        derived_dir = derived_dir / "dry_run"
    raw_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    df = load_v3_matched_pair_table(v3_cfg["dataset"]["path"])
    logger.info(describe_loaded_frame(df, "V3-RQ3 matched-pair table", v3_cfg["dataset"]["path"]))

    bootstrap_n = v3_cfg.get("path_mediation", {}).get("eval_bootstrap_n", 1000)

    fam_key, target_model_id = resolve_instruct_target_from_args(
        args,
        Path(args.models_config),
        fallback_family=v3_cfg.get("target_family"),
    )

    num_layers = resolve_architecture_dims(target_model_id)[0]
    discovery_res = None
    confirmation_res = None
    full_output = None
    cache_hit = False

    out_raw = raw_dir / f"v3_path_mediation_{fam_key}.json"
    manifest_path = raw_dir / f"manifest_rq3_{fam_key}.json"

    manifest_config = {
        "analysis_role": "discovery_mediation",
        "family": fam_key,
        "model_id": target_model_id,
        "dataset_path": str(v3_cfg["dataset"]["path"]),
        "subsample": args.subsample,
        "bootstrap_n": bootstrap_n,
        "seed": v3_cfg.get("seed", 42),
        "dry_run": bool(args.dry_run),
    }

    if not args.force and out_raw.exists() and not args.dry_run:
        try:
            with open(out_raw, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached and "confirmation" in cached:
                expected_config_hash = compute_string_or_dict_hash(manifest_config)
                expected_dataset_hash = compute_string_or_dict_hash(str(v3_cfg["dataset"]["path"]))
                if not cached.get("dry_run", False) and is_manifest_matching(
                    str(manifest_path),
                    expected_model_name=target_model_id,
                    expected_config_hash=expected_config_hash,
                    expected_dataset_hash=expected_dataset_hash,
                    expected_code_version=DEFAULT_CODE_VERSION,
                    expected_dry_run=False,
                ):
                    logger.info(f"Loaded existing results from {out_raw}. Skipping computation.")
                    full_output = cached
                    discovery_res = cached["discovery"]
                    confirmation_res = cached["confirmation"]
                    cache_hit = True
        except Exception as e:
            logger.warning(f"Cache check failed for {out_raw}: {e}")

    if full_output is None or discovery_res is None or confirmation_res is None:
        if args.dry_run:
            logger.info("Executing mock path mediation analysis (--dry-run specified)...")
            base_seed = v3_cfg.get("seed", 42)
            discovery_res = simulate_path_mediation_discovery(df.head(len(df) // 2), num_layers, seed=base_seed + 1)
            confirmation_res = simulate_path_mediation_confirmation(
                df.tail(len(df) // 2),
                mediator_layer=discovery_res["mediator_layer"],
                num_layers=num_layers,
                bootstrap_n=bootstrap_n,
                seed=base_seed + 2,
            )
        else:
            logger.info(f"Executing REAL path mediation analysis on {target_model_id}...")
            discovery_res, confirmation_res = run_real_path_mediation(
                df=df,
                model_id=target_model_id,
                device=args.device,
                subsample=args.subsample,
                bootstrap_n=bootstrap_n,
                v3_cfg=v3_cfg,
            )

        n_intervention = int(args.subsample if args.subsample and args.subsample > 0 else len(df))
        full_output = {
            "model_id": target_model_id,
            "family": fam_key,
            "num_layers": num_layers,
            "dry_run": bool(args.dry_run),
            "n_dataset_total": int(len(df)),
            "n_intervention_samples": n_intervention,
            "discovery": discovery_res,
            "confirmation": confirmation_res,
        }
        with open(out_raw, "w", encoding="utf-8") as f:
            json.dump(full_output, f, indent=2)
        logger.info(f"Saved path mediation raw results to {out_raw}")

    # Save manifest only on fresh computation to prevent washing old artifacts
    if not cache_hit:
        n_intervention_manifest = int(full_output.get("n_intervention_samples", len(df)))
        manifest = create_run_manifest(
            run_type="v3_rq3_path_mediation",
            model_name=target_model_id,
            config=manifest_config,
            metadata={
                "n_dataset_total": int(len(df)),
                "n_intervention_samples": n_intervention_manifest,
                "mediator_layer": confirmation_res["mediator_layer"],
                "mediator_relative_depth": confirmation_res.get("mediator_relative_depth", float(confirmation_res["mediator_layer"] / (num_layers - 1))),
                "valence_attenuation": confirmation_res["valence"]["mediated_attenuation"]["mean"],
                "arousal_attenuation": confirmation_res["arousal"]["mediated_attenuation"]["mean"],
                "valence_attenuation_ratio": confirmation_res["valence"]["attenuation_ratio"]["mean"],
                "arousal_attenuation_ratio": confirmation_res["arousal"]["attenuation_ratio"]["mean"],
            },
            dry_run=bool(args.dry_run),
        )
        manifest.save(raw_dir / f"manifest_rq3_{fam_key}.json")
        logger.info(f"Saved RQ3 manifest to {raw_dir / f'manifest_rq3_{fam_key}.json'}")

    out_summary = derived_dir / "v3_path_mediation_summary.json"
    summary_output = {
        "mediator_layer": confirmation_res["mediator_layer"],
        "mediator_relative_depth": confirmation_res.get("mediator_relative_depth", float(confirmation_res["mediator_layer"] / (num_layers - 1))),
        "n_total": confirmation_res["n_total"],
        "n_valid_ratio_v": confirmation_res["n_valid_ratio_v"],
        "n_valid_ratio_a": confirmation_res["n_valid_ratio_a"],
        # Primary: Absolute mediated attenuation
        "valence_mediated_attenuation": confirmation_res["valence"]["mediated_attenuation"]["mean"],
        "valence_mediated_attenuation_ci": [
            confirmation_res["valence"]["mediated_attenuation"]["ci_lower"],
            confirmation_res["valence"]["mediated_attenuation"]["ci_upper"],
        ],
        "arousal_mediated_attenuation": confirmation_res["arousal"]["mediated_attenuation"]["mean"],
        "arousal_mediated_attenuation_ci": [
            confirmation_res["arousal"]["mediated_attenuation"]["ci_lower"],
            confirmation_res["arousal"]["mediated_attenuation"]["ci_upper"],
        ],
        # Secondary: Attenuation ratio
        "valence_attenuation_ratio": confirmation_res["valence"]["attenuation_ratio"]["mean"],
        "valence_attenuation_ci": [
            confirmation_res["valence"]["attenuation_ratio"]["ci_lower"],
            confirmation_res["valence"]["attenuation_ratio"]["ci_upper"],
        ],
        "arousal_attenuation_ratio": confirmation_res["arousal"]["attenuation_ratio"]["mean"],
        "arousal_attenuation_ci": [
            confirmation_res["arousal"]["attenuation_ratio"]["ci_lower"],
            confirmation_res["arousal"]["attenuation_ratio"]["ci_upper"],
        ],
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, indent=2)
    logger.info(f"Saved path mediation summary to {out_summary}")

    # Generate frozen confirmatory sites artifact after RQ3 completion
    # Integrating sufficiency site (RQ1), temporal site (RQ2), and mediation site (RQ3)
    rq1_results_path = raw_dir / "v3_rq1_results.json"
    rq2_sites_path = derived_dir / "v3_rq2_causal_sites.json"

    suff_rel_depth = 0.50
    if rq1_results_path.exists():
        try:
            with open(rq1_results_path, "r", encoding="utf-8") as f:
                rq1_data = json.load(f)
            if "resolved_relative_depth" in rq1_data:
                suff_rel_depth = float(rq1_data["resolved_relative_depth"])
            elif "relative_depth" in rq1_data:
                suff_rel_depth = float(rq1_data["relative_depth"])
            elif "config" in rq1_data and "relative_depth" in rq1_data["config"]:
                suff_rel_depth = float(rq1_data["config"]["relative_depth"])
            elif not args.dry_run:
                raise ValueError(f"RQ1 results at {rq1_results_path} lack resolved_relative_depth.")
        except Exception as e:
            if not args.dry_run:
                raise ValueError(f"Failed to load resolved_relative_depth from RQ1 results: {e}")
    elif not args.dry_run:
        raise FileNotFoundError(f"RQ1 results artifact required at {rq1_results_path} for canonical confirmatory sites.")

    temp_rel_depth_v = 0.65
    temp_rel_depth_a = 0.65
    target_stages = ["pre_V", "pre_A"]
    stage_v = "pre_V"
    stage_a = "pre_A"
    if rq2_sites_path.exists():
        try:
            with open(rq2_sites_path, "r", encoding="utf-8") as f:
                rq2_data = json.load(f)
            temp_rel_depth_v = float(rq2_data.get("temporal_relative_depth_v", rq2_data.get("temporal_relative_depth", 0.65)))
            temp_rel_depth_a = float(rq2_data.get("temporal_relative_depth_a", rq2_data.get("temporal_relative_depth", 0.65)))
            target_stages = rq2_data.get("target_stages", target_stages)
            stage_v = str(rq2_data.get("temporal_stage_v", rq2_data.get("causal_peak_stage_v", "pre_V")))
            stage_a = str(rq2_data.get("temporal_stage_a", rq2_data.get("causal_peak_stage_a", "pre_A")))
        except Exception as e:
            if not args.dry_run:
                raise ValueError(f"Failed to load RQ2 sites from {rq2_sites_path}: {e}")
    elif not args.dry_run:
        raise FileNotFoundError(f"RQ2 sites artifact required at {rq2_sites_path} for canonical confirmatory sites.")

    med_rel_depth = float(confirmation_res.get("mediator_relative_depth", float(confirmation_res["mediator_layer"] / (num_layers - 1)) if num_layers > 1 else 0.65))

    frozen_sites = {
        "discovery_model": target_model_id,
        "discovery_family": fam_key,
        "generation_stage": "post_rq3_canonical",
        "sufficiency_relative_depth": suff_rel_depth,
        "temporal_relative_depth_v": temp_rel_depth_v,
        "temporal_stage_v": stage_v,
        "temporal_relative_depth_a": temp_rel_depth_a,
        "temporal_stage_a": stage_a,
        "temporal_relative_depth": temp_rel_depth_v,  # 互換用
        "a_priori_test_stage_v": rq2_data.get("a_priori_test_stage_v", "pre_V") if rq2_sites_path.exists() else "pre_V",
        "a_priori_test_stage_a": rq2_data.get("a_priori_test_stage_a", "pre_A") if rq2_sites_path.exists() else "pre_A",
        "mediation_relative_depth": med_rel_depth,
        "target_stages": target_stages,
        "causal_peak_stage_v": stage_v,
        "causal_peak_stage_a": stage_a,
    }
    frozen_sites_path = derived_dir / "frozen_confirmatory_sites.json"
    with open(frozen_sites_path, "w", encoding="utf-8") as f:
        json.dump(frozen_sites, f, indent=2)
    logger.info(f"Successfully generated canonical frozen confirmatory sites artifact at {frozen_sites_path}")


if __name__ == "__main__":
    main()
