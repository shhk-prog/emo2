#!/usr/bin/env python3
"""
V3-RQ3: 刺激提示時情動表現から自己報告ロジットへの Path Mediation 解析
モデル: Qwen 2.5 (1.5B) Instruct
プロトコル:
  - 刺激データを Discovery (50%) と Confirmation (50%) に厳格分割 (Data-splitting)
  - Discovery セットで Mediator 層 (l_med*) を自動選定
  - Confirmation セットで固定した Mediator 層を 2D 部分空間除去 (P_A = Q Q^T) で遮断
  - Total Effect (TE), Natural Direct Effect (NDE), Natural Indirect Effect (NIE), Mediation Ratio を算出
  - Bootstrap 95% 信頼区間による統計的検証
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
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
from affective_empathy_eval.models.adapters import get_model_adapter
from affective_empathy_eval.models.registry import (
    add_model_selection_args,
    get_registry,
    resolve_architecture_dims,
    resolve_models_from_args,
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
    parser.add_argument("--subsample", type=int, default=40, help="Number of pairs per split for evaluation")
    add_model_selection_args(parser)
    return parser.parse_args()


def simulate_path_mediation_discovery(
    discovery_df: pd.DataFrame,
    num_layers: int,
) -> Dict[str, Any]:
    """
    dry-run用: Discovery セットを用いた刺激提示時ピークおよび Mediator 層候補の模擬選定
    """
    rng = np.random.default_rng(101)
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
    }


def simulate_path_mediation_confirmation(
    confirmation_df: pd.DataFrame,
    mediator_layer: int,
    bootstrap_n: int = 1000,
) -> Dict[str, Any]:
    """
    dry-run用: Confirmation セットにおいて固定された Mediator 層を遮断し、媒介効果を模擬推定
    """
    rng = np.random.default_rng(202)
    n = len(confirmation_df)

    te_samples_v = rng.normal(1.25, 0.18, n)
    te_samples_a = rng.normal(1.05, 0.16, n)

    nde_samples_v = rng.normal(0.32, 0.12, n)
    nde_samples_a = rng.normal(0.28, 0.11, n)

    nie_samples_v = te_samples_v - nde_samples_v
    nie_samples_a = te_samples_a - nde_samples_a

    ratio_samples_v = nie_samples_v / np.clip(te_samples_v, 1e-5, None)
    ratio_samples_a = nie_samples_a / np.clip(te_samples_a, 1e-5, None)

    te_v_mean, te_v_low, te_v_high = compute_bootstrap_ci(te_samples_v, n_boot=bootstrap_n)
    nde_v_mean, nde_v_low, nde_v_high = compute_bootstrap_ci(nde_samples_v, n_boot=bootstrap_n)
    nie_v_mean, nie_v_low, nie_v_high = compute_bootstrap_ci(nie_samples_v, n_boot=bootstrap_n)
    ratio_v_mean, ratio_v_low, ratio_v_high = compute_bootstrap_ci(ratio_samples_v, n_boot=bootstrap_n)

    te_a_mean, te_a_low, te_a_high = compute_bootstrap_ci(te_samples_a, n_boot=bootstrap_n)
    nde_a_mean, nde_a_low, nde_a_high = compute_bootstrap_ci(nde_samples_a, n_boot=bootstrap_n)
    nie_a_mean, nie_a_low, nie_a_high = compute_bootstrap_ci(nie_samples_a, n_boot=bootstrap_n)
    ratio_a_mean, ratio_a_low, ratio_a_high = compute_bootstrap_ci(ratio_samples_a, n_boot=bootstrap_n)

    return {
        "mediator_layer": mediator_layer,
        "valence": {
            "total_affective_shift": {"mean": te_v_mean, "ci_lower": te_v_low, "ci_upper": te_v_high},
            "residual_shift_after_blocking": {"mean": nde_v_mean, "ci_lower": nde_v_low, "ci_upper": nde_v_high},
            "mediated_attenuation": {"mean": nie_v_mean, "ci_lower": nie_v_low, "ci_upper": nie_v_high},
            "attenuation_ratio": {"mean": ratio_v_mean, "ci_lower": ratio_v_low, "ci_upper": ratio_v_high},
            # Backward compatibility aliases
            "total_effect": {"mean": te_v_mean, "ci_lower": te_v_low, "ci_upper": te_v_high},
            "natural_direct_effect": {"mean": nde_v_mean, "ci_lower": nde_v_low, "ci_upper": nde_v_high},
            "natural_indirect_effect": {"mean": nie_v_mean, "ci_lower": nie_v_low, "ci_upper": nie_v_high},
            "mediation_ratio": {"mean": ratio_v_mean, "ci_lower": ratio_v_low, "ci_upper": ratio_v_high},
        },
        "arousal": {
            "total_affective_shift": {"mean": te_a_mean, "ci_lower": te_a_low, "ci_upper": te_a_high},
            "residual_shift_after_blocking": {"mean": nde_a_mean, "ci_lower": nde_a_low, "ci_upper": nde_a_high},
            "mediated_attenuation": {"mean": nie_a_mean, "ci_lower": nie_a_low, "ci_upper": nie_a_high},
            "attenuation_ratio": {"mean": ratio_a_mean, "ci_lower": ratio_a_low, "ci_upper": ratio_a_high},
            # Backward compatibility aliases
            "total_effect": {"mean": te_a_mean, "ci_lower": te_a_low, "ci_upper": te_a_high},
            "natural_direct_effect": {"mean": nde_a_mean, "ci_lower": nde_a_low, "ci_upper": nde_a_high},
            "natural_indirect_effect": {"mean": nie_a_mean, "ci_lower": nie_a_low, "ci_upper": nie_a_high},
            "mediation_ratio": {"mean": ratio_a_mean, "ci_lower": ratio_a_low, "ci_upper": ratio_a_high},
        },
    }


def run_real_path_mediation(
    df: pd.DataFrame,
    model_id: str,
    device: str = "cpu",
    subsample: int = 40,
    bootstrap_n: int = 1000,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    実モデルを用いた V3-RQ3 Path Mediation 解析
    1. Discovery split で全層のデコード・因果変位から Mediator 層 l_med* を自動選定
    2. Confirmation split で固定した l_med* の情動部分空間を除去し、TE, NDE, NIE, MR を測定
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
    fam_cfg = registry.get_family(model_id)
    adapter = get_model_adapter(model, fam_cfg)
    num_layers = fam_cfg.num_layers
    relative_depths = [l / (num_layers - 1) if num_layers > 1 else 0.0 for l in range(num_layers)]

    # 1. 厳格な 50/50 Data Splitting (seed 42, pair_id に基づく Group split)
    seed = 42
    rng = np.random.RandomState(seed)
    if "pair_id" in df.columns and df["pair_id"].nunique() > 1:
        unique_pairs = df["pair_id"].unique()
        rng.shuffle(unique_pairs)
        half_pairs = len(unique_pairs) // 2
        disc_pairs = set(unique_pairs[:half_pairs])
        disc_df = df[df["pair_id"].isin(disc_pairs)].head(subsample).copy().reset_index(drop=True)
        conf_df = df[~df["pair_id"].isin(disc_pairs)].head(subsample).copy().reset_index(drop=True)
        logger.info(f"Group split on pair_id: {len(disc_pairs)} pairs Discovery ({len(disc_df)} samples), {len(unique_pairs) - half_pairs} pairs Confirmation ({len(conf_df)} samples)")
    else:
        indices = rng.permutation(len(df))
        half = len(df) // 2
        disc_df = df.iloc[indices[:half]].head(subsample).copy().reset_index(drop=True)
        conf_df = df.iloc[indices[half:]].head(subsample).copy().reset_index(drop=True)
        logger.info(f"Index split: {len(disc_df)} samples Discovery, {len(conf_df)} samples Confirmation")

    candidates = build_va_candidates()

    # 2. Discovery: ピーク層および Mediator 層の自動同定（探索的 site selection）
    # 注: Discovery サブセット内での方向推定と評価は候補層スクリーニングであり、
    # 最終的な媒介推論は完全に独立した Confirmation サブセットで実行されます。
    logger.info(f"Running Discovery stage (exploratory site selection) on {len(disc_df)} samples across {num_layers} layers...")
    d_stim_profile = []
    c_gen_profile = []

    disc_texts = [str(t) for t in disc_df["text"]]
    y_v_disc = disc_df["reader_V"].values

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
        # Held-out 5-fold cross-validation R^2
        from sklearn.model_selection import KFold
        n_splits = min(5, len(H_s))
        if n_splits > 1:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            preds = np.zeros_like(y_v_disc)
            for train_idx, val_idx in kf.split(H_s):
                ridge = Ridge(alpha=10.0).fit(H_s[train_idx], y_v_disc[train_idx])
                preds[val_idx] = ridge.predict(H_s[val_idx])
            ss_res = np.sum((y_v_disc - preds)**2)
            ss_tot = np.sum((y_v_disc - np.mean(y_v_disc))**2) + 1e-6
            r2 = max(0.0, float(1.0 - ss_res / ss_tot))
        else:
            r2 = 0.0
        d_stim_profile.append(r2)

        # 実 activation intervention による因果的変位 C(l) = |Delta Report(l)| の実測 (探索的スクリーニング)
        eval_k = min(8, len(disc_texts))
        delta_reports = []
        if H_s.shape[0] >= 2 and np.std(y_v_disc) > 1e-4:
            # 層 l での Valence 方向ベクトル
            ridge_dir = Ridge(alpha=10.0).fit(H_s, y_v_disc)
            d_l = ridge_dir.coef_
            norm_d = np.linalg.norm(d_l)
            if norm_d > 1e-6:
                d_l = d_l / norm_d
            else:
                d_l = np.zeros_like(d_l)
        else:
            d_l = np.zeros(H_s.shape[1])

        with torch.no_grad():
            for k_idx in range(eval_k):
                text_k = disc_texts[k_idx]
                p_k = build_prompt(text_k, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
                enc_k = encode_prompt_canonical(tokenizer, p_k, device=device)
                anch_k = find_semantic_anchors(enc_k["input_ids"][0].tolist(), tokenizer, text_k)
                patch_pos_k = anch_k["prompt_end"]

                # a. Clean expected report
                _, probs_clean_k = compute_sequence_likelihoods_for_candidates(
                    model=model, tokenizer=tokenizer, prompt=p_k, candidates=candidates, device=device, batch_size=81
                )
                ev_c, ea_c = compute_expected_va(probs_clean_k, candidates)

                # b. Intervened expected report (alpha=1.0 along d_l)
                h_orig = H_s[k_idx]
                h_patched = h_orig + d_l * 1.0
                patch_tensor_k = torch.tensor(h_patched, dtype=torch.float32, device=device)

                with ActivationHookManager(adapter) as hook_mgr:
                    hook_mgr.register_patch_hook(
                        layer_idx=l,
                        patch_tensor=patch_tensor_k,
                        token_indices=patch_pos_k,
                        hook_point=HookPoint.POST_MLP_RESID,
                    )
                    _, probs_int_k = compute_sequence_likelihoods_for_candidates(
                        model=model, tokenizer=tokenizer, prompt=p_k, candidates=candidates, device=device, batch_size=81
                    )
                ev_i, ea_i = compute_expected_va(probs_int_k, candidates)
                delta_norm = float(np.sqrt((ev_i - ev_c)**2 + (ea_i - ea_c)**2))
                delta_reports.append(delta_norm)

        c_score = float(np.mean(delta_reports)) if delta_reports else 0.0
        c_gen_profile.append(c_score)

    stim_peak_layer = int(np.argmax(d_stim_profile))
    mediator_layer = int(np.argmax(c_gen_profile))
    logger.info(f"Discovery Result: stim_peak_layer={stim_peak_layer}, mediator_layer={mediator_layer}")

    discovery_res = {
        "stim_peak_layer": stim_peak_layer,
        "stim_peak_depth": relative_depths[stim_peak_layer],
        "mediator_layer": mediator_layer,
        "mediator_depth": relative_depths[mediator_layer],
        "d_stim_profile": [float(x) for x in d_stim_profile],
        "c_gen_profile": [float(x) for x in c_gen_profile],
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
    dirs = extract_conditional_directions(H_med, disc_df["reader_V"].values, disc_df["reader_A"].values, alpha=1.0)
    Q_sub = compute_orthonormal_subspace([dirs["direction_v"], dirs["direction_a"]])  # (D, 2)

    # matched-neutral 表現の抽出 (Discovery split)
    disc_neu_hiddens = []
    with torch.no_grad():
        for _, row in disc_df.iterrows():
            neu_text = None
            if "neutral_text" in row and str(row["neutral_text"]).strip():
                neu_text = str(row["neutral_text"])
            elif "text_neutral" in row and str(row["text_neutral"]).strip():
                neu_text = str(row["text_neutral"])
            elif "pair_id" in disc_df.columns:
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
                        layer_idx=mediator_layer,
                        hook_point=HookPoint.POST_MLP_RESID,
                        token_indices=anchors_neu["prompt_end"],
                        key="h_med_neu",
                    )
                    _ = model(**enc_neu)
                    disc_neu_hiddens.append(hook_mgr.captured_activations["h_med_neu"].cpu().float().numpy().ravel())
    if len(disc_neu_hiddens) > 0:
        mu_neu = np.mean(disc_neu_hiddens, axis=0)
    else:
        mu_neu = np.mean(H_med, axis=0)

    # Confirmation セットで自然な情動変位 (Total affective shift) と Mediator 遮断後の残差変位 (Residual shift) を実測
    te_v_list, te_a_list = [], []
    nde_v_list, nde_a_list = [], []

    with torch.no_grad():
        for _, row in conf_df.iterrows():
            text = str(row["text"])
            prompt = build_prompt(text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
            enc = encode_prompt_canonical(tokenizer, prompt, device=device)
            anchors = find_semantic_anchors(enc["input_ids"][0].tolist(), tokenizer, text)
            patch_pos = anchors["prompt_end"]

            # matched-neutral の特定と baseline 自己報告の計測
            neutral_text = None
            if "neutral_text" in row and str(row["neutral_text"]).strip():
                neutral_text = str(row["neutral_text"])
            elif "text_neutral" in row and str(row["text_neutral"]).strip():
                neutral_text = str(row["text_neutral"])
            elif "pair_id" in conf_df.columns:
                pair_matches = df[(df["pair_id"] == row["pair_id"]) & (df.get("condition", pd.Series()) == "neutral")]
                if len(pair_matches) > 0:
                    neutral_text = str(pair_matches.iloc[0]["text"])

            if neutral_text is not None and len(neutral_text.strip()) > 0:
                p_neu = build_prompt(neutral_text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
                _, probs_neu = compute_sequence_likelihoods_for_candidates(
                    model=model, tokenizer=tokenizer, prompt=p_neu, candidates=candidates, device=device, batch_size=81
                )
                ev_neu, ea_neu = compute_expected_va(probs_neu, candidates)
            else:
                # neutral baseline がない場合はエラーを送出（Primaryでは固定5.0へのサイレントfallbackは禁止）
                raise ValueError(f"Missing matched-neutral baseline for stimulus: {row.get('stimulus_id', row.get('id', 'unknown'))}")

            # a. Clean baseline (Total affective shift: 自己報告の情動変位 |ev_clean - ev_neu|)
            _, probs_clean = compute_sequence_likelihoods_for_candidates(
                model=model, tokenizer=tokenizer, prompt=prompt, candidates=candidates, device=device, batch_size=81
            )
            ev_clean, ea_clean = compute_expected_va(probs_clean, candidates)
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
                _, probs_abl = compute_sequence_likelihoods_for_candidates(
                    model=model, tokenizer=tokenizer, prompt=prompt, candidates=candidates, device=device, batch_size=81
                )
            ev_abl, ea_abl = compute_expected_va(probs_abl, candidates)
            nde_v = abs(ev_abl - ev_neu)
            nde_a = abs(ea_abl - ea_neu)
            nde_v_list.append(nde_v)
            nde_a_list.append(nde_a)

    te_samples_v = np.array(te_v_list)
    te_samples_a = np.array(te_a_list)
    nde_samples_v = np.array(nde_v_list)
    nde_samples_a = np.array(nde_a_list)

    nie_samples_v = te_samples_v - nde_samples_v
    nie_samples_a = te_samples_a - nde_samples_a

    ratio_samples_v = nie_samples_v / np.clip(te_samples_v, 1e-5, None)
    ratio_samples_a = nie_samples_a / np.clip(te_samples_a, 1e-5, None)

    te_v_mean, te_v_low, te_v_high = compute_bootstrap_ci(te_samples_v, n_boot=bootstrap_n)
    nde_v_mean, nde_v_low, nde_v_high = compute_bootstrap_ci(nde_samples_v, n_boot=bootstrap_n)
    nie_v_mean, nie_v_low, nie_v_high = compute_bootstrap_ci(nie_samples_v, n_boot=bootstrap_n)
    ratio_v_mean, ratio_v_low, ratio_v_high = compute_bootstrap_ci(ratio_samples_v, n_boot=bootstrap_n)

    te_a_mean, te_a_low, te_a_high = compute_bootstrap_ci(te_samples_a, n_boot=bootstrap_n)
    nde_a_mean, nde_a_low, nde_a_high = compute_bootstrap_ci(nde_samples_a, n_boot=bootstrap_n)
    nie_a_mean, nie_a_low, nie_a_high = compute_bootstrap_ci(nie_samples_a, n_boot=bootstrap_n)
    ratio_a_mean, ratio_a_low, ratio_a_high = compute_bootstrap_ci(ratio_samples_a, n_boot=bootstrap_n)

    confirmation_res = {
        "mediator_layer": mediator_layer,
        "valence": {
            "total_affective_shift": {"mean": te_v_mean, "ci_lower": te_v_low, "ci_upper": te_v_high},
            "residual_shift_after_blocking": {"mean": nde_v_mean, "ci_lower": nde_v_low, "ci_upper": nde_v_high},
            "mediated_attenuation": {"mean": nie_v_mean, "ci_lower": nie_v_low, "ci_upper": nie_v_high},
            "attenuation_ratio": {"mean": ratio_v_mean, "ci_lower": ratio_v_low, "ci_upper": ratio_v_high},
            # Backward compatibility aliases
            "total_effect": {"mean": te_v_mean, "ci_lower": te_v_low, "ci_upper": te_v_high},
            "natural_direct_effect": {"mean": nde_v_mean, "ci_lower": nde_v_low, "ci_upper": nde_v_high},
            "natural_indirect_effect": {"mean": nie_v_mean, "ci_lower": nie_v_low, "ci_upper": nie_v_high},
            "mediation_ratio": {"mean": ratio_v_mean, "ci_lower": ratio_v_low, "ci_upper": ratio_v_high},
        },
        "arousal": {
            "total_affective_shift": {"mean": te_a_mean, "ci_lower": te_a_low, "ci_upper": te_a_high},
            "residual_shift_after_blocking": {"mean": nde_a_mean, "ci_lower": nde_a_low, "ci_upper": nde_a_high},
            "mediated_attenuation": {"mean": nie_a_mean, "ci_lower": nie_a_low, "ci_upper": nie_a_high},
            "attenuation_ratio": {"mean": ratio_a_mean, "ci_lower": ratio_a_low, "ci_upper": ratio_a_high},
            # Backward compatibility aliases
            "total_effect": {"mean": te_a_mean, "ci_lower": te_a_low, "ci_upper": te_a_high},
            "natural_direct_effect": {"mean": nde_a_mean, "ci_lower": nde_a_low, "ci_upper": nde_a_high},
            "natural_indirect_effect": {"mean": nie_a_mean, "ci_lower": nie_a_low, "ci_upper": nie_a_high},
            "mediation_ratio": {"mean": ratio_a_mean, "ci_lower": ratio_a_low, "ci_upper": ratio_a_high},
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
    raw_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(v3_cfg["dataset"]["path"])

    bootstrap_n = v3_cfg.get("path_mediation", {}).get("eval_bootstrap_n", 1000)

    target_models = resolve_models_from_args(args, Path(args.models_config))
    if args.family and args.family.lower() in target_models:
        fam_key = args.family.lower()
        target_model_id = target_models[fam_key].instruct_model.model_id
    elif list(target_models.values()):
        fam_key = list(target_models.keys())[0]
        target_model_id = list(target_models.values())[0].instruct_model.model_id
    else:
        fam_key = "qwen"
        target_model_id = v3_cfg.get("target_model", "Qwen/Qwen2.5-1.5B-Instruct")

    num_layers = resolve_architecture_dims(target_model_id)[0]

    if args.dry_run:
        logger.info("Executing mock path mediation analysis (--dry-run specified)...")
        discovery_res = simulate_path_mediation_discovery(df.head(len(df) // 2), num_layers)
        confirmation_res = simulate_path_mediation_confirmation(
            df.tail(len(df) // 2),
            mediator_layer=discovery_res["mediator_layer"],
            bootstrap_n=bootstrap_n,
        )
    else:
        logger.info(f"Executing REAL path mediation analysis on {target_model_id}...")
        discovery_res, confirmation_res = run_real_path_mediation(
            df=df,
            model_id=target_model_id,
            device=args.device,
            subsample=args.subsample,
            bootstrap_n=bootstrap_n,
        )

    out_raw = raw_dir / f"v3_path_mediation_{fam_key}.json"
    full_output = {
        "model_id": target_model_id,
        "family": fam_key,
        "num_layers": num_layers,
        "discovery": discovery_res,
        "confirmation": confirmation_res,
    }
    with open(out_raw, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    logger.info(f"Saved path mediation raw results to {out_raw}")

    out_summary = derived_dir / "v3_path_mediation_summary.json"
    summary_output = {
        "mediator_layer": confirmation_res["mediator_layer"],
        "valence_mediation_ratio": confirmation_res["valence"]["mediation_ratio"]["mean"],
        "valence_mediation_ci": [
            confirmation_res["valence"]["mediation_ratio"]["ci_lower"],
            confirmation_res["valence"]["mediation_ratio"]["ci_upper"],
        ],
        "arousal_mediation_ratio": confirmation_res["arousal"]["mediation_ratio"]["mean"],
        "arousal_mediation_ci": [
            confirmation_res["arousal"]["mediation_ratio"]["ci_lower"],
            confirmation_res["arousal"]["mediation_ratio"]["ci_upper"],
        ],
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, indent=2)
    logger.info(f"Saved path mediation summary to {out_summary}")

    logger.info(f"Valence Mediation Ratio: {summary_output['valence_mediation_ratio']:.3f} "
                f"(95% CI: [{summary_output['valence_mediation_ci'][0]:.3f}, {summary_output['valence_mediation_ci'][1]:.3f}])")


if __name__ == "__main__":
    main()
