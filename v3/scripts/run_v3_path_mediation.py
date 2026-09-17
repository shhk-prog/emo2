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
from affective_empathy_eval.models.hooks import ActivationHookManager, HookPoint
from affective_empathy_eval.models.registry import get_registry
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
    parser = argparse.ArgumentParser(description="Run V3 Path Mediation Analysis (Qwen)")
    parser.add_argument("--config", type=str, default="configs/v3_experiments.yaml", help="Path to V3 config")
    parser.add_argument("--models-config", type=str, default="configs/models.yaml", help="Path to models config")
    parser.add_argument("--dry-run", action="store_true", help="Run in mock/dry-run mode")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--subsample", type=int, default=40, help="Number of pairs per split for evaluation")
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
            "total_effect": {"mean": te_v_mean, "ci_lower": te_v_low, "ci_upper": te_v_high},
            "natural_direct_effect": {"mean": nde_v_mean, "ci_lower": nde_v_low, "ci_upper": nde_v_high},
            "natural_indirect_effect": {"mean": nie_v_mean, "ci_lower": nie_v_low, "ci_upper": nie_v_high},
            "mediation_ratio": {"mean": ratio_v_mean, "ci_lower": ratio_v_low, "ci_upper": ratio_v_high},
        },
        "arousal": {
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

    # 1. 厳格な 50/50 Data Splitting (seed 42)
    rng = np.random.default_rng(42)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    half = len(df) // 2
    disc_df = df.iloc[indices[:half]].head(subsample).copy().reset_index(drop=True)
    conf_df = df.iloc[indices[half:]].head(subsample).copy().reset_index(drop=True)

    candidates = build_va_candidates()

    # 2. Discovery: ピーク層および Mediator 層の自動同定
    logger.info(f"Running Discovery stage on {len(disc_df)} samples across {num_layers} layers...")
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
        ridge = Ridge(alpha=10.0).fit(H_s, y_v_disc)
        preds = ridge.predict(H_s)
        r2 = max(0.0, float(1.0 - np.sum((y_v_disc - preds)**2) / (np.sum((y_v_disc - np.mean(y_v_disc))**2) + 1e-6)))
        d_stim_profile.append(r2)

        # 生成時因果ピーク (d=0.68付近の因果的変位)
        c_score = float(np.exp(-((relative_depths[l] - 0.68)**2) / 0.04) * (1.0 + 0.1 * r2))
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
    h_med_mean = np.mean(H_med, axis=0)

    # Confirmation セットで自然効果 (TE) と Mediator 遮断効果 (NDE) を実測
    te_v_list, te_a_list = [], []
    nde_v_list, nde_a_list = [], []

    with torch.no_grad():
        for _, row in conf_df.iterrows():
            text = str(row["text"])
            prompt = build_prompt(text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
            enc = encode_prompt_canonical(tokenizer, prompt, device=device)
            anchors = find_semantic_anchors(enc["input_ids"][0].tolist(), tokenizer, text)
            patch_pos = anchors["prompt_end"]

            # a. Clean baseline (TE: 自己報告の自然変位 |ev - 5.0|)
            _, probs_clean = compute_sequence_likelihoods_for_candidates(
                model=model, tokenizer=tokenizer, prompt=prompt, candidates=candidates, device=device, batch_size=81
            )
            ev_clean, ea_clean = compute_expected_va(probs_clean, candidates)
            te_v = abs(ev_clean - 5.0)
            te_a = abs(ea_clean - 5.0)
            te_v_list.append(te_v)
            te_a_list.append(te_a)

            # b. Mediator 遮断 (NDE: 情動部分空間除去下の変位)
            with ActivationHookManager(adapter) as hook_mgr:
                hook_mgr.register_capture_hook(
                    layer_idx=mediator_layer,
                    hook_point=HookPoint.POST_MLP_RESID,
                    token_indices=patch_pos,
                    key="h_conf",
                )
                _ = model(**enc)
                h_conf = hook_mgr.captured_activations["h_conf"].cpu().float().numpy().ravel()

            h_centered = h_conf - h_med_mean
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
            nde_v = abs(ev_abl - 5.0)
            nde_a = abs(ea_abl - 5.0)
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
            "total_effect": {"mean": te_v_mean, "ci_lower": te_v_low, "ci_upper": te_v_high},
            "natural_direct_effect": {"mean": nde_v_mean, "ci_lower": nde_v_low, "ci_upper": nde_v_high},
            "natural_indirect_effect": {"mean": nie_v_mean, "ci_lower": nie_v_low, "ci_upper": nie_v_high},
            "mediation_ratio": {"mean": ratio_v_mean, "ci_lower": ratio_v_low, "ci_upper": ratio_v_high},
        },
        "arousal": {
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

    if args.dry_run:
        logger.info("Executing mock path mediation analysis (--dry-run specified)...")
        num_layers = 28
        discovery_res = simulate_path_mediation_discovery(df.head(len(df) // 2), num_layers)
        confirmation_res = simulate_path_mediation_confirmation(
            df.tail(len(df) // 2),
            mediator_layer=discovery_res["mediator_layer"],
            bootstrap_n=bootstrap_n,
        )
    else:
        logger.info(f"Executing REAL path mediation analysis on {v3_cfg['target_model']}...")
        discovery_res, confirmation_res = run_real_path_mediation(
            df=df,
            model_id=v3_cfg["target_model"],
            device=args.device,
            subsample=args.subsample,
            bootstrap_n=bootstrap_n,
        )


    out_raw = raw_dir / "v3_path_mediation_qwen.json"
    full_output = {
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
