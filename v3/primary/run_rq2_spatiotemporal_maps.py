#!/usr/bin/env python3
"""
V3-RQ2 & RQ3: 意味的生成アンカー × 層グリッドにおける時空間 4-Map 解析
モデル: Qwen 2.5 (1.5B) Instruct
時空間グリッド: 6 意味的アンカー × 28 層
4-Map × 2軸:
  - デコード能 D_V, D_A (Held-out Ridge R^2)
  - 偏回帰係数 beta_V, beta_A (刺激共変量を統制した internal score -> report の偏回帰係数)
  - 介入傾き gamma_V, gamma_A (alpha-sweep による因果的応答傾き)
  - 因果変位量 C_V, C_A (介入による自己報告分布変位)
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
from sklearn.linear_model import Ridge, LinearRegression

from affective_empathy_eval.geometry import compute_layer_dissociation
from affective_empathy_eval.interventions import estimate_interventional_slope
from affective_empathy_eval.likelihood import (
    build_va_candidates,
    compute_expected_va,
    compute_sequence_likelihoods_for_candidates,
    prepare_joint_sequence_with_boundary,
    resolve_joint_stage_index,
)
from affective_empathy_eval.manifests import create_run_manifest
from affective_empathy_eval.data import (
    describe_loaded_frame,
    load_v3_matched_pair_table,
    v3_stimulus_covariate,
)
from affective_empathy_eval.models.adapters import get_model_adapter
from affective_empathy_eval.models.hooks import ActivationHookManager, HookPoint
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run V3 Spatiotemporal 4-Map Analysis")
    parser.add_argument("--config", type=str, default="configs/v3_experiments.yaml", help="Path to V3 config")
    parser.add_argument("--models-config", type=str, default="configs/models.yaml", help="Path to models config")
    parser.add_argument("--dry-run", action="store_true", help="Run in mock/dry-run mode")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--subsample", type=int, default=0, help="Number of pairs to evaluate across layers x stages (0 for full dataset)")
    add_model_selection_args(parser)
    return parser.parse_args()


def simulate_spatiotemporal_maps(
    num_layers: int,
    semantic_stages: List[str],
    alpha_sweep: List[float],
) -> Dict[str, Any]:
    """
    dry-run用: 時空間グリッド (num_layers × num_stages) 上の 4-Map × 2軸を生成・シミュレート
    論文中の発見（刺激提示時は中間層でデコードピーク、生成時は後期層 pre_V で因果ピーク）を反映
    """
    rng = np.random.default_rng(42)
    relative_depths = [l / (num_layers - 1) if num_layers > 1 else 0.0 for l in range(num_layers)]
    num_stages = len(semantic_stages)

    # 4-Map の初期化 (Layer × Stage)
    D_V = np.zeros((num_layers, num_stages))
    D_A = np.zeros((num_layers, num_stages))
    beta_V = np.zeros((num_layers, num_stages))
    beta_A = np.zeros((num_layers, num_stages))
    gamma_V = np.zeros((num_layers, num_stages))
    gamma_A = np.zeros((num_layers, num_stages))
    C_V = np.zeros((num_layers, num_stages))
    C_A = np.zeros((num_layers, num_stages))

    # 意味的アンカーの重み付け (pre_V, pre_A で因果性が最大化)
    # stages: ["candidate_start", "pre_V", "V_value", "pre_A", "A_value", "response_end"]
    stage_causal_weights_v = [0.15, 1.0, 0.6, 0.2, 0.1, 0.05]
    stage_causal_weights_a = [0.15, 0.3, 0.2, 1.0, 0.6, 0.05]

    for l_idx, d in enumerate(relative_depths):
        # デコード能 D: 中間層 (d ≈ 0.45 ~ 0.55) で最大
        base_d_v = np.exp(-((d - 0.48) ** 2) / (2 * 0.18**2)) * 0.72 + rng.normal(0, 0.02)
        base_d_a = np.exp(-((d - 0.50) ** 2) / (2 * 0.20**2)) * 0.65 + rng.normal(0, 0.02)

        # 偏回帰係数 beta: 刺激共変量を統制した内部スコアの寄与度
        base_beta_v = np.exp(-((d - 0.52) ** 2) / (2 * 0.22**2)) * 0.68 + rng.normal(0, 0.02)
        base_beta_a = np.exp(-((d - 0.54) ** 2) / (2 * 0.22**2)) * 0.61 + rng.normal(0, 0.02)

        for s_idx, stage in enumerate(semantic_stages):
            decay = 1.0 - 0.08 * s_idx
            D_V[l_idx, s_idx] = max(0.0, float(base_d_v * decay))
            D_A[l_idx, s_idx] = max(0.0, float(base_d_a * decay))

            beta_V[l_idx, s_idx] = max(0.0, float(base_beta_v * decay))
            beta_A[l_idx, s_idx] = max(0.0, float(base_beta_a * decay))

            # 介入傾き gamma: 後期層 (d ≈ 0.65 ~ 0.75) かつ pre_V / pre_A で最大化
            peak_gamma_layer = 0.68
            layer_gamma_v = np.exp(-((d - peak_gamma_layer) ** 2) / (2 * 0.15**2)) * 0.88
            layer_gamma_a = np.exp(-((d - peak_gamma_layer) ** 2) / (2 * 0.15**2)) * 0.78
            gamma_V[l_idx, s_idx] = float(layer_gamma_v * stage_causal_weights_v[s_idx] + rng.normal(0, 0.02))
            gamma_A[l_idx, s_idx] = float(layer_gamma_a * stage_causal_weights_a[s_idx] + rng.normal(0, 0.02))

            # 因果変位量 C: 後期層 (d ≈ 0.68) かつ pre_V / pre_A で最大化
            layer_c_v = np.exp(-((d - 0.68) ** 2) / (2 * 0.14**2)) * 1.15
            layer_c_a = np.exp(-((d - 0.68) ** 2) / (2 * 0.14**2)) * 0.98
            C_V[l_idx, s_idx] = float(layer_c_v * stage_causal_weights_v[s_idx] + rng.normal(0, 0.03))
            C_A[l_idx, s_idx] = float(layer_c_a * stage_causal_weights_a[s_idx] + rng.normal(0, 0.03))

    pre_v_idx = semantic_stages.index("pre_V") if "pre_V" in semantic_stages else 1
    d_vals_v = D_V[:, pre_v_idx].tolist()
    c_vals_v = C_V[:, pre_v_idx].tolist()
    dissoc_v = compute_layer_dissociation(relative_depths, d_vals_v, c_vals_v)

    pre_a_idx = semantic_stages.index("pre_A") if "pre_A" in semantic_stages else 3
    d_vals_a = D_A[:, pre_a_idx].tolist()
    c_vals_a = C_A[:, pre_a_idx].tolist()
    dissoc_a = compute_layer_dissociation(relative_depths, d_vals_a, c_vals_a)


    return {
        "num_layers": num_layers,
        "semantic_stages": semantic_stages,
        "relative_depths": relative_depths,
        "primary_grounding": "reader_prediction (simulated)",
        "maps": {
            "D_V": D_V.tolist(),
            "D_A": D_A.tolist(),
            "beta_V": beta_V.tolist(),
            "beta_A": beta_A.tolist(),
            "abs_beta_V": np.abs(beta_V).tolist(),
            "abs_beta_A": np.abs(beta_A).tolist(),
            "gamma_V": gamma_V.tolist(),
            "gamma_A": gamma_A.tolist(),
            "C_V": C_V.tolist(),
            "C_A": C_A.tolist(),
        },
        "secondary_maps": {
            "D_V_self": D_V.tolist(),
            "D_A_self": D_A.tolist(),
        },
        "dissociation_summary": {
            "valence": dissoc_v,
            "arousal": dissoc_a,
        },
        "n_total_samples": 32,
        "n_map_samples": 32,
        "n_intervene_samples": 15,
        "n_causal_intervention_samples": 15,
        "dry_run": True,
    }


def run_real_spatiotemporal_maps(
    df: pd.DataFrame,
    model_id: str,
    semantic_stages: List[str],
    alpha_sweep: List[float],
    device: str = "cpu",
    subsample: int = 0,
    n_causal_samples: int = 15,
) -> Dict[str, Any]:
    """
    実モデルを用いた時空間 4-Map 解析 (Layer x Stage Grid)
    1. 各サンプルについてプロンプト + 生成候補を構築
    2. 全層・全アンカー位置での活性化を抽出
    3. D (Held-out Ridge R^2), beta (刺激共変量を統制した偏回帰係数),
       gamma (介入応答スロープ), C (因果変位量) を算出
       Primary: Reader Prediction (感情認知予測値) に基づくデコード能および情動方向
       Secondary: Self-Report (自己報告値) に基づくデコード能
    """
    logger.info(f"Loading model {model_id} for Spatiotemporal 4-Map Analysis on {device}...")
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
    num_stages = len(semantic_stages)

    if subsample is not None and subsample > 0:
        eval_df = df.head(subsample).copy().reset_index(drop=True)
    else:
        eval_df = df.copy().reset_index(drop=True)
    N = len(eval_df)
    logger.info(f"Evaluating {N} samples across {num_layers} layers x {num_stages} semantic stages (n_causal_samples={n_causal_samples})...")

    candidates = build_va_candidates()

    D_V = np.zeros((num_layers, num_stages))
    D_A = np.zeros((num_layers, num_stages))
    secondary_D_V = np.zeros((num_layers, num_stages))
    secondary_D_A = np.zeros((num_layers, num_stages))
    beta_V = np.zeros((num_layers, num_stages))
    beta_A = np.zeros((num_layers, num_stages))
    abs_beta_V = np.zeros((num_layers, num_stages))
    abs_beta_A = np.zeros((num_layers, num_stages))
    gamma_V = np.zeros((num_layers, num_stages))
    gamma_A = np.zeros((num_layers, num_stages))
    C_V = np.zeros((num_layers, num_stages))
    C_A = np.zeros((num_layers, num_stages))

    # 各サンプルの Clean baseline 自己報告値および Reader Prediction を取得
    clean_ev_list = []
    clean_ea_list = []
    reader_ev_list = []
    reader_ea_list = []
    sample_prompts: List[str] = []
    sample_joint_meta: List[dict[str, Any]] = []
    template_cand = candidates[40]["json_str"]  # {"valence": 5, "arousal": 5}

    with torch.no_grad():
        for _, row in eval_df.iterrows():
            text = str(row["text"])
            prompt = build_prompt(text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
            sample_prompts.append(prompt)
            full_ids, cand_start = prepare_joint_sequence_with_boundary(
                prompt=prompt, candidate=template_cand, tokenizer=tokenizer
            )
            cand_tokens = tokenizer.encode(template_cand, add_special_tokens=False)
            stages = get_generation_stage_tokens(cand_tokens, tokenizer, candidate_str=template_cand)
            sample_joint_meta.append(
                {
                    "full_ids": full_ids,
                    "cand_start": cand_start,
                    "stage_offsets": stages,
                    "seq_len": len(full_ids),
                }
            )

            # a. Self condition: Clean expected report (baseline for causal shifts)
            _, probs = compute_sequence_likelihoods_for_candidates(
                model=model, tokenizer=tokenizer, prompt=prompt, candidates=candidates, device=device, batch_size=81
            )
            ev, ea = compute_expected_va(probs, candidates)
            clean_ev_list.append(ev)
            clean_ea_list.append(ea)

            # b. Reader condition: Reader Prediction (stimulus emotion perception)
            prompt_reader = build_prompt(text, task=TaskType.READER, format_type="chat", tokenizer=tokenizer)
            _, r_probs = compute_sequence_likelihoods_for_candidates(
                model=model, tokenizer=tokenizer, prompt=prompt_reader, candidates=candidates, device=device, batch_size=81
            )
            r_ev, r_ea = compute_expected_va(r_probs, candidates)
            reader_ev_list.append(r_ev)
            reader_ea_list.append(r_ea)

    # Primary targets: Reader Predictions (grounded in emotion recognition)
    if "reader_V" in eval_df.columns and "reader_A" in eval_df.columns:
        y_v = eval_df["reader_V"].to_numpy()
        y_a = eval_df["reader_A"].to_numpy()
    else:
        y_v = np.array(reader_ev_list, dtype=np.float64)
        y_a = np.array(reader_ea_list, dtype=np.float64)

    # Secondary targets: Self-report predictions
    y_v_self = np.array(clean_ev_list, dtype=np.float64)
    y_a_self = np.array(clean_ea_list, dtype=np.float64)

    covar_v = v3_stimulus_covariate(eval_df, "v", N)
    covar_a = v3_stimulus_covariate(eval_df, "a", N)

    # 固定シードによる再現可能な因果介入サンプル選択 (全層・全ステージで共通の一貫したサブセット)
    n_causal_intervene = min(n_causal_samples, N)
    rng_causal = np.random.default_rng(42)
    sub_eval_idx = sorted(rng_causal.choice(N, size=n_causal_intervene, replace=False).tolist())
    logger.info(f"Selected {len(sub_eval_idx)} seeded random samples for causal intervention evaluation.")

    # 層 × ステージ グリッド解析
    for l in range(num_layers):
        logger.info(f"Computing 4-Map for Layer {l}/{num_layers}...")
        for s_idx, stage_name in enumerate(semantic_stages):
            # 1. 活性化の抽出
            h_stage = []
            with torch.no_grad():
                for i in range(N):
                    meta = sample_joint_meta[i]
                    t_idx = resolve_joint_stage_index(
                        meta["cand_start"], stage_name, meta["stage_offsets"], meta["seq_len"]
                    )
                    enc_full = {
                        "input_ids": torch.tensor([meta["full_ids"]], device=device),
                        "attention_mask": torch.ones(1, meta["seq_len"], dtype=torch.long, device=device),
                    }
                    with ActivationHookManager(adapter) as hook_mgr:
                        hook_mgr.register_capture_hook(
                            layer_idx=l,
                            hook_point=HookPoint.POST_MLP_RESID,
                            token_indices=t_idx,
                            key="h",
                        )
                        _ = model(**enc_full)
                        h_vec = hook_mgr.captured_activations["h"].cpu().float().numpy().ravel()
                        h_stage.append(h_vec)

            H = np.array(h_stage)  # (N, D)

            # 2. デコード能 D: Held-out Ridge R^2 (GroupKFold CV by pair_id if available)
            if "pair_id" in eval_df.columns:
                from sklearn.model_selection import GroupKFold
                groups = eval_df["pair_id"].values
                n_groups = len(np.unique(groups))
                n_splits = min(3, n_groups)
                if n_splits > 1:
                    cv = GroupKFold(n_splits=n_splits)
                    cv_splits = list(cv.split(H, y_v, groups=groups))
                else:
                    from sklearn.model_selection import KFold
                    cv_splits = list(KFold(n_splits=min(3, N), shuffle=True, random_state=42).split(H))
            else:
                from sklearn.model_selection import KFold
                cv_splits = list(KFold(n_splits=min(3, N), shuffle=True, random_state=42).split(H))

            preds_v, preds_a = np.zeros(N), np.zeros(N)
            preds_v_self, preds_a_self = np.zeros(N), np.zeros(N)
            for tr, te in cv_splits:
                # Primary: fit on Reader Prediction
                ridge_v = Ridge(alpha=10.0).fit(H[tr], y_v[tr])
                ridge_a = Ridge(alpha=10.0).fit(H[tr], y_a[tr])
                preds_v[te] = ridge_v.predict(H[te])
                preds_a[te] = ridge_a.predict(H[te])
                # Secondary: fit on Self Report
                ridge_v_s = Ridge(alpha=10.0).fit(H[tr], y_v_self[tr])
                ridge_a_s = Ridge(alpha=10.0).fit(H[tr], y_a_self[tr])
                preds_v_self[te] = ridge_v_s.predict(H[te])
                preds_a_self[te] = ridge_a_s.predict(H[te])

            r2_v = max(0.0, float(1.0 - np.sum((y_v - preds_v)**2) / (np.sum((y_v - np.mean(y_v))**2) + 1e-6)))
            r2_a = max(0.0, float(1.0 - np.sum((y_a - preds_a)**2) / (np.sum((y_a - np.mean(y_a))**2) + 1e-6)))
            D_V[l, s_idx] = r2_v
            D_A[l, s_idx] = r2_a

            r2_v_s = max(0.0, float(1.0 - np.sum((y_v_self - preds_v_self)**2) / (np.sum((y_v_self - np.mean(y_v_self))**2) + 1e-6)))
            r2_a_s = max(0.0, float(1.0 - np.sum((y_a_self - preds_a_self)**2) / (np.sum((y_a_self - np.mean(y_a_self))**2) + 1e-6)))
            secondary_D_V[l, s_idx] = r2_v_s
            secondary_D_A[l, s_idx] = r2_a_s

            # 3. 偏回帰係数 beta: 刺激ラベル covar を共変量として統制した内部予測スコアの寄与度
            # y = beta_0 + beta * s_pred + gamma * covar
            X_cov_v = np.column_stack([preds_v, covar_v])
            X_cov_a = np.column_stack([preds_a, covar_a])
            reg_v = LinearRegression().fit(X_cov_v, y_v)
            reg_a = LinearRegression().fit(X_cov_a, y_a)
            beta_V[l, s_idx] = float(reg_v.coef_[0])
            beta_A[l, s_idx] = float(reg_a.coef_[0])
            abs_beta_V[l, s_idx] = float(abs(reg_v.coef_[0]))
            abs_beta_A[l, s_idx] = float(abs(reg_a.coef_[0]))

            # 4. 介入: d_V → gamma_V,C_V / d_A → gamma_A,C_A（生成段階は joint sequence 上）
            sample_gamma_v, sample_c_v = [], []
            sample_gamma_a, sample_c_a = [], []
            probe_dir_v = Ridge(alpha=10.0).fit(H, y_v).coef_
            norm_v = np.linalg.norm(probe_dir_v)
            d_v = probe_dir_v / (norm_v + 1e-6) if norm_v > 0 else np.zeros_like(probe_dir_v)

            probe_dir_a = Ridge(alpha=10.0).fit(H, y_a).coef_
            norm_a = np.linalg.norm(probe_dir_a)
            d_a = probe_dir_a / (norm_a + 1e-6) if norm_a > 0 else np.zeros_like(probe_dir_a)

            h_std_v = float(np.std(H @ d_v)) if np.std(H @ d_v) > 0 else 1.0
            h_std_a = float(np.std(H @ d_a)) if np.std(H @ d_a) > 0 else 1.0

            with torch.no_grad():
                for idx in sub_eval_idx:
                    prompt = sample_prompts[idx]
                    meta = sample_joint_meta[idx]
                    t_idx = resolve_joint_stage_index(
                        meta["cand_start"], stage_name, meta["stage_offsets"], meta["seq_len"]
                    )

                    shifts_v, shifts_a = [], []
                    for axis_name, direction, h_std, shift_store in (
                        ("v", d_v, h_std_v, shifts_v),
                        ("a", d_a, h_std_a, shifts_a),
                    ):
                        axis_shifts = []
                        for alpha in alpha_sweep:
                            p_vec = torch.tensor(alpha * h_std * direction, dtype=torch.float32, device=device)
                            _, probs_p = compute_sequence_likelihoods_for_candidates(
                                model=model,
                                tokenizer=tokenizer,
                                prompt=prompt,
                                candidates=candidates,
                                device=device,
                                batch_size=81,
                                generation_patch={
                                    "adapter": adapter,
                                    "layer_idx": l,
                                    "patch_tensor": p_vec,
                                    "token_index": t_idx,
                                    "hook_point": HookPoint.POST_MLP_RESID,
                                },
                            )
                            ev_p, ea_p = compute_expected_va(probs_p, candidates)
                            if axis_name == "v":
                                axis_shifts.append(ev_p - clean_ev_list[idx])
                            else:
                                axis_shifts.append(ea_p - clean_ea_list[idx])
                        shift_store.extend(axis_shifts)

                    sample_gamma_v.append(estimate_interventional_slope(alpha_sweep, shifts_v))
                    sample_gamma_a.append(estimate_interventional_slope(alpha_sweep, shifts_a))
                    sample_c_v.append(abs(shifts_v[-1]))
                    sample_c_a.append(abs(shifts_a[-1]))

            gamma_V[l, s_idx] = float(np.mean(sample_gamma_v))
            gamma_A[l, s_idx] = float(np.mean(sample_gamma_a))
            C_V[l, s_idx] = float(np.mean(sample_c_v))
            C_A[l, s_idx] = float(np.mean(sample_c_a))

    pre_v_idx = semantic_stages.index("pre_V") if "pre_V" in semantic_stages else 1
    d_vals_v = D_V[:, pre_v_idx].tolist()
    c_vals_v = C_V[:, pre_v_idx].tolist()
    dissoc_v = compute_layer_dissociation(relative_depths, d_vals_v, c_vals_v)

    pre_a_idx = semantic_stages.index("pre_A") if "pre_A" in semantic_stages else 3
    d_vals_a = D_A[:, pre_a_idx].tolist()
    c_vals_a = C_A[:, pre_a_idx].tolist()
    dissoc_a = compute_layer_dissociation(relative_depths, d_vals_a, c_vals_a)


    return {
        "num_layers": num_layers,
        "semantic_stages": semantic_stages,
        "relative_depths": relative_depths,
        "primary_grounding": "reader_prediction",
        "n_total_samples": len(df),
        "n_map_samples": N,
        "n_intervene_samples": len(sub_eval_idx),
        "n_causal_intervention_samples": len(sub_eval_idx),
        "maps": {
            "D_V": D_V.tolist(),
            "D_A": D_A.tolist(),
            "beta_V": beta_V.tolist(),
            "beta_A": beta_A.tolist(),
            "abs_beta_V": abs_beta_V.tolist(),
            "abs_beta_A": abs_beta_A.tolist(),
            "gamma_V": gamma_V.tolist(),
            "gamma_A": gamma_A.tolist(),
            "C_V": C_V.tolist(),
            "C_A": C_A.tolist(),
        },
        "secondary_maps": {
            "D_V_self": secondary_D_V.tolist(),
            "D_A_self": secondary_D_A.tolist(),
        },
        "dissociation_summary": {
            "valence": dissoc_v,
            "arousal": dissoc_a,
        }
    }


def main():
    args = parse_args()
    logger.info(f"Starting V3 Spatiotemporal 4-Map Analysis (dry_run={args.dry_run})")

    with open(args.config, "r", encoding="utf-8") as f:
        v3_cfg = yaml.safe_load(f)

    raw_dir = Path(v3_cfg["output"]["raw_dir"])
    derived_dir = Path(v3_cfg["output"]["derived_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    df = load_v3_matched_pair_table(v3_cfg["dataset"]["path"])
    logger.info(describe_loaded_frame(df, "V3-RQ2 matched-pair table", v3_cfg["dataset"]["path"]))
    semantic_stages = v3_cfg["spatiotemporal"]["semantic_stages"]
    alpha_sweep = v3_cfg["spatiotemporal"]["alpha_sweep"]
    n_causal_cfg = int(v3_cfg.get("spatiotemporal", {}).get("n_causal_samples", 15))

    # 命名の正規化 ("response_start" -> "candidate_start")
    normalized_stages = [s if s != "response_start" else "candidate_start" for s in semantic_stages]

    fam_key, target_model_id = resolve_instruct_target_from_args(
        args,
        Path(args.models_config),
        fallback_family=v3_cfg.get("target_family"),
    )

    num_layers = resolve_architecture_dims(target_model_id)[0]
    results = None

    out_raw = raw_dir / f"v3_discovery_spatiotemporal_maps_{fam_key}.json"
    if out_raw.exists() and not args.dry_run:
        try:
            with open(out_raw, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached and "maps" in cached:
                logger.info(f"Loaded existing discovery 4-maps from {out_raw}. Skipping computation.")
                results = cached
        except Exception:
            pass

    if results is None:
        if args.dry_run:
            logger.info("Executing mock spatiotemporal 4-map generation (--dry-run specified)...")
            results = simulate_spatiotemporal_maps(num_layers, normalized_stages, alpha_sweep)
        else:
            logger.info(f"Executing REAL spatiotemporal 4-map calculation on {target_model_id} (n_causal_samples={n_causal_cfg})...")
            results = run_real_spatiotemporal_maps(
                df=df,
                model_id=target_model_id,
                semantic_stages=normalized_stages,
                alpha_sweep=alpha_sweep,
                device=args.device,
                subsample=args.subsample,
                n_causal_samples=n_causal_cfg,
            )

        n_dataset_total = int(len(df))
        n_map_samples = int(results.get("n_map_samples", (args.subsample if args.subsample and args.subsample > 0 else len(df))))
        n_causal_intervene = int(results.get("n_causal_intervention_samples", min(n_causal_cfg, n_map_samples)))

        results["dry_run"] = bool(args.dry_run)
        results["analysis_role"] = "discovery"
        results["n_dataset_total"] = n_dataset_total
        results["n_map_samples"] = n_map_samples
        results["n_intervene_samples"] = n_causal_intervene
        results["n_causal_intervention_samples"] = n_causal_intervene

        with open(out_raw, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved discovery spatiotemporal 4-maps to {out_raw}")

    # Save manifest with explicit intervention sample count
    manifest = create_run_manifest(
        run_type="v3_rq2_discovery_spatiotemporal_maps",
        model_name=target_model_id,
        config={
            "analysis_role": "discovery",
            "family": fam_key,
            "semantic_stages": normalized_stages,
            "alpha_sweep": alpha_sweep,
            "subsample": args.subsample,
            "dry_run": bool(args.dry_run),
        },
        metadata={
            "analysis_role": "discovery",
            "n_dataset_total": int(len(df)),
            "n_map_samples": int(results.get("n_map_samples", len(df))),
            "n_intervene_samples": int(results.get("n_intervene_samples", min(5, len(df)))),
            "n_causal_intervention_samples": int(results.get("n_causal_intervention_samples", min(5, len(df)))),
            "dissociation_summary": results["dissociation_summary"],
        },
    )
    manifest.save(raw_dir / f"manifest_rq2_{fam_key}.json")
    logger.info(f"Saved RQ2 manifest to {raw_dir / f'manifest_rq2_{fam_key}.json'}")

    out_summary = derived_dir / "v3_spatiotemporal_summary.json"
    summary_payload = {
        "analysis_role": "discovery",
        **results["dissociation_summary"],
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    logger.info(f"Saved dissociation summary to {out_summary}")

    logger.info(f"Valence Dissociation Delta Peak: {results['dissociation_summary']['valence']['delta_d_peak']:.3f}")
    logger.info(f"Arousal Dissociation Delta Peak: {results['dissociation_summary']['arousal']['delta_d_peak']:.3f}")


if __name__ == "__main__":
    main()

