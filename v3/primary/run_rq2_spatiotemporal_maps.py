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
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import Ridge, LinearRegression

from affective_empathy_eval.geometry import compute_layer_dissociation
from affective_empathy_eval.interventions import estimate_interventional_slope
from affective_empathy_eval.likelihood import (
    build_va_candidates,
    compute_expected_va,
    compute_sequence_likelihoods_for_candidates,
)
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run V3 Spatiotemporal 4-Map Analysis")
    parser.add_argument("--config", type=str, default="configs/v3_experiments.yaml", help="Path to V3 config")
    parser.add_argument("--models-config", type=str, default="configs/models.yaml", help="Path to models config")
    parser.add_argument("--dry-run", action="store_true", help="Run in mock/dry-run mode")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--subsample", type=int, default=30, help="Number of pairs to evaluate across layers x stages")
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
        "maps": {
            "D_V": D_V.tolist(),
            "D_A": D_A.tolist(),
            "beta_V": beta_V.tolist(),
            "beta_A": beta_A.tolist(),
            "gamma_V": gamma_V.tolist(),
            "gamma_A": gamma_A.tolist(),
            "C_V": C_V.tolist(),
            "C_A": C_A.tolist(),
        },
        "dissociation_summary": {
            "valence": dissoc_v,
            "arousal": dissoc_a,
        }
    }


def run_real_spatiotemporal_maps(
    df: pd.DataFrame,
    model_id: str,
    semantic_stages: List[str],
    alpha_sweep: List[float],
    device: str = "cpu",
    subsample: int = 30,
) -> Dict[str, Any]:
    """
    実モデルを用いた時空間 4-Map 解析 (Layer x Stage Grid)
    1. 各サンプルについてプロンプト + 生成候補を構築
    2. 全層・全アンカー位置での活性化を抽出
    3. D (Held-out Ridge R^2), beta (刺激共変量を統制した偏回帰係数),
       gamma (介入応答スロープ), C (因果変位量) を算出
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
    fam_cfg = registry.get_family(model_id)
    adapter = get_model_adapter(model, fam_cfg)
    num_layers = fam_cfg.num_layers
    relative_depths = [l / (num_layers - 1) if num_layers > 1 else 0.0 for l in range(num_layers)]
    num_stages = len(semantic_stages)

    eval_df = df.head(subsample).copy().reset_index(drop=True)
    N = len(eval_df)
    logger.info(f"Evaluating {N} samples across {num_layers} layers x {num_stages} semantic stages...")

    candidates = build_va_candidates()

    D_V = np.zeros((num_layers, num_stages))
    D_A = np.zeros((num_layers, num_stages))
    beta_V = np.zeros((num_layers, num_stages))
    beta_A = np.zeros((num_layers, num_stages))
    gamma_V = np.zeros((num_layers, num_stages))
    gamma_A = np.zeros((num_layers, num_stages))
    C_V = np.zeros((num_layers, num_stages))
    C_A = np.zeros((num_layers, num_stages))

    # 各サンプルの Clean baseline 自己報告値を取得
    clean_ev_list = []
    clean_ea_list = []
    clean_prompt_tokens = []
    sample_stage_indices = []

    with torch.no_grad():
        for _, row in eval_df.iterrows():
            text = str(row["text"])
            prompt = build_prompt(text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
            enc = encode_prompt_canonical(tokenizer, prompt, device=device)
            p_ids = enc["input_ids"][0].tolist()
            clean_prompt_tokens.append(p_ids)

            # 生成時ステージ位置の特定
            sample_cand = candidates[40]["json_str"]  # 代表候補: {"valence": 5, "arousal": 5}
            cand_tokens = tokenizer.encode(sample_cand, add_special_tokens=False)
            stages = get_generation_stage_tokens(cand_tokens, tokenizer, candidate_str=sample_cand)

            # プロンプト終端からの相対オフセットに変換
            p_end = len(p_ids) - 1
            stage_map = {
                "candidate_start": p_end + 1,
                "pre_V": p_end + 1 + stages["pre_V"],
                "V_value": p_end + 1 + stages["V_value"],
                "pre_A": p_end + 1 + stages["pre_A"],
                "A_value": p_end + 1 + stages["A_value"],
                "response_end": p_end + 1 + stages["response_end"],
            }
            sample_stage_indices.append(stage_map)

            _, probs = compute_sequence_likelihoods_for_candidates(
                model=model, tokenizer=tokenizer, prompt=prompt, candidates=candidates, device=device, batch_size=81
            )
            ev, ea = compute_expected_va(probs, candidates)
            clean_ev_list.append(ev)
            clean_ea_list.append(ea)

    y_v = np.array(clean_ev_list)
    y_a = np.array(clean_ea_list)
    covar_v = eval_df["reader_V"].values
    covar_a = eval_df["reader_A"].values

    # 層 × ステージ グリッド解析
    for l in range(num_layers):
        logger.info(f"Computing 4-Map for Layer {l}/{num_layers}...")
        for s_idx, stage_name in enumerate(semantic_stages):
            # 1. 活性化の抽出
            h_stage = []
            with torch.no_grad():
                for i, row in eval_df.iterrows():
                    text = str(row["text"])
                    prompt = build_prompt(text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
                    cand = candidates[40]["json_str"]
                    full_text = prompt + cand
                    enc_full = encode_prompt_canonical(tokenizer, full_text, device=device)
                    t_idx = sample_stage_indices[i].get(stage_name, enc_full["input_ids"].shape[1] - 1)
                    t_idx = min(t_idx, enc_full["input_ids"].shape[1] - 1)

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

            # 2. デコード能 D: Held-out Ridge R^2 (5-fold CV)
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            r2_v_folds, r2_a_folds = [], []
            preds_v, preds_a = np.zeros(N), np.zeros(N)
            for tr, te in kf.split(H):
                ridge_v = Ridge(alpha=10.0).fit(H[tr], y_v[tr])
                ridge_a = Ridge(alpha=10.0).fit(H[tr], y_a[tr])
                preds_v[te] = ridge_v.predict(H[te])
                preds_a[te] = ridge_a.predict(H[te])
            r2_v = max(0.0, float(1.0 - np.sum((y_v - preds_v)**2) / (np.sum((y_v - np.mean(y_v))**2) + 1e-6)))
            r2_a = max(0.0, float(1.0 - np.sum((y_a - preds_a)**2) / (np.sum((y_a - np.mean(y_a))**2) + 1e-6)))
            D_V[l, s_idx] = r2_v
            D_A[l, s_idx] = r2_a

            # 3. 偏回帰係数 beta: 刺激ラベル covar を共変量として統制した内部予測スコアの寄与度
            # y = beta_0 + beta * s_pred + gamma * covar
            X_cov_v = np.column_stack([preds_v, covar_v])
            X_cov_a = np.column_stack([preds_a, covar_a])
            reg_v = LinearRegression().fit(X_cov_v, y_v)
            reg_a = LinearRegression().fit(X_cov_a, y_a)
            beta_V[l, s_idx] = float(abs(reg_v.coef_[0]))
            beta_A[l, s_idx] = float(abs(reg_a.coef_[0]))

            # 4. 介入傾き gamma & 因果変位量 C (代表刺激 5 サンプルで高速測定)
            sample_gamma_v, sample_c_v = [], []
            sample_gamma_a, sample_c_a = [], []
            probe_dir_v = Ridge(alpha=10.0).fit(H, y_v).coef_
            norm_v = np.linalg.norm(probe_dir_v)
            d_v = probe_dir_v / (norm_v + 1e-6) if norm_v > 0 else np.zeros_like(probe_dir_v)

            probe_dir_a = Ridge(alpha=10.0).fit(H, y_a).coef_
            norm_a = np.linalg.norm(probe_dir_a)
            d_a = probe_dir_a / (norm_a + 1e-6) if norm_a > 0 else np.zeros_like(probe_dir_a)

            h_std = float(np.std(H @ d_v)) if np.std(H @ d_v) > 0 else 1.0

            sub_eval_idx = list(range(min(5, N)))
            with torch.no_grad():
                for idx in sub_eval_idx:
                    text = str(eval_df.loc[idx, "text"])
                    prompt = build_prompt(text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
                    t_idx = sample_stage_indices[idx].get(stage_name, len(clean_prompt_tokens[idx]) - 1)
                    t_idx = min(t_idx, len(clean_prompt_tokens[idx]) - 1)

                    shifts_v, shifts_a = [], []
                    for alpha in alpha_sweep:
                        p_vec = torch.tensor(alpha * h_std * d_v, dtype=torch.float32, device=device)
                        with ActivationHookManager(adapter) as hook_mgr:
                            hook_mgr.register_patch_hook(
                                layer_idx=l,
                                patch_tensor=p_vec,
                                token_indices=t_idx,
                                hook_point=HookPoint.POST_MLP_RESID,
                            )
                            _, probs_p = compute_sequence_likelihoods_for_candidates(
                                model=model, tokenizer=tokenizer, prompt=prompt, candidates=candidates, device=device, batch_size=81
                            )
                        ev_p, ea_p = compute_expected_va(probs_p, candidates)
                        shifts_v.append(ev_p - clean_ev_list[idx])
                        shifts_a.append(ea_p - clean_ea_list[idx])

                    sl_v = estimate_interventional_slope(alpha_sweep, shifts_v)
                    sl_a = estimate_interventional_slope(alpha_sweep, shifts_a)
                    sample_gamma_v.append(sl_v)
                    sample_gamma_a.append(sl_a)
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
        "maps": {
            "D_V": D_V.tolist(),
            "D_A": D_A.tolist(),
            "beta_V": beta_V.tolist(),
            "beta_A": beta_A.tolist(),
            "gamma_V": gamma_V.tolist(),
            "gamma_A": gamma_A.tolist(),
            "C_V": C_V.tolist(),
            "C_A": C_A.tolist(),
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

    df = pd.read_csv(v3_cfg["dataset"]["path"])
    semantic_stages = v3_cfg["spatiotemporal"]["semantic_stages"]
    alpha_sweep = v3_cfg["spatiotemporal"]["alpha_sweep"]

    # 命名の正規化 ("response_start" -> "candidate_start")
    normalized_stages = [s if s != "response_start" else "candidate_start" for s in semantic_stages]

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
        logger.info("Executing mock spatiotemporal 4-map generation (--dry-run specified)...")
        results = simulate_spatiotemporal_maps(num_layers, normalized_stages, alpha_sweep)
    else:
        logger.info(f"Executing REAL spatiotemporal 4-map calculation on {target_model_id}...")
        results = run_real_spatiotemporal_maps(
            df=df,
            model_id=target_model_id,
            semantic_stages=normalized_stages,
            alpha_sweep=alpha_sweep,
            device=args.device,
            subsample=args.subsample,
        )

    out_raw = raw_dir / f"v3_spatiotemporal_maps_{fam_key}.json"
    with open(out_raw, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved spatiotemporal 4-maps to {out_raw}")

    out_summary = derived_dir / "v3_spatiotemporal_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(results["dissociation_summary"], f, indent=2)
    logger.info(f"Saved dissociation summary to {out_summary}")

    logger.info(f"Valence Dissociation Delta Peak: {results['dissociation_summary']['valence']['delta_d_peak']:.3f}")
    logger.info(f"Arousal Dissociation Delta Peak: {results['dissociation_summary']['arousal']['delta_d_peak']:.3f}")


if __name__ == "__main__":
    main()

