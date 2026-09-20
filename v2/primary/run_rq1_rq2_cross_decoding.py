#!/usr/bin/env python3
"""
v2/primary/run_rq1_rq2_cross_decoding.py

V2-RQ1 & RQ2: Post-training による表現幾何の変化と Reader–Self 共有性の再編
4モデルファミリー (Qwen 2.5, Llama 3.2, Gemma 3, OLMo 2) × 2水準 (Base, Instruct)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # --dry-run は transformers 未導入環境でも起動できるようにする
    AutoModelForCausalLM = None  # type: ignore[misc, assignment]
    AutoTokenizer = None  # type: ignore[misc, assignment]

from affective_empathy_eval.data import describe_loaded_frame, dry_run_va_label_vector
from affective_empathy_eval.manifests import (
    DEFAULT_CODE_VERSION,
    compute_string_or_dict_hash,
    create_run_manifest,
    is_manifest_matching,
)
from affective_empathy_eval.geometry import (
    compute_center_of_mass,
    compute_peak_depth,
    compute_relative_depth,
    compute_rsa_correlation,
    eval_held_out_cross_decoding,
    eval_held_out_procrustes,
    train_and_eval_held_out_probe,
)
from affective_empathy_eval.models.adapters import get_model_adapter
from affective_empathy_eval.models.hooks import ActivationHookManager, HookPoint
from affective_empathy_eval.models.registry import (
    add_model_selection_args,
    get_registry,
    resolve_models_from_args,
)
from affective_empathy_eval.prompts import (
    TaskType,
    build_prompt,
    encode_prompt_canonical,
    find_semantic_anchors,
)
from affective_empathy_eval.statistics import compute_bootstrap_ci, paired_family_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run V2-RQ1 & RQ2 Cross-decoding and Geometry Analysis")
    parser.add_argument("--config", type=str, default="configs/v2_experiments.yaml", help="Path to V2 config")
    parser.add_argument("--models-config", type=str, default="configs/models.yaml", help="Path to models config")
    parser.add_argument("--dry-run", action="store_true", help="Run in mock/dry-run mode without loading full weights")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples for quick testing")
    parser.add_argument("--force", action="store_true", help="Force recomputation even if valid cached results exist")
    add_model_selection_args(parser)
    return parser.parse_args()


def load_dataset(csv_path: str, max_samples: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    logger.info(describe_loaded_frame(df, "V2-RQ1/RQ2 dataset", csv_path))
    if max_samples is not None and len(df) > max_samples:
        df = df.iloc[:max_samples].copy()
        logger.info(f"Applied max_samples={max_samples}: n_rows={len(df)}")
    return df


def split_dataset(df: pd.DataFrame, train_ratio: float = 0.7, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    if "pair_id" in df.columns:
        # Group Split by pair_id: 同一ヴィネット・最小対が train/test に跨る交絡を完全に排除
        unique_pairs = df["pair_id"].unique()
        rng.shuffle(unique_pairs)
        n_train_pairs = int(len(unique_pairs) * train_ratio)
        train_pairs = set(unique_pairs[:n_train_pairs])
        train_mask = df["pair_id"].isin(train_pairs)
        train_df = df[train_mask].copy().reset_index(drop=True)
        test_df = df[~train_mask].copy().reset_index(drop=True)
        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError(f"Train ({len(train_df)}) or test ({len(test_df)}) split is empty in pair-aware split.")
        return train_df, test_df
    else:
        n = len(df)
        indices = np.arange(n)
        rng.shuffle(indices)
        n_train = int(n * train_ratio)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        train_df = df.iloc[train_indices].copy().reset_index(drop=True)
        test_df = df.iloc[test_indices].copy().reset_index(drop=True)
        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError(f"Train ({len(train_df)}) or test ({len(test_df)}) split is empty in permutation split.")
        return train_df, test_df


def extract_activations_for_model(
    model: Any,
    tokenizer: Any,
    df: pd.DataFrame,
    task: TaskType,
    format_type: str,
    device: str = "cpu",
    is_dry_run: bool = False,
    hidden_dim: int = 1536,
    num_layers: int = 28,
) -> dict[int, np.ndarray]:
    """
    データセットの全刺激文に対して、各層の残差ストリーム活性化（意味的アンカー: prompt_end）を抽出
    戻り値: {layer_idx: (N, hidden_dim)}
    """
    N = len(df)
    if is_dry_run:
        # モック活性化（Valence / Arousal と相関を持たせた合成表現）
        rng = np.random.default_rng(42)
        v_vals = dry_run_va_label_vector(df, "reader_V", N)
        activations = {}
        for l in range(num_layers):
            rel_d = l / max(1, num_layers - 1)
            # 中間層で相関が高くなる合成プロファイル
            signal_strength = np.exp(-((rel_d - 0.5) ** 2) / 0.08)
            v_dir = rng.standard_normal(hidden_dim)
            v_dir /= np.linalg.norm(v_dir)
            noise = rng.standard_normal((N, hidden_dim)) * 0.5
            h = np.outer(v_vals, v_dir) * signal_strength + noise
            activations[l] = h.astype(np.float32)
        return activations

    adapter = get_model_adapter(model)
    actual_layers = adapter.get_num_layers()
    layer_acts: dict[int, list[np.ndarray]] = {l: [] for l in range(actual_layers)}

    model.eval()
    with torch.no_grad():
        for _, row in df.iterrows():
            text = str(row["text"])
            prompt = build_prompt(text=text, task=task, format_type=format_type, tokenizer=tokenizer)
            enc = encode_prompt_canonical(tokenizer, prompt, device=device)
            input_ids = enc["input_ids"]

            anchors = find_semantic_anchors(enc["input_ids"][0].tolist(), tokenizer, text)
            anchor_pos = anchors["prompt_end"]  # Aligned with RQ3 causal patching token position

            with ActivationHookManager(adapter) as hook_mgr:
                for l in range(actual_layers):
                    hook_mgr.register_capture_hook(
                        layer_idx=l,
                        hook_point=HookPoint.POST_MLP_RESID,
                        token_indices=anchor_pos,
                        key=f"layer_{l}",
                    )
                _ = model(input_ids)
                for l in range(actual_layers):
                    act = hook_mgr.captured_activations[f"layer_{l}"].squeeze(1).float().cpu().numpy()[0]
                    layer_acts[l].append(act)

    return {l: np.array(acts, dtype=np.float32) for l, acts in layer_acts.items()}


def analyze_v2_geometry_and_sharing(
    acts_train: dict[str, dict[int, np.ndarray]],
    acts_test: dict[str, dict[int, np.ndarray]],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_layers: int,
    ridge_alpha: float = 1.0,
) -> dict[str, Any]:
    """
    V2-RQ1 & RQ2 の包括的幾何・共有性解析 (Valence & Arousal の2軸対応)
    acts keys: 'base_reader', 'base_self', 'inst_reader', 'inst_self',
    および matched_plain がある場合は 'inst_matched_reader', 'inst_matched_self'
    """
    targets = {}
    if "reader_V" in train_df.columns and "reader_V" in test_df.columns:
        targets["valence"] = (train_df["reader_V"].values, test_df["reader_V"].values)
    else:
        targets["valence"] = (np.zeros(len(train_df)), np.zeros(len(test_df)))

    if "reader_A" in train_df.columns and "reader_A" in test_df.columns:
        targets["arousal"] = (train_df["reader_A"].values, test_df["reader_A"].values)

    has_matched = "inst_matched_reader" in acts_train and "inst_matched_self" in acts_train

    results: dict[str, Any] = {
        "rq1_geometry": {
            "rsa_metric": "rsa_similarity",
            # 1. Base plain <-> Instruct native-chat (Secondary Comparison)
            "reader_distortion": [],
            "self_distortion": [],
            "rsa_reader": [],
            "rsa_self": [],
        },
        "rq2_sharing": {},
    }

    if has_matched:
        # 2. Base plain <-> Instruct matched-plain (Primary Comparison: Prompt-format-controlled Base-Instruct)
        results["rq1_geometry"]["reader_distortion_matched"] = []
        results["rq1_geometry"]["self_distortion_matched"] = []
        results["rq1_geometry"]["rsa_reader_matched"] = []
        results["rq1_geometry"]["rsa_self_matched"] = []
        # 3. Instruct chat <-> Instruct plain (Prompt format / Chat template control)
        results["rq1_geometry"]["reader_distortion_format"] = []
        results["rq1_geometry"]["self_distortion_format"] = []
        results["rq1_geometry"]["rsa_reader_format"] = []
        results["rq1_geometry"]["rsa_self_format"] = []

    for axis in targets.keys():
        results["rq2_sharing"][axis] = {
            "base_r2_reader": [],
            "base_r2_self": [],
            "base_cross_r_to_s": [],
            "base_cross_s_to_r": [],
            "base_sharing": [],  # (R^2_{BR->BS} + R^2_{BS->BR}) / 2
            "inst_r2_reader": [],
            "inst_r2_self": [],
            "inst_cross_r_to_s": [],
            "inst_cross_s_to_r": [],
            "inst_sharing": [],  # (R^2_{IR->IS} + R^2_{IS->IR}) / 2
            "delta_sharing": [],  # Primary: Inst_Sharing - Base_Sharing
            "delta_delta_cross": [],  # Secondary
        }
        if has_matched:
            results["rq2_sharing"][axis].update({
                "inst_matched_r2_reader": [],
                "inst_matched_r2_self": [],
                "inst_matched_cross_r_to_s": [],
                "inst_matched_cross_s_to_r": [],
                "inst_matched_sharing": [],
                "delta_sharing_matched": [],  # Inst_Matched_Sharing - Base_Sharing (純事後学習)
                "delta_sharing_format": [],   # Inst_Sharing - Inst_Matched_Sharing (プロンプト効果)
            })

    depths = [compute_relative_depth(l, num_layers) for l in range(num_layers)]

    logger.info(f"Running held-out geometry & cross-decoding analysis across {num_layers} layers (has_matched={has_matched})...")
    for l in range(num_layers):
        if l % 5 == 0 or l == num_layers - 1:
            logger.info(f"  Evaluating layer {l}/{num_layers - 1} (d={depths[l]:.2f})...")

        # 1. V2-RQ1 幾何歪み (Procrustes) & RSA: Base ↔ Instruct Chat
        proc_r = eval_held_out_procrustes(
            acts_train["base_reader"][l], acts_train["inst_reader"][l],
            acts_test["base_reader"][l], acts_test["inst_reader"][l],
        )
        proc_s = eval_held_out_procrustes(
            acts_train["base_self"][l], acts_train["inst_self"][l],
            acts_test["base_self"][l], acts_test["inst_self"][l],
        )
        rsa_r = compute_rsa_correlation(acts_test["base_reader"][l], acts_test["inst_reader"][l])
        rsa_s = compute_rsa_correlation(acts_test["base_self"][l], acts_test["inst_self"][l])

        results["rq1_geometry"]["reader_distortion"].append(proc_r)
        results["rq1_geometry"]["self_distortion"].append(proc_s)
        results["rq1_geometry"]["rsa_reader"].append(rsa_r)
        results["rq1_geometry"]["rsa_self"].append(rsa_s)

        # Matched plain 制御条件の幾何解析
        if has_matched:
            # Base ↔ Inst Matched
            proc_r_m = eval_held_out_procrustes(
                acts_train["base_reader"][l], acts_train["inst_matched_reader"][l],
                acts_test["base_reader"][l], acts_test["inst_matched_reader"][l],
            )
            proc_s_m = eval_held_out_procrustes(
                acts_train["base_self"][l], acts_train["inst_matched_self"][l],
                acts_test["base_self"][l], acts_test["inst_matched_self"][l],
            )
            rsa_r_m = compute_rsa_correlation(acts_test["base_reader"][l], acts_test["inst_matched_reader"][l])
            rsa_s_m = compute_rsa_correlation(acts_test["base_self"][l], acts_test["inst_matched_self"][l])

            results["rq1_geometry"]["reader_distortion_matched"].append(proc_r_m)
            results["rq1_geometry"]["self_distortion_matched"].append(proc_s_m)
            results["rq1_geometry"]["rsa_reader_matched"].append(rsa_r_m)
            results["rq1_geometry"]["rsa_self_matched"].append(rsa_s_m)

            # Inst Chat ↔ Inst Matched (Format Effect)
            proc_r_f = eval_held_out_procrustes(
                acts_train["inst_matched_reader"][l], acts_train["inst_reader"][l],
                acts_test["inst_matched_reader"][l], acts_test["inst_reader"][l],
            )
            proc_s_f = eval_held_out_procrustes(
                acts_train["inst_matched_self"][l], acts_train["inst_self"][l],
                acts_test["inst_matched_self"][l], acts_test["inst_self"][l],
            )
            rsa_r_f = compute_rsa_correlation(acts_test["inst_matched_reader"][l], acts_test["inst_reader"][l])
            rsa_s_f = compute_rsa_correlation(acts_test["inst_matched_self"][l], acts_test["inst_self"][l])

            results["rq1_geometry"]["reader_distortion_format"].append(proc_r_f)
            results["rq1_geometry"]["self_distortion_format"].append(proc_s_f)
            results["rq1_geometry"]["rsa_reader_format"].append(rsa_r_f)
            results["rq1_geometry"]["rsa_self_format"].append(rsa_s_f)

        # 2. V2-RQ2 Reader-Self 共有性 (Held-out Decodability & Cross-decoding)
        for axis, (y_tr, y_te) in targets.items():
            base_r_r2 = train_and_eval_held_out_probe(
                acts_train["base_reader"][l], y_tr, acts_test["base_reader"][l], y_te, alpha=ridge_alpha
            )
            base_s_r2 = train_and_eval_held_out_probe(
                acts_train["base_self"][l], y_tr, acts_test["base_self"][l], y_te, alpha=ridge_alpha
            )
            base_r_to_s = eval_held_out_cross_decoding(
                acts_train["base_reader"][l], y_tr, acts_test["base_self"][l], y_te, alpha=ridge_alpha
            )
            base_s_to_r = eval_held_out_cross_decoding(
                acts_train["base_self"][l], y_tr, acts_test["base_reader"][l], y_te, alpha=ridge_alpha
            )
            base_share = float((base_r_to_s + base_s_to_r) / 2.0)

            inst_r_r2 = train_and_eval_held_out_probe(
                acts_train["inst_reader"][l], y_tr, acts_test["inst_reader"][l], y_te, alpha=ridge_alpha
            )
            inst_s_r2 = train_and_eval_held_out_probe(
                acts_train["inst_self"][l], y_tr, acts_test["inst_self"][l], y_te, alpha=ridge_alpha
            )
            inst_r_to_s = eval_held_out_cross_decoding(
                acts_train["inst_reader"][l], y_tr, acts_test["inst_self"][l], y_te, alpha=ridge_alpha
            )
            inst_s_to_r = eval_held_out_cross_decoding(
                acts_train["inst_self"][l], y_tr, acts_test["inst_reader"][l], y_te, alpha=ridge_alpha
            )
            inst_share = float((inst_r_to_s + inst_s_to_r) / 2.0)
            delta_share = float(inst_share - base_share)

            # Secondary 指標: 双方向 cross-decoding の変化差 (Reader->Self の変化 vs Self->Reader の変化)
            dd = float((inst_r_to_s - base_r_to_s) - (inst_s_to_r - base_s_to_r))

            res_axis = results["rq2_sharing"][axis]
            res_axis["base_r2_reader"].append(base_r_r2)
            res_axis["base_r2_self"].append(base_s_r2)
            res_axis["base_cross_r_to_s"].append(base_r_to_s)
            res_axis["base_cross_s_to_r"].append(base_s_to_r)
            res_axis["base_sharing"].append(base_share)

            res_axis["inst_r2_reader"].append(inst_r_r2)
            res_axis["inst_r2_self"].append(inst_s_r2)
            res_axis["inst_cross_r_to_s"].append(inst_r_to_s)
            res_axis["inst_cross_s_to_r"].append(inst_s_to_r)
            res_axis["inst_sharing"].append(inst_share)
            res_axis["delta_sharing"].append(delta_share)
            res_axis["delta_delta_cross"].append(dd)

            if has_matched:
                inst_m_r_r2 = train_and_eval_held_out_probe(
                    acts_train["inst_matched_reader"][l], y_tr, acts_test["inst_matched_reader"][l], y_te, alpha=ridge_alpha
                )
                inst_m_s_r2 = train_and_eval_held_out_probe(
                    acts_train["inst_matched_self"][l], y_tr, acts_test["inst_matched_self"][l], y_te, alpha=ridge_alpha
                )
                inst_m_r_to_s = eval_held_out_cross_decoding(
                    acts_train["inst_matched_reader"][l], y_tr, acts_test["inst_matched_self"][l], y_te, alpha=ridge_alpha
                )
                inst_m_s_to_r = eval_held_out_cross_decoding(
                    acts_train["inst_matched_self"][l], y_tr, acts_test["inst_matched_reader"][l], y_te, alpha=ridge_alpha
                )
                inst_m_share = float((inst_m_r_to_s + inst_m_s_to_r) / 2.0)

                res_axis["inst_matched_r2_reader"].append(inst_m_r_r2)
                res_axis["inst_matched_r2_self"].append(inst_m_s_r2)
                res_axis["inst_matched_cross_r_to_s"].append(inst_m_r_to_s)
                res_axis["inst_matched_cross_s_to_r"].append(inst_m_s_to_r)
                res_axis["inst_matched_sharing"].append(inst_m_share)
                res_axis["delta_sharing_matched"].append(float(inst_m_share - base_share))
                res_axis["delta_sharing_format"].append(float(inst_share - inst_m_share))

    # 重心とピーク
    primary_axis = "valence" if "valence" in targets else list(targets.keys())[0]
    p_sharing = results["rq2_sharing"][primary_axis]

    # Primary summary metrics (Matched Plain: Base plain vs Instruct matched-plain)
    if has_matched and "inst_matched_cross_r_to_s" in p_sharing:
        primary_matched = {
            "com_distortion_reader": compute_center_of_mass(results["rq1_geometry"].get("reader_distortion_matched", results["rq1_geometry"]["reader_distortion"]), depths),
            "com_distortion_self": compute_center_of_mass(results["rq1_geometry"].get("self_distortion_matched", results["rq1_geometry"]["self_distortion"]), depths),
            "peak_depth_base_cross": compute_peak_depth(p_sharing["base_cross_r_to_s"], depths),
            "peak_depth_inst_cross": compute_peak_depth(p_sharing["inst_matched_cross_r_to_s"], depths),
            "peak_depth_delta_share": compute_peak_depth(p_sharing["delta_sharing_matched"], depths),
            "mean_delta_sharing": float(np.mean(p_sharing["delta_sharing_matched"])),
        }
    else:
        primary_matched = {
            "com_distortion_reader": compute_center_of_mass(results["rq1_geometry"]["reader_distortion"], depths),
            "com_distortion_self": compute_center_of_mass(results["rq1_geometry"]["self_distortion"], depths),
            "peak_depth_base_cross": compute_peak_depth(p_sharing["base_cross_r_to_s"], depths),
            "peak_depth_inst_cross": compute_peak_depth(p_sharing["inst_cross_r_to_s"], depths),
            "peak_depth_delta_share": compute_peak_depth(p_sharing["delta_sharing"], depths),
            "mean_delta_sharing": float(np.mean(p_sharing["delta_sharing"])),
        }

    # Secondary summary metrics (Native Chat)
    secondary_native = {
        "com_distortion_reader": compute_center_of_mass(results["rq1_geometry"]["reader_distortion"], depths),
        "com_distortion_self": compute_center_of_mass(results["rq1_geometry"]["self_distortion"], depths),
        "peak_depth_base_cross": compute_peak_depth(p_sharing["base_cross_r_to_s"], depths),
        "peak_depth_inst_cross": compute_peak_depth(p_sharing["inst_cross_r_to_s"], depths),
        "peak_depth_delta_share": compute_peak_depth(p_sharing["delta_sharing"], depths),
        "mean_delta_sharing": float(np.mean(p_sharing["delta_sharing"])),
    }

    # Format Confound Effect (Native Chat - Matched Plain)
    format_effect = {
        "mean_delta_sharing_format": float(np.mean(p_sharing["delta_sharing_format"])) if has_matched and "delta_sharing_format" in p_sharing else 0.0,
    }

    results["summary_metrics"] = {
        "primary_matched_plain": primary_matched,
        "secondary_native_chat": secondary_native,
        "format_effect": format_effect,
        # Flatten primary matched-plain metrics to root for backward compatibility
        **primary_matched,
        "mean_delta_sharing_matched": primary_matched["mean_delta_sharing"],
        "mean_delta_sharing_format": format_effect["mean_delta_sharing_format"],
    }

    results["relative_depths"] = [float(d) for d in depths]
    results["num_layers"] = num_layers

    return results


def main():
    args = parse_args()
    logger.info(f"Starting V2-RQ1 & RQ2 analysis (dry_run={args.dry_run}, device={args.device})")

    # 設定読み込み
    with open(args.config, "r", encoding="utf-8") as f:
        v2_config = yaml.safe_load(f)

    target_models = resolve_models_from_args(args, Path(args.models_config))
    data_path = Path(v2_config["dataset"]["path"])
    seed = int(v2_config["seed"])

    dataset_df = load_dataset(str(data_path), max_samples=args.max_samples)
    if args.dry_run:
        dataset_df = dataset_df.head(32).copy()
        logger.info(f"[DRY-RUN] Scaled down dataset to N={len(dataset_df)} for fast smoke testing.")
    train_df, test_df = split_dataset(
        dataset_df,
        train_ratio=v2_config["dataset"]["train_ratio"],
        seed=seed,
    )
    logger.info(f"Dataset loaded: total={len(dataset_df)}, train={len(train_df)}, held_out_test={len(test_df)}")

    target_families = list(target_models.keys())
    raw_dir = Path(v2_config["output"]["raw_dir"])
    derived_dir = Path(v2_config["output"]["derived_dir"])
    if args.dry_run:
        raw_dir = raw_dir / "dry_run"
        derived_dir = derived_dir / "dry_run"
    raw_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    all_family_results = {}

    from affective_empathy_eval.io import save_experiment_result, is_experiment_completed

    for fam_id, fam_cfg in target_models.items():
        fam_out_path = raw_dir / f"v2_geometry_{fam_id}.json"
        rq1_out_path = raw_dir / f"v2_rq1_decodability_preservation_{fam_id}.json"
        rq2_out_path = raw_dir / f"v2_rq2_geometry_transformation_{fam_id}.json"
        manifest_path = raw_dir / f"manifest_geometry_{fam_id}.json"
        config_payload = {
            "family_id": fam_id,
            "base_model": fam_cfg.base_model.model_id,
            "base_revision": fam_cfg.base_model.revision,
            "instruct_model": fam_cfg.instruct_model.model_id,
            "instruct_revision": fam_cfg.instruct_model.revision,
            "seed": seed,
            "max_samples": args.max_samples,
            "dry_run": bool(args.dry_run),
            "dtype": getattr(fam_cfg, "inference_dtype", "bfloat16"),
        }
        exp_cfg_hash = compute_string_or_dict_hash(config_payload)
        exp_ds_hash = compute_file_hash(data_path) if data_path.exists() else compute_string_or_dict_hash(str(data_path))

        if not args.force and not args.dry_run and manifest_path.exists():
            from affective_empathy_eval.manifests import is_manifest_matching
            manifest_valid = is_manifest_matching(
                manifest_path=str(manifest_path),
                expected_config_hash=exp_cfg_hash,
                expected_dataset_hash=exp_ds_hash,
                expected_dry_run=False,
            )
            if manifest_valid:
                check_rq1 = rq1_out_path if rq1_out_path.exists() else fam_out_path
                check_rq2 = rq2_out_path if rq2_out_path.exists() else fam_out_path
                if is_experiment_completed(str(check_rq1), manifest_path=str(manifest_path)) and is_experiment_completed(str(check_rq2), manifest_path=str(manifest_path)):
                    try:
                        if fam_out_path.exists():
                            with open(fam_out_path, "r", encoding="utf-8") as f:
                                cached_res = json.load(f)
                        else:
                            with open(rq1_out_path, "r", encoding="utf-8") as f1, open(rq2_out_path, "r", encoding="utf-8") as f2:
                                cached_res = {**json.load(f1), **json.load(f2)}
                        logger.info(f"Loaded existing validated results matching manifest for {fam_id}. Skipping computation.")
                        all_family_results[fam_id] = cached_res
                        continue
                    except Exception as e:
                        logger.warning(f"Cache check failed for {fam_id}: {e}")


        eff_num_layers = min(fam_cfg.num_layers, 4) if args.dry_run else fam_cfg.num_layers
        eff_hidden_dim = min(fam_cfg.hidden_dim, 64) if args.dry_run else fam_cfg.hidden_dim

        logger.info(f"--- Processing Family: {fam_id} ({fam_cfg.family_name}, {eff_num_layers} layers, dim={eff_hidden_dim}) ---")

        # 抽出条件: Base/Instruct × Reader/Self
        conditions = [
            ("base", "reader", TaskType.READER, "plain"),
            ("base", "self", TaskType.SELF, "plain"),
            ("inst", "reader", TaskType.READER, "chat"),
            ("inst", "self", TaskType.SELF, "chat"),
        ]
        # 設定に matched_plain が含まれる場合は Instruct plain 条件も追加
        formats_cfg = v2_config.get("formats", ["native"])
        if "matched_plain" in formats_cfg:
            conditions.extend([
                ("inst_matched", "reader", TaskType.READER, "plain"),
                ("inst_matched", "self", TaskType.SELF, "plain"),
            ])

        acts_train = {}
        acts_test = {}

        for align_prefix, task_name, task_type, default_fmt in conditions:
            key = f"{align_prefix}_{task_name}"
            model_spec = fam_cfg.get_model_spec("base" if "base" in align_prefix else "instruct")
            logger.info(f"Extracting activations for {key} (model={model_spec.model_id}, format={default_fmt})...")

            if args.dry_run:
                model = None
                tokenizer = None
            else:
                logger.info(f"Loading weights for {model_spec.model_id} onto {args.device}...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_spec.model_id,
                    revision=model_spec.revision,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_spec.model_id,
                    revision=model_spec.revision,
                    torch_dtype=torch.bfloat16 if "cuda" in args.device else torch.float32,
                    device_map=args.device if "cuda" in args.device else None,
                )

            acts_train[key] = extract_activations_for_model(
                model=model,
                tokenizer=tokenizer,
                df=train_df,
                task=task_type,
                format_type=default_fmt,
                device=args.device,
                is_dry_run=args.dry_run,
                hidden_dim=eff_hidden_dim,
                num_layers=eff_num_layers,
            )
            acts_test[key] = extract_activations_for_model(
                model=model,
                tokenizer=tokenizer,
                df=test_df,
                task=task_type,
                format_type=default_fmt,
                device=args.device,
                is_dry_run=args.dry_run,
                hidden_dim=eff_hidden_dim,
                num_layers=eff_num_layers,
            )

            if model is not None:
                del model
                if "cuda" in args.device:
                    torch.cuda.empty_cache()

        # 幾何・共有性解析
        fam_res = analyze_v2_geometry_and_sharing(
            acts_train=acts_train,
            acts_test=acts_test,
            train_df=train_df,
            test_df=test_df,
            num_layers=eff_num_layers,
            ridge_alpha=v2_config["decoding"]["ridge_alpha"],
        )
        fam_res["dry_run"] = bool(args.dry_run)
        all_family_results[fam_id] = fam_res

        # 各ファミリーごとの結果保存 (モジュラー分離)
        rq1_payload = {
            "family_id": fam_id,
            "relative_depths": fam_res["relative_depths"],
            "num_layers": fam_res["num_layers"],
            "rq1_geometry": fam_res.get("rq1_geometry", {}),
            "summary_metrics": {
                "primary_matched_plain": {
                    "com_distortion_reader": fam_res["summary_metrics"]["primary_matched_plain"]["com_distortion_reader"],
                    "com_distortion_self": fam_res["summary_metrics"]["primary_matched_plain"]["com_distortion_self"],
                }
            },
        }
        rq2_payload = {
            "family_id": fam_id,
            "relative_depths": fam_res["relative_depths"],
            "num_layers": fam_res["num_layers"],
            "rq2_sharing": fam_res.get("rq2_sharing", {}),
            "summary_metrics": fam_res["summary_metrics"],
        }

        save_experiment_result(
            output_path=str(rq1_out_path),
            payload=rq1_payload,
            stage="v2",
            experiment_id="v2_rq1_decodability_preservation",
            status="success",
            success=True,
            metadata={"family_id": fam_id, "dry_run": bool(args.dry_run)},
        )
        save_experiment_result(
            output_path=str(rq2_out_path),
            payload=rq2_payload,
            stage="v2",
            experiment_id="v2_rq2_geometry_transformation",
            status="success",
            success=True,
            metadata={"family_id": fam_id, "dry_run": bool(args.dry_run)},
        )

        # 互換用 combined 保存
        fam_out_path = raw_dir / f"v2_geometry_{fam_id}.json"
        fam_out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fam_out_path, "w", encoding="utf-8") as f:
            json.dump(fam_res, f, indent=2)
        logger.info(f"Saved family results to {fam_out_path}, {rq1_out_path}, and {rq2_out_path}")

        # Manifest 保存
        manifest = create_run_manifest(
            run_type="v2_geometry",
            model_name=fam_cfg.family_name,
            config=config_payload,
            metadata={"num_layers": eff_num_layers, "hidden_dim": eff_hidden_dim},
            dataset_path=str(data_path),
            candidate_space="N/A",
            measurement_space="prompt_end_hidden_state",
            seed=seed,
            dry_run=bool(args.dry_run),
        )
        manifest.save(str(raw_dir / f"manifest_geometry_{fam_id}.json"))


    # 4ファミリー統合要約（Bootstrap CI & Paired comparison）
    summary_path = derived_dir / "v2_cross_family_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # 指標リストの収集 (Item 6: Primary matched-plain metrics for cross-family bootstrap)
    fams = list(all_family_results.keys())

    # 1) Primary: Matched Plain
    com_dist_r_prim = [all_family_results[f]["summary_metrics"]["primary_matched_plain"]["com_distortion_reader"] for f in fams]
    com_dist_s_prim = [all_family_results[f]["summary_metrics"]["primary_matched_plain"]["com_distortion_self"] for f in fams]
    pk_base_cross_prim = [all_family_results[f]["summary_metrics"]["primary_matched_plain"]["peak_depth_base_cross"] for f in fams]
    pk_inst_cross_prim = [all_family_results[f]["summary_metrics"]["primary_matched_plain"]["peak_depth_inst_cross"] for f in fams]
    mean_delta_share_prim = [all_family_results[f]["summary_metrics"]["primary_matched_plain"]["mean_delta_sharing"] for f in fams]

    # 2) Secondary: Native Chat
    com_dist_r_sec = [all_family_results[f]["summary_metrics"]["secondary_native_chat"]["com_distortion_reader"] for f in fams]
    com_dist_s_sec = [all_family_results[f]["summary_metrics"]["secondary_native_chat"]["com_distortion_self"] for f in fams]
    pk_inst_cross_sec = [all_family_results[f]["summary_metrics"]["secondary_native_chat"]["peak_depth_inst_cross"] for f in fams]
    mean_delta_share_sec = [all_family_results[f]["summary_metrics"]["secondary_native_chat"]["mean_delta_sharing"] for f in fams]

    n_boot = 10 if args.dry_run else (v2_config.get("bootstrap", {}).get("n_boot") or v2_config.get("statistics", {}).get("n_boot", 1000))

    # Bootstrap 信頼区間の計算 (Primary)
    pt_com_r, com_r_low, com_r_up = compute_bootstrap_ci(com_dist_r_prim, n_boot=n_boot)
    pt_com_s, com_s_low, com_s_up = compute_bootstrap_ci(com_dist_s_prim, n_boot=n_boot)
    pt_base_pk, base_pk_low, base_pk_up = compute_bootstrap_ci(pk_base_cross_prim, n_boot=n_boot)
    pt_inst_pk, inst_pk_low, inst_pk_up = compute_bootstrap_ci(pk_inst_cross_prim, n_boot=n_boot)
    pt_share, share_low, share_up = compute_bootstrap_ci(mean_delta_share_prim, n_boot=n_boot)

    # Bootstrap 信頼区間の計算 (Secondary)
    pt_com_r_sec, com_r_sec_low, com_r_sec_up = compute_bootstrap_ci(com_dist_r_sec, n_boot=n_boot)
    pt_com_s_sec, com_s_sec_low, com_s_sec_up = compute_bootstrap_ci(com_dist_s_sec, n_boot=n_boot)
    pt_inst_pk_sec, inst_pk_sec_low, inst_pk_sec_up = compute_bootstrap_ci(pk_inst_cross_sec, n_boot=n_boot)
    pt_share_sec, share_sec_low, share_sec_up = compute_bootstrap_ci(mean_delta_share_sec, n_boot=n_boot)

    # Paired comparison (Base vs Instruct cross-decoding peak shift) on Primary
    paired_peak_comp = paired_family_comparison(pk_inst_cross_prim, pk_base_cross_prim)

    summary_data = {
        "families": fams,
        "per_family_summary": {fam_id: res["summary_metrics"] for fam_id, res in all_family_results.items()},
        "primary_matched_plain": {
            "bootstrap_ci_95": {
                "com_distortion_reader": {"mean": pt_com_r, "ci_lower": com_r_low, "ci_upper": com_r_up},
                "com_distortion_self": {"mean": pt_com_s, "ci_lower": com_s_low, "ci_upper": com_s_up},
                "peak_depth_base_cross": {"mean": pt_base_pk, "ci_lower": base_pk_low, "ci_upper": base_pk_up},
                "peak_depth_inst_cross": {"mean": pt_inst_pk, "ci_lower": inst_pk_low, "ci_upper": inst_pk_up},
                "mean_delta_sharing": {"mean": pt_share, "ci_lower": share_low, "ci_upper": share_up},
            },
            "paired_peak_depth_comparison": paired_peak_comp,
        },
        "secondary_native_chat": {
            "bootstrap_ci_95": {
                "com_distortion_reader": {"mean": pt_com_r_sec, "ci_lower": com_r_sec_low, "ci_upper": com_r_sec_up},
                "com_distortion_self": {"mean": pt_com_s_sec, "ci_lower": com_s_sec_low, "ci_upper": com_s_sec_up},
                "peak_depth_inst_cross": {"mean": pt_inst_pk_sec, "ci_lower": inst_pk_sec_low, "ci_upper": inst_pk_sec_up},
                "mean_delta_sharing": {"mean": pt_share_sec, "ci_lower": share_sec_low, "ci_upper": share_sec_up},
            },
        },
        # Root fallback for backward compatibility
        "bootstrap_ci_95": {
            "com_distortion_reader": {"mean": pt_com_r, "ci_lower": com_r_low, "ci_upper": com_r_up},
            "com_distortion_self": {"mean": pt_com_s, "ci_lower": com_s_low, "ci_upper": com_s_up},
            "peak_depth_base_cross": {"mean": pt_base_pk, "ci_lower": base_pk_low, "ci_upper": base_pk_up},
            "peak_depth_inst_cross": {"mean": pt_inst_pk, "ci_lower": inst_pk_low, "ci_upper": inst_pk_up},
            "mean_delta_sharing": {"mean": pt_share, "ci_lower": share_low, "ci_upper": share_up},
        },
        "paired_peak_depth_comparison": paired_peak_comp,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"All done! Integrated summary with Bootstrap CIs saved to {summary_path}")


if __name__ == "__main__":
    main()
