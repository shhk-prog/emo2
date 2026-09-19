#!/usr/bin/env python3
"""
V2-RQ3: Post-training による因果回路の再配置とピーク解離解析
4モデルファミリー (Qwen 2.5, Llama 3.2, Gemma 3, OLMo 2) × 4条件 (Base/Inst × Reader/Self)
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

from affective_empathy_eval.geometry import (
    compute_center_of_mass,
    compute_dissociation_metrics,
    compute_peak_depth,
    compute_relative_depth,
)
from affective_empathy_eval.interventions import compute_causal_leverage
from affective_empathy_eval.likelihood import (
    build_va_candidates,
    compute_expected_va,
    compute_sequence_likelihoods_for_candidates,
)
from affective_empathy_eval.data import describe_loaded_frame
from affective_empathy_eval.manifests import (
    create_run_manifest,
    is_manifest_matching,
    compute_string_or_dict_hash,
    DEFAULT_CODE_VERSION,
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
from affective_empathy_eval.statistics import fit_sample_level_lmm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run V2-RQ3 Causal Map and Peak Dissociation Analysis"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/v2_experiments.yaml",
        help="Path to V2 config",
    )
    parser.add_argument(
        "--models-config",
        type=str,
        default="configs/models.yaml",
        help="Path to models config",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in mock/dry-run mode without loading full weights",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use (cpu or cuda)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Limit number of samples"
    )
    add_model_selection_args(parser)
    return parser.parse_args()


from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

def run_causal_patching_for_model(
    model: Any,
    tokenizer: Any,
    df: pd.DataFrame,
    task: TaskType,
    format_type: str,
    device: str = "cpu",
    is_dry_run: bool = False,
    num_layers: int = 24,
) -> dict[str, Any]:
    """
    指定モデルに対する因果パッチング実験の実行
    戻り値: {
        "c_v": [...],
        "c_a": [...],
        "c_v_zero": [...],
        "c_a_zero": [...],
        "pair_level": [
            {
                "sample_idx": int,
                "pair_id": str,
                "layer": int,
                "relative_depth": float,
                "c_v": float,
                "c_a": float,
                "c_v_zero": float,
                "c_a_zero": float,
                "fold_id": int,
                "direction_fit_split": "train",
                "evaluation_split": "test",
            },
            ...
        ]
    }
    """
    if is_dry_run:
        # モック因果プロファイル: 後期層で causal leverage が立ち上がる（Decodability との解離）
        rng = np.random.default_rng(123)
        depths = [l / max(1, num_layers - 1) for l in range(num_layers)]
        c_v_mean = [
            float(1.5 / (1.0 + np.exp(-12.0 * (d - 0.75))))
            for d in depths
        ]
        c_a_mean = [
            float(1.2 / (1.0 + np.exp(-10.0 * (d - 0.70))))
            for d in depths
        ]
        c_v_zero = [
            float(2.0 / (1.0 + np.exp(-8.0 * (d - 0.60))))
            for d in depths
        ]
        c_a_zero = [
            float(1.8 / (1.0 + np.exp(-8.0 * (d - 0.60))))
            for d in depths
        ]

        pair_records = []
        for i, row in df.iterrows():
            p_id = row.get("pair_id", f"pair_{i}")
            fold_id = i % 5
            for l in range(num_layers):
                noise_v = float(rng.normal(0, 0.08))
                noise_a = float(rng.normal(0, 0.08))
                cv = max(0.0, c_v_mean[l] + noise_v)
                ca = max(0.0, c_a_mean[l] + noise_a)
                cvz = max(0.0, c_v_zero[l] + noise_v)
                caz = max(0.0, c_a_zero[l] + noise_v)
                pair_records.append({
                    "sample_idx": i,
                    "pair_id": str(p_id),
                    "layer": l,
                    "relative_depth": depths[l],
                    "c_v": cv,
                    "c_a": ca,
                    "c_v_zero": cvz,
                    "c_a_zero": caz,
                    "fold_id": fold_id,
                    "direction_fit_split": "train",
                    "evaluation_split": "test",
                })

        return {
            "c_v": c_v_mean,
            "c_a": c_a_mean,
            "c_v_mean": c_v_mean,
            "c_a_mean": c_a_mean,
            "c_v_zero": c_v_zero,
            "c_a_zero": c_a_zero,
            "pair_level": pair_records,
        }

    adapter = get_model_adapter(model)
    actual_layers = adapter.get_num_layers()
    candidates = build_va_candidates()

    model.eval()
    layer_shifts_v = [[] for _ in range(actual_layers)]
    layer_shifts_a = [[] for _ in range(actual_layers)]
    layer_shifts_v_zero = [[] for _ in range(actual_layers)]
    layer_shifts_a_zero = [[] for _ in range(actual_layers)]
    pair_records = []

    # Step 1: Clean runs 及び各層の prompt_end 活性化の収集
    clean_ev_list = []
    clean_ea_list = []
    sample_prompts = []
    sample_anchors = []
    sample_pids = []
    layer_activations = [[] for _ in range(actual_layers)]

    with torch.no_grad():
        for i, row in df.iterrows():
            text = str(row["text"])
            p_id = str(row.get("pair_id", f"pair_{i}"))
            prompt = build_prompt(
                text=text, task=task, format_type=format_type, tokenizer=tokenizer
            )
            enc = encode_prompt_canonical(tokenizer, prompt, device=device)
            input_ids = enc["input_ids"]
            anchors = find_semantic_anchors(
                input_ids[0].tolist(), tokenizer, text
            )
            patch_pos = anchors["prompt_end"]

            sample_prompts.append(prompt)
            sample_anchors.append(patch_pos)
            sample_pids.append(p_id)

            # Clean run (ベースライン出力期待値: 81候補 Joint Sequence-Likelihood Protocol)
            _, probs_clean = compute_sequence_likelihoods_for_candidates(
                model=model, tokenizer=tokenizer, prompt=prompt, candidates=candidates, device=device
            )
            ev_clean, ea_clean = compute_expected_va(probs_clean, candidates)
            clean_ev_list.append(ev_clean)
            clean_ea_list.append(ea_clean)

            # 活性化収集
            with ActivationHookManager(adapter) as hook_mgr:
                for l in range(actual_layers):
                    hook_mgr.register_capture_hook(
                        layer_idx=l,
                        hook_point=HookPoint.POST_MLP_RESID,
                        token_indices=patch_pos,
                        key=f"layer_{l}",
                    )
                _ = model(**enc)
                caps = hook_mgr.get_captured()
                for l in range(actual_layers):
                    act = caps[f"layer_{l}"].squeeze(0).squeeze(0).cpu().to(torch.float32).numpy()
                    layer_activations[l].append(act)

    # Step 2: 目的変数の準備（外部ラベル優先、なければモデルのクリーン出力期待値）
    if "reader_V" in df.columns:
        y_v = df["reader_V"].to_numpy(dtype=np.float64)
        y_a = df["reader_A"].to_numpy(dtype=np.float64) if "reader_A" in df.columns else y_v
    else:
        y_v = np.array(clean_ev_list, dtype=np.float64)
        y_a = np.array(clean_ea_list, dtype=np.float64)

    # Step 3: 各層での Cross-Fitting 方向推定と介入実行 (5-fold CV)
    hidden_dim = model.config.hidden_size
    zero_patch = torch.zeros(1, 1, hidden_dim, device=device)

    seed = 42
    n_samples = len(sample_prompts)
    n_splits = min(5, n_samples) if n_samples >= 2 else 1
    if n_splits > 1:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(kf.split(np.arange(n_samples)))
    else:
        folds = [(np.arange(n_samples), np.arange(n_samples))]

    for l in range(actual_layers):
        rel_d = compute_relative_depth(l, actual_layers)
        H_l = np.array(layer_activations[l], dtype=np.float64)

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            H_train = H_l[train_idx]
            y_v_train = y_v[train_idx]
            y_a_train = y_a[train_idx]

            # Train fold で情動特異的介入方向 d_V, d_A を推定
            if H_train.shape[0] >= 4 and np.std(y_v_train) > 1e-4:
                reg_v = Ridge(alpha=10.0).fit(H_train, y_v_train)
                d_v = reg_v.coef_
                norm_v = np.linalg.norm(d_v)
                d_v = d_v / norm_v if norm_v > 1e-6 else np.zeros_like(d_v)
            else:
                d_v = np.zeros(H_train.shape[1])

            if H_train.shape[0] >= 4 and np.std(y_a_train) > 1e-4:
                reg_a = Ridge(alpha=10.0).fit(H_train, y_a_train)
                d_a = reg_a.coef_
                norm_a = np.linalg.norm(d_a)
                d_a = d_a / norm_a if norm_a > 1e-6 else np.zeros_like(d_a)
            else:
                d_a = np.zeros(H_train.shape[1])

            proj_v = H_train @ d_v
            std_v = float(np.std(proj_v)) if np.std(proj_v) > 1e-6 else float(np.std(H_train))
            std_v = max(1e-4, std_v)

            proj_a = H_train @ d_a
            std_a = float(np.std(proj_a)) if np.std(proj_a) > 1e-6 else float(np.std(H_train))
            std_a = max(1e-4, std_a)

            d_v_t = torch.tensor(d_v, dtype=torch.float32, device=device).reshape(1, 1, -1)
            d_a_t = torch.tensor(d_a, dtype=torch.float32, device=device).reshape(1, 1, -1)

            # Test fold で介入評価
            for i in test_idx:
                prompt = sample_prompts[i]
                patch_pos = sample_anchors[i]
                p_id = sample_pids[i]
                ev_clean = clean_ev_list[i]
                ea_clean = clean_ea_list[i]

                # 3-1. Primary: 情動特異的介入 (d_V 加算注入 -> C_V 測定)
                with ActivationHookManager(adapter) as hook_mgr:
                    hook_mgr.register_direction_intervention_hook(
                        layer_idx=l,
                        direction=d_v_t,
                        alpha=1.0,
                        hidden_std=std_v,
                        token_indices=patch_pos,
                        hook_point=HookPoint.POST_MLP_RESID,
                        mode="inject",
                    )
                    _, probs_v = compute_sequence_likelihoods_for_candidates(
                        model=model, tokenizer=tokenizer, prompt=prompt, candidates=candidates, device=device
                    )
                    ev_v, _ = compute_expected_va(probs_v, candidates)
                    cv, _ = compute_causal_leverage(ev_v, ev_clean)

                # 3-2. Primary: 情動特異的介入 (d_A 加算注入 -> C_A 測定)
                with ActivationHookManager(adapter) as hook_mgr:
                    hook_mgr.register_direction_intervention_hook(
                        layer_idx=l,
                        direction=d_a_t,
                        alpha=1.0,
                        hidden_std=std_a,
                        token_indices=patch_pos,
                        hook_point=HookPoint.POST_MLP_RESID,
                        mode="inject",
                    )
                    _, probs_a = compute_sequence_likelihoods_for_candidates(
                        model=model, tokenizer=tokenizer, prompt=prompt, candidates=candidates, device=device
                    )
                    _, ea_a = compute_expected_va(probs_a, candidates)
                    ca, _ = compute_causal_leverage(ea_a, ea_clean)

                # 3-3. Secondary: 非特異的ゼロアブレーション統制
                with ActivationHookManager(adapter) as hook_mgr:
                    hook_mgr.register_patch_hook(
                        layer_idx=l,
                        patch_tensor=zero_patch,
                        token_indices=patch_pos,
                        hook_point=HookPoint.POST_MLP_RESID,
                    )
                    _, probs_zero = compute_sequence_likelihoods_for_candidates(
                        model=model, tokenizer=tokenizer, prompt=prompt, candidates=candidates, device=device
                    )
                    ev_z, ea_z = compute_expected_va(probs_zero, candidates)
                    cv_z, _ = compute_causal_leverage(ev_z, ev_clean)
                    ca_z, _ = compute_causal_leverage(ea_z, ea_clean)

                layer_shifts_v[l].append(cv)
                layer_shifts_a[l].append(ca)
                layer_shifts_v_zero[l].append(cv_z)
                layer_shifts_a_zero[l].append(ca_z)

                pair_records.append({
                    "sample_idx": int(i),
                    "pair_id": p_id,
                    "layer": l,
                    "relative_depth": rel_d,
                    "c_v": cv,
                    "c_a": ca,
                    "c_v_zero": cv_z,
                    "c_a_zero": ca_z,
                    "fold_id": fold_idx,
                    "direction_fit_split": "train",
                    "evaluation_split": "test",
                })


    c_v_m = [float(np.mean(shifts)) if len(shifts) > 0 else 0.0 for shifts in layer_shifts_v]
    c_a_m = [float(np.mean(shifts)) if len(shifts) > 0 else 0.0 for shifts in layer_shifts_a]
    c_v_z_m = [float(np.mean(shifts)) if len(shifts) > 0 else 0.0 for shifts in layer_shifts_v_zero]
    c_a_z_m = [float(np.mean(shifts)) if len(shifts) > 0 else 0.0 for shifts in layer_shifts_a_zero]

    return {
        "c_v": c_v_m,
        "c_a": c_a_m,
        "c_v_mean": c_v_m,
        "c_a_mean": c_a_m,
        "c_v_zero": c_v_z_m,
        "c_a_zero": c_a_z_m,
        "pair_level": pair_records,
    }


def main():
    args = parse_args()
    logger.info(
        f"Starting V2-RQ3 Causal Map analysis (dry_run={args.dry_run}, device={args.device})"
    )

    with open(args.config, "r", encoding="utf-8") as f:
        v2_config = yaml.safe_load(f)

    target_models = resolve_models_from_args(args, Path(args.models_config))
    raw_dir = Path(v2_config["output"]["raw_dir"])
    derived_dir = Path(v2_config["output"]["derived_dir"])
    if args.dry_run:
        raw_dir = raw_dir / "dry_run"
        derived_dir = derived_dir / "dry_run"
    raw_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    pair_dir = derived_dir / "pair_level"
    pair_dir.mkdir(parents=True, exist_ok=True)

    # 刺激データセット読み込み
    df = pd.read_csv(v2_config["dataset"]["path"])
    logger.info(describe_loaded_frame(df, "V2-RQ3 dataset", v2_config["dataset"]["path"]))
    if args.dry_run:
        df = df.head(32).copy()
        logger.info(f"[DRY-RUN] Scaled down dataset to N={len(df)} for fast smoke testing.")
    elif args.max_samples is not None:
        df = df.iloc[: args.max_samples].copy()
        logger.info(f"Applied max_samples={args.max_samples}: n_rows={len(df)}")

    all_causal_results = {}
    all_pair_level_records = []

    for fam_id, fam_cfg in target_models.items():
        out_path = raw_dir / f"v2_causal_map_{fam_id}.json"
        manifest_path = raw_dir / f"manifest_causal_map_{fam_id}.json"
        fam_pair_path = pair_dir / f"v2_causal_pair_level_{fam_id}.csv"

        manifest_config = {
            "v2_config": v2_config,
            "family_id": fam_id,
            "family_name": fam_cfg.family_name,
            "model_set": args.model_set,
            "max_samples": args.max_samples,
            "dry_run": bool(args.dry_run),
            "base_model_id": fam_cfg.base_model.model_id,
            "instruct_model_id": fam_cfg.instruct_model.model_id,
            "dataset_path": str(v2_config["dataset"]["path"]),
            "seed": v2_config.get("seed", 42),
        }

        if out_path.exists() and not args.dry_run:
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if cached and "causal_maps" in cached:
                    expected_config_hash = compute_string_or_dict_hash(manifest_config)
                    expected_dataset_hash = compute_string_or_dict_hash(str(v2_config["dataset"]["path"]))
                    if not cached.get("dry_run", False) and is_manifest_matching(
                        str(manifest_path),
                        expected_model_name=fam_cfg.family_name,
                        expected_config_hash=expected_config_hash,
                        expected_dataset_hash=expected_dataset_hash,
                        expected_code_version=DEFAULT_CODE_VERSION,
                        expected_dry_run=False,
                    ):
                        if fam_pair_path.exists():
                            logger.info(f"Loaded existing results and pair-level CSV for {fam_id}. Skipping computation.")
                            all_causal_results[fam_id] = cached
                            cached_pairs_df = pd.read_csv(fam_pair_path)
                            all_pair_level_records.extend(cached_pairs_df.to_dict("records"))
                            continue
                        else:
                            logger.warning(f"Pair-level artifact missing for {fam_id} ({fam_pair_path}) => cache invalid, recomputing.")
            except Exception as e:
                logger.warning(f"Cache check failed for {fam_id}: {e}")


        eff_num_layers = min(fam_cfg.num_layers, 4) if args.dry_run else fam_cfg.num_layers
        logger.info(
            f"--- Running Causal Maps for Family: {fam_id} ({eff_num_layers} layers) ---"
        )

        depths = [
            compute_relative_depth(l, eff_num_layers)
            for l in range(eff_num_layers)
        ]
        fam_causal = {}
        model_groups = [
            ("base", "plain", [("reader", TaskType.READER, "plain"), ("self", TaskType.SELF, "plain")]),
            ("inst", "matched_plain", [("reader", TaskType.READER, "plain"), ("self", TaskType.SELF, "plain")]),
            ("inst", "native_chat", [("reader", TaskType.READER, "chat"), ("self", TaskType.SELF, "chat")]),
        ]

        for align_prefix, format_cond, task_configs in model_groups:
            model_spec = fam_cfg.get_model_spec("base" if align_prefix == "base" else "instruct")
            if args.dry_run:
                model = None
                tokenizer = None
            else:
                logger.info(f"Loading weights for {model_spec.model_id} ({format_cond}) onto {args.device}...")
                tokenizer = AutoTokenizer.from_pretrained(model_spec.model_id)
                model = AutoModelForCausalLM.from_pretrained(
                    model_spec.model_id,
                    torch_dtype=torch.bfloat16 if "cuda" in args.device else torch.float32,
                    device_map=args.device if "cuda" in args.device else None,
                )

            for task_str, task_type, fmt in task_configs:
                cond_key = f"{align_prefix}_{format_cond}_{task_str}"
                logger.info(f"Causal patching for {cond_key}...")
                res = run_causal_patching_for_model(
                    model=model,
                    tokenizer=tokenizer,
                    df=df,
                    task=task_type,
                    format_type=fmt,
                    num_layers=eff_num_layers,
                    device=args.device,
                    is_dry_run=args.dry_run,
                )
                causal_entry = {
                    "c_v": res["c_v"],
                    "c_a": res["c_a"],
                    "c_v_zero": res.get("c_v_zero", []),
                    "c_a_zero": res.get("c_a_zero", []),
                }
                fam_causal[cond_key] = causal_entry

                # 後方互換性エイリアス (matched と native を厳密に分離)
                if format_cond == "plain" and align_prefix == "base":
                    fam_causal[f"base_{task_str}"] = causal_entry
                elif format_cond == "native_chat" and align_prefix == "inst":
                    fam_causal[f"inst_native_{task_str}"] = causal_entry
                    fam_causal[f"deprecated_inst_{task_str}_native_alias"] = causal_entry
                elif format_cond == "matched_plain" and align_prefix == "inst":
                    fam_causal[f"inst_matched_{task_str}"] = causal_entry

                # pair-level 記録の集約
                for prec in res.get("pair_level", []):
                    all_pair_level_records.append({
                        "family": fam_id,
                        "alignment": align_prefix,
                        "format_condition": format_cond,
                        "task": task_str,
                        "pair_id": prec["pair_id"],
                        "layer": prec["layer"],
                        "relative_depth": prec["relative_depth"],
                        "c_v": prec["c_v"],
                        "c_a": prec["c_a"],
                        "c_v_zero": prec.get("c_v_zero", 0.0),
                        "c_a_zero": prec.get("c_a_zero", 0.0),
                    })

            if not args.dry_run and model is not None:
                del model
                del tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # RQ1/RQ2のデコードプロファイルと突合して解離量を計算（matched-plain を Primary、native-chat を Secondary）
        geom_path = raw_dir / f"v2_geometry_{fam_id}.json"
        dissoc_results = {}
        if geom_path.exists():
            with open(geom_path, "r", encoding="utf-8") as f:
                geom_data = json.load(f)
            sharing_data = geom_data.get("rq2_sharing", {})

            for axis in ["valence", "arousal"]:
                axis_sharing = sharing_data.get(axis, {})
                c_field = "c_v" if axis == "valence" else "c_a"
                dissoc_results[axis] = {}

                conditions_map = [
                    ("base_reader", "base_r2_reader", "base_reader"),
                    ("base_self", "base_r2_self", "base_self"),
                    ("inst_matched_reader", "inst_matched_r2_reader", "inst_matched_reader"),
                    ("inst_matched_self", "inst_matched_r2_self", "inst_matched_self"),
                    ("inst_native_reader", "inst_native_r2_reader", "inst_native_reader"),
                    ("inst_native_self", "inst_native_r2_self", "inst_native_self"),
                ]
                for cond_name, d_key, c_key in conditions_map:
                    if d_key in axis_sharing and c_key in fam_causal:
                        d_prof = axis_sharing[d_key]
                        c_prof = fam_causal[c_key][c_field]
                        d_metrics = compute_dissociation_metrics(d_prof, c_prof, depths)
                        dissoc_results[axis][cond_name] = d_metrics

                # 事後学習に伴う解離の差分の比較 (Post-training-associated difference in dissociation)
                base_self_key = "base_self"
                base_reader_key = "base_reader"
                inst_self_key = "inst_matched_self"
                inst_reader_key = "inst_matched_reader"

                if (
                    base_self_key in dissoc_results[axis]
                    and inst_self_key in dissoc_results[axis]
                    and base_reader_key in dissoc_results[axis]
                    and inst_reader_key in dissoc_results[axis]
                ):
                    delta_d_is = dissoc_results[axis][inst_self_key]["delta_d_star"]
                    delta_d_bs = dissoc_results[axis][base_self_key]["delta_d_star"]
                    delta_bar_is = dissoc_results[axis][inst_self_key]["delta_bar_d"]
                    delta_bar_bs = dissoc_results[axis][base_self_key]["delta_bar_d"]

                    delta_d_ir = dissoc_results[axis][inst_reader_key]["delta_d_star"]
                    delta_d_br = dissoc_results[axis][base_reader_key]["delta_d_star"]
                    delta_bar_ir = dissoc_results[axis][inst_reader_key]["delta_bar_d"]
                    delta_bar_br = dissoc_results[axis][base_reader_key]["delta_bar_d"]

                    dissoc_results[axis]["post_training_comparison_matched"] = {
                        "delta_d_star_self_change": float(delta_d_is - delta_d_bs),
                        "delta_bar_d_self_change": float(delta_bar_is - delta_bar_bs),
                        "delta_d_star_reader_change": float(delta_d_ir - delta_d_br),
                        "delta_bar_d_reader_change": float(delta_bar_ir - delta_bar_br),
                        "diff_of_diffs_peak": float((delta_d_is - delta_d_bs) - (delta_d_ir - delta_d_br)),
                        "diff_of_diffs_com": float((delta_bar_is - delta_bar_bs) - (delta_bar_ir - delta_bar_br)),
                    }
                    dissoc_results[axis]["post_training_comparison"] = {
                        "deprecated_alias_of": "post_training_comparison_matched"
                    }
                    logger.info(
                        f"Matched dissociation changes ({fam_id} {axis}): Self Delta d* shift={delta_d_is - delta_d_bs:.3f}, Reader Delta d* shift={delta_d_ir - delta_d_br:.3f}"
                    )

                inst_self_nat = "inst_native_self"
                inst_reader_nat = "inst_native_reader"
                if (
                    base_self_key in dissoc_results[axis]
                    and inst_self_nat in dissoc_results[axis]
                    and base_reader_key in dissoc_results[axis]
                    and inst_reader_nat in dissoc_results[axis]
                ):
                    delta_d_is_n = dissoc_results[axis][inst_self_nat]["delta_d_star"]
                    delta_d_bs = dissoc_results[axis][base_self_key]["delta_d_star"]
                    delta_bar_is_n = dissoc_results[axis][inst_self_nat]["delta_bar_d"]
                    delta_bar_bs = dissoc_results[axis][base_self_key]["delta_bar_d"]

                    delta_d_ir_n = dissoc_results[axis][inst_reader_nat]["delta_d_star"]
                    delta_d_br = dissoc_results[axis][base_reader_key]["delta_d_star"]
                    delta_bar_ir_n = dissoc_results[axis][inst_reader_nat]["delta_bar_d"]
                    delta_bar_br = dissoc_results[axis][base_reader_key]["delta_bar_d"]

                    dissoc_results[axis]["post_training_comparison_native"] = {
                        "delta_d_star_self_change": float(delta_d_is_n - delta_d_bs),
                        "delta_bar_d_self_change": float(delta_bar_is_n - delta_bar_bs),
                        "delta_d_star_reader_change": float(delta_d_ir_n - delta_d_br),
                        "delta_bar_d_reader_change": float(delta_bar_ir_n - delta_bar_br),
                        "diff_of_diffs_peak": float((delta_d_is_n - delta_d_bs) - (delta_d_ir_n - delta_d_br)),
                        "diff_of_diffs_com": float((delta_bar_is_n - delta_bar_bs) - (delta_bar_ir_n - delta_bar_br)),
                    }

        fam_output = {
            "family_id": fam_id,
            "num_layers": fam_cfg.num_layers,
            "relative_depths": depths,
            "causal_maps": fam_causal,
            "dissociation": dissoc_results,
            "summary": {
                # Primary: matched-plain
                "base_reader_c_v_peak": compute_peak_depth(fam_causal["base_reader"]["c_v"], depths),
                "base_self_c_v_peak": compute_peak_depth(fam_causal["base_self"]["c_v"], depths),
                "inst_matched_reader_c_v_peak": compute_peak_depth(fam_causal["inst_matched_reader"]["c_v"], depths),
                "inst_matched_self_c_v_peak": compute_peak_depth(fam_causal["inst_matched_self"]["c_v"], depths),
                "base_self_c_v_com": compute_center_of_mass(fam_causal["base_self"]["c_v"], depths),
                "inst_matched_self_c_v_com": compute_center_of_mass(fam_causal["inst_matched_self"]["c_v"], depths),
                # Secondary: native-chat
                "inst_native_reader_c_v_peak": compute_peak_depth(fam_causal["inst_native_reader"]["c_v"], depths),
                "inst_native_self_c_v_peak": compute_peak_depth(fam_causal["inst_native_self"]["c_v"], depths),
                "inst_native_self_c_v_com": compute_center_of_mass(fam_causal["inst_native_self"]["c_v"], depths),
            },
        }
        fam_output["dry_run"] = bool(args.dry_run)
        all_causal_results[fam_id] = fam_output

        out_path = raw_dir / f"v2_causal_map_{fam_id}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(fam_output, f, indent=2)
        logger.info(f"Saved causal map to {out_path}")

        # Save per-family pair-level CSV to prevent data loss on cache hit
        fam_records = [r for r in all_pair_level_records if r["family"] == fam_id]
        if fam_records:
            fam_pair_df = pd.DataFrame(fam_records)
            fam_pair_df.to_csv(fam_pair_path, index=False)
            logger.info(f"Saved family pair-level records ({len(fam_pair_df)} rows) to {fam_pair_path}")

        # Manifest 保存
        manifest = create_run_manifest(
            run_type="v2_causal_map",
            model_name=fam_cfg.family_name,
            config=manifest_config,
            metadata={"num_layers": eff_num_layers},
            dataset_path=str(v2_config["dataset"]["path"]),
            dry_run=bool(args.dry_run),
        )
        manifest.save(str(raw_dir / f"manifest_causal_map_{fam_id}.json"))


    # 1. pair-level long-form CSV の保存
    pair_csv_path = derived_dir / "v2_causal_pair_level.csv"
    df_pair = pd.DataFrame(all_pair_level_records)
    df_pair.to_csv(pair_csv_path, index=False)
    logger.info(f"Saved pair-level causal records ({len(df_pair)} rows) to {pair_csv_path}")

    # 2. LMM 検定の実行: Primary (matched-plain) と Secondary (native-chat) に完全分離
    df_primary = df_pair[
        ((df_pair["alignment"] == "base") & (df_pair["format_condition"] == "plain"))
        | ((df_pair["alignment"] == "inst") & (df_pair["format_condition"] == "matched_plain"))
    ].copy()

    df_native = df_pair[
        ((df_pair["alignment"] == "base") & (df_pair["format_condition"] == "plain"))
        | ((df_pair["alignment"] == "inst") & (df_pair["format_condition"] == "native_chat"))
    ].copy()

    has_multi_family = len(df_pair["family"].unique()) > 1 if "family" in df_pair.columns else False
    formula_v = "c_v ~ C(family) + C(alignment) * C(task) * relative_depth" if has_multi_family else "c_v ~ C(alignment) * C(task) * relative_depth"
    formula_a = "c_a ~ C(family) + C(alignment) * C(task) * relative_depth" if has_multi_family else "c_a ~ C(alignment) * C(task) * relative_depth"

    lmm_summary: Dict[str, Any] = {
        "primary_matched_plain": {},
        "secondary_native_chat": {},
    }

    # Primary LMM
    try:
        logger.info(f"Fitting Primary LMM (matched-plain, N={len(df_primary)}) for Valence...")
        lmm_v_p = fit_sample_level_lmm(df=df_primary, formula=formula_v, groups="pair_id")
        lmm_summary["primary_matched_plain"]["valence"] = {
            "converged": lmm_v_p["converged"],
            "formula": formula_v,
            "params": lmm_v_p["params"],
            "pvalues": lmm_v_p["pvalues"],
            "conf_int": lmm_v_p["conf_int"],
        }
        logger.info(f"Fitting Primary LMM (matched-plain, N={len(df_primary)}) for Arousal...")
        lmm_a_p = fit_sample_level_lmm(df=df_primary, formula=formula_a, groups="pair_id")
        lmm_summary["primary_matched_plain"]["arousal"] = {
            "converged": lmm_a_p["converged"],
            "formula": formula_a,
            "params": lmm_a_p["params"],
            "pvalues": lmm_a_p["pvalues"],
            "conf_int": lmm_a_p["conf_int"],
        }
    except Exception as e:
        logger.warning(f"Primary LMM fitting failed: {e}")
        lmm_summary["primary_matched_plain"]["error"] = str(e)

    # Secondary LMM
    if not df_native.empty and len(df_native["alignment"].unique()) >= 2:
        try:
            logger.info(f"Fitting Secondary LMM (native-chat, N={len(df_native)}) for Valence...")
            lmm_v_n = fit_sample_level_lmm(df=df_native, formula=formula_v, groups="pair_id")
            lmm_summary["secondary_native_chat"]["valence"] = {
                "converged": lmm_v_n["converged"],
                "formula": formula_v,
                "params": lmm_v_n["params"],
                "pvalues": lmm_v_n["pvalues"],
                "conf_int": lmm_v_n["conf_int"],
            }
            logger.info(f"Fitting Secondary LMM (native-chat, N={len(df_native)}) for Arousal...")
            lmm_a_n = fit_sample_level_lmm(df=df_native, formula=formula_a, groups="pair_id")
            lmm_summary["secondary_native_chat"]["arousal"] = {
                "converged": lmm_a_n["converged"],
                "formula": formula_a,
                "params": lmm_a_n["params"],
                "pvalues": lmm_a_n["pvalues"],
                "conf_int": lmm_a_n["conf_int"],
            }
        except Exception as e:
            logger.warning(f"Secondary LMM fitting failed: {e}")
            lmm_summary["secondary_native_chat"]["error"] = str(e)

    # 3. 統合要約
    summary_path = derived_dir / "v2_causal_dissociation_summary.json"
    final_output = {
        "per_family": all_causal_results,
        "lmm_statistical_tests": lmm_summary,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    logger.info(f"All causal maps completed! Saved summary with LMM to {summary_path}")


if __name__ == "__main__":
    main()

