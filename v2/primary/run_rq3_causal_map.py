#!/usr/bin/env python3
"""
V2-RQ3: Post-training による因果回路の再配置とピーク解離解析
4モデルファミリー (Qwen 2.5, Llama 3.2, Gemma 2, Mistral) × 4条件 (Base/Inst × Reader/Self)
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
from transformers import AutoModelForCausalLM, AutoTokenizer

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
from affective_empathy_eval.models.adapters import get_model_adapter
from affective_empathy_eval.models.hooks import ActivationHookManager, HookPoint
from affective_empathy_eval.models.registry import get_registry
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
    parser.add_argument(
        "--family", type=str, default=None, help="Target specific family (e.g. Qwen)"
    )
    return parser.parse_args()


def run_causal_patching_for_model(
    model: Any,
    tokenizer: Any,
    df: pd.DataFrame,
    task: TaskType,
    format_type: str,
    num_layers: int,
    device: str = "cpu",
    is_dry_run: bool = False,
) -> dict[str, Any]:
    """
    各層の残差ストリームを中立ベースライン活性化で置換（パッチング）し、
    出力期待値の変化量 C_V, C_A を層平均およびサンプル（ペア）単位で測定
    戻り値: {
        "c_v": [layer0..layerL],
        "c_a": [layer0..layerL],
        "pair_level": [{"pair_id": ..., "layer": ..., "c_v": ..., "c_a": ...}]
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

        pair_records = []
        for i, row in df.iterrows():
            p_id = row.get("pair_id", f"pair_{i}")
            for l in range(num_layers):
                noise_v = float(rng.normal(0, 0.08))
                noise_a = float(rng.normal(0, 0.08))
                cv = max(0.0, c_v_mean[l] + noise_v)
                ca = max(0.0, c_a_mean[l] + noise_a)
                pair_records.append({
                    "sample_idx": i,
                    "pair_id": str(p_id),
                    "layer": l,
                    "relative_depth": depths[l],
                    "c_v": cv,
                    "c_a": ca,
                })

        return {
            "c_v": c_v_mean,
            "c_a": c_a_mean,
            "c_v_mean": c_v_mean,
            "c_a_mean": c_a_mean,
            "pair_level": pair_records,
        }

    adapter = get_model_adapter(model)
    actual_layers = adapter.get_num_layers()
    candidates = build_va_candidates()

    model.eval()
    layer_shifts_v = [[] for _ in range(actual_layers)]
    layer_shifts_a = [[] for _ in range(actual_layers)]
    pair_records = []

    with torch.no_grad():
        for i, row in df.iterrows():
            text = str(row["text"])
            p_id = str(row.get("pair_id", f"pair_{i}"))
            prompt = build_prompt(
                text=text, task=task, format_type=format_type, tokenizer=tokenizer
            )
            # 正準トークナイズ
            enc = encode_prompt_canonical(tokenizer, prompt, device=device)
            input_ids = enc["input_ids"]
            anchors = find_semantic_anchors(
                input_ids[0].tolist(), tokenizer, text
            )
            patch_pos = anchors["prompt_end"]  # プロンプト末尾（生成直前プレフィックス終端）

            # 1. Clean run (ベースライン出力期待値: 81候補 Joint Sequence-Likelihood Protocol)
            _, probs_clean = compute_sequence_likelihoods_for_candidates(
                model=model, tokenizer=tokenizer, prompt=prompt, candidates=candidates, device=device
            )
            ev_clean, ea_clean = compute_expected_va(probs_clean, candidates)

            # 2. 各層へのパッチング（Zero Ablation: prompt_end token をゼロ置換）
            hidden_dim = model.config.hidden_size
            zero_patch = torch.zeros(1, 1, hidden_dim, device=device)

            for l in range(actual_layers):
                rel_d = compute_relative_depth(l, actual_layers)
                with ActivationHookManager(adapter) as hook_mgr:
                    hook_mgr.register_patch_hook(
                        layer_idx=l,
                        patch_tensor=zero_patch,
                        token_indices=patch_pos,
                        hook_point=HookPoint.POST_MLP_RESID,
                    )
                    _, probs_patched = compute_sequence_likelihoods_for_candidates(
                        model=model, tokenizer=tokenizer, prompt=prompt, candidates=candidates, device=device
                    )
                    ev_patch, ea_patch = compute_expected_va(probs_patched, candidates)

                    cv, _ = compute_causal_leverage(ev_patch, ev_clean)
                    ca, _ = compute_causal_leverage(ea_patch, ea_clean)
                    layer_shifts_v[l].append(cv)
                    layer_shifts_a[l].append(ca)

                    pair_records.append({
                        "sample_idx": i,
                        "pair_id": p_id,
                        "layer": l,
                        "relative_depth": rel_d,
                        "c_v": cv,
                        "c_a": ca,
                    })

    c_v_m = [float(np.mean(shifts)) if len(shifts) > 0 else 0.0 for shifts in layer_shifts_v]
    c_a_m = [float(np.mean(shifts)) if len(shifts) > 0 else 0.0 for shifts in layer_shifts_a]

    return {
        "c_v": c_v_m,
        "c_a": c_a_m,
        "c_v_mean": c_v_m,
        "c_a_mean": c_a_m,
        "pair_level": pair_records,
    }


def main():
    args = parse_args()
    logger.info(
        f"Starting V2-RQ3 Causal Map analysis (dry_run={args.dry_run}, device={args.device})"
    )

    with open(args.config, "r", encoding="utf-8") as f:
        v2_config = yaml.safe_load(f)

    registry = get_registry(Path(args.models_config))
    raw_dir = Path(v2_config["output"]["raw_dir"])
    derived_dir = Path(v2_config["output"]["derived_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    # 刺激データセット読み込み
    df = pd.read_csv(v2_config["dataset"]["path"])
    if args.max_samples is not None:
        df = df.iloc[: args.max_samples].copy()

    target_families = [args.family] if args.family else v2_config["families"]
    all_causal_results = {}
    all_pair_level_records = []

    for fam_id in target_families:
        fam_cfg = registry.get_family(fam_id)
        logger.info(
            f"--- Running Causal Maps for Family: {fam_id} ({fam_cfg.num_layers} layers) ---"
        )

        depths = [
            compute_relative_depth(l, fam_cfg.num_layers)
            for l in range(fam_cfg.num_layers)
        ]
        fam_causal = {}
        model_groups = [
            ("base", [("reader", TaskType.READER, "plain"), ("self", TaskType.SELF, "plain")]),
            ("inst", [("reader", TaskType.READER, "chat"), ("self", TaskType.SELF, "chat")]),
        ]

        for align_prefix, task_configs in model_groups:
            model_spec = fam_cfg.get_model_spec("base" if align_prefix == "base" else "instruct")
            if args.dry_run:
                model = None
                tokenizer = None
            else:
                logger.info(f"Loading weights for {model_spec.model_id} onto {args.device}...")
                tokenizer = AutoTokenizer.from_pretrained(model_spec.model_id)
                model = AutoModelForCausalLM.from_pretrained(
                    model_spec.model_id,
                    torch_dtype=torch.bfloat16 if "cuda" in args.device else torch.float32,
                    device_map=args.device if "cuda" in args.device else None,
                )

            for task_str, task_type, fmt in task_configs:
                cond_key = f"{align_prefix}_{task_str}"
                logger.info(f"Causal patching for {cond_key}...")
                res = run_causal_patching_for_model(
                    model=model,
                    tokenizer=tokenizer,
                    df=df,
                    task=task_type,
                    format_type=fmt,
                    num_layers=fam_cfg.num_layers,
                    device=args.device,
                    is_dry_run=args.dry_run,
                )
                fam_causal[cond_key] = {
                    "c_v": res["c_v"],
                    "c_a": res["c_a"],
                }

                # pair-level 記録の集約
                for prec in res.get("pair_level", []):
                    all_pair_level_records.append({
                        "family": fam_id,
                        "alignment": align_prefix,
                        "task": task_str,
                        "pair_id": prec["pair_id"],
                        "layer": prec["layer"],
                        "relative_depth": prec["relative_depth"],
                        "c_v": prec["c_v"],
                        "c_a": prec["c_a"],
                    })

            if not args.dry_run and model is not None:
                del model
                del tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # RQ1/RQ2のデコードプロファイルと突合して4条件すべての解離量を計算（BR, BS, IR, IS）
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

                # 4条件すべてで計算
                conditions_map = [
                    ("base_reader", "base_r2_reader", "base_reader"),
                    ("base_self", "base_r2_self", "base_self"),
                    ("inst_reader", "inst_r2_reader", "inst_reader"),
                    ("inst_self", "inst_r2_self", "inst_self"),
                ]
                for cond_name, d_key, c_key in conditions_map:
                    if d_key in axis_sharing and c_key in fam_causal:
                        d_prof = axis_sharing[d_key]
                        c_prof = fam_causal[c_key][c_field]
                        d_metrics = compute_dissociation_metrics(d_prof, c_prof, depths)
                        dissoc_results[axis][cond_name] = d_metrics

                # 事後学習による解離の変化量の比較 (Post-training reorganizes dissociation)
                if "base_self" in dissoc_results[axis] and "inst_self" in dissoc_results[axis]:
                    delta_d_is = dissoc_results[axis]["inst_self"]["delta_d_star"]
                    delta_d_bs = dissoc_results[axis]["base_self"]["delta_d_star"]
                    delta_bar_is = dissoc_results[axis]["inst_self"]["delta_bar_d"]
                    delta_bar_bs = dissoc_results[axis]["base_self"]["delta_bar_d"]

                    delta_d_ir = dissoc_results[axis]["inst_reader"]["delta_d_star"] if "inst_reader" in dissoc_results[axis] else 0.0
                    delta_d_br = dissoc_results[axis]["base_reader"]["delta_d_star"] if "base_reader" in dissoc_results[axis] else 0.0
                    delta_bar_ir = dissoc_results[axis]["inst_reader"]["delta_bar_d"] if "inst_reader" in dissoc_results[axis] else 0.0
                    delta_bar_br = dissoc_results[axis]["base_reader"]["delta_bar_d"] if "base_reader" in dissoc_results[axis] else 0.0

                    dissoc_results[axis]["post_training_comparison"] = {
                        "delta_d_star_self_change": float(delta_d_is - delta_d_bs),
                        "delta_bar_d_self_change": float(delta_bar_is - delta_bar_bs),
                        "delta_d_star_reader_change": float(delta_d_ir - delta_d_br),
                        "delta_bar_d_reader_change": float(delta_bar_ir - delta_bar_br),
                        "diff_of_diffs_peak": float((delta_d_is - delta_d_bs) - (delta_d_ir - delta_d_br)),
                        "diff_of_diffs_com": float((delta_bar_is - delta_bar_bs) - (delta_bar_ir - delta_bar_br)),
                    }
                    logger.info(
                        f"Dissociation changes ({fam_id} {axis}): Self Delta d* shift={delta_d_is - delta_d_bs:.3f}, Reader Delta d* shift={delta_d_ir - delta_d_br:.3f}"
                    )

        fam_output = {
            "family_id": fam_id,
            "num_layers": fam_cfg.num_layers,
            "relative_depths": depths,
            "causal_maps": fam_causal,
            "dissociation": dissoc_results,
            "summary": {
                "base_reader_c_v_peak": compute_peak_depth(fam_causal["base_reader"]["c_v"], depths),
                "base_self_c_v_peak": compute_peak_depth(fam_causal["base_self"]["c_v"], depths),
                "inst_reader_c_v_peak": compute_peak_depth(fam_causal["inst_reader"]["c_v"], depths),
                "inst_self_c_v_peak": compute_peak_depth(fam_causal["inst_self"]["c_v"], depths),
                "base_self_c_v_com": compute_center_of_mass(fam_causal["base_self"]["c_v"], depths),
                "inst_self_c_v_com": compute_center_of_mass(fam_causal["inst_self"]["c_v"], depths),
            },
        }
        all_causal_results[fam_id] = fam_output

        out_path = raw_dir / f"v2_causal_map_{fam_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(fam_output, f, indent=2)
        logger.info(f"Saved causal map to {out_path}")

    # 1. pair-level long-form CSV の保存
    pair_csv_path = derived_dir / "v2_causal_pair_level.csv"
    df_pair = pd.DataFrame(all_pair_level_records)
    df_pair.to_csv(pair_csv_path, index=False)
    logger.info(f"Saved pair-level causal records ({len(df_pair)} rows) to {pair_csv_path}")

    # 2. LMM 検定の実行: C ~ Alignment * Task * Depth + (1 | pair_id)
    lmm_summary = {}
    try:
        logger.info("Fitting LMM for Valence causal leverage...")
        lmm_v = fit_sample_level_lmm(
            df=df_pair,
            formula="c_v ~ C(alignment) * C(task) * relative_depth",
            groups="pair_id",
        )
        lmm_summary["valence"] = {
            "converged": lmm_v["converged"],
            "params": lmm_v["params"],
            "pvalues": lmm_v["pvalues"],
            "conf_int": lmm_v["conf_int"],
        }
        logger.info("Fitting LMM for Arousal causal leverage...")
        lmm_a = fit_sample_level_lmm(
            df=df_pair,
            formula="c_a ~ C(alignment) * C(task) * relative_depth",
            groups="pair_id",
        )
        lmm_summary["arousal"] = {
            "converged": lmm_a["converged"],
            "params": lmm_a["params"],
            "pvalues": lmm_a["pvalues"],
            "conf_int": lmm_a["conf_int"],
        }
    except Exception as e:
        logger.warning(f"LMM fitting encountered an issue: {e}")
        lmm_summary["error"] = str(e)

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

