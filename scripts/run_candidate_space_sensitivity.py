#!/usr/bin/env python3
"""
scripts/run_candidate_space_sensitivity.py

729 VAD vs 81 VA Candidate-Space Sensitivity Analysis.
Evaluates whether directional affective shifts (Delta V, Delta A) between clinical emotional
stimuli and matched neutral controls are preserved between the 729 VAD (3D) and 81 VA (2D)
candidate spaces under identical model, prompt, and stimulus conditions on AIPsy matched pairs.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch

from affective_empathy_eval.likelihood import (
    build_va_candidates,
    build_vad_candidates,
    compute_expected_va,
    compute_sequence_likelihoods_for_candidates,
)
from affective_empathy_eval.manifests import create_run_manifest
from affective_empathy_eval.models.registry import get_registry, add_model_selection_args
from affective_empathy_eval.prompts import TaskType, build_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_expected_vad_values(likelihoods: List[float], vad_cand_dicts: List[Dict[str, Any]]) -> Tuple[float, float, float]:
    """729候補の対数尤度から Softmax 確率分布を求め E[V], E[A], E[D] を算出"""
    l_arr = np.array(likelihoods, dtype=np.float64)
    l_max = np.max(l_arr)
    probs = np.exp(l_arr - l_max)
    probs = probs / np.sum(probs)

    e_v = float(np.sum(probs * [c["valence"] for c in vad_cand_dicts]))
    e_a = float(np.sum(probs * [c["arousal"] for c in vad_cand_dicts]))
    e_d = float(np.sum(probs * [c["dominance"] for c in vad_cand_dicts]))
    return e_v, e_a, e_d


def run_sensitivity_simulation(n_pairs: int = 20, seed: int = 42) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """dry-run / テスト用シミュレーション: AIPsy matched pair Delta V, Delta A の感度分析"""
    rng = np.random.default_rng(seed)

    # 729 空間での真の情動変位 Delta V, Delta A
    delta_v_729 = rng.uniform(-2.5, 2.5, size=n_pairs)
    delta_a_729 = rng.uniform(-2.0, 2.0, size=n_pairs)

    # 81 空間では高相関（r > 0.95）かつ微小なノイズを伴う変位
    delta_v_81 = delta_v_729 * 0.98 + rng.normal(0, 0.10, size=n_pairs)
    delta_a_81 = delta_a_729 * 0.97 + rng.normal(0, 0.12, size=n_pairs)

    r_v, _ = pearsonr(delta_v_729, delta_v_81)
    rho_v, _ = spearmanr(delta_v_729, delta_v_81)
    r_a, _ = pearsonr(delta_a_729, delta_a_81)
    rho_a, _ = spearmanr(delta_a_729, delta_a_81)

    dir_agree_v = float(np.mean(np.sign(delta_v_729) == np.sign(delta_v_81)))
    dir_agree_a = float(np.mean(np.sign(delta_a_729) == np.sign(delta_a_81)))

    records = []
    for i in range(n_pairs):
        records.append({
            "pair_id": f"sim_pair_{i:03d}",
            "delta_v_729": float(delta_v_729[i]),
            "delta_a_729": float(delta_a_729[i]),
            "delta_v_81": float(delta_v_81[i]),
            "delta_a_81": float(delta_a_81[i]),
            "diff_delta_v": float(delta_v_729[i] - delta_v_81[i]),
            "diff_delta_a": float(delta_a_729[i] - delta_a_81[i]),
        })

    summary = {
        "status": "success",
        "dry_run": True,
        "n_pairs": n_pairs,
        "valence_metrics": {
            "pearson_r": float(r_v),
            "spearman_rho": float(rho_v),
            "direction_agreement": dir_agree_v,
            "mae": float(np.mean(np.abs(delta_v_729 - delta_v_81))),
        },
        "arousal_metrics": {
            "pearson_r": float(r_a),
            "spearman_rho": float(rho_a),
            "direction_agreement": dir_agree_a,
            "mae": float(np.mean(np.abs(delta_a_729 - delta_a_81))),
        },
    }
    return summary, records


def run_sensitivity_analysis(
    model_id: str,
    stimuli_path: str,
    model_revision: Optional[str] = None,
    n_pairs: int = 20,
    device: str = "cpu",
    task: str = "self",
    seed: int = 42,
    dry_run: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """実モデルを用いた 729 vs 81 感度分析の実行 (AIPsy matched pair Delta V, Delta A)"""
    if dry_run:
        return run_sensitivity_simulation(n_pairs=n_pairs, seed=seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model {model_id} (revision={model_revision}) on {device} (task={task})...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=model_revision,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_revision,
        torch_dtype=torch.bfloat16 if is_cuda else torch.float32,
        device_map=device if is_cuda else None,
        trust_remote_code=True,
    )
    model.eval()

    df = pd.read_csv(stimuli_path)
    if "pair_id" not in df.columns:
        raise ValueError(f"Stimuli dataset at {stimuli_path} must contain 'pair_id' column for matched pair sensitivity analysis.")

    # Select unique pairs
    unique_pairs = df["pair_id"].dropna().unique()
    rng = np.random.default_rng(seed)
    selected_pairs = rng.choice(unique_pairs, size=min(n_pairs, len(unique_pairs)), replace=False)

    cands_81 = build_va_candidates()
    cands_729 = build_vad_candidates()
    cands_729_strs = [c["json_str"] for c in cands_729]

    task_type = TaskType.SELF if task.lower() == "self" else TaskType.READER

    records = []
    with torch.no_grad():
        for pair_id in selected_pairs:
            pair_rows = df[df["pair_id"] == pair_id]
            intensity = pair_rows["intensity"].astype(str).str.lower()
            neu_rows = pair_rows[intensity.eq("none")]
            clin_rows = pair_rows[intensity.isin(["peak", "clinical"])]

            if len(neu_rows) != 1 or len(clin_rows) != 1:
                raise ValueError(
                    f"Pair {pair_id} does not have exactly 1 neutral row and 1 clinical/peak row. "
                    f"Found {len(neu_rows)} neutral, {len(clin_rows)} clinical rows."
                )

            neu_text = str(neu_rows.iloc[0]["text"])
            clin_text = str(clin_rows.iloc[0]["text"])

            prompt_neu = build_prompt(neu_text, task=task_type, format_type="chat", tokenizer=tokenizer)
            prompt_clin = build_prompt(clin_text, task=task_type, format_type="chat", tokenizer=tokenizer)

            # --- 81 VA candidates ---
            lik_neu_81, _ = compute_sequence_likelihoods_for_candidates(
                model=model, tokenizer=tokenizer, prompt=prompt_neu, candidates=cands_81, device=device, batch_size=81
            )
            v_neu_81, a_neu_81 = compute_expected_va(lik_neu_81, cands_81)

            lik_clin_81, _ = compute_sequence_likelihoods_for_candidates(
                model=model, tokenizer=tokenizer, prompt=prompt_clin, candidates=cands_81, device=device, batch_size=81
            )
            v_clin_81, a_clin_81 = compute_expected_va(lik_clin_81, cands_81)

            delta_v_81 = v_clin_81 - v_neu_81
            delta_a_81 = a_clin_81 - a_neu_81

            # --- 729 VAD candidates ---
            lik_neu_729, _ = compute_sequence_likelihoods_for_candidates(
                model=model, tokenizer=tokenizer, prompt=prompt_neu, candidates=cands_729_strs, device=device, batch_size=81
            )
            v_neu_729, a_neu_729, _ = compute_expected_vad_values(lik_neu_729, cands_729)

            lik_clin_729, _ = compute_sequence_likelihoods_for_candidates(
                model=model, tokenizer=tokenizer, prompt=prompt_clin, candidates=cands_729_strs, device=device, batch_size=81
            )
            v_clin_729, a_clin_729, _ = compute_expected_vad_values(lik_clin_729, cands_729)

            delta_v_729 = v_clin_729 - v_neu_729
            delta_a_729 = a_clin_729 - a_neu_729

            records.append({
                "pair_id": pair_id,
                "delta_v_729": delta_v_729,
                "delta_a_729": delta_a_729,
                "delta_v_81": delta_v_81,
                "delta_a_81": delta_a_81,
                "diff_delta_v": delta_v_729 - delta_v_81,
                "diff_delta_a": delta_a_729 - delta_a_81,
            })

    dv_729 = np.array([r["delta_v_729"] for r in records])
    dv_81 = np.array([r["delta_v_81"] for r in records])
    da_729 = np.array([r["delta_a_729"] for r in records])
    da_81 = np.array([r["delta_a_81"] for r in records])

    r_v, _ = pearsonr(dv_729, dv_81) if np.std(dv_729) > 0 and np.std(dv_81) > 0 else (np.nan, np.nan)
    rho_v, _ = spearmanr(dv_729, dv_81) if np.std(dv_729) > 0 and np.std(dv_81) > 0 else (np.nan, np.nan)
    r_a, _ = pearsonr(da_729, da_81) if np.std(da_729) > 0 and np.std(da_81) > 0 else (np.nan, np.nan)
    rho_a, _ = spearmanr(da_729, da_81) if np.std(da_729) > 0 and np.std(da_81) > 0 else (np.nan, np.nan)

    dir_agree_v = float(np.mean(np.sign(dv_729) == np.sign(dv_81))) if len(dv_729) > 0 else 0.0
    dir_agree_a = float(np.mean(np.sign(da_729) == np.sign(da_81))) if len(da_729) > 0 else 0.0

    summary = {
        "status": "success",
        "dry_run": False,
        "model_id": model_id,
        "model_revision": model_revision or "main",
        "task": task,
        "stimuli_path": stimuli_path,
        "n_pairs": len(records),
        "valence_metrics": {
            "pearson_r": float(r_v),
            "spearman_rho": float(rho_v),
            "direction_agreement": dir_agree_v,
            "mae": float(np.mean(np.abs(dv_729 - dv_81))) if len(dv_729) > 0 else 0.0,
        },
        "arousal_metrics": {
            "pearson_r": float(r_a),
            "spearman_rho": float(rho_a),
            "direction_agreement": dir_agree_a,
            "mae": float(np.mean(np.abs(da_729 - da_81))) if len(da_729) > 0 else 0.0,
        },
    }
    return summary, records


def main():
    parser = argparse.ArgumentParser(description="729 VAD vs 81 VA Candidate-Space Sensitivity Analysis on Matched Pairs")
    parser.add_argument("--task", type=str, default="self", choices=["reader", "self"], help="Task condition: reader or self")
    parser.add_argument("--stimuli-path", type=str, default="v1/data/processed/aipsy_4split_all.csv", help="AIPsy matched pair CSV")
    parser.add_argument("--n-samples", type=int, default=20, help="Number of evaluation pairs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Run simulation dry-run")
    parser.add_argument("--out-dir", type=str, default="results/derived/candidate_space_sensitivity")
    parser.add_argument("--model-revision", type=str, default=None, help="HuggingFace model git commit SHA or branch")
    add_model_selection_args(parser)
    args = parser.parse_args()

    reg = get_registry()
    target_family = args.family or "qwen"
    target_revision = args.model_revision
    if target_family in reg.families:
        fam_cfg = reg.families[target_family]
        target_model_id = fam_cfg.instruct_model.model_id
        if not target_revision:
            target_revision = fam_cfg.instruct_model.revision
    else:
        target_model_id = args.model_id or "Qwen/Qwen2.5-1.5B-Instruct"

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary, records = run_sensitivity_analysis(
        model_id=target_model_id,
        stimuli_path=args.stimuli_path,
        model_revision=target_revision,
        n_pairs=args.n_samples,
        device=args.device,
        task=args.task,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    with open(out_path / "sensitivity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(records).to_csv(out_path / "sensitivity_comparison_records.csv", index=False)

    manifest = create_run_manifest(
        run_type="candidate_space_sensitivity",
        model_name=target_model_id,
        model_revision=target_revision or "main",
        config={
            "n_pairs": args.n_samples,
            "seed": args.seed,
            "model_revision": target_revision,
            "dry_run": args.dry_run,
        },
        metadata=summary,
        dataset_path=args.stimuli_path,
        candidate_space="VAD_729_vs_VA_81",
        intervention_version="none",
        dry_run=args.dry_run,
    )
    manifest.save(str(out_path / "manifest.json"))

    print("\n" + "=" * 60)
    print("Candidate-Space Sensitivity Analysis on Matched Pairs Complete")
    print(f"Valence (Delta V): Pearson r = {summary['valence_metrics']['pearson_r']:.4f}, Dir Agreement = {summary['valence_metrics']['direction_agreement']:.2%}")
    print(f"Arousal (Delta A): Pearson r = {summary['arousal_metrics']['pearson_r']:.4f}, Dir Agreement = {summary['arousal_metrics']['direction_agreement']:.2%}")
    print(f"Saved results to {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
