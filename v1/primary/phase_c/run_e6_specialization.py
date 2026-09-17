#!/usr/bin/env python3
"""
v1/primary/phase_c/run_e6_specialization.py

V1 Phase C Primary Script: E6 Targeted Ablation & Double Dissociation Analysis
Evaluates whether Reader and Self recruit task-specific causal sub-circuits
through Targeted Ablation and Linear Mixed-Effects Model (LMM) interaction testing:
  Outcome ~ Task * SiteType + (1 | pair)
"""

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from affective_empathy_eval.intervention import PyTorchActivationPatcher
from affective_empathy_eval.manifests import create_run_manifest


def format_prompt(
    tokenizer, text: str, task_type: str, is_instruct: bool
) -> str:
    if task_type == "reader":
        instruction = "Read the following text and estimate the affective response that this text is likely to evoke in an average human reader."
    elif task_type == "self":
        instruction = "Read the following text and report your affective state."
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    user_content = (
        f"{instruction}\n\n"
        f"Text: {text}\n\n"
        f"Respond strictly in JSON format with 'valence', 'arousal', and 'dominance' keys (integers from 1 to 9)."
    )

    if is_instruct:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
    return (
        f"Task: Evaluate emotional Valence, Arousal, and Dominance (1-9).\n\n"
        f"{instruction}\n\n"
        f"Text: {text}\n\n"
        f"Output:\n"
    )


def build_vad_candidates():
    candidates = []
    vad_triplets = []
    for v in range(1, 10):
        for a in range(1, 10):
            for d in range(1, 10):
                cand_str = (
                    f'{{"valence": {v}, "arousal": {a}, "dominance": {d}}}'
                )
                candidates.append(cand_str)
                vad_triplets.append((v, a, d))
    return candidates, np.array(vad_triplets)


def get_prompt_end_position(tokenizer, prompt: str) -> int:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return len(prompt_ids) - 1


@torch.no_grad()
def evaluate_expected_va_batch(
    model,
    tokenizer,
    prompts: List[str],
    candidates: List[str],
    vad_triplets: np.ndarray,
    device: str = "cuda",
    sub_batch_size: int = 81,
) -> Tuple[np.ndarray, np.ndarray]:
    from affective_empathy_eval.likelihood import (
        compute_sequence_likelihoods_for_candidates,
    )

    exp_v_list, exp_a_list = [], []
    for prompt in prompts:
        log_ll, probs = compute_sequence_likelihoods_for_candidates(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            candidates=candidates,
            device=device,
            batch_size=sub_batch_size,
            normalize_length=False,
        )
        exp_v_list.append(float(np.sum(probs * vad_triplets[:, 0])))
        exp_a_list.append(float(np.sum(probs * vad_triplets[:, 1])))
    return np.array(exp_v_list), np.array(exp_a_list)


def run_lmm_interaction_test(df_long: pd.DataFrame) -> Dict[str, Any]:
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf

        formula = "impact ~ C(task, Treatment(reference='Reader')) * C(site_type, Treatment(reference='ReaderSite'))"
        md = smf.mixedlm(formula, df_long, groups=df_long["pair_id"])
        mdf = md.fit()
        interaction_term = [
            idx for idx in mdf.pvalues.index if ":" in idx or " * " in idx
        ]
        p_val = float(mdf.pvalues[interaction_term[0]]) if interaction_term else 1.0
        coef = float(mdf.params[interaction_term[0]]) if interaction_term else 0.0

        return {
            "model_type": "Linear Mixed-Effects Model (LMM)",
            "formula": formula,
            "p_interaction": p_val,
            "coef_interaction": coef,
            "summary_text": str(mdf.summary()),
        }
    except Exception as ex:
        # Paired t-test fallback on double difference
        piv = df_long.pivot_table(
            index="pair_id", columns=["task", "site_type"], values="impact"
        )
        double_diff = (
            piv[("Reader", "ReaderSite")]
            - piv[("Self", "ReaderSite")]
            - (piv[("Reader", "SelfSite")] - piv[("Self", "SelfSite")])
        )
        t_stat, p_val = stats.ttest_1samp(double_diff.dropna(), 0.0)
        return {
            "model_type": f"Paired t-test on Double Difference (Fallback due to: {ex})",
            "p_interaction": float(p_val),
            "coef_interaction": float(np.mean(double_diff)),
            "summary_text": f"t={t_stat:.4f}, p={p_val:.4e}",
        }


def main():
    parser = argparse.ArgumentParser(
        description="V1 Primary Phase C: E6 Targeted Ablation & Double Dissociation"
    )
    parser.add_argument(
        "--model-id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct"
    )
    parser.add_argument(
        "--model-prefix", type=str, default="qwen2.5_1.5b_instruct"
    )
    parser.add_argument("--reader-layer", type=int, default=14)
    parser.add_argument("--self-layer", type=int, default=16)
    parser.add_argument(
        "--ablation-type",
        type=str,
        choices=["zero", "mean"],
        default="zero",
    )
    parser.add_argument(
        "--split-eval",
        type=str,
        choices=["confirmation", "all"],
        default="confirmation",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="v1/results/derived/v1_phase_c_prompt_end",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model_dir = os.path.join(args.out_dir, args.model_prefix)
    os.makedirs(model_dir, exist_ok=True)

    is_instruct = (
        "instruct" in args.model_id.lower()
        or "chat" in args.model_id.lower()
        or "it" in args.model_id.lower()
    )

    print(
        f"=== Starting V1 Phase C E6 Double Dissociation: {args.model_id} ==="
    )
    print(
        f"Reader-Site Layer: {args.reader_layer} | Self-Site Layer: {args.self_layer}"
    )

    candidates, vad_triplets = build_vad_candidates()

    aipsy_path = Path("v1/data/processed/aipsy_4split_all.csv")
    if not aipsy_path.exists():
        aipsy_path = Path("data/processed/aipsy_4split_all.csv")

    df_aipsy = pd.read_csv(aipsy_path)
    df_clin = df_aipsy[df_aipsy["split"] == "clinical"].copy()
    df_neut = df_aipsy[df_aipsy["split"] == "neutral"].copy()
    merged = pd.merge(
        df_clin, df_neut, on="pair_id", suffixes=("_aff", "_neu")
    ).dropna(subset=["text_aff", "text_neu"])

    if args.limit > 0:
        merged = merged.head(args.limit)

    eval_df = merged.reset_index(drop=True)
    n_pairs = len(eval_df)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        device_map="auto" if args.device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    num_layers = getattr(model.config, "num_hidden_layers", None)
    if num_layers is None and hasattr(model.config, "n_layer"):
        num_layers = model.config.n_layer

    prompts_r_aff = [
        format_prompt(tokenizer, t, "reader", is_instruct)
        for t in eval_df["text_aff"]
    ]
    prompts_s_aff = [
        format_prompt(tokenizer, t, "self", is_instruct)
        for t in eval_df["text_aff"]
    ]

    print("Computing baseline intact predictions...")
    base_v_r, _ = evaluate_expected_va_batch(
        model, tokenizer, prompts_r_aff, candidates, vad_triplets, device=args.device
    )
    base_v_s, _ = evaluate_expected_va_batch(
        model, tokenizer, prompts_s_aff, candidates, vad_triplets, device=args.device
    )

    long_records = []
    print("Executing 2x2 Targeted Ablations...")
    for p_idx in tqdm(range(n_pairs), desc="Ablation Pairs"):
        p_id = eval_df.loc[p_idx, "pair_id"]
        pos_r = get_prompt_end_position(tokenizer, prompts_r_aff[p_idx])
        pos_s = get_prompt_end_position(tokenizer, prompts_s_aff[p_idx])

        # Ablate Reader-Site
        with PyTorchActivationPatcher(
            model,
            args.reader_layer,
            None,
            position=pos_r,
            intervention_type=args.ablation_type,
        ):
            abl_vr_rs, _ = evaluate_expected_va_batch(
                model,
                tokenizer,
                [prompts_r_aff[p_idx]],
                candidates,
                vad_triplets,
                device=args.device,
            )
        with PyTorchActivationPatcher(
            model,
            args.reader_layer,
            None,
            position=pos_s,
            intervention_type=args.ablation_type,
        ):
            abl_vs_rs, _ = evaluate_expected_va_batch(
                model,
                tokenizer,
                [prompts_s_aff[p_idx]],
                candidates,
                vad_triplets,
                device=args.device,
            )

        # Ablate Self-Site
        with PyTorchActivationPatcher(
            model,
            args.self_layer,
            None,
            position=pos_r,
            intervention_type=args.ablation_type,
        ):
            abl_vr_ss, _ = evaluate_expected_va_batch(
                model,
                tokenizer,
                [prompts_r_aff[p_idx]],
                candidates,
                vad_triplets,
                device=args.device,
            )
        with PyTorchActivationPatcher(
            model,
            args.self_layer,
            None,
            position=pos_s,
            intervention_type=args.ablation_type,
        ):
            abl_vs_ss, _ = evaluate_expected_va_batch(
                model,
                tokenizer,
                [prompts_s_aff[p_idx]],
                candidates,
                vad_triplets,
                device=args.device,
            )

        long_records.append(
            {
                "pair_id": p_id,
                "task": "Reader",
                "site_type": "ReaderSite",
                "site_layer": args.reader_layer,
                "impact": float(abs(abl_vr_rs[0] - base_v_r[p_idx])),
            }
        )
        long_records.append(
            {
                "pair_id": p_id,
                "task": "Self",
                "site_type": "ReaderSite",
                "site_layer": args.reader_layer,
                "impact": float(abs(abl_vs_rs[0] - base_v_s[p_idx])),
            }
        )
        long_records.append(
            {
                "pair_id": p_id,
                "task": "Reader",
                "site_type": "SelfSite",
                "site_layer": args.self_layer,
                "impact": float(abs(abl_vr_ss[0] - base_v_r[p_idx])),
            }
        )
        long_records.append(
            {
                "pair_id": p_id,
                "task": "Self",
                "site_type": "SelfSite",
                "site_layer": args.self_layer,
                "impact": float(abs(abl_vs_ss[0] - base_v_s[p_idx])),
            }
        )

    df_long = pd.DataFrame(long_records)
    long_csv_path = os.path.join(
        model_dir, "e6_double_dissociation_trials.csv"
    )
    df_long.to_csv(long_csv_path, index=False)

    stat_results = run_lmm_interaction_test(df_long)
    cell_means = df_long.groupby(["task", "site_type"])["impact"].mean()
    mean_r_rs = float(cell_means.loc["Reader", "ReaderSite"])
    mean_s_rs = float(cell_means.loc["Self", "ReaderSite"])
    mean_r_ss = float(cell_means.loc["Reader", "SelfSite"])
    mean_s_ss = float(cell_means.loc["Self", "SelfSite"])
    has_crossover = (mean_r_rs > mean_s_rs) and (mean_s_ss > mean_r_ss)

    stat_results["cell_means"] = {
        "Reader_ReaderSite": mean_r_rs,
        "Self_ReaderSite": mean_s_rs,
        "Reader_SelfSite": mean_r_ss,
        "Self_SelfSite": mean_s_ss,
    }
    stat_results["has_crossover"] = bool(has_crossover)

    with open(os.path.join(model_dir, "e6_lmm_results.json"), "w") as f:
        json.dump(
            {k: v for k, v in stat_results.items() if k != "summary_text"},
            f,
            indent=2,
        )

    # Save manifest
    manifest = create_run_manifest(
        run_type="v1_phase_c_e6",
        model_name=args.model_id,
        config={
            "model_prefix": args.model_prefix,
            "reader_layer": args.reader_layer,
            "self_layer": args.self_layer,
            "ablation_type": args.ablation_type,
            "limit": args.limit,
        },
        metadata=stat_results,
    )
    manifest.save(os.path.join(model_dir, "manifest_e6.json"))
    print(f"E6 analysis complete. Saved to {model_dir}")


if __name__ == "__main__":
    main()
