#!/usr/bin/env python3
"""
v1/primary/phase_c/run_e6_specialization.py

V1 Phase C Primary Script: E6 Targeted Ablation & Task-Specific Causal Specialization
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
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # --dry-run は transformers 未導入環境でも起動できるようにする
    AutoModelForCausalLM = None  # type: ignore[misc, assignment]
    AutoTokenizer = None  # type: ignore[misc, assignment]

from affective_empathy_eval.intervention import PyTorchActivationPatcher
from affective_empathy_eval.manifests import create_run_manifest
from affective_empathy_eval.models.registry import (
    add_model_selection_args,
    resolve_architecture_dims,
    resolve_single_model_from_args,
)


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


def select_sites_from_e3(
    e3_csv_path: str, num_layers: int
) -> Tuple[Optional[int], Optional[int], str, Dict[str, Any]]:
    """
    Select Reader-selective and Self-selective sites deterministically from E3 Discovery results
    based on task selectivity contrast:
      S_R(l) = C_R(l) - C_S(l)
      S_S(l) = C_S(l) - C_R(l)

    If distinct task-selective sites are not identified (e.g., max selectivity <= 0,
    or reader_layer == self_layer), returns (None, None, "no_distinct_sites_identified", info).
    """
    if not os.path.exists(e3_csv_path):
        raise FileNotFoundError(
            f"E3 Discovery results not found at '{e3_csv_path}'. "
            "E6 requires valid E3 Discovery causal map CSV to select task-selective sites. "
            "Heuristic fallback is strictly prohibited."
        )

    df = pd.read_csv(e3_csv_path)
    mag_r_col = (
        "discovery_mag_reader"
        if "discovery_mag_reader" in df.columns
        else ("magnitude_reader" if "magnitude_reader" in df.columns else None)
    )
    mag_s_col = (
        "discovery_mag_self"
        if "discovery_mag_self" in df.columns
        else ("magnitude_self" if "magnitude_self" in df.columns else None)
    )

    if mag_r_col is None or mag_s_col is None or "layer" not in df.columns:
        raise ValueError(
            f"E3 CSV at '{e3_csv_path}' is missing required magnitude columns "
            f"('{mag_r_col}', '{mag_s_col}') or 'layer' column."
        )

    # Task selectivity contrast
    selectivity_r = df[mag_r_col] - df[mag_s_col]
    selectivity_s = df[mag_s_col] - df[mag_r_col]

    reader_peak_idx = selectivity_r.idxmax()
    self_peak_idx = selectivity_s.idxmax()

    reader_peak_l = int(df.loc[reader_peak_idx, "layer"])
    self_peak_l = int(df.loc[self_peak_idx, "layer"])

    max_sel_r = float(selectivity_r.max())
    max_sel_s = float(selectivity_s.max())

    info = {
        "max_selectivity_reader": max_sel_r,
        "max_selectivity_self": max_sel_s,
        "reader_selective_layer": reader_peak_l,
        "self_selective_layer": self_peak_l,
    }

    # 科学的判定: 同一レイヤーまたは選択性が正でない場合は No-Go
    if reader_peak_l == self_peak_l or max_sel_r <= 0.0 or max_sel_s <= 0.0:
        return None, None, "no_distinct_sites_identified", info

    return reader_peak_l, self_peak_l, "task_selectivity_contrast", info


def main():
    parser = argparse.ArgumentParser(
        description="V1 Primary Phase C: E6 Targeted Ablation & Causal Specialization"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Hugging Face model ID. Required unless --family is set.",
    )
    parser.add_argument(
        "--model-prefix",
        type=str,
        default=None,
        help="Output prefix. Derived from --family or --model-id if omitted.",
    )
    parser.add_argument(
        "--e3-csv",
        type=str,
        default=None,
        help="Path to e3_causal_map.csv from which to extract Discovery peaks",
    )
    parser.add_argument("--reader-layer", type=int, default=None)
    parser.add_argument("--self-layer", type=int, default=None)
    parser.add_argument(
        "--ablation-type",
        type=str,
        choices=["zero", "mean"],
        default="zero",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for 50/50 Discovery/Confirmation split (matching E3/E4)",
    )
    parser.add_argument(
        "--split-eval",
        type=str,
        choices=["confirmation", "all"],
        default="confirmation",
        help="Split to evaluate targeted ablation on (default: confirmation)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Sample limit (0 for full dataset)")
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
    parser.add_argument("--is-instruct", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mock dry-run mode for quick pipeline smoke testing",
    )
    add_model_selection_args(parser)
    args = parser.parse_args()
    args.model_id, args.model_prefix = resolve_single_model_from_args(args)

    if args.dry_run:
        args.out_dir = os.path.join(args.out_dir, "dry_run")

    os.makedirs(args.out_dir, exist_ok=True)
    model_dir = os.path.join(args.out_dir, args.model_prefix)
    os.makedirs(model_dir, exist_ok=True)

    is_instruct = (
        args.is_instruct
        or "instruct" in args.model_id.lower()
        or "chat" in args.model_id.lower()
        or "it" in args.model_id.lower()
    )

    try:
        num_layers, _ = resolve_architecture_dims(args.model_id)
    except Exception as ex:
        logger.error(
            f"Failed to resolve architecture dimensions for '{args.model_id}': {ex}"
        )
        raise

    # 動的レイヤー決定 (E3 Discovery 結果または指定)
    e3_csv_path = args.e3_csv or os.path.join(model_dir, "e3_causal_map.csv")
    site_selection_method = "manual"
    selectivity_info: Dict[str, Any] = {}

    if args.reader_layer is None or args.self_layer is None:
        if args.dry_run and not os.path.exists(e3_csv_path):
            # Standalone dry-run mock sites when E3 CSV does not exist yet
            r_l, s_l = 0, min(14, num_layers - 1)
            site_selection_method = "dry_run_mock"
            selectivity_info = {"mock": True}
        else:
            r_l, s_l, site_selection_method, selectivity_info = select_sites_from_e3(
                e3_csv_path, num_layers
            )

        if site_selection_method == "no_distinct_sites_identified":
            print(
                f"=== V1 Phase C E6: No distinct task-selective sites identified for {args.model_id} ==="
            )
            print(f"Selectivity Info: {selectivity_info}")
            print("Scientific Decision: Negative Result (No-Go for targeted ablation). Skipping ablation.")
            negative_result = {
                "status": "negative_result_no_distinct_sites",
                "distinct_sites_found": False,
                "site_selection_method": "task_selectivity_contrast",
                "reason": "Task selectivity contrast did not identify distinct sites for Reader and Self.",
                "selectivity_info": selectivity_info,
                "p_value_interaction": None,
                "coef_interaction": None,
                "has_crossover": False,
            }
            with open(os.path.join(model_dir, "e6_lmm_results.json"), "w") as f:
                json.dump(negative_result, f, indent=2)
            manifest = create_run_manifest(
                run_type="v1_phase_c_e6_specialization",
                model_name=args.model_id,
                config={
                    "model_prefix": args.model_prefix,
                    "site_selection_method": site_selection_method,
                    "split_eval": args.split_eval,
                    "dry_run": args.dry_run,
                },
                metadata=negative_result,
            )
            manifest.save(os.path.join(model_dir, "manifest_e6.json"))
            print(f"Recorded negative result to {model_dir}/e6_lmm_results.json and manifest_e6.json")
            return

        if args.reader_layer is None:
            args.reader_layer = r_l
        if args.self_layer is None:
            args.self_layer = s_l

    print(
        f"=== Starting V1 Phase C E6 Causal Specialization: {args.model_id} ==="
    )
    print(
        f"Site Selection: {site_selection_method} (Source: {e3_csv_path})"
    )
    print(
        f"Reader-Site Layer: {args.reader_layer} | Self-Site Layer: {args.self_layer}"
    )
    print(
        f"Evaluation Split: {args.split_eval} (Seed: {args.split_seed})"
    )

    if args.dry_run:
        print(f"[DRY-RUN] V1 Phase C E6 for Model: {args.model_id}")
        stat_results = {
            "p_value_interaction": 0.001,
            "coef_interaction": 0.42,
            "cell_means": {
                "Reader_ReaderSite": 0.55,
                "Self_ReaderSite": 0.20,
                "Reader_SelfSite": 0.18,
                "Self_SelfSite": 0.62,
            },
            "has_crossover": True,
            "site_selection_method": site_selection_method,
            "reader_layer": args.reader_layer,
            "self_layer": args.self_layer,
            "split_eval": args.split_eval,
        }
        with open(os.path.join(model_dir, "e6_lmm_results.json"), "w") as f:
            json.dump(stat_results, f, indent=2)
        dry_df = pd.DataFrame(
            [
                {
                    "pair_id": "dry_pair_1",
                    "eval_split": args.split_eval,
                    "task": "Reader",
                    "site_type": "ReaderSite",
                    "site_layer": args.reader_layer,
                    "impact": 0.55,
                }
            ]
        )
        dry_df.to_csv(
            os.path.join(model_dir, "e6_specialization_trials.csv"),
            index=False,
        )
        manifest = create_run_manifest(
            run_type="v1_phase_c_e6_specialization",
            model_name=args.model_id,
            config={
                "model_prefix": args.model_prefix,
                "reader_layer": args.reader_layer,
                "self_layer": args.self_layer,
                "site_selection_method": site_selection_method,
                "split_eval": args.split_eval,
                "split_seed": args.split_seed,
                "dry_run": True,
            },
            metadata=stat_results,
        )
        manifest.save(os.path.join(model_dir, "manifest_e6.json"))
        print(f"[DRY-RUN] Completed E6 mock output in {model_dir}")
        return

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

    # 50/50 Group split by pair_id matching E3/E4
    rng_split = np.random.default_rng(args.split_seed)
    unique_pairs = merged["pair_id"].unique()
    perm_pairs = rng_split.permutation(unique_pairs)
    split_cut = len(perm_pairs) // 2
    discovery_pairs_set = set(perm_pairs[:split_cut])
    
    merged["eval_split"] = [
        "discovery" if pid in discovery_pairs_set else "confirmation"
        for pid in merged["pair_id"]
    ]

    if args.split_eval == "confirmation":
        eval_df = merged[merged["eval_split"] == "confirmation"].reset_index(drop=True)
        print(f"Targeted Ablation strictly evaluated on Confirmation split: N={len(eval_df)} pairs (Discovery N={len(discovery_pairs_set)} reserved for site selection)")
    else:
        eval_df = merged.reset_index(drop=True)
        print(f"Targeted Ablation evaluated on ALL pairs: N={len(eval_df)} pairs")

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
    spec_csv_path = os.path.join(
        model_dir, "e6_specialization_trials.csv"
    )
    df_long.to_csv(spec_csv_path, index=False)

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
    stat_results["site_selection_method"] = site_selection_method
    stat_results["selectivity_info"] = selectivity_info
    stat_results["reader_layer"] = args.reader_layer
    stat_results["self_layer"] = args.self_layer
    stat_results["split_eval"] = args.split_eval
    stat_results["n_evaluation_pairs"] = len(eval_df)

    with open(os.path.join(model_dir, "e6_lmm_results.json"), "w") as f:
        json.dump(
            {k: v for k, v in stat_results.items() if k != "summary_text"},
            f,
            indent=2,
        )

    # Save manifest
    manifest = create_run_manifest(
        run_type="v1_phase_c_e6_specialization",
        model_name=args.model_id,
        config={
            "model_prefix": args.model_prefix,
            "reader_layer": args.reader_layer,
            "self_layer": args.self_layer,
            "site_selection_method": site_selection_method,
            "ablation_type": args.ablation_type,
            "split_eval": args.split_eval,
            "split_seed": args.split_seed,
            "limit": args.limit,
        },
        metadata=stat_results,
    )
    manifest.save(os.path.join(model_dir, "manifest_e6.json"))
    print(f"E6 causal specialization analysis complete. Saved to {model_dir}")


if __name__ == "__main__":
    main()
