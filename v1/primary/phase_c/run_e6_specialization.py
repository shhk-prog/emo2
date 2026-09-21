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
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import torch
from tqdm import tqdm
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # --dry-run は transformers 未導入環境でも起動できるようにする
    AutoModelForCausalLM = None  # type: ignore[misc, assignment]
    AutoTokenizer = None  # type: ignore[misc, assignment]

from affective_empathy_eval.io import is_experiment_completed, save_experiment_result
from affective_empathy_eval.intervention import PyTorchActivationPatcher
from affective_empathy_eval.likelihood import build_vad_candidates
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
            normalize_length=True,
        )
        exp_v_list.append(float(np.sum(probs * vad_triplets[:, 0])))
        exp_a_list.append(float(np.sum(probs * vad_triplets[:, 1])))
    return np.array(exp_v_list), np.array(exp_a_list)


def run_lmm_interaction_test(df_long: pd.DataFrame, target_metric: str = "impact_va") -> Dict[str, Any]:
    metric = target_metric if target_metric in df_long.columns else "impact"
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf

        formula = f"{metric} ~ C(task, Treatment(reference='Reader')) * C(site_type, Treatment(reference='ReaderSite'))"
        md = smf.mixedlm(formula, df_long, groups=df_long["pair_id"])
        mdf = md.fit()
        interaction_term = [
            idx for idx in mdf.pvalues.index if ":" in idx or " * " in idx
        ]
        p_val = float(mdf.pvalues[interaction_term[0]]) if interaction_term else 1.0
        coef = float(mdf.params[interaction_term[0]]) if interaction_term else 0.0

        return {
            "model_type": "Linear Mixed-Effects Model (LMM)",
            "target_metric": metric,
            "formula": formula,
            "p_interaction": p_val,
            "coef_interaction": coef,
            "summary_text": str(mdf.summary()),
        }
    except Exception as ex:
        # Paired t-test fallback on double difference
        piv = df_long.pivot_table(
            index="pair_id", columns=["task", "site_type"], values=metric
        )
        double_diff = (
            piv[("Reader", "ReaderSite")]
            - piv[("Self", "ReaderSite")]
            - (piv[("Reader", "SelfSite")] - piv[("Self", "SelfSite")])
        )
        t_stat, p_val = stats.ttest_1samp(double_diff.dropna(), 0.0)
        return {
            "model_type": f"Paired t-test on Double Difference (Fallback due to: {ex})",
            "target_metric": metric,
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
    selectivity_r = (df[mag_r_col] - df[mag_s_col]).dropna()
    selectivity_s = (df[mag_s_col] - df[mag_r_col]).dropna()

    if selectivity_r.empty or selectivity_s.empty:
        return None, None, "no_distinct_sites_identified", {"error": "All selectivity values are NaN"}

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
        help="Seed for Discovery/Confirmation split (matching E3/E4)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/v1_experiments.yaml",
        help="Path to experiment configuration YAML",
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
    parser.add_argument(
        "--model-revision",
        type=str,
        default=None,
        help="Specific HuggingFace model git commit SHA or branch",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional unique run_id for results organization",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if output files already exist",
    )
    add_model_selection_args(parser)
    args = parser.parse_args()
    args.model_id, args.model_prefix = resolve_single_model_from_args(args)

    # Production safety valve: ensure fixed model revision is resolved
    if args.model_revision is None and not getattr(args, "dry_run", False):
        from affective_empathy_eval.models.registry import get_registry
        registry = get_registry()
        fam_cfg = registry.get_family_by_model_id(args.model_id)
        if fam_cfg:
            for spec in (fam_cfg.base_model, fam_cfg.instruct_model):
                if spec.model_id == args.model_id and spec.revision:
                    args.model_revision = spec.revision
                    break
        if not args.model_revision:
            raise ValueError(
                f"Production run requires explicit fixed model revision for '{args.model_id}', but none was provided or resolved from registry."
            )

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

    aipsy_check_path = Path("v1/data/processed/aipsy_4split_all.csv")
    if not aipsy_check_path.exists():
        aipsy_check_path = Path("data/processed/aipsy_4split_all.csv")

    try:
        num_layers, _ = resolve_architecture_dims(args.model_id)
    except Exception as ex:
        logger.error(
            f"Failed to resolve architecture dimensions for '{args.model_id}': {ex}"
        )
        raise

    from affective_empathy_eval.manifests import (
        is_manifest_matching,
        compute_file_hash,
        compute_string_or_dict_hash,
    )

    dataset_hash = compute_file_hash(aipsy_check_path) if aipsy_check_path.exists() else "unknown"

    # 動的レイヤー決定 (E3 Discovery 結果または指定)
    e3_csv_path = args.e3_csv or os.path.join(model_dir, "e3_causal_map.csv")
    e3_hash = compute_file_hash(e3_csv_path) if os.path.exists(e3_csv_path) else "unknown"
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
            save_experiment_result(
                output_path=os.path.join(model_dir, f"v1_e6_double_dissociation_{args.model_prefix}.json"),
                payload=negative_result,
                stage="v1",
                experiment_id="v1_e6_double_dissociation",
                status="success",
                success=True,
                metadata={"model_id": args.model_id, "model_prefix": args.model_prefix, "result": "negative_no_distinct_sites"},
            )
            with open(os.path.join(model_dir, "e6_lmm_results.json"), "w") as f:
                json.dump(negative_result, f, indent=2)
            manifest = create_run_manifest(
                run_type="v1_phase_c_e6_specialization",
                model_name=args.model_id,
                model_revision=args.model_revision or "main",
                config={
                    "model_prefix": args.model_prefix,
                    "model_id": args.model_id,
                    "model_revision": args.model_revision or "main",
                    "site_selection_method": site_selection_method,
                    "split_eval": args.split_eval,
                    "dataset_hash": dataset_hash,
                    "e3_hash": e3_hash,
                    "dry_run": args.dry_run,
                },
                metadata=negative_result,
                candidate_space="VAD_729",
                measurement_space="VA_expectation_from_VAD_729",
                intervention_version="none",
                run_id=args.run_id,
                dry_run=args.dry_run,
            )
            manifest.save(os.path.join(model_dir, "manifest_e6.json"))
            print(f"Recorded negative result to {model_dir}/e6_lmm_results.json")
            return

        if args.reader_layer is None:
            args.reader_layer = r_l
        if args.self_layer is None:
            args.self_layer = s_l

    manifest_config = {
        "model_prefix": args.model_prefix,
        "model_id": args.model_id,
        "model_revision": args.model_revision or "main",
        "reader_layer": args.reader_layer,
        "self_layer": args.self_layer,
        "site_selection_method": site_selection_method,
        "ablation_type": args.ablation_type,
        "split_eval": args.split_eval,
        "split_seed": args.split_seed,
        "limit": args.limit,
        "dataset_hash": dataset_hash,
        "e3_hash": e3_hash,
    }
    expected_config_hash = compute_string_or_dict_hash(manifest_config)

    # Early skip if already completed and valid
    modular_e6_json = os.path.join(model_dir, f"v1_e6_double_dissociation_{args.model_prefix}.json")
    manifest_path = os.path.join(model_dir, "manifest_e6.json")
    lmm_path = os.path.join(model_dir, "e6_lmm_results.json")

    if not args.force and not args.dry_run and os.path.exists(manifest_path):
        manifest_valid = is_manifest_matching(
            manifest_path=manifest_path,
            expected_model_name=args.model_id,
            expected_config_hash=expected_config_hash,
            expected_dataset_hash=dataset_hash,
            expected_model_revision=args.model_revision,
            expected_dry_run=False,
        )
        if manifest_valid:
            target_check = modular_e6_json if os.path.exists(modular_e6_json) else lmm_path
            if is_experiment_completed(target_check, manifest_path=manifest_path):
                print(
                    f"[SKIP] Validated Phase C E6 results matching manifest found in {model_dir}. "
                    f"Skipping computation for {args.model_prefix}. Use --force to rerun."
                )
                return

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
        save_experiment_result(
            output_path=modular_e6_json,
            payload=stat_results,
            stage="v1",
            experiment_id="v1_e6_double_dissociation",
            status="success",
            success=True,
            metadata={"model_id": args.model_id, "model_prefix": args.model_prefix, "dry_run": True},
        )
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
                    "impact_va": 0.55,
                    "impact_v": 0.45,
                    "impact_a": 0.30,
                    "delta_v": 0.45,
                    "delta_a": 0.30,
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
            model_revision=args.model_revision or "main",
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
            candidate_space="VAD_729",
            measurement_space="VA_expectation_from_VAD_729",
            intervention_version="prompt_end_normalized",
            run_id=args.run_id,
            dry_run=True,
        )
        manifest.save(os.path.join(model_dir, "manifest_e6.json"))
        print(f"[DRY-RUN] Completed E6 mock output in {model_dir}")
        return

    cand_dicts = build_vad_candidates()
    candidates = [c["json_str"] for c in cand_dicts]
    vad_triplets = np.array([(c["valence"], c["arousal"], c["dominance"]) for c in cand_dicts])

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

    # Group split by pair_id matching E3/E4 from config
    phase_c_cfg = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            phase_c_cfg = yaml.safe_load(f).get("phase_c", {})
    discovery_ratio = float(phase_c_cfg.get("discovery_ratio", 0.5))

    rng_split = np.random.default_rng(args.split_seed)
    unique_pairs = merged["pair_id"].unique()
    perm_pairs = rng_split.permutation(unique_pairs)
    split_cut = int(round(len(perm_pairs) * discovery_ratio))
    discovery_pairs_set = set(perm_pairs[:split_cut])
    confirmation_pairs_set = set(perm_pairs[split_cut:])
    assert discovery_pairs_set.isdisjoint(confirmation_pairs_set), (
        "Data leakage! Discovery and Confirmation pair sets must be strictly disjoint."
    )
    
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

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_cuda = str(args.device).startswith("cuda") and torch.cuda.is_available()
    actual_torch_dtype = (
        torch.bfloat16
        if is_cuda and torch.cuda.is_bf16_supported()
        else (torch.float16 if is_cuda else torch.float32)
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        torch_dtype=actual_torch_dtype,
        device_map=args.device if is_cuda else None,
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

    print("Computing baseline intact predictions (VA)...")
    base_v_r, base_a_r = evaluate_expected_va_batch(
        model, tokenizer, prompts_r_aff, candidates, vad_triplets, device=args.device
    )
    base_v_s, base_a_s = evaluate_expected_va_batch(
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
            abl_vr_rs, abl_ar_rs = evaluate_expected_va_batch(
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
            abl_vs_rs, abl_as_rs = evaluate_expected_va_batch(
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
            abl_vr_ss, abl_ar_ss = evaluate_expected_va_batch(
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
            abl_vs_ss, abl_as_ss = evaluate_expected_va_batch(
                model,
                tokenizer,
                [prompts_s_aff[p_idx]],
                candidates,
                vad_triplets,
                device=args.device,
            )

        # Compute 2D VA displacements and scalar impacts
        dv_r_rs = float(abl_vr_rs[0] - base_v_r[p_idx])
        da_r_rs = float(abl_ar_rs[0] - base_a_r[p_idx])
        imp_v_r_rs = abs(dv_r_rs)
        imp_a_r_rs = abs(da_r_rs)
        imp_va_r_rs = float(np.sqrt(dv_r_rs**2 + da_r_rs**2))

        dv_s_rs = float(abl_vs_rs[0] - base_v_s[p_idx])
        da_s_rs = float(abl_as_rs[0] - base_a_s[p_idx])
        imp_v_s_rs = abs(dv_s_rs)
        imp_a_s_rs = abs(da_s_rs)
        imp_va_s_rs = float(np.sqrt(dv_s_rs**2 + da_s_rs**2))

        dv_r_ss = float(abl_vr_ss[0] - base_v_r[p_idx])
        da_r_ss = float(abl_ar_ss[0] - base_a_r[p_idx])
        imp_v_r_ss = abs(dv_r_ss)
        imp_a_r_ss = abs(da_r_ss)
        imp_va_r_ss = float(np.sqrt(dv_r_ss**2 + da_r_ss**2))

        dv_s_ss = float(abl_vs_ss[0] - base_v_s[p_idx])
        da_s_ss = float(abl_as_ss[0] - base_a_s[p_idx])
        imp_v_s_ss = abs(dv_s_ss)
        imp_a_s_ss = abs(da_s_ss)
        imp_va_s_ss = float(np.sqrt(dv_s_ss**2 + da_s_ss**2))

        long_records.append(
            {
                "pair_id": p_id,
                "task": "Reader",
                "site_type": "ReaderSite",
                "site_layer": args.reader_layer,
                "impact": imp_va_r_rs,
                "impact_va": imp_va_r_rs,
                "impact_v": imp_v_r_rs,
                "impact_a": imp_a_r_rs,
                "delta_v": dv_r_rs,
                "delta_a": da_r_rs,
            }
        )
        long_records.append(
            {
                "pair_id": p_id,
                "task": "Self",
                "site_type": "ReaderSite",
                "site_layer": args.reader_layer,
                "impact": imp_va_s_rs,
                "impact_va": imp_va_s_rs,
                "impact_v": imp_v_s_rs,
                "impact_a": imp_a_s_rs,
                "delta_v": dv_s_rs,
                "delta_a": da_s_rs,
            }
        )
        long_records.append(
            {
                "pair_id": p_id,
                "task": "Reader",
                "site_type": "SelfSite",
                "site_layer": args.self_layer,
                "impact": imp_va_r_ss,
                "impact_va": imp_va_r_ss,
                "impact_v": imp_v_r_ss,
                "impact_a": imp_a_r_ss,
                "delta_v": dv_r_ss,
                "delta_a": da_r_ss,
            }
        )
        long_records.append(
            {
                "pair_id": p_id,
                "task": "Self",
                "site_type": "SelfSite",
                "site_layer": args.self_layer,
                "impact": imp_va_s_ss,
                "impact_va": imp_va_s_ss,
                "impact_v": imp_v_s_ss,
                "impact_a": imp_a_s_ss,
                "delta_v": dv_s_ss,
                "delta_a": da_s_ss,
            }
        )

        # Incremental save
        spec_csv_path = os.path.join(
            model_dir, "e6_specialization_trials.csv"
        )
        pd.DataFrame(long_records).to_csv(spec_csv_path, index=False)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df_long = pd.DataFrame(long_records)

    # Primary LMM interaction test on impact_va
    stat_results = run_lmm_interaction_test(df_long, target_metric="impact_va")
    # Secondary tests on individual V and A axes
    stat_results_v = run_lmm_interaction_test(df_long, target_metric="impact_v")
    stat_results_a = run_lmm_interaction_test(df_long, target_metric="impact_a")
    stat_results["secondary_v"] = stat_results_v
    stat_results["secondary_a"] = stat_results_a

    cell_means_va = df_long.groupby(["task", "site_type"])["impact_va"].mean()
    cell_means_v = df_long.groupby(["task", "site_type"])["impact_v"].mean()
    cell_means_a = df_long.groupby(["task", "site_type"])["impact_a"].mean()

    mean_r_rs = float(cell_means_va.loc["Reader", "ReaderSite"])
    mean_s_rs = float(cell_means_va.loc["Self", "ReaderSite"])
    mean_r_ss = float(cell_means_va.loc["Reader", "SelfSite"])
    mean_s_ss = float(cell_means_va.loc["Self", "SelfSite"])
    has_crossover = (mean_r_rs > mean_s_rs) and (mean_s_ss > mean_r_ss)

    stat_results["cell_means_va"] = {
        "Reader_ReaderSite": mean_r_rs,
        "Self_ReaderSite": mean_s_rs,
        "Reader_SelfSite": mean_r_ss,
        "Self_SelfSite": mean_s_ss,
    }
    stat_results["cell_means"] = stat_results["cell_means_va"]
    stat_results["cell_means_v"] = {
        "Reader_ReaderSite": float(cell_means_v.loc["Reader", "ReaderSite"]),
        "Self_ReaderSite": float(cell_means_v.loc["Self", "ReaderSite"]),
        "Reader_SelfSite": float(cell_means_v.loc["Reader", "SelfSite"]),
        "Self_SelfSite": float(cell_means_v.loc["Self", "SelfSite"]),
    }
    stat_results["cell_means_a"] = {
        "Reader_ReaderSite": float(cell_means_a.loc["Reader", "ReaderSite"]),
        "Self_ReaderSite": float(cell_means_a.loc["Self", "ReaderSite"]),
        "Reader_SelfSite": float(cell_means_a.loc["Reader", "SelfSite"]),
        "Self_SelfSite": float(cell_means_a.loc["Self", "SelfSite"]),
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

    save_experiment_result(
        output_path=modular_e6_json,
        payload={k: v for k, v in stat_results.items() if k != "summary_text"},
        stage="v1",
        experiment_id="v1_e6_double_dissociation",
        status="success",
        success=True,
        metadata={
            "model_id": args.model_id,
            "model_prefix": args.model_prefix,
            "reader_layer": args.reader_layer,
            "self_layer": args.self_layer,
            "site_selection_method": site_selection_method,
            "has_crossover": bool(has_crossover),
        },
    )

    # Save manifest
    manifest = create_run_manifest(
        run_type="v1_phase_c_e6_specialization",
        model_name=args.model_id,
        model_revision=args.model_revision or "main",
        config=manifest_config,
        dataset_hash=dataset_hash,
        metadata=stat_results,
        candidate_space="VAD_729",
        measurement_space="VA_expectation_from_VAD_729",
        actual_dtype=str(actual_torch_dtype).replace("torch.", ""),
        intervention_version="prompt_end_normalized",
        run_id=args.run_id,
        dry_run=args.dry_run,
    )
    manifest.save(os.path.join(model_dir, "manifest_e6.json"))
    print(f"E6 causal specialization analysis complete. Saved to {model_dir}")


if __name__ == "__main__":
    main()
