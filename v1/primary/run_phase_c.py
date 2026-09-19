#!/usr/bin/env python3
"""
v1/primary/run_phase_c.py

V1 Phase C Primary Script: Causal Interventions and Interchangeability
Evaluates:
  - E3: Shared Causal Map (Magnitude + 2D Directional Vector Cosine Similarity across layers)
  - E4: Causal Interchangeability (Matched vs. Random Difference-Vector Patching on Candidate Layers)

Strict Features:
  - Prompt-End Normalized: strictly uses `add_special_tokens=False` and `prompt_end = len(prompt_ids) - 1`
  - Canonical relative depth: d = l / (num_layers - 1)
  - Zero-forward optimization for alpha = 0.0
  - Pre-caching mechanism for clean baselines & representations
  - 50:50 Discovery / Confirmation split with exact derangement
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cosine
import torch
from tqdm import tqdm
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # --dry-run は transformers 未導入環境でも起動できるようにする
    AutoModelForCausalLM = None  # type: ignore[misc, assignment]
    AutoTokenizer = None  # type: ignore[misc, assignment]

from affective_empathy_eval.intervention import PyTorchActivationPatcher
from affective_empathy_eval.likelihood import (
    build_vad_candidates,
    compute_sequence_likelihoods_for_candidates,
)
from affective_empathy_eval.manifests import create_run_manifest
from affective_empathy_eval.models.registry import (
    add_model_selection_args,
    resolve_architecture_dims,
    resolve_single_model_from_args,
)
from affective_empathy_eval.affect_directions import AIPSY_EXPECTED_DIRECTION
from affective_empathy_eval.geometry import get_block_hidden_state
from affective_empathy_eval.statistics import (
    compute_bivariate_bootstrap_ci,
    compute_paired_cohen_dz,
    generate_derangement,
)


def get_git_commit() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return commit.decode("utf-8").strip()
    except Exception:
        return "unknown"


def compute_cache_metadata(
    model_id: str,
    tokenizer: Any,
    df: pd.DataFrame,
    prompts_sample: List[str],
    intervention_position: str = "prompt_end",
    candidate_schema: str = "vad_triplets_729",
) -> Dict[str, Any]:
    dataset_str = "".join(
        df["pair_id"].astype(str)
        + df["text_aff"].astype(str)
        + df["text_neu"].astype(str)
    )
    dataset_hash = hashlib.sha256(dataset_str.encode("utf-8")).hexdigest()[:16]

    prompt_str = "".join(prompts_sample)
    prompt_hash = hashlib.sha256(prompt_str.encode("utf-8")).hexdigest()[:16]

    tok_name = tokenizer.__class__.__name__ if tokenizer is not None else "unknown"
    vocab_size = getattr(tokenizer, "vocab_size", -1) if tokenizer is not None else -1

    return {
        "model_id": str(model_id),
        "git_commit": get_git_commit(),
        "dataset_hash": dataset_hash,
        "prompt_hash": prompt_hash,
        "candidate_schema": candidate_schema,
        "intervention_position": intervention_position,
        "tokenizer": f"{tok_name}_v{vocab_size}",
    }


def validate_cache_metadata(meta_file: str, current_meta: Dict[str, Any]) -> bool:
    if not os.path.exists(meta_file):
        return False
    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            cached_meta = json.load(f)
        for k, v in current_meta.items():
            if cached_meta.get(k) != v:
                print(
                    f"[CACHE MISMATCH in {os.path.basename(meta_file)}] {k}: cached={cached_meta.get(k)} != current={v}"
                )
                return False
        return True
    except Exception as ex:
        print(f"[CACHE ERROR] Failed reading {meta_file}: {ex}")
        return False


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
                messages, tokenize=False, add_generation_prompt=True
            )
    else:
        return (
            f"Task: Evaluate emotional Valence, Arousal, and Dominance (1-9).\n\n"
            f"{instruction}\n\n"
            f"Text: {text}\n\n"
            f"Output:\n"
        )


def get_prompt_end_position(tokenizer, prompt: str) -> int:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) == 0:
        raise ValueError("Prompt encoded to 0 tokens.")
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
    exp_v_list = []
    exp_a_list = []

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
        ev = float(np.sum(probs * vad_triplets[:, 0]))
        ea = float(np.sum(probs * vad_triplets[:, 1]))
        exp_v_list.append(ev)
        exp_a_list.append(ea)

    return np.array(exp_v_list), np.array(exp_a_list)


@torch.no_grad()
def extract_hidden_states(
    model,
    tokenizer,
    prompts: List[str],
    device: str = "cuda",
    batch_size: int = 16,
) -> Dict[int, np.ndarray]:
    all_layer_vecs: Dict[int, List[np.ndarray]] = {}

    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        encoded = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(device)
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            output_hidden_states=True,
        )
        seq_lengths = encoded["attention_mask"].sum(dim=1) - 1
        num_blocks = getattr(model.config, "num_hidden_layers", len(outputs.hidden_states) - 1)
        for block_idx in range(num_blocks):
            layer_tensor = get_block_hidden_state(outputs.hidden_states, block_idx)
            if block_idx not in all_layer_vecs:
                all_layer_vecs[block_idx] = []
            for b in range(len(chunk)):
                vec = (
                    layer_tensor[b, seq_lengths[b].item(), :]
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                )
                all_layer_vecs[block_idx].append(vec)

        del encoded, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {l: np.array(vecs) for l, vecs in all_layer_vecs.items()}


def main():
    parser = argparse.ArgumentParser(
        description="V1 Primary Phase C: Causal Interventions"
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
        "--mode", type=str, choices=["all", "e3", "e4"], default="all"
    )
    parser.add_argument("--limit", type=int, default=0, help="Sample limit (0 for full dataset)")
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[-1.0, 0.0, 0.5, 1.0, 1.5],
    )
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument(
        "--candidate-layers", type=int, nargs="+", default=None
    )
    parser.add_argument("--num-e4-layers", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--sub-batch-size", type=int, default=81)
    parser.add_argument("--cache-dir", type=str, default="v1/results/cache")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--force", action="store_true")
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

    if args.full:
        args.limit = 0

    if args.dry_run:
        args.out_dir = os.path.join(args.out_dir, "dry_run")
        args.cache_dir = os.path.join(args.cache_dir, "dry_run")

    os.makedirs(args.out_dir, exist_ok=True)
    model_dir = os.path.join(args.out_dir, args.model_prefix)
    os.makedirs(model_dir, exist_ok=True)

    cache_model_dir = os.path.join(args.cache_dir, args.model_prefix)
    if not args.no_cache:
        os.makedirs(cache_model_dir, exist_ok=True)

    is_instruct = (
        args.is_instruct
        or "instruct" in args.model_id.lower()
        or "chat" in args.model_id.lower()
        or "it" in args.model_id.lower()
    )

    print(
        f"=== Starting V1 Phase C Causal Intervention (Prompt-End Normalized) ==="
    )
    print(f"Model ID: {args.model_id} | Mode: {args.mode}")

    if args.dry_run:
        print(f"[DRY-RUN] V1 Phase C for Model: {args.model_id}")
        e3_records = [
            {
                "layer": 0,
                "relative_depth": 0.0,
                "delta_h_cosine": 0.85,
                "magnitude_reader": 0.30,
                "magnitude_self": 0.25,
                "discovery_mag_reader": 0.32,
                "discovery_mag_self": 0.26,
                "r2_causal_effect": 0.45,
            },
            {
                "layer": 14,
                "relative_depth": 0.5,
                "delta_h_cosine": 0.92,
                "magnitude_reader": 0.65,
                "magnitude_self": 0.70,
                "discovery_mag_reader": 0.68,
                "discovery_mag_self": 0.72,
                "r2_causal_effect": 0.65,
            },
        ]
        e4_records = [
            {
                "layer": 14,
                "relative_depth": 0.5,
                "alpha": 1.0,
                "aligned_matched_shift_V": 0.35,
                "aligned_matched_shift_A": 0.25,
                "aligned_random_shift_V": 0.05,
                "aligned_random_shift_A": 0.02,
                "aligned_specificity_V": 0.30,
                "aligned_specificity_A": 0.23,
                "aligned_cohen_dz_V": 1.15,
                "aligned_cohen_dz_A": 0.95,
                "transfer_ratio_V": 0.70,
                "transfer_ratio_A": 0.65,
                "matched_shift_V": 0.35,
                "matched_shift_A": 0.25,
                "random_shift_V": 0.05,
                "random_shift_A": 0.02,
                "specificity_V": 0.30,
                "specificity_A": 0.23,
                "cohen_dz_V": 1.15,
                "cohen_dz_A": 0.95,
            }
        ]
        pd.DataFrame(e3_records).to_csv(
            os.path.join(model_dir, "e3_causal_map.csv"), index=False
        )
        pd.DataFrame(e4_records).to_csv(
            os.path.join(model_dir, "e4_interchangeability_results.csv"), index=False
        )
        manifest = create_run_manifest(
            run_type="v1_phase_c",
            model_name=args.model_id,
            config={
                "model_prefix": args.model_prefix,
                "mode": args.mode,
                "limit": args.limit,
                "dry_run": True,
            },
            candidate_space="VAD_729",
            dry_run=True,
        )
        manifest.save(os.path.join(model_dir, "manifest.json"))
        print(f"[DRY-RUN] Completed Phase C mock output in {model_dir}")
        return

    cand_dicts = build_vad_candidates()
    candidates = [c["json_str"] for c in cand_dicts]
    vad_triplets = np.array(
        [[c["valence"], c["arousal"], c["dominance"]] for c in cand_dicts]
    )

    aipsy_path = Path("v1/data/processed/aipsy_4split_all.csv")
    if not aipsy_path.exists():
        aipsy_path = Path("data/processed/aipsy_4split_all.csv")

    df_aipsy = pd.read_csv(aipsy_path)
    df_clin = df_aipsy[df_aipsy["split"] == "clinical"].copy()
    df_neut = df_aipsy[df_aipsy["split"] == "neutral"].copy()

    merged = pd.merge(
        df_clin, df_neut, on="pair_id", suffixes=("_aff", "_neu")
    )
    merged = merged.dropna(subset=["text_aff", "text_neu"]).reset_index(
        drop=True
    )

    if args.limit > 0:
        merged = merged.head(args.limit)

    n_pairs = len(merged)
    print(f"Loaded {n_pairs} matched clinical-neutral pairs.")

    rng_split = np.random.default_rng(args.split_seed)
    unique_pairs = merged["pair_id"].unique()
    perm_pairs = rng_split.permutation(unique_pairs)
    split_cut = len(perm_pairs) // 2
    discovery_pairs_set = set(perm_pairs[:split_cut])
    confirmation_pairs_set = set(perm_pairs[split_cut:])

    merged["eval_split"] = [
        "discovery" if pid in discovery_pairs_set else "confirmation"
        for pid in merged["pair_id"]
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_cuda = str(args.device).startswith("cuda") and torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if is_cuda else torch.float32,
        device_map=args.device if is_cuda else None,
        trust_remote_code=True,
    )
    model.eval()

    num_layers = getattr(model.config, "num_hidden_layers", None)
    if num_layers is None and hasattr(model.config, "n_layer"):
        num_layers = model.config.n_layer

    if args.all_layers:
        target_layers = list(range(num_layers))
    elif args.layers is not None:
        target_layers = [l for l in args.layers if l < num_layers]
    else:
        target_layers = list(range(0, num_layers, max(1, num_layers // 10)))

    prompts_r_aff = [
        format_prompt(tokenizer, t, "reader", is_instruct)
        for t in merged["text_aff"]
    ]
    prompts_r_neu = [
        format_prompt(tokenizer, t, "reader", is_instruct)
        for t in merged["text_neu"]
    ]
    prompts_s_aff = [
        format_prompt(tokenizer, t, "self", is_instruct)
        for t in merged["text_aff"]
    ]
    prompts_s_neu = [
        format_prompt(tokenizer, t, "self", is_instruct)
        for t in merged["text_neu"]
    ]

    # Compute current run cache metadata
    current_cache_meta = compute_cache_metadata(
        model_id=args.model_id,
        tokenizer=tokenizer,
        df=merged,
        prompts_sample=prompts_r_aff[:10] + prompts_s_aff[:10],
        intervention_position="prompt_end",
        candidate_schema="vad_triplets_729",
    )

    # Baselines
    baseline_cache_file = os.path.join(
        cache_model_dir, f"baselines_n{n_pairs}.npz"
    )
    baseline_meta_file = os.path.join(
        cache_model_dir, f"baselines_n{n_pairs}_meta.json"
    )
    loaded_cache_baselines = False
    if (
        not args.no_cache
        and not args.force
        and os.path.exists(baseline_cache_file)
        and validate_cache_metadata(baseline_meta_file, current_cache_meta)
    ):
        try:
            bcache = np.load(baseline_cache_file)
            clean_ev_s_neu = bcache["clean_ev_s_neu"]
            clean_ea_s_neu = bcache["clean_ea_s_neu"]
            clean_ev_s_aff = bcache["clean_ev_s_aff"]
            clean_ea_s_aff = bcache["clean_ea_s_aff"]
            clean_ev_r_neu = bcache["clean_ev_r_neu"]
            clean_ea_r_neu = bcache["clean_ea_r_neu"]
            clean_ev_r_aff = bcache["clean_ev_r_aff"]
            clean_ea_r_aff = bcache["clean_ea_r_aff"]
            loaded_cache_baselines = True
            print(f"Loaded validated clean baselines from {baseline_cache_file}")
        except Exception:
            pass

    if not loaded_cache_baselines:
        print("Computing clean baselines...")
        clean_ev_s_neu, clean_ea_s_neu = evaluate_expected_va_batch(
            model,
            tokenizer,
            prompts_s_neu,
            candidates,
            vad_triplets,
            device=args.device,
            sub_batch_size=args.sub_batch_size,
        )
        clean_ev_s_aff, clean_ea_s_aff = evaluate_expected_va_batch(
            model,
            tokenizer,
            prompts_s_aff,
            candidates,
            vad_triplets,
            device=args.device,
            sub_batch_size=args.sub_batch_size,
        )
        clean_ev_r_neu, clean_ea_r_neu = evaluate_expected_va_batch(
            model,
            tokenizer,
            prompts_r_neu,
            candidates,
            vad_triplets,
            device=args.device,
            sub_batch_size=args.sub_batch_size,
        )
        clean_ev_r_aff, clean_ea_r_aff = evaluate_expected_va_batch(
            model,
            tokenizer,
            prompts_r_aff,
            candidates,
            vad_triplets,
            device=args.device,
            sub_batch_size=args.sub_batch_size,
        )

        if not args.no_cache:
            np.savez_compressed(
                baseline_cache_file,
                clean_ev_s_neu=clean_ev_s_neu,
                clean_ea_s_neu=clean_ea_s_neu,
                clean_ev_s_aff=clean_ev_s_aff,
                clean_ea_s_aff=clean_ea_s_aff,
                clean_ev_r_neu=clean_ev_r_neu,
                clean_ea_r_neu=clean_ea_r_neu,
                clean_ev_r_aff=clean_ev_r_aff,
                clean_ea_r_aff=clean_ea_r_aff,
            )
            with open(baseline_meta_file, "w", encoding="utf-8") as f:
                json.dump(current_cache_meta, f, indent=2)

    # Hidden States
    hidden_cache_file = os.path.join(
        cache_model_dir, f"hidden_states_n{n_pairs}.npz"
    )
    hidden_meta_file = os.path.join(
        cache_model_dir, f"hidden_states_n{n_pairs}_meta.json"
    )
    loaded_cache_hidden = False
    if (
        not args.no_cache
        and not args.force
        and os.path.exists(hidden_cache_file)
        and validate_cache_metadata(hidden_meta_file, current_cache_meta)
    ):
        try:
            hcache = np.load(hidden_cache_file)
            reps_r_aff = {
                int(k.split("_")[-1]): hcache[k]
                for k in hcache.files
                if k.startswith("r_aff_")
            }
            reps_r_neu = {
                int(k.split("_")[-1]): hcache[k]
                for k in hcache.files
                if k.startswith("r_neu_")
            }
            reps_s_aff = {
                int(k.split("_")[-1]): hcache[k]
                for k in hcache.files
                if k.startswith("s_aff_")
            }
            reps_s_neu = {
                int(k.split("_")[-1]): hcache[k]
                for k in hcache.files
                if k.startswith("s_neu_")
            }
            loaded_cache_hidden = True
            print(f"Loaded validated representations from {hidden_cache_file}")
        except Exception:
            pass

    if not loaded_cache_hidden:
        print("Extracting representations...")
        reps_r_aff = extract_hidden_states(
            model, tokenizer, prompts_r_aff, device=args.device
        )
        reps_r_neu = extract_hidden_states(
            model, tokenizer, prompts_r_neu, device=args.device
        )
        reps_s_aff = extract_hidden_states(
            model, tokenizer, prompts_s_aff, device=args.device
        )
        reps_s_neu = extract_hidden_states(
            model, tokenizer, prompts_s_neu, device=args.device
        )

        if not args.no_cache:
            save_dict = {}
            for l, arr in reps_r_aff.items():
                save_dict[f"r_aff_{l}"] = arr
            for l, arr in reps_r_neu.items():
                save_dict[f"r_neu_{l}"] = arr
            for l, arr in reps_s_aff.items():
                save_dict[f"s_aff_{l}"] = arr
            for l, arr in reps_s_neu.items():
                save_dict[f"s_neu_{l}"] = arr
            np.savez_compressed(hidden_cache_file, **save_dict)
            with open(hidden_meta_file, "w", encoding="utf-8") as f:
                json.dump(current_cache_meta, f, indent=2)

    # E3: Shared Causal Map
    e3_csv_path = os.path.join(model_dir, "e3_causal_map.csv")
    e3_pair_csv_path = os.path.join(model_dir, "e3_causal_map_pair_level.csv")
    e3_causal_records = []
    e3_pair_records = []

    if args.mode in ["all", "e3"]:
        print("Running E3: Shared Causal Map...")
        for l in tqdm(target_layers, desc="E3 Target Layers"):
            rel_d = l / (num_layers - 1) if num_layers > 1 else 0.0
            delta_h_r = reps_r_aff[l] - reps_r_neu[l]
            delta_h_s = reps_s_aff[l] - reps_s_neu[l]

            shifts_v_self = []
            shifts_a_self = []
            shifts_v_reader = []
            shifts_a_reader = []
            pairwise_cosines = []

            for p_idx in range(n_pairs):
                p_id = merged.loc[p_idx, "pair_id"]
                split_tag = merged.loc[p_idx, "eval_split"]

                pos_s = get_prompt_end_position(
                    tokenizer, prompts_s_neu[p_idx]
                )
                pos_r = get_prompt_end_position(
                    tokenizer, prompts_r_neu[p_idx]
                )

                diff_s = delta_h_s[p_idx]
                with PyTorchActivationPatcher(
                    model,
                    l,
                    diff_s,
                    patch_weight=1.0,
                    position=pos_s,
                    intervention_type="add",
                ):
                    ptc_ev_s, ptc_ea_s = evaluate_expected_va_batch(
                        model,
                        tokenizer,
                        [prompts_s_neu[p_idx]],
                        candidates,
                        vad_triplets,
                        device=args.device,
                        sub_batch_size=args.sub_batch_size,
                    )
                    sv = float(ptc_ev_s[0] - clean_ev_s_neu[p_idx])
                    sa = float(ptc_ea_s[0] - clean_ea_s_neu[p_idx])
                    shifts_v_self.append(sv)
                    shifts_a_self.append(sa)

                diff_r = delta_h_r[p_idx]
                with PyTorchActivationPatcher(
                    model,
                    l,
                    diff_r,
                    patch_weight=1.0,
                    position=pos_r,
                    intervention_type="add",
                ):
                    ptc_ev_r, ptc_ea_r = evaluate_expected_va_batch(
                        model,
                        tokenizer,
                        [prompts_r_neu[p_idx]],
                        candidates,
                        vad_triplets,
                        device=args.device,
                        sub_batch_size=args.sub_batch_size,
                    )
                    rv = float(ptc_ev_r[0] - clean_ev_r_neu[p_idx])
                    ra = float(ptc_ea_r[0] - clean_ea_r_neu[p_idx])
                    shifts_v_reader.append(rv)
                    shifts_a_reader.append(ra)

                mag_sp = np.sqrt(sv**2 + sa**2)
                mag_rp = np.sqrt(rv**2 + ra**2)
                denom_p = mag_sp * mag_rp
                pair_cos = (
                    float((sv * rv + sa * ra) / denom_p)
                    if denom_p > 1e-6
                    else np.nan
                )
                pairwise_cosines.append(pair_cos)

                e3_pair_records.append(
                    {
                        "layer": l,
                        "relative_depth": rel_d,
                        "pair_id": p_id,
                        "eval_split": split_tag,
                        "shift_V_self": sv,
                        "shift_A_self": sa,
                        "magnitude_self": mag_sp,
                        "shift_V_reader": rv,
                        "shift_A_reader": ra,
                        "magnitude_reader": mag_rp,
                        "pairwise_cos": pair_cos,
                    }
                )

            mean_sv = float(np.mean(shifts_v_self))
            mean_sa = float(np.mean(shifts_a_self))
            mean_rv = float(np.mean(shifts_v_reader))
            mean_ra = float(np.mean(shifts_a_reader))

            mag_s = float(np.sqrt(mean_sv**2 + mean_sa**2))
            mag_r = float(np.sqrt(mean_rv**2 + mean_ra**2))
            denom_mean = mag_s * mag_r
            cos_sim_of_means = (
                float((mean_sv * mean_rv + mean_sa * mean_ra) / denom_mean)
                if denom_mean > 1e-6
                else 0.0
            )

            valid_cos = [c for c in pairwise_cosines if not np.isnan(c)]
            mean_pairwise_cos = (
                float(np.mean(valid_cos)) if len(valid_cos) > 0 else 0.0
            )

            disc_idx = [
                i
                for i in range(n_pairs)
                if merged.loc[i, "eval_split"] == "discovery"
            ]
            conf_idx = [
                i
                for i in range(n_pairs)
                if merged.loc[i, "eval_split"] == "confirmation"
            ]
            disc_mag_s = (
                float(
                    np.mean(
                        [
                            np.sqrt(
                                shifts_v_self[i] ** 2 + shifts_a_self[i] ** 2
                            )
                            for i in disc_idx
                        ]
                    )
                )
                if disc_idx
                else mag_s
            )
            disc_mag_r = (
                float(
                    np.mean(
                        [
                            np.sqrt(
                                shifts_v_reader[i] ** 2
                                + shifts_a_reader[i] ** 2
                            )
                            for i in disc_idx
                        ]
                    )
                )
                if disc_idx
                else mag_r
            )
            conf_mag_s = (
                float(
                    np.mean(
                        [
                            np.sqrt(
                                shifts_v_self[i] ** 2 + shifts_a_self[i] ** 2
                            )
                            for i in conf_idx
                        ]
                    )
                )
                if conf_idx
                else mag_s
            )
            conf_mag_r = (
                float(
                    np.mean(
                        [
                            np.sqrt(
                                shifts_v_reader[i] ** 2
                                + shifts_a_reader[i] ** 2
                            )
                            for i in conf_idx
                        ]
                    )
                )
                if conf_idx
                else mag_r
            )

            e3_causal_records.append(
                {
                    "layer": l,
                    "relative_depth": rel_d,
                    "magnitude_self": mag_s,
                    "magnitude_reader": mag_r,
                    "shift_V_self": mean_sv,
                    "shift_A_self": mean_sa,
                    "shift_V_reader": mean_rv,
                    "shift_A_reader": mean_ra,
                    "directional_cosine_similarity": cos_sim_of_means,
                    "mean_pairwise_directional_cosine": mean_pairwise_cos,
                    "discovery_mag_self": disc_mag_s,
                    "discovery_mag_reader": disc_mag_r,
                    "confirmation_mag_self": conf_mag_s,
                    "confirmation_mag_reader": conf_mag_r,
                }
            )

        pd.DataFrame(e3_causal_records).to_csv(e3_csv_path, index=False)
        pd.DataFrame(e3_pair_records).to_csv(e3_pair_csv_path, index=False)
        print(f"E3 results saved to {e3_csv_path}")

    # E4: Interchangeability
    if args.mode in ["all", "e4"]:
        df_e3 = pd.DataFrame(e3_causal_records)
        if df_e3.empty and os.path.exists(e3_csv_path):
            df_e3 = pd.read_csv(e3_csv_path)

        if args.candidate_layers is not None and len(args.candidate_layers) > 0:
            e4_layers = sorted(list(set(args.candidate_layers)))
        elif not df_e3.empty:
            reader_peak_l = int(
                df_e3.loc[df_e3["discovery_mag_reader"].idxmax(), "layer"]
            )
            self_peak_l = int(
                df_e3.loc[df_e3["discovery_mag_self"].idxmax(), "layer"]
            )
            near_l1 = max(0, min(reader_peak_l - 1, num_layers - 1))
            near_l2 = max(0, min(self_peak_l + 1, num_layers - 1))
            late_l = int(num_layers * 0.85)
            e4_layers = sorted(
                list(
                    set([reader_peak_l, self_peak_l, near_l1, near_l2, late_l])
                )
            )
        else:
            e4_layers = sorted(
                list(
                    set([int(num_layers * d) for d in [0.45, 0.60, 0.75, 0.85]])
                )
            )

        e4_csv_path = os.path.join(
            model_dir, "e4_interchangeability_results.csv"
        )
        e4_pair_csv_path = os.path.join(
            model_dir, "e4_interchangeability_pair_level.csv"
        )
        e4_patching_records = []
        e4_pair_records = []

        conf_indices = [
            i
            for i in range(n_pairs)
            if merged.loc[i, "eval_split"] == "confirmation"
        ]
        if len(conf_indices) < 5:
            if args.dry_run:
                print(f"[DRY-RUN] Small confirmation sample (N={len(conf_indices)}); using available confirmation pairs or fallback.")
                if len(conf_indices) < 2:
                    conf_indices = list(range(n_pairs))
            else:
                raise ValueError(
                    f"Insufficient confirmation pairs for E4: found {len(conf_indices)}, required >= 5. "
                    f"Aborting to prevent data leakage between discovery and confirmation splits."
                )

        n_conf = len(conf_indices)
        rng_derange = np.random.default_rng(args.split_seed + 100)
        deranged_sub_indices = generate_derangement(n_conf, rng_derange)
        random_indices_map = {
            conf_indices[i]: conf_indices[deranged_sub_indices[i]]
            for i in range(n_conf)
        }
        EXPECTED_DIRECTION = AIPSY_EXPECTED_DIRECTION

        print(f"Running E4 Interchangeability on Layers {e4_layers} (with Same-Task Controls)...")
        for l in e4_layers:
            rel_d = l / (num_layers - 1) if num_layers > 1 else 0.0
            delta_h_r = reps_r_aff[l] - reps_r_neu[l]
            delta_h_s = reps_s_aff[l] - reps_s_neu[l]

            for alpha in args.alphas:
                # Raw signed shifts
                matched_shifts_v = []
                matched_shifts_a = []
                random_shifts_v = []
                random_shifts_a = []
                self_self_shifts_v = []
                self_self_shifts_a = []
                reader_reader_shifts_v = []
                reader_reader_shifts_a = []
                self_reader_shifts_v = []
                self_reader_shifts_a = []

                # Target-direction aligned shifts (Primary to avoid sign cancellation)
                aligned_matched_shifts_v = []
                aligned_matched_shifts_a = []
                aligned_random_shifts_v = []
                aligned_random_shifts_a = []
                aligned_self_self_shifts_v = []
                aligned_self_self_shifts_a = []
                aligned_reader_reader_shifts_v = []
                aligned_reader_reader_shifts_a = []

                is_zero_alpha = abs(alpha) < 1e-9

                for p_idx in conf_indices:
                    p_id = merged.loc[p_idx, "pair_id"]
                    pos_s = get_prompt_end_position(
                        tokenizer, prompts_s_neu[p_idx]
                    )
                    pos_r = get_prompt_end_position(
                        tokenizer, prompts_r_neu[p_idx]
                    )

                    if is_zero_alpha:
                        m_sv, m_sa = 0.0, 0.0
                        r_sv, r_sa = 0.0, 0.0
                        ss_v, ss_a = 0.0, 0.0
                        rr_v, rr_a = 0.0, 0.0
                        sr_v, sr_a = 0.0, 0.0
                    else:
                        # 1. Primary: Reader -> Self (Cross-task Matched)
                        diff_matched = delta_h_r[p_idx]
                        with PyTorchActivationPatcher(
                            model,
                            l,
                            diff_matched,
                            patch_weight=alpha,
                            position=pos_s,
                            intervention_type="add",
                        ):
                            m_ev, m_ea = evaluate_expected_va_batch(
                                model,
                                tokenizer,
                                [prompts_s_neu[p_idx]],
                                candidates,
                                vad_triplets,
                                device=args.device,
                                sub_batch_size=args.sub_batch_size,
                            )
                            m_sv = float(m_ev[0] - clean_ev_s_neu[p_idx])
                            m_sa = float(m_ea[0] - clean_ea_s_neu[p_idx])

                        # 2. Control 1: Reader -> Self (Random Permuted)
                        rnd_idx = random_indices_map[p_idx]
                        diff_random = delta_h_r[rnd_idx]
                        with PyTorchActivationPatcher(
                            model,
                            l,
                            diff_random,
                            patch_weight=alpha,
                            position=pos_s,
                            intervention_type="add",
                        ):
                            rnd_ev, rnd_ea = evaluate_expected_va_batch(
                                model,
                                tokenizer,
                                [prompts_s_neu[p_idx]],
                                candidates,
                                vad_triplets,
                                device=args.device,
                                sub_batch_size=args.sub_batch_size,
                            )
                            r_sv = float(rnd_ev[0] - clean_ev_s_neu[p_idx])
                            r_sa = float(rnd_ea[0] - clean_ea_s_neu[p_idx])

                        # 3. Same-task Control: Self -> Self (Upper bound of causal influence)
                        diff_s = delta_h_s[p_idx]
                        with PyTorchActivationPatcher(
                            model,
                            l,
                            diff_s,
                            patch_weight=alpha,
                            position=pos_s,
                            intervention_type="add",
                        ):
                            ss_ev, ss_ea = evaluate_expected_va_batch(
                                model,
                                tokenizer,
                                [prompts_s_neu[p_idx]],
                                candidates,
                                vad_triplets,
                                device=args.device,
                                sub_batch_size=args.sub_batch_size,
                            )
                            ss_v = float(ss_ev[0] - clean_ev_s_neu[p_idx])
                            ss_a = float(ss_ea[0] - clean_ea_s_neu[p_idx])

                        # 4. Same-task Control: Reader -> Reader
                        with PyTorchActivationPatcher(
                            model,
                            l,
                            diff_matched,
                            patch_weight=alpha,
                            position=pos_r,
                            intervention_type="add",
                        ):
                            rr_ev, rr_ea = evaluate_expected_va_batch(
                                model,
                                tokenizer,
                                [prompts_r_neu[p_idx]],
                                candidates,
                                vad_triplets,
                                device=args.device,
                                sub_batch_size=args.sub_batch_size,
                            )
                            rr_v = float(rr_ev[0] - clean_ev_r_neu[p_idx])
                            rr_a = float(rr_ea[0] - clean_ea_r_neu[p_idx])

                        # 5. Reverse Cross-task: Self -> Reader
                        with PyTorchActivationPatcher(
                            model,
                            l,
                            diff_s,
                            patch_weight=alpha,
                            position=pos_r,
                            intervention_type="add",
                        ):
                            sr_ev, sr_ea = evaluate_expected_va_batch(
                                model,
                                tokenizer,
                                [prompts_r_neu[p_idx]],
                                candidates,
                                vad_triplets,
                                device=args.device,
                                sub_batch_size=args.sub_batch_size,
                            )
                            sr_v = float(sr_ev[0] - clean_ev_r_neu[p_idx])
                            sr_a = float(sr_ea[0] - clean_ea_r_neu[p_idx])

                    matched_shifts_v.append(m_sv)
                    matched_shifts_a.append(m_sa)
                    random_shifts_v.append(r_sv)
                    random_shifts_a.append(r_sa)
                    self_self_shifts_v.append(ss_v)
                    self_self_shifts_a.append(ss_a)
                    reader_reader_shifts_v.append(rr_v)
                    reader_reader_shifts_a.append(rr_a)
                    self_reader_shifts_v.append(sr_v)
                    self_reader_shifts_a.append(sr_a)

                    # Direction alignment for this pair
                    emo_name = str(
                        merged.loc[p_idx, "target_emotion_aff"]
                        if "target_emotion_aff" in merged.columns
                        else (
                            merged.loc[p_idx, "target_emotion"]
                            if "target_emotion" in merged.columns
                            else merged.loc[p_idx, "emotion_aff"]
                        )
                    ).lower()
                    sign_v = EXPECTED_DIRECTION.get(emo_name, {}).get("V")
                    sign_a = EXPECTED_DIRECTION.get(emo_name, {}).get("A")

                    aligned_m_sv = sign_v * m_sv if sign_v is not None else np.nan
                    aligned_r_sv = sign_v * r_sv if sign_v is not None else np.nan
                    aligned_ss_v = sign_v * ss_v if sign_v is not None else np.nan
                    aligned_rr_v = sign_v * rr_v if sign_v is not None else np.nan

                    aligned_m_sa = sign_a * m_sa if sign_a is not None else np.nan
                    aligned_r_sa = sign_a * r_sa if sign_a is not None else np.nan
                    aligned_ss_a = sign_a * ss_a if sign_a is not None else np.nan
                    aligned_rr_a = sign_a * rr_a if sign_a is not None else np.nan

                    if sign_v is not None:
                        aligned_matched_shifts_v.append(aligned_m_sv)
                        aligned_random_shifts_v.append(aligned_r_sv)
                        aligned_self_self_shifts_v.append(aligned_ss_v)
                        aligned_reader_reader_shifts_v.append(aligned_rr_v)

                    if sign_a is not None:
                        aligned_matched_shifts_a.append(aligned_m_sa)
                        aligned_random_shifts_a.append(aligned_r_sa)
                        aligned_self_self_shifts_a.append(aligned_ss_a)
                        aligned_reader_reader_shifts_a.append(aligned_rr_a)

                    e4_pair_records.append(
                        {
                            "layer": l,
                            "alpha": alpha,
                            "pair_id": p_id,
                            "emotion": emo_name,
                            "sign_V": sign_v,
                            "sign_A": sign_a,
                            # Primary: target direction aligned
                            "aligned_matched_shift_V": aligned_m_sv,
                            "aligned_matched_shift_A": aligned_m_sa,
                            "aligned_random_shift_V": aligned_r_sv,
                            "aligned_random_shift_A": aligned_r_sa,
                            "aligned_self_self_shift_V": aligned_ss_v,
                            "aligned_self_self_shift_A": aligned_ss_a,
                            "aligned_reader_reader_shift_V": aligned_rr_v,
                            "aligned_reader_reader_shift_A": aligned_rr_a,
                            "aligned_specificity_V": (aligned_m_sv - aligned_r_sv) if not np.isnan(aligned_m_sv) else np.nan,
                            "aligned_specificity_A": (aligned_m_sa - aligned_r_sa) if not np.isnan(aligned_m_sa) else np.nan,
                            # Secondary: raw signed
                            "matched_shift_V": m_sv,
                            "matched_shift_A": m_sa,
                            "random_shift_V": r_sv,
                            "random_shift_A": r_sa,
                            "self_self_shift_V": ss_v,
                            "self_self_shift_A": ss_a,
                            "reader_reader_shift_V": rr_v,
                            "reader_reader_shift_A": rr_a,
                            "self_reader_shift_V": sr_v,
                            "self_reader_shift_A": sr_a,
                            "specificity_V": m_sv - r_sv,
                            "specificity_A": m_sa - r_sa,
                        }
                    )

                # Primary aligned means
                mean_al_m_v = float(np.mean(aligned_matched_shifts_v)) if aligned_matched_shifts_v else 0.0
                mean_al_m_a = float(np.mean(aligned_matched_shifts_a)) if aligned_matched_shifts_a else 0.0
                mean_al_rnd_v = float(np.mean(aligned_random_shifts_v)) if aligned_random_shifts_v else 0.0
                mean_al_rnd_a = float(np.mean(aligned_random_shifts_a)) if aligned_random_shifts_a else 0.0
                mean_al_ss_v = float(np.mean(aligned_self_self_shifts_v)) if aligned_self_self_shifts_v else 0.0
                mean_al_ss_a = float(np.mean(aligned_self_self_shifts_a)) if aligned_self_self_shifts_a else 0.0
                mean_al_rr_v = float(np.mean(aligned_reader_reader_shifts_v)) if aligned_reader_reader_shifts_v else 0.0
                mean_al_rr_a = float(np.mean(aligned_reader_reader_shifts_a)) if aligned_reader_reader_shifts_a else 0.0

                al_spec_v = mean_al_m_v - mean_al_rnd_v
                al_spec_a = mean_al_m_a - mean_al_rnd_a

                # Transfer Ratio from aligned shifts with denominator stability guard
                if abs(mean_al_ss_v) >= 0.05:
                    transfer_ratio_v = float(mean_al_m_v / (mean_al_ss_v + 1e-6))
                else:
                    transfer_ratio_v = np.nan

                if abs(mean_al_ss_a) >= 0.05:
                    transfer_ratio_a = float(mean_al_m_a / (mean_al_ss_a + 1e-6))
                else:
                    transfer_ratio_a = np.nan

                # Primary stats on aligned shifts
                if not is_zero_alpha and len(aligned_matched_shifts_v) > 2:
                    al_dz_v = compute_paired_cohen_dz(aligned_matched_shifts_v, aligned_random_shifts_v, ddof=1)
                    al_ci_low_v, al_ci_high_v = compute_bivariate_bootstrap_ci(
                        aligned_matched_shifts_v, aligned_random_shifts_v, stat_fn=lambda x, y: np.mean(x - y), n_bootstraps=1000
                    )
                    try:
                        _, al_p_val_v = stats.ttest_rel(aligned_matched_shifts_v, aligned_random_shifts_v)
                    except Exception:
                        al_p_val_v = 1.0
                else:
                    al_dz_v, al_ci_low_v, al_ci_high_v, al_p_val_v = 0.0, 0.0, 0.0, 1.0

                if not is_zero_alpha and len(aligned_matched_shifts_a) > 2:
                    al_dz_a = compute_paired_cohen_dz(aligned_matched_shifts_a, aligned_random_shifts_a, ddof=1)
                    al_ci_low_a, al_ci_high_a = compute_bivariate_bootstrap_ci(
                        aligned_matched_shifts_a, aligned_random_shifts_a, stat_fn=lambda x, y: np.mean(x - y), n_bootstraps=1000
                    )
                    try:
                        _, al_p_val_a = stats.ttest_rel(aligned_matched_shifts_a, aligned_random_shifts_a)
                    except Exception:
                        al_p_val_a = 1.0
                else:
                    al_dz_a, al_ci_low_a, al_ci_high_a, al_p_val_a = 0.0, 0.0, 0.0, 1.0

                # Secondary raw signed means
                mean_m_v = float(np.mean(matched_shifts_v))
                mean_m_a = float(np.mean(matched_shifts_a))
                mean_rnd_v = float(np.mean(random_shifts_v))
                mean_rnd_a = float(np.mean(random_shifts_a))
                mean_ss_v = float(np.mean(self_self_shifts_v))
                mean_ss_a = float(np.mean(self_self_shifts_a))
                mean_rr_v = float(np.mean(reader_reader_shifts_v))
                mean_rr_a = float(np.mean(reader_reader_shifts_a))
                mean_sr_v = float(np.mean(self_reader_shifts_v))
                mean_sr_a = float(np.mean(self_reader_shifts_a))

                spec_v = mean_m_v - mean_rnd_v
                spec_a = mean_m_a - mean_rnd_a

                if not is_zero_alpha and len(matched_shifts_v) > 2:
                    dz_v = compute_paired_cohen_dz(matched_shifts_v, random_shifts_v, ddof=1)
                    dz_a = compute_paired_cohen_dz(matched_shifts_a, random_shifts_a, ddof=1)
                    ci_low_v, ci_high_v = compute_bivariate_bootstrap_ci(
                        matched_shifts_v, random_shifts_v, stat_fn=lambda x, y: np.mean(x - y), n_bootstraps=1000
                    )
                    ci_low_a, ci_high_a = compute_bivariate_bootstrap_ci(
                        matched_shifts_a, random_shifts_a, stat_fn=lambda x, y: np.mean(x - y), n_bootstraps=1000
                    )
                    try:
                        _, p_val_v = stats.ttest_rel(matched_shifts_v, random_shifts_v)
                        _, p_val_a = stats.ttest_rel(matched_shifts_a, random_shifts_a)
                    except Exception:
                        p_val_v, p_val_a = 1.0, 1.0
                else:
                    dz_v, dz_a = 0.0, 0.0
                    ci_low_v, ci_high_v = 0.0, 0.0
                    ci_low_a, ci_high_a = 0.0, 0.0
                    p_val_v, p_val_a = 1.0, 1.0

                e4_patching_records.append(
                    {
                        "layer": l,
                        "relative_depth": rel_d,
                        "alpha": alpha,
                        # Primary: Direction-aligned effects
                        "aligned_matched_shift_V": mean_al_m_v,
                        "aligned_matched_shift_A": mean_al_m_a,
                        "aligned_random_shift_V": mean_al_rnd_v,
                        "aligned_random_shift_A": mean_al_rnd_a,
                        "aligned_self_self_shift_V": mean_al_ss_v,
                        "aligned_self_self_shift_A": mean_al_ss_a,
                        "aligned_reader_reader_shift_V": mean_al_rr_v,
                        "aligned_reader_reader_shift_A": mean_al_rr_a,
                        "aligned_specificity_V": al_spec_v,
                        "aligned_specificity_A": al_spec_a,
                        "aligned_cohen_dz_V": al_dz_v,
                        "aligned_cohen_dz_A": al_dz_a,
                        "aligned_ci_95_low_V": al_ci_low_v,
                        "aligned_ci_95_high_V": al_ci_high_v,
                        "aligned_ci_95_low_A": al_ci_low_a,
                        "aligned_ci_95_high_A": al_ci_high_a,
                        "aligned_p_val_V": al_p_val_v,
                        "aligned_p_val_A": al_p_val_a,
                        "transfer_ratio_V": transfer_ratio_v,
                        "transfer_ratio_A": transfer_ratio_a,
                        # Secondary: Raw signed shifts
                        "matched_shift_V": mean_m_v,
                        "matched_shift_A": mean_m_a,
                        "random_shift_V": mean_rnd_v,
                        "random_shift_A": mean_rnd_a,
                        "self_self_shift_V": mean_ss_v,
                        "self_self_shift_A": mean_ss_a,
                        "reader_reader_shift_V": mean_rr_v,
                        "reader_reader_shift_A": mean_rr_a,
                        "self_reader_shift_V": mean_sr_v,
                        "self_reader_shift_A": mean_sr_a,
                        "specificity_V": spec_v,
                        "specificity_A": spec_a,
                        "cohen_dz_V": dz_v,
                        "cohen_dz_A": dz_a,
                        "ci_95_low_V": ci_low_v,
                        "ci_95_high_V": ci_high_v,
                        "ci_95_low_A": ci_low_a,
                        "ci_95_high_A": ci_high_a,
                        "p_val_V": p_val_v,
                        "p_val_A": p_val_a,
                    }
                )

        pd.DataFrame(e4_patching_records).to_csv(e4_csv_path, index=False)
        pd.DataFrame(e4_pair_records).to_csv(e4_pair_csv_path, index=False)
        print(f"E4 results saved to {e4_csv_path}")

    # Save manifest
    manifest = create_run_manifest(
        run_type="v1_phase_c",
        model_name=args.model_id,
        config={
            "model_prefix": args.model_prefix,
            "mode": args.mode,
            "limit": args.limit,
            "alphas": args.alphas,
            "num_layers": num_layers,
            "split_seed": args.split_seed,
            "zero_forward_optimized": True,
        },
        candidate_space="VAD_729",
    )
    manifest.save(os.path.join(model_dir, "manifest.json"))
    print(f"Phase C completed. Results saved to {model_dir}")


if __name__ == "__main__":
    main()
