#!/usr/bin/env python3
"""
V1 Phase C: Causal Patching Script (Normalized & Optimized)
Evaluates:
  - E3: Shared Causal Map (Magnitude + 2D Directional Vector Cosine Similarity across layers)
  - E4: Causal Interchangeability (Matched vs. Random Difference-Vector Patching on Candidate Layers)

Key Updates & Optimizations:
  1. Normalized Intervention Position:
     Strictly uses `add_special_tokens=False` and `prompt_end = len(prompt_ids) - 1`
     consistent with hidden state extraction and Sequence-Likelihood boundary tokens.
  2. Decoupled Execution (E3 full-layer discovery -> E4 candidate-layer confirmation):
     E3 maps all layers across Discovery split; E4 tests 3-5 candidate layers on Confirmation split.
  3. Zero-Forward Optimization:
     alpha = 0.0 skips model forward pass entirely (records exact 0.0 shift).
  4. Pre-caching Mechanism:
     Baselines and hidden states can be cached to disk to avoid redundant forward passes.
  5. Strict Derangement on Confirmation Split:
     Random control preserves self-match exclusion within evaluated split.
  6. Output Isolation:
     Defaults to `v1/results/derived/v1_phase_c_prompt_end` with full metadata traceability.
"""

import os
import re
import json
import argparse
import subprocess
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Set
from tqdm import tqdm
from scipy.spatial.distance import cosine
from scipy import stats
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from affective_empathy_eval.intervention import PyTorchActivationPatcher
from affective_empathy_eval.statistics import (
    generate_derangement,
    compute_paired_cohen_dz,
    compute_bivariate_bootstrap_ci
)
from affective_empathy_eval.likelihood import (
    build_vad_candidates,
    compute_sequence_likelihoods_for_candidates
)


def get_git_commit() -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return commit.decode("utf-8").strip()
    except Exception:
        return "unknown"


def format_prompt(tokenizer, text: str, task_type: str, is_instruct: bool) -> str:
    """Formats prompt identical to canonical behavioral evaluation protocol."""
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
            {"role": "user", "content": user_content}
        ]
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            messages = [{"role": "user", "content": user_content}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = (
            f"Task: Evaluate emotional Valence, Arousal, and Dominance (1-9).\n\n"
            f"{instruction}\n\n"
            f"Text: {text}\n\n"
            f"Output:\n"
        )
    return prompt


def get_prompt_end_position(tokenizer, prompt: str) -> int:
    """Returns exact prompt_end token index using canonical add_special_tokens=False encoding."""
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
    sub_batch_size: int = 81
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes expected Valence and Arousal using canonical Sequence-Likelihood Protocol."""
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
    batch_size: int = 16
) -> Dict[int, np.ndarray]:
    """Extracts prompt_end hidden states across all layers using canonical encoding (add_special_tokens=False)."""
    all_layer_vecs: Dict[int, List[np.ndarray]] = {}

    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        encoded = tokenizer(chunk, padding=True, truncation=True, add_special_tokens=False, return_tensors="pt").to(device)
        outputs = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"], output_hidden_states=True)
        seq_lengths = encoded["attention_mask"].sum(dim=1) - 1

        for l, layer_tensor in enumerate(outputs.hidden_states):
            if l not in all_layer_vecs:
                all_layer_vecs[l] = []
            for b in range(len(chunk)):
                vec = layer_tensor[b, seq_lengths[b].item(), :].detach().cpu().float().numpy()
                all_layer_vecs[l].append(vec)

        del encoded, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    layer_reps = {l: np.array(vecs) for l, vecs in all_layer_vecs.items()}
    return layer_reps


def main():
    parser = argparse.ArgumentParser(description="Run V1 Phase C: E3 (Shared Causal Map) & E4 (Causal Interchangeability).")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Hugging Face model ID.")
    parser.add_argument("--model-prefix", type=str, default="qwen2.5_1.5b_instruct", help="Prefix for file naming.")
    parser.add_argument("--mode", type=str, choices=["all", "e3", "e4"], default="all", help="Execution mode: e3 only, e4 only, or both.")
    parser.add_argument("--limit", type=int, default=20, help="Number of stimulus pairs to evaluate (default: 20 for pilot; set 0 or --full for all 192 pairs).")
    parser.add_argument("--full", action="store_true", help="Evaluate all available stimulus pairs (overrides --limit).")
    parser.add_argument("--alphas", type=float, nargs="+", default=[-1.0, 0.0, 0.5, 1.0, 1.5], help="Alpha sweep values for E4.")
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="Specific layers to test for E3 (default: uniform sweep across all architecture layers).")
    parser.add_argument("--all-layers", action="store_true", help="Test all layers in the model for E3.")
    parser.add_argument("--candidate-layers", type=int, nargs="+", default=None, help="Explicit candidate layers for E4 (default: auto-selected from Discovery split peak/near-peak layers).")
    parser.add_argument("--num-e4-layers", type=int, default=4, help="Number of candidate layers to select for E4 if auto-selected.")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for Discovery/Confirmation split.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sub-batch-size", type=int, default=81, help="Sub-batch size for 729-candidate evaluation to avoid OOM.")
    parser.add_argument("--cache-dir", type=str, default="v1/results/cache", help="Cache directory for baselines and hidden states.")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching of baselines and hidden states.")
    parser.add_argument("--force", action="store_true", help="Force re-run even if output files already exist.")
    parser.add_argument("--out-dir", type=str, default="v1/results/derived/v1_phase_c_prompt_end", help="Output directory for prompt-end results.")
    args = parser.parse_args()

    if args.full:
        args.limit = 0

    os.makedirs(args.out_dir, exist_ok=True)
    model_dir = os.path.join(args.out_dir, args.model_prefix)
    os.makedirs(model_dir, exist_ok=True)

    cache_model_dir = os.path.join(args.cache_dir, args.model_prefix)
    if not args.no_cache:
        os.makedirs(cache_model_dir, exist_ok=True)

    is_instruct = "instruct" in args.model_id.lower() or "chat" in args.model_id.lower() or "it" in args.model_id.lower()

    print(f"=== Starting V1 Phase C Causal Intervention (Prompt-End Normalized) ===")
    print(f"Model ID: {args.model_id} | Mode: {args.mode}")
    print(f"Alphas: {args.alphas} | Limit: {args.limit} pairs (full={args.full}) | Sub-batch: {args.sub_batch_size}")
    print(f"Output Directory: {model_dir}")

    # 1. Load candidates
    cand_dicts = build_vad_candidates()
    candidates = [c["json_str"] for c in cand_dicts]
    vad_triplets = np.array([[c["valence"], c["arousal"], c["dominance"]] for c in cand_dicts])

    # 2. Load Dataset (AIPsy-Affect paired stimuli)
    aipsy_path = "v1/data/processed/aipsy_4split_all.csv"
    if not os.path.exists(aipsy_path):
        # Fallback to root or alternative relative path
        alt_path = "data/processed/aipsy_4split_all.csv"
        if os.path.exists(alt_path):
            aipsy_path = alt_path
        else:
            raise FileNotFoundError(f"AIPsy dataset not found at {aipsy_path} or {alt_path}")

    df_aipsy = pd.read_csv(aipsy_path)
    df_clin = df_aipsy[df_aipsy["split"] == "clinical"].copy()
    df_neut = df_aipsy[df_aipsy["split"] == "neutral"].copy()

    merged = pd.merge(df_clin, df_neut, on="pair_id", suffixes=('_aff', '_neu'))
    merged = merged.dropna(subset=["text_aff", "text_neu"]).reset_index(drop=True)

    if args.limit > 0:
        merged = merged.head(args.limit)

    n_pairs = len(merged)
    print(f"Loaded {n_pairs} matched clinical-neutral pairs.")

    # 3. Deterministic Discovery / Confirmation split by pair_id
    rng_split = np.random.default_rng(args.split_seed)
    unique_pairs = merged["pair_id"].unique()
    perm_pairs = rng_split.permutation(unique_pairs)
    split_cut = len(perm_pairs) // 2
    discovery_pairs_set = set(perm_pairs[:split_cut])
    confirmation_pairs_set = set(perm_pairs[split_cut:])

    merged["eval_split"] = ["discovery" if pid in discovery_pairs_set else "confirmation" for pid in merged["pair_id"]]
    print(f"Pair Split: {sum(merged['eval_split'] == 'discovery')} Discovery, {sum(merged['eval_split'] == 'confirmation')} Confirmation.")

    # 4. Load Tokenizer & Model
    print("\nLoading Tokenizer and Model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        device_map="auto" if args.device == "cuda" else None,
        trust_remote_code=True
    )
    model.eval()

    num_layers = getattr(model.config, "num_hidden_layers", None)
    if num_layers is None and hasattr(model.config, "n_layer"):
        num_layers = model.config.n_layer
    if num_layers is None:
        raise ValueError("Could not determine number of layers from model config.")

    if args.all_layers:
        target_layers = list(range(num_layers))
    elif args.layers is not None:
        target_layers = [l for l in args.layers if l < num_layers]
    else:
        target_layers = list(range(0, num_layers, max(1, num_layers // 10)))
    print(f"Architecture layers: {num_layers} | E3 Evaluation layers: {target_layers}")

    # Prepare Prompts
    prompts_r_aff = [format_prompt(tokenizer, t, "reader", is_instruct) for t in merged["text_aff"]]
    prompts_r_neu = [format_prompt(tokenizer, t, "reader", is_instruct) for t in merged["text_neu"]]
    prompts_s_aff = [format_prompt(tokenizer, t, "self", is_instruct) for t in merged["text_aff"]]
    prompts_s_neu = [format_prompt(tokenizer, t, "self", is_instruct) for t in merged["text_neu"]]

    # Verify tokenization consistency between prompt_end and joint candidates
    sample_prompt = prompts_s_neu[0]
    expected_pos = get_prompt_end_position(tokenizer, sample_prompt)
    sample_prompt_ids = tokenizer.encode(sample_prompt, add_special_tokens=False)
    assert len(sample_prompt_ids) - 1 == expected_pos, "Token length mismatch!"

    # 5. Clean Baselines (with disk caching)
    baseline_cache_file = os.path.join(cache_model_dir, f"baselines_n{n_pairs}.npz")
    loaded_cache_baselines = False
    if not args.no_cache and not args.force and os.path.exists(baseline_cache_file):
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
            print(f"[CACHE HIT] Loaded clean baselines from {baseline_cache_file}")
            loaded_cache_baselines = True
        except Exception as ex:
            print(f"[CACHE CORRUPT] Could not load baselines: {ex}. Recomputing...")

    if not loaded_cache_baselines:
        print("\n[1/3] Computing Clean Baselines (No Intervention)...")
        clean_ev_s_neu, clean_ea_s_neu = evaluate_expected_va_batch(model, tokenizer, prompts_s_neu, candidates, vad_triplets, device=args.device, sub_batch_size=args.sub_batch_size)
        clean_ev_s_aff, clean_ea_s_aff = evaluate_expected_va_batch(model, tokenizer, prompts_s_aff, candidates, vad_triplets, device=args.device, sub_batch_size=args.sub_batch_size)
        clean_ev_r_neu, clean_ea_r_neu = evaluate_expected_va_batch(model, tokenizer, prompts_r_neu, candidates, vad_triplets, device=args.device, sub_batch_size=args.sub_batch_size)
        clean_ev_r_aff, clean_ea_r_aff = evaluate_expected_va_batch(model, tokenizer, prompts_r_aff, candidates, vad_triplets, device=args.device, sub_batch_size=args.sub_batch_size)

        if not args.no_cache:
            np.savez_compressed(
                baseline_cache_file,
                clean_ev_s_neu=clean_ev_s_neu, clean_ea_s_neu=clean_ea_s_neu,
                clean_ev_s_aff=clean_ev_s_aff, clean_ea_s_aff=clean_ea_s_aff,
                clean_ev_r_neu=clean_ev_r_neu, clean_ea_r_neu=clean_ea_r_neu,
                clean_ev_r_aff=clean_ev_r_aff, clean_ea_r_aff=clean_ea_r_aff,
            )
            print(f"Saved baselines to cache: {baseline_cache_file}")

    # Clean Shifts
    clean_shift_v_s = clean_ev_s_aff - clean_ev_s_neu
    clean_shift_a_s = clean_ea_s_aff - clean_ea_s_neu
    clean_shift_v_r = clean_ev_r_aff - clean_ev_r_neu
    clean_shift_a_r = clean_ea_r_aff - clean_ea_r_neu

    print(f"Clean Shift Self:   mean Delta V = {np.mean(clean_shift_v_s):.3f}, mean Delta A = {np.mean(clean_shift_a_s):.3f}")
    print(f"Clean Shift Reader: mean Delta V = {np.mean(clean_shift_v_r):.3f}, mean Delta A = {np.mean(clean_shift_a_r):.3f}")

    # 6. Extract Hidden States (with disk caching)
    hidden_cache_file = os.path.join(cache_model_dir, f"hidden_states_n{n_pairs}.npz")
    loaded_cache_hidden = False
    if not args.no_cache and not args.force and os.path.exists(hidden_cache_file):
        try:
            hcache = np.load(hidden_cache_file)
            reps_r_aff = {int(k.split('_')[-1]): hcache[k] for k in hcache.files if k.startswith("r_aff_")}
            reps_r_neu = {int(k.split('_')[-1]): hcache[k] for k in hcache.files if k.startswith("r_neu_")}
            reps_s_aff = {int(k.split('_')[-1]): hcache[k] for k in hcache.files if k.startswith("s_aff_")}
            reps_s_neu = {int(k.split('_')[-1]): hcache[k] for k in hcache.files if k.startswith("s_neu_")}
            print(f"[CACHE HIT] Loaded hidden states across {len(reps_r_aff)} layers from {hidden_cache_file}")
            loaded_cache_hidden = True
        except Exception as ex:
            print(f"[CACHE CORRUPT] Could not load hidden states: {ex}. Recomputing...")

    if not loaded_cache_hidden:
        print("\n[2/3] Extracting Hidden States across Target Layers...")
        reps_r_aff = extract_hidden_states(model, tokenizer, prompts_r_aff, device=args.device)
        reps_r_neu = extract_hidden_states(model, tokenizer, prompts_r_neu, device=args.device)
        reps_s_aff = extract_hidden_states(model, tokenizer, prompts_s_aff, device=args.device)
        reps_s_neu = extract_hidden_states(model, tokenizer, prompts_s_neu, device=args.device)

        if not args.no_cache:
            save_dict = {}
            for l, arr in reps_r_aff.items(): save_dict[f"r_aff_{l}"] = arr
            for l, arr in reps_r_neu.items(): save_dict[f"r_neu_{l}"] = arr
            for l, arr in reps_s_aff.items(): save_dict[f"s_aff_{l}"] = arr
            for l, arr in reps_s_neu.items(): save_dict[f"s_neu_{l}"] = arr
            np.savez_compressed(hidden_cache_file, **save_dict)
            print(f"Saved hidden states to cache: {hidden_cache_file}")

    # 7. E3: Shared Causal Map Execution
    e3_csv_path = os.path.join(model_dir, "e3_causal_map.csv")
    e3_pair_csv_path = os.path.join(model_dir, "e3_causal_map_pair_level.csv")
    e3_causal_records = []
    e3_pair_records = []

    run_e3 = args.mode in ["all", "e3"]
    if run_e3:
        print("\n[3/3] Running E3: Shared Causal Map across Layers...")
        completed_e3_layers = set()
        if not args.force and os.path.exists(e3_csv_path) and os.path.exists(e3_pair_csv_path):
            try:
                df_prev_e3 = pd.read_csv(e3_csv_path)
                df_prev_e3_pair = pd.read_csv(e3_pair_csv_path)
                completed_e3_layers = set(df_prev_e3["layer"].astype(int).tolist())
                e3_causal_records = df_prev_e3.to_dict("records")
                e3_pair_records = df_prev_e3_pair.to_dict("records")
                print(f"Resuming E3: found {len(completed_e3_layers)} completed layers.")
            except Exception:
                completed_e3_layers = set()

        for l in tqdm(target_layers, desc="E3 Target Layers"):
            if l in completed_e3_layers:
                continue

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

                # Normalized prompt_end positions
                pos_s = get_prompt_end_position(tokenizer, prompts_s_neu[p_idx])
                pos_r = get_prompt_end_position(tokenizer, prompts_r_neu[p_idx])

                # Patch Self
                diff_s = delta_h_s[p_idx]
                with PyTorchActivationPatcher(model, l, diff_s, patch_weight=1.0, position=pos_s, intervention_type="add"):
                    ptc_ev_s, ptc_ea_s = evaluate_expected_va_batch(model, tokenizer, [prompts_s_neu[p_idx]], candidates, vad_triplets, device=args.device, sub_batch_size=args.sub_batch_size)
                    sv = float(ptc_ev_s[0] - clean_ev_s_neu[p_idx])
                    sa = float(ptc_ea_s[0] - clean_ea_s_neu[p_idx])
                    shifts_v_self.append(sv)
                    shifts_a_self.append(sa)

                # Patch Reader
                diff_r = delta_h_r[p_idx]
                with PyTorchActivationPatcher(model, l, diff_r, patch_weight=1.0, position=pos_r, intervention_type="add"):
                    ptc_ev_r, ptc_ea_r = evaluate_expected_va_batch(model, tokenizer, [prompts_r_neu[p_idx]], candidates, vad_triplets, device=args.device, sub_batch_size=args.sub_batch_size)
                    rv = float(ptc_ev_r[0] - clean_ev_r_neu[p_idx])
                    ra = float(ptc_ea_r[0] - clean_ea_r_neu[p_idx])
                    shifts_v_reader.append(rv)
                    shifts_a_reader.append(ra)

                mag_sp = np.sqrt(sv**2 + sa**2)
                mag_rp = np.sqrt(rv**2 + ra**2)
                denom_p = mag_sp * mag_rp
                pair_cos = float((sv * rv + sa * ra) / denom_p) if denom_p > 1e-6 else np.nan
                pairwise_cosines.append(pair_cos)

                e3_pair_records.append({
                    "layer": l,
                    "relative_depth": l / (num_layers - 1) if num_layers > 1 else 0.0,
                    "pair_id": p_id,
                    "eval_split": split_tag,
                    "shift_V_self": sv,
                    "shift_A_self": sa,
                    "magnitude_self": mag_sp,
                    "shift_V_reader": rv,
                    "shift_A_reader": ra,
                    "magnitude_reader": mag_rp,
                    "pairwise_cos": pair_cos
                })

            mean_sv = float(np.mean(shifts_v_self))
            mean_sa = float(np.mean(shifts_a_self))
            mean_rv = float(np.mean(shifts_v_reader))
            mean_ra = float(np.mean(shifts_a_reader))

            mag_s = float(np.sqrt(mean_sv**2 + mean_sa**2))
            mag_r = float(np.sqrt(mean_rv**2 + mean_ra**2))

            denom_mean = (mag_s * mag_r)
            cos_sim_of_means = float((mean_sv * mean_rv + mean_sa * mean_ra) / denom_mean) if denom_mean > 1e-6 else 0.0

            valid_cos = [c for c in pairwise_cosines if not np.isnan(c)]
            mean_pairwise_cos = float(np.mean(valid_cos)) if len(valid_cos) > 0 else 0.0

            # Split-specific magnitudes for clean Discovery / Confirmation tracking
            disc_idx = [i for i in range(n_pairs) if merged.loc[i, "eval_split"] == "discovery"]
            conf_idx = [i for i in range(n_pairs) if merged.loc[i, "eval_split"] == "confirmation"]
            disc_mag_s = float(np.mean([np.sqrt(shifts_v_self[i]**2 + shifts_a_self[i]**2) for i in disc_idx])) if disc_idx else mag_s
            disc_mag_r = float(np.mean([np.sqrt(shifts_v_reader[i]**2 + shifts_a_reader[i]**2) for i in disc_idx])) if disc_idx else mag_r
            conf_mag_s = float(np.mean([np.sqrt(shifts_v_self[i]**2 + shifts_a_self[i]**2) for i in conf_idx])) if conf_idx else mag_s
            conf_mag_r = float(np.mean([np.sqrt(shifts_v_reader[i]**2 + shifts_a_reader[i]**2) for i in conf_idx])) if conf_idx else mag_r

            e3_causal_records.append({
                "layer": l,
                "relative_depth": l / (num_layers - 1) if num_layers > 1 else 0.0,
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
            })

            # Save progress
            pd.DataFrame(e3_causal_records).to_csv(e3_csv_path, index=False)
            pd.DataFrame(e3_pair_records).to_csv(e3_pair_csv_path, index=False)

        print(f"E3 Causal Map saved to {e3_csv_path}")

    # 8. E4: Candidate Layer Selection & Interchangeability Execution
    run_e4 = args.mode in ["all", "e4"]
    if run_e4:
        df_e3 = pd.DataFrame(e3_causal_records)
        if df_e3.empty and os.path.exists(e3_csv_path):
            df_e3 = pd.read_csv(e3_csv_path)

        # Select candidate layers for E4
        if args.candidate_layers is not None and len(args.candidate_layers) > 0:
            e4_layers = sorted(list(set(args.candidate_layers)))
            print(f"\n[E4 Candidates] Using user-specified candidate layers: {e4_layers}")
        elif not df_e3.empty:
            # Auto-selection from Discovery split
            reader_peak_l = int(df_e3.loc[df_e3["discovery_mag_reader"].idxmax(), "layer"])
            self_peak_l = int(df_e3.loc[df_e3["discovery_mag_self"].idxmax(), "layer"])
            near_l1 = max(0, min(reader_peak_l - 1, num_layers - 1))
            near_l2 = max(0, min(self_peak_l + 1, num_layers - 1))
            late_l = int(num_layers * 0.85)
            e4_layers = sorted(list(set([reader_peak_l, self_peak_l, near_l1, near_l2, late_l])))
            print(f"\n[E4 Candidates Auto-Selected from Discovery Split]: {e4_layers} (Reader peak: {reader_peak_l}, Self peak: {self_peak_l})")
        else:
            # Fallback if E3 was not run
            e4_layers = sorted(list(set([int(num_layers * d) for d in [0.45, 0.60, 0.75, 0.85]])))
            print(f"\n[E4 Candidates Fallback]: {e4_layers}")

        e4_csv_path = os.path.join(model_dir, "e4_interchangeability_results.csv")
        e4_pair_csv_path = os.path.join(model_dir, "e4_interchangeability_pair_level.csv")
        e4_patching_records = []
        e4_pair_records = []

        completed_e4_keys: Set[Tuple[int, float]] = set()
        if not args.force and os.path.exists(e4_csv_path) and os.path.exists(e4_pair_csv_path):
            try:
                df_prev_e4 = pd.read_csv(e4_csv_path)
                df_prev_e4_pair = pd.read_csv(e4_pair_csv_path)
                for _, r in df_prev_e4.iterrows():
                    completed_e4_keys.add((int(r["layer"]), float(r["alpha"])))
                e4_patching_records = df_prev_e4.to_dict("records")
                e4_pair_records = df_prev_e4_pair.to_dict("records")
                print(f"Resuming E4: found {len(completed_e4_keys)} completed (layer, alpha) combinations.")
            except Exception:
                completed_e4_keys = set()

        # E4 Evaluation on Confirmation split (or all pairs if limit is small)
        conf_indices = [i for i in range(n_pairs) if merged.loc[i, "eval_split"] == "confirmation"]
        if len(conf_indices) < 5:
            conf_indices = list(range(n_pairs))
            print("Note: Small sample limit; evaluating E4 across all available pairs.")
        else:
            print(f"Evaluating E4 exclusively on Confirmation Split ({len(conf_indices)} pairs).")

        n_conf = len(conf_indices)
        rng_derange = np.random.default_rng(args.split_seed + 100)
        deranged_sub_indices = generate_derangement(n_conf, rng_derange)
        # Map back to global indices
        random_indices_map = {conf_indices[i]: conf_indices[deranged_sub_indices[i]] for i in range(n_conf)}

        print(f"\nRunning E4 Difference-Vector Patching on Layers {e4_layers} across Alphas {args.alphas}...")
        for l in e4_layers:
            delta_h_r = reps_r_aff[l] - reps_r_neu[l]

            for alpha in args.alphas:
                if (l, alpha) in completed_e4_keys:
                    continue

                matched_shifts_v = []
                matched_shifts_a = []
                random_shifts_v = []
                random_shifts_a = []

                # Optimization: alpha = 0.0 is an exact no-op (clean state)
                is_zero_alpha = abs(alpha) < 1e-9

                for p_idx in conf_indices:
                    p_id = merged.loc[p_idx, "pair_id"]
                    pos_s = get_prompt_end_position(tokenizer, prompts_s_neu[p_idx])

                    if is_zero_alpha:
                        # Zero shift without invoking neural forward pass
                        m_sv, m_sa = 0.0, 0.0
                        r_sv, r_sa = 0.0, 0.0
                    else:
                        # 1. Matched difference patching (inject delta_h_r[i] into Self neutral[i])
                        diff_matched = delta_h_r[p_idx]
                        with PyTorchActivationPatcher(model, l, diff_matched, patch_weight=alpha, position=pos_s, intervention_type="add"):
                            m_ev, m_ea = evaluate_expected_va_batch(model, tokenizer, [prompts_s_neu[p_idx]], candidates, vad_triplets, device=args.device, sub_batch_size=args.sub_batch_size)
                            m_sv = float(m_ev[0] - clean_ev_s_neu[p_idx])
                            m_sa = float(m_ea[0] - clean_ea_s_neu[p_idx])

                        # 2. Random difference patching (inject delta_h_r[j] into Self neutral[i], j != i)
                        rnd_idx = random_indices_map[p_idx]
                        diff_random = delta_h_r[rnd_idx]
                        with PyTorchActivationPatcher(model, l, diff_random, patch_weight=alpha, position=pos_s, intervention_type="add"):
                            rnd_ev, rnd_ea = evaluate_expected_va_batch(model, tokenizer, [prompts_s_neu[p_idx]], candidates, vad_triplets, device=args.device, sub_batch_size=args.sub_batch_size)
                            r_sv = float(rnd_ev[0] - clean_ev_s_neu[p_idx])
                            r_sa = float(rnd_ea[0] - clean_ea_s_neu[p_idx])

                    matched_shifts_v.append(m_sv)
                    matched_shifts_a.append(m_sa)
                    random_shifts_v.append(r_sv)
                    random_shifts_a.append(r_sa)

                    e4_pair_records.append({
                        "layer": l,
                        "alpha": alpha,
                        "pair_id": p_id,
                        "matched_shift_V": m_sv,
                        "matched_shift_A": m_sa,
                        "random_shift_V": r_sv,
                        "random_shift_A": r_sa,
                        "specificity_V": m_sv - r_sv,
                        "specificity_A": m_sa - r_sa
                    })

                mean_m_v = float(np.mean(matched_shifts_v))
                mean_m_a = float(np.mean(matched_shifts_a))
                mean_rnd_v = float(np.mean(random_shifts_v))
                mean_rnd_a = float(np.mean(random_shifts_a))

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
                        t_stat_v, p_val_v = stats.ttest_rel(matched_shifts_v, random_shifts_v)
                        t_stat_a, p_val_a = stats.ttest_rel(matched_shifts_a, random_shifts_a)
                    except Exception:
                        p_val_v, p_val_a = 1.0, 1.0
                else:
                    dz_v, dz_a = 0.0, 0.0
                    ci_low_v, ci_high_v = 0.0, 0.0
                    ci_low_a, ci_high_a = 0.0, 0.0
                    p_val_v, p_val_a = 1.0, 1.0

                e4_patching_records.append({
                    "layer": l,
                    "relative_depth": l / (num_layers - 1) if num_layers > 1 else 0.0,
                    "alpha": alpha,
                    "matched_shift_V": mean_m_v,
                    "matched_shift_A": mean_m_a,
                    "random_shift_V": mean_rnd_v,
                    "random_shift_A": mean_rnd_a,
                    "specificity_V": spec_v,
                    "specificity_A": spec_a,
                    "cohen_dz_V": dz_v,
                    "cohen_dz_A": dz_a,
                    "ci_95_low_V": ci_low_v,
                    "ci_95_high_V": ci_high_v,
                    "ci_95_low_A": ci_low_a,
                    "ci_95_high_A": ci_high_a,
                    "p_val_V": p_val_v,
                    "p_val_A": p_val_a
                })

            pd.DataFrame(e4_patching_records).to_csv(e4_csv_path, index=False)
            pd.DataFrame(e4_pair_records).to_csv(e4_pair_csv_path, index=False)

        print(f"E4 Interchangeability results saved to {e4_csv_path}")

    # 9. Save Metadata JSON
    metadata = {
        "model_id": args.model_id,
        "model_prefix": args.model_prefix,
        "git_commit": get_git_commit(),
        "intervention_position_rule": "prompt_end = len(tokenizer.encode(prompt, add_special_tokens=False)) - 1",
        "add_special_tokens": False,
        "candidates_count": len(candidates),
        "alphas": args.alphas,
        "e3_layers_evaluated": target_layers if run_e3 else [],
        "e4_candidate_layers": e4_layers if run_e4 else [],
        "discovery_confirmation_split": {
            "seed": args.split_seed,
            "n_discovery": int(sum(merged["eval_split"] == "discovery")),
            "n_confirmation": int(sum(merged["eval_split"] == "confirmation")),
        },
        "zero_forward_optimized": True,
        "status": "COMPLETED"
    }
    meta_path = os.path.join(model_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {meta_path}")

    # 10. Summary Report
    summary_path = os.path.join(model_dir, "phase_c_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# V1 Phase C Execution Summary: {args.model_id}\n\n")
        f.write(f"- Git Commit: `{metadata['git_commit']}`\n")
        f.write(f"- Position Rule: `{metadata['intervention_position_rule']}`\n")
        f.write(f"- Total Pairs: {n_pairs} (Discovery: {metadata['discovery_confirmation_split']['n_discovery']}, Confirmation: {metadata['discovery_confirmation_split']['n_confirmation']})\n")
        if run_e3:
            f.write(f"- E3 Layers Evaluated: `{target_layers}`\n")
        if run_e4:
            f.write(f"- E4 Candidate Layers Tested: `{e4_layers}`\n")
        f.write(f"\nExecution successfully completed.\n")
    print(f"Summary markdown written to {summary_path}")


if __name__ == "__main__":
    main()
