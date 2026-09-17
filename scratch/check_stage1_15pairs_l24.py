#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

project_root = Path("/mnt/nas/home/hiromi/src/emo")
sys.path.insert(0, str(project_root))

from v2.src.likelihood import generate_81_candidates
from v3.src.ot_utils import compute_joint_ot_recovery
from v3.src.model_utils import (
    get_component_module,
    get_patch_hook,
    format_base_prompt
)
from v3.scripts.run_aligned_cross_model_patching import get_3way_split
from v3.scripts.run_generation_time_causal_sweep import (
    PREFIX_STR,
    build_generation_prefix_inputs,
    compute_conditional_candidate_logprobs,
    extract_generation_time_activation
)

def main():
    device = "cuda:0"
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    model.eval()

    df = pd.read_csv("v3/data/aipsy_strict_expanded.csv")
    train_df, val_df, test_df = get_3way_split(df)
    test_peaks = test_df[test_df['intensity'] == 'peak'].copy()
    test_neutrals = test_df[test_df['intensity'] == 'neutral'].copy()

    valid_pairs = []
    for pid in test_peaks['pair_id'].unique():
        p_row = test_peaks[test_peaks['pair_id'] == pid]
        n_row = test_neutrals[test_neutrals['pair_id'] == pid]
        if len(p_row) > 0 and len(n_row) > 0:
            valid_pairs.append((pid, p_row.iloc[0], n_row.iloc[0]))

    print(f"Total pairs in held-out test: {len(valid_pairs)}")
    first_15_pairs = valid_pairs[:15]

    candidates, _ = generate_81_candidates()
    suffixes = [c.replace('{"valence": ', '') for c in candidates]

    # Pre-cache generation-time baseline distributions for 15 pairs
    base_cache_gen = {}
    for pid, peak_row, neut_row in first_15_pairs:
        p_prompt = format_base_prompt(peak_row['text'])
        n_prompt = format_base_prompt(neut_row['text'])
        _, tpos, _ = build_generation_prefix_inputs(tokenizer, n_prompt, PREFIX_STR, device=device)
        pp_mat = compute_conditional_candidate_logprobs(model, tokenizer, p_prompt, suffixes, tpos, device=device, normalize_length=True)
        pn_mat = compute_conditional_candidate_logprobs(model, tokenizer, n_prompt, suffixes, tpos, device=device, normalize_length=True)
        base_cache_gen[pid] = {
            "p_peak": pp_mat,
            "p_neut": pn_mat,
            "tpos": tpos
        }

    # Evaluate Layer 24 Residual
    l = 24
    comp = 'resid'
    target_mod = get_component_module(model, l, comp)

    gen_recs = []
    pair_details = []
    for idx, (pid, peak_row, neut_row) in enumerate(first_15_pairs):
        p_prompt = format_base_prompt(peak_row['text'])
        n_prompt = format_base_prompt(neut_row['text'])
        tpos = base_cache_gen[pid]["tpos"]

        src_val, _ = extract_generation_time_activation(model, tokenizer, p_prompt, l, comp=comp, device=device)
        p_hook = get_patch_hook(src_val, tpos)
        p_mat = compute_conditional_candidate_logprobs(
            model, tokenizer, n_prompt, suffixes, tpos,
            hook_fn=p_hook, hook_module=target_mod, device=device, normalize_length=True
        )
        r_ot, d_patch_peak, d_neut_peak, is_valid = compute_joint_ot_recovery(
            p_mat, base_cache_gen[pid]["p_peak"], base_cache_gen[pid]["p_neut"], eps_rec=0.05
        )
        if is_valid and r_ot is not None:
            gen_recs.append(r_ot)
            pair_details.append((pid, r_ot * 100.0))
        else:
            pair_details.append((pid, "INVALID (d_neut_peak < 0.05)"))

    print(f"\n--- First 15 Pairs (Stage 1 subset) on Layer 24 Residual ---")
    for pid, res in pair_details:
        if isinstance(res, float):
            print(f"  Pair {pid}: {res:6.2f}%")
        else:
            print(f"  Pair {pid}: {res}")

    print(f"\nMean Recovery (Stage 1 subset): {np.mean(gen_recs)*100.0:.2f}%")
    print(f"Median Recovery (Stage 1 subset): {np.median(gen_recs)*100.0:.2f}%")
    print(f"Valid pairs: {len(gen_recs)} / {len(first_15_pairs)}")

if __name__ == "__main__":
    main()
