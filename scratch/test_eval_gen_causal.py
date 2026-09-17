#!/usr/bin/env python3
import sys
from pathlib import Path
project_root = Path("/mnt/nas/home/hiromi/src/emo")
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from v3.scripts.run_generation_time_causal_sweep import (
    evaluate_generation_causal_effect,
    get_3way_split,
    generate_81_suffixes,
    format_base_prompt,
    build_generation_prefix_inputs,
    compute_conditional_candidate_logprobs,
    PREFIX_STR
)

def main():
    device = "cuda:0"
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
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

    eval_pairs = valid_pairs[:15]
    suffixes, _ = generate_81_suffixes()

    base_cache = {}
    for pid, peak_row, neut_row in eval_pairs:
        peak_prompt = format_base_prompt(peak_row['text'])
        neut_prompt = format_base_prompt(neut_row['text'])
        _, p_target_pos, _ = build_generation_prefix_inputs(tokenizer, peak_prompt, PREFIX_STR, device=device)
        _, n_target_pos, _ = build_generation_prefix_inputs(tokenizer, neut_prompt, PREFIX_STR, device=device)
        p_peak = compute_conditional_candidate_logprobs(
            model, tokenizer, peak_prompt, suffixes, p_target_pos,
            device=device, normalize_length=True
        )
        p_neut = compute_conditional_candidate_logprobs(
            model, tokenizer, neut_prompt, suffixes, n_target_pos,
            device=device, normalize_length=True
        )
        base_cache[pid] = {"p_peak": p_peak, "p_neut": p_neut}

    l = 24
    for comp in ["mlp", "attn", "resid"]:
        res = evaluate_generation_causal_effect(
            model, tokenizer, eval_pairs, l, comp, suffixes,
            base_cache=base_cache,
            device=device, normalize_length=True, eps_rec=0.05
        )
        print(f"Direct call to evaluate_generation_causal_effect: Layer {l} | {comp.upper()}: Joint OT Rec={res['mean_ot_recovery']:.2f}% (Med: {res['median_ot_recovery']:.2f}%), Valid={res['valid_pairs']}")

if __name__ == "__main__":
    main()
