#!/usr/bin/env python3
"""
v3/scripts/run_generation_multilayer_residual.py
================================================
Evaluates multi-layer simultaneous Residual Stream patching at Generation-time.

Tests whether late residual causal leverage is additive/distributed across layers
(increasing recovery from 53% toward 80-100%) or redundant/saturating.

Evaluated Conditions (all at generation prefix final token):
1. L18_only: [18]
2. L20_only: [20]
3. L24_only: [24]
4. L18_L20: [18, 20]
5. L20_L24: [20, 24]
6. L18_L20_L24: [18, 20, 24]
7. L18_to_L24_contiguous: [18, 19, 20, 21, 22, 23, 24]
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoModelForCausalLM, AutoTokenizer
from v2.src.likelihood import generate_81_candidates
from v3.src.model_utils import (
    get_component_module,
    get_patch_hook,
    format_base_prompt
)
from v3.src.ot_utils import compute_joint_ot_recovery
from v3.scripts.run_aligned_cross_model_patching import get_3way_split
from v3.scripts.run_generation_time_causal_sweep import (
    PREFIX_STR,
    build_generation_prefix_inputs,
    extract_generation_time_activation
)

CONDITIONS = {
    "L18_only": [18],
    "L20_only": [20],
    "L24_only": [24],
    "L18_L20": [18, 20],
    "L20_L24": [20, 24],
    "L18_L20_L24": [18, 20, 24],
    "L18_to_L24_contiguous": [18, 19, 20, 21, 22, 23, 24],
}

def compute_conditional_candidate_logprobs_multihook(
    model, tokenizer, base_prompt, suffixes, target_pos,
    hook_pairs=None, device="cuda", normalize_length=False
):
    """
    Computes conditional log-likelihood for all 81 suffixes in a SINGLE forward pass,
    with multiple simultaneous forward hooks registered across multiple layers.
    """
    prefix_prompt = base_prompt + PREFIX_STR
    prefix_tokens = tokenizer(prefix_prompt, add_special_tokens=False).input_ids
    prefix_len = len(prefix_tokens)
    
    full_seqs = [prefix_tokens + tokenizer.encode(s, add_special_tokens=False) for s in suffixes]
    max_len = max(len(s) for s in full_seqs)
    
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    padded_ids = []
    attention_masks = []
    suffix_lens = []
    
    for s in full_seqs:
        s_len = len(s)
        pad_len = max_len - s_len
        padded_ids.append(s + [pad_id] * pad_len)
        attention_masks.append([1] * s_len + [0] * pad_len)
        suffix_lens.append(s_len - prefix_len)
        
    input_tensor = torch.tensor(padded_ids, dtype=torch.long, device=device)
    attn_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)
    
    handles = []
    if hook_pairs is not None:
        for hook_fn, hook_module in hook_pairs:
            handles.append(hook_module.register_forward_hook(hook_fn))
            
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_tensor, attention_mask=attn_mask, use_cache=False)
            logits = outputs.logits
            log_probs = torch.log_softmax(logits, dim=-1)
            
            cand_lps = []
            for b_idx, s_len in enumerate(suffix_lens):
                target_tokens = input_tensor[b_idx, prefix_len : prefix_len + s_len]
                step_log_p = log_probs[b_idx, prefix_len - 1 : prefix_len + s_len - 1, :]
                selected_p = step_log_p.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
                lp = selected_p.sum().item()
                if normalize_length and s_len > 0:
                    lp = lp / s_len
                cand_lps.append(lp)
    finally:
        for h in handles:
            h.remove()
            
    cand_lps = np.array(cand_lps)
    max_lp = np.max(cand_lps)
    probs = np.exp(cand_lps - max_lp)
    probs = probs / np.sum(probs)
    return probs.reshape((9, 9))

def bootstrap_ci(arr, n_boot=2000, ci=95):
    arr = np.array(arr)
    if len(arr) == 0:
        return 0.0, 0.0
    boots = []
    n = len(arr)
    for _ in range(n_boot):
        sample = np.random.choice(arr, size=n, replace=True)
        boots.append(np.mean(sample))
    lower = np.percentile(boots, (100 - ci) / 2)
    upper = np.percentile(boots, 100 - (100 - ci) / 2)
    return float(lower), float(upper)

def main():
    parser = argparse.ArgumentParser(description="Multi-layer Generation-Time Residual Patching")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data_path", type=str, default="v3/data/aipsy_strict_expanded.csv")
    parser.add_argument("--output_path", type=str, default="v3/results/generation_multilayer_residual_results.csv")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--eps_rec", type=float, default=0.05)
    args = parser.parse_args()

    print(f"Loading {args.model_name} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if "cuda" in args.device else torch.float32,
        device_map=args.device if "cuda" in args.device else None
    )
    model.eval()

    df = pd.read_csv(args.data_path)
    train_df, val_df, test_df = get_3way_split(df)
    test_peaks = test_df[test_df['intensity'] == 'peak'].copy()
    test_neutrals = test_df[test_df['intensity'] == 'neutral'].copy()

    valid_pairs = []
    for pid in test_peaks['pair_id'].unique():
        p_row = test_peaks[test_peaks['pair_id'] == pid]
        n_row = test_neutrals[test_neutrals['pair_id'] == pid]
        if len(p_row) > 0 and len(n_row) > 0:
            valid_pairs.append((pid, p_row.iloc[0], n_row.iloc[0]))

    print(f"Total available test pairs: {len(valid_pairs)}")
    candidates, _ = generate_81_candidates()
    suffixes = [c.replace('{"valence": ', '') for c in candidates]

    # Pre-cache unpatched baseline distributions for all 39 pairs
    print("Pre-computing Generation-Time baseline distributions for all pairs...")
    base_cache = {}
    for pid, peak_row, neut_row in valid_pairs:
        p_prompt = format_base_prompt(peak_row['text'])
        n_prompt = format_base_prompt(neut_row['text'])
        _, tpos, _ = build_generation_prefix_inputs(tokenizer, n_prompt, PREFIX_STR, device=args.device)
        pp_mat = compute_conditional_candidate_logprobs_multihook(model, tokenizer, p_prompt, suffixes, tpos, device=args.device)
        pn_mat = compute_conditional_candidate_logprobs_multihook(model, tokenizer, n_prompt, suffixes, tpos, device=args.device)
        base_cache[pid] = {
            "p_peak": pp_mat,
            "p_neut": pn_mat,
            "tpos": tpos
        }

    print("\n--- Running Multi-Layer Residual Patching Sweep ---")
    results = []
    for cond_name, layers in CONDITIONS.items():
        print(f"\nEvaluating Condition: {cond_name} (Layers: {layers})...")
        recs = []
        for pid, peak_row, neut_row in valid_pairs:
            p_prompt = format_base_prompt(peak_row['text'])
            n_prompt = format_base_prompt(neut_row['text'])
            tpos = base_cache[pid]["tpos"]

            hook_pairs = []
            for l in layers:
                target_mod = get_component_module(model, l, 'resid')
                src_val, _ = extract_generation_time_activation(model, tokenizer, p_prompt, l, comp='resid', device=args.device)
                hook_fn = get_patch_hook(src_val, tpos)
                hook_pairs.append((hook_fn, target_mod))

            p_mat = compute_conditional_candidate_logprobs_multihook(
                model, tokenizer, n_prompt, suffixes, tpos,
                hook_pairs=hook_pairs, device=args.device
            )

            r_ot, _, _, is_valid = compute_joint_ot_recovery(
                p_mat, base_cache[pid]["p_peak"], base_cache[pid]["p_neut"], eps_rec=args.eps_rec
            )
            if is_valid and r_ot is not None:
                recs.append(r_ot)

        mean_r = float(np.mean(recs)) * 100.0 if len(recs) > 0 else 0.0
        med_r = float(np.median(recs)) * 100.0 if len(recs) > 0 else 0.0
        ci_low, ci_high = bootstrap_ci(np.array(recs) * 100.0)
        q25 = float(np.percentile(recs, 25)) * 100.0 if len(recs) > 0 else 0.0
        q75 = float(np.percentile(recs, 75)) * 100.0 if len(recs) > 0 else 0.0
        pos_frac = float(np.mean(np.array(recs) > 0.001)) * 100.0 if len(recs) > 0 else 0.0

        print(f"  {cond_name:22s} => Mean: {mean_r:5.2f}% (Med: {med_r:5.2f}%) | 95% CI: [{ci_low:5.1f}%, {ci_high:5.1f}%] | IQR: [{q25:5.1f}%, {q75:5.1f}%] | Pos: {pos_frac:4.1f}% (N={len(recs)}/{len(valid_pairs)})")

        results.append({
            "condition": cond_name,
            "layers": str(layers),
            "num_layers": len(layers),
            "total_pairs": len(valid_pairs),
            "valid_pairs": len(recs),
            "mean_recovery": mean_r,
            "median_recovery": med_r,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "iqr_25": q25,
            "iqr_75": q75,
            "positive_fraction": pos_frac,
        })

    out_df = pd.DataFrame(results)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved multi-layer residual results to: {out_path}")

if __name__ == "__main__":
    main()
