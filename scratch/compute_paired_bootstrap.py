"""
scratch/compute_paired_bootstrap.py
Computes paired bootstrap 95% CI for:
  - G_{L15, MLP}
  - G_{L24, Resid}
  - \Delta G = G_{L24, Resid} - G_{L15, MLP}
across all 39 test pairs.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from v2.src.likelihood import generate_81_candidates
from v3.src.ot_utils import compute_joint_ot_recovery
from v3.src.model_utils import (
    get_model_layers,
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

def bootstrap_ci(arr, n_boot=10000, seed=42):
    np.random.seed(seed)
    boots = []
    n = len(arr)
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        boots.append(np.mean(arr[idx]))
    boots = np.array(boots)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    data_path = "v3/data/aipsy_strict_expanded.csv"

    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32,
        device_map=device if "cuda" in device else None
    )
    model.eval()

    df = pd.read_csv(data_path)
    _, _, test_df = get_3way_split(df)
    test_peaks = test_df[test_df['intensity'] == 'peak'].copy()
    test_neutrals = test_df[test_df['intensity'] == 'neutral'].copy()

    valid_pairs = []
    for pid in test_peaks['pair_id'].unique():
        p_row = test_peaks[test_peaks['pair_id'] == pid]
        n_row = test_neutrals[test_neutrals['pair_id'] == pid]
        if len(p_row) > 0 and len(n_row) > 0:
            valid_pairs.append((pid, p_row.iloc[0], n_row.iloc[0]))

    print(f"Total test pairs: {len(valid_pairs)}")
    candidates, _ = generate_81_candidates()
    suffixes = [c.replace('{"valence": ', '') for c in candidates]

    print("Pre-computing Generation-Time baseline distributions...")
    base_cache_gen = {}
    for pid, peak_row, neut_row in valid_pairs:
        p_prompt = format_base_prompt(peak_row['text'])
        n_prompt = format_base_prompt(neut_row['text'])
        _, tpos, _ = build_generation_prefix_inputs(tokenizer, n_prompt, PREFIX_STR, device=device)
        pp_mat = compute_conditional_candidate_logprobs(model, tokenizer, p_prompt, suffixes, tpos, device=device)
        pn_mat = compute_conditional_candidate_logprobs(model, tokenizer, n_prompt, suffixes, tpos, device=device)
        base_cache_gen[pid] = {"p_peak": pp_mat, "p_neut": pn_mat}

    layers_dict = get_model_layers(model)

    # 1. L15 MLP
    print("Evaluating L15 MLP...")
    target_mod_l15 = get_component_module(model, 15, "mlp")
    l15_mlp_recs = []
    pids = []
    for pid, peak_row, neut_row in valid_pairs:
        pids.append(pid)
        p_prompt = format_base_prompt(peak_row['text'])
        n_prompt = format_base_prompt(neut_row['text'])
        _, tpos, _ = build_generation_prefix_inputs(tokenizer, n_prompt, PREFIX_STR, device=device)
        src_val, _ = extract_generation_time_activation(model, tokenizer, p_prompt, 15, comp="mlp", device=device)
        p_hook = get_patch_hook(src_val, tpos)
        p_mat = compute_conditional_candidate_logprobs(
            model, tokenizer, n_prompt, suffixes, tpos,
            hook_fn=p_hook, hook_module=target_mod_l15, device=device
        )
        r_ot, _, _, is_valid = compute_joint_ot_recovery(
            p_mat, base_cache_gen[pid]["p_peak"], base_cache_gen[pid]["p_neut"], eps_rec=0.05
        )
        l15_mlp_recs.append(r_ot * 100.0 if (is_valid and r_ot is not None) else 0.0)

    # 2. L24 Resid
    print("Evaluating L24 Resid...")
    target_mod_l24 = get_component_module(model, 24, "resid")
    l24_resid_recs = []
    for pid, peak_row, neut_row in valid_pairs:
        p_prompt = format_base_prompt(peak_row['text'])
        n_prompt = format_base_prompt(neut_row['text'])
        _, tpos, _ = build_generation_prefix_inputs(tokenizer, n_prompt, PREFIX_STR, device=device)
        src_val, _ = extract_generation_time_activation(model, tokenizer, p_prompt, 24, comp="resid", device=device)
        p_hook = get_patch_hook(src_val, tpos)
        p_mat = compute_conditional_candidate_logprobs(
            model, tokenizer, n_prompt, suffixes, tpos,
            hook_fn=p_hook, hook_module=target_mod_l24, device=device
        )
        r_ot, _, _, is_valid = compute_joint_ot_recovery(
            p_mat, base_cache_gen[pid]["p_peak"], base_cache_gen[pid]["p_neut"], eps_rec=0.05
        )
        l24_resid_recs.append(r_ot * 100.0 if (is_valid and r_ot is not None) else 0.0)

    l15_arr = np.array(l15_mlp_recs)
    l24_arr = np.array(l24_resid_recs)
    delta_arr = l24_arr - l15_arr

    ci_l15 = bootstrap_ci(l15_arr)
    ci_l24 = bootstrap_ci(l24_arr)
    ci_delta = bootstrap_ci(delta_arr)

    # Save pair-level results for reproducibility
    pair_df = pd.DataFrame({
        "pair_id": pids,
        "l15_mlp_recovery": l15_arr,
        "l24_resid_recovery": l24_arr,
        "delta_g": delta_arr
    })
    pair_df.to_csv("v3/results/focused_causal_sweep_39pairs_pair_level.csv", index=False)
    print("Saved pair-level results to v3/results/focused_causal_sweep_39pairs_pair_level.csv")

    print("\n" + "="*50)
    print(f"L15 MLP Mean: {np.mean(l15_arr):.2f}%, 95% CI: [{ci_l15[0]:.2f}%, {ci_l15[1]:.2f}%]")
    print(f"L24 Resid Mean: {np.mean(l24_arr):.2f}%, 95% CI: [{ci_l24[0]:.2f}%, {ci_l24[1]:.2f}%]")
    print(f"Paired Delta G Mean: {np.mean(delta_arr):.2f}%, 95% CI: [{ci_delta[0]:.2f}%, {ci_delta[1]:.2f}%]")
    print("="*50)

if __name__ == "__main__":
    main()
