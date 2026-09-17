import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from v2.src.likelihood import generate_81_candidates, compute_expected_va
from v3.src.batch_likelihood import compute_likelihoods_batched
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

device = "cuda:0"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
data_path = "v3/data/aipsy_strict_expanded.csv"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device
)
model.eval()

df = pd.read_csv(data_path)
train_df, val_df, test_df = get_3way_split(df)
test_peaks = test_df[test_df['intensity'] == 'peak'].copy()
test_neutrals = test_df[test_df['intensity'] == 'neutral'].copy()

valid_pairs = []
for pid in test_peaks['pair_id'].unique():
    p_row = test_peaks[test_peaks['pair_id'] == pid]
    n_row = test_neutrals[test_neutrals['pair_id'] == pid]
    if len(p_row) > 0 and len(n_row) > 0:
        valid_pairs.append((pid, p_row.iloc[0], n_row.iloc[0]))

candidates, va_pairs = generate_81_candidates()
suffixes = [c.replace('{"valence": ', '') for c in candidates]

# Pre-cache generation-time baseline distributions EXACTLY as in run_focused_39pairs_sweep.py
base_cache_gen = {}
for pid, peak_row, neut_row in valid_pairs:
    p_prompt = format_base_prompt(peak_row['text'])
    n_prompt = format_base_prompt(neut_row['text'])
    _, tpos, _ = build_generation_prefix_inputs(tokenizer, n_prompt, PREFIX_STR, device=device)
    pp_mat = compute_conditional_candidate_logprobs(model, tokenizer, p_prompt, suffixes, tpos, device=device)
    pn_mat = compute_conditional_candidate_logprobs(model, tokenizer, n_prompt, suffixes, tpos, device=device)
    base_cache_gen[pid] = {
        "p_peak": pp_mat,
        "p_neut": pn_mat
    }

# L15 MLP
target_mod_l15 = get_component_module(model, 15, "mlp")
gen_recs_l15 = []
for pid, peak_row, neut_row in valid_pairs:
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
    if is_valid and r_ot is not None:
        gen_recs_l15.append(r_ot)

print(f"L15 MLP valid count: {len(gen_recs_l15)}")
print(f"L15 MLP Mean: {np.mean(gen_recs_l15)*100:.4f}%")
print(f"L15 MLP Med : {np.median(gen_recs_l15)*100:.4f}%")

# L24 Resid
target_mod_l24 = get_component_module(model, 24, "resid")
gen_recs_l24 = []
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
    if is_valid and r_ot is not None:
        gen_recs_l24.append(r_ot)

print(f"L24 Resid valid count: {len(gen_recs_l24)}")
print(f"L24 Resid Mean: {np.mean(gen_recs_l24)*100:.4f}%")
print(f"L24 Resid Med : {np.median(gen_recs_l24)*100:.4f}%")
