"""
run_within_model_positive_control.py

Purpose:
Validate whether within-model activation patching from Peak to Neutral stimulus
can causally recover the first-person report distribution within the Instruct model.
Tests multiple intervention sites:
1. MLP output (last token) - original setting
2. Residual stream (last token)
3. Residual stream (all prompt tokens)
4. Multi-layer residual stream (L13-16, last token)
5. Multi-layer residual stream (L13-16, all prompt tokens)
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
import numpy as np
import pandas as pd
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import wasserstein_distance
from v2.src.likelihood import generate_81_candidates, compute_likelihoods_for_candidates, compute_expected_va

def compute_2d_emd(p1, p2):
    """
    Computes Earth Mover's Distance between two 9x9 probability distributions.
    Approximated as sum of 1D Wasserstein distances along V and A marginals.
    """
    v_marg1 = np.sum(p1, axis=1)
    v_marg2 = np.sum(p2, axis=1)
    a_marg1 = np.sum(p1, axis=0)
    a_marg2 = np.sum(p2, axis=0)
    coords = np.arange(1, 10)
    emd_v = wasserstein_distance(coords, coords, v_marg1, v_marg2)
    emd_a = wasserstein_distance(coords, coords, a_marg1, a_marg2)
    return emd_v + emd_a

def format_prompt(text):
    return f"Read the following text and report your affective state.\n\nText: {text}\n\nRespond strictly in JSON format with 'valence' and 'arousal' keys (1-9)."

def get_last_token_patch_hook(source_tensor, target_pos):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            if target_pos < h.shape[1]:
                h[0, target_pos, :] = source_tensor.to(h.dtype)
            return (h,) + output[1:]
        else:
            if target_pos < output.shape[1]:
                output[0, target_pos, :] = source_tensor.to(output.dtype)
            return output
    return hook

def get_all_tokens_patch_hook(source_tensor):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            # Match lengths from the end
            src_len = source_tensor.shape[1]
            tgt_len = h.shape[1]
            min_len = min(src_len, tgt_len)
            h[0, tgt_len - min_len:tgt_len, :] = source_tensor[0, src_len - min_len:src_len, :].to(h.dtype)
            return (h,) + output[1:]
        else:
            src_len = source_tensor.shape[1]
            tgt_len = output.shape[1]
            min_len = min(src_len, tgt_len)
            output[0, tgt_len - min_len:tgt_len, :] = source_tensor[0, src_len - min_len:src_len, :].to(output.dtype)
            return output
    return hook

def run_within_model_experiment(model, tokenizer, test_df, candidates, va_pairs, device="cuda", normalize_length=False):
    """
    Evaluates Peak -> Neutral within-model patching across pair_id groups.
    """
    results = []
    
    # Filter pairs that have both peak and neutral
    pair_ids = test_df['pair_id'].unique()
    valid_pairs = []
    for pid in pair_ids:
        pdf = test_df[test_df['pair_id'] == pid]
        if 'peak' in pdf['intensity'].values and 'neutral' in pdf['intensity'].values:
            peak_row = pdf[pdf['intensity'] == 'peak'].iloc[0]
            neutral_row = pdf[pdf['intensity'] == 'neutral'].iloc[0]
            valid_pairs.append((pid, peak_row, neutral_row))
            
    print(f"Found {len(valid_pairs)} complete peak-neutral pairs in held-out test split.")
    
    protocols = [
        ("mlp_last_token_L15", [15], "mlp", False),
        ("resid_last_token_L15", [15], "resid", False),
        ("resid_all_tokens_L15", [15], "resid", True),
        ("multi_resid_last_L13_16", [13, 14, 15, 16], "resid", False),
        ("multi_resid_all_L13_16", [13, 14, 15, 16], "resid", True)
    ]
    
    for proto_name, layers, comp, all_tokens in protocols:
        print(f"\n--- Testing Protocol: {proto_name} ---")
        recoveries = []
        shifts = []
        
        for pid, peak_row, neutral_row in valid_pairs[:20]: # Test on first 20 pairs for validation
            peak_prompt = format_prompt(peak_row['text'])
            neutral_prompt = format_prompt(neutral_row['text'])
            
            # 1. Source (Peak) report distribution
            l_peak, _ = compute_likelihoods_for_candidates(model, tokenizer, peak_prompt, candidates, device=device, normalize_length=normalize_length)
            ev_peak, _, _, _, p_peak = compute_expected_va(l_peak, va_pairs)
            
            # 2. Target (Neutral) report distribution
            l_neut, _ = compute_likelihoods_for_candidates(model, tokenizer, neutral_prompt, candidates, device=device, normalize_length=normalize_length)
            ev_neut, _, _, _, p_neut = compute_expected_va(l_neut, va_pairs)
            
            # Extract activations from Peak run
            extracted_acts = {}
            handles = []
            
            peak_inputs = tokenizer(peak_prompt, return_tensors="pt").to(device)
            peak_last_pos = peak_inputs.input_ids.shape[1] - 1
            
            for l in layers:
                if comp == "mlp":
                    mod = model.model.layers[l].mlp
                else:
                    mod = model.model.layers[l] # full layer output (residual stream)
                    
                def make_extract_hook(layer_idx):
                    def hook(m, inp, out):
                        val = out[0] if isinstance(out, tuple) else out
                        extracted_acts[layer_idx] = val.detach().clone()
                    return hook
                handles.append(mod.register_forward_hook(make_extract_hook(l)))
                
            with torch.no_grad():
                model(**peak_inputs)
            for h in handles:
                h.remove()
                
            # 3. Patch into Neutral run
            patch_handles = []
            neutral_inputs = tokenizer(neutral_prompt, return_tensors="pt").to(device)
            neutral_last_pos = neutral_inputs.input_ids.shape[1] - 1
            
            for l in layers:
                if comp == "mlp":
                    target_mod = model.model.layers[l].mlp
                else:
                    target_mod = model.model.layers[l]
                    
                src_act = extracted_acts[l]
                if all_tokens:
                    patch_handles.append(target_mod.register_forward_hook(get_all_tokens_patch_hook(src_act)))
                else:
                    patch_handles.append(target_mod.register_forward_hook(
                        get_last_token_patch_hook(src_act[0, peak_last_pos, :], neutral_last_pos)
                    ))
                    
            # Compute report distribution under patch
            l_patch, _ = compute_likelihoods_for_candidates(model, tokenizer, neutral_prompt, candidates, device=device, normalize_length=normalize_length)
            ev_patch, _, _, _, p_patch = compute_expected_va(l_patch, va_pairs)
            
            for h in patch_handles:
                h.remove()
                
            # Compute recovery
            # 2D EMD using normalized probability distributions
            p_source_mat = np.array(p_peak).reshape(9, 9)
            p_target_mat = np.array(p_neut).reshape(9, 9)
            p_patch_mat = np.array(p_patch).reshape(9, 9)
            
            baseline_emd = compute_2d_emd(p_target_mat, p_source_mat)
            patch_emd = compute_2d_emd(p_patch_mat, p_source_mat)
            
            if baseline_emd > 1e-6:
                rec = 1.0 - (patch_emd / baseline_emd)
            else:
                rec = 0.0
            recoveries.append(rec)
            
            # Expected Valence Shift
            denom = ev_peak - ev_neut
            shift = (ev_patch - ev_neut) / denom if abs(denom) > 1e-4 else 0.0
            shifts.append(shift)
            
        mean_rec = np.mean(recoveries) * 100
        median_rec = np.median(recoveries) * 100
        mean_shift = np.mean(shifts) * 100
        print(f"Protocol: {proto_name} | Mean EMD Recovery: {mean_rec:.2f}% | Median: {median_rec:.2f}% | EV Shift: {mean_shift:.2f}%")
        
        results.append({
            "protocol": proto_name,
            "mean_emd_recovery_pct": mean_rec,
            "median_emd_recovery_pct": median_rec,
            "mean_ev_shift_pct": mean_shift
        })
        
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data_path", type=str, default="v3/data/aipsy_strict_expanded.csv")
    parser.add_argument("--output_path", type=str, default="v3/results/within_model_positive_control_results.csv")
    parser.add_argument("--normalize_length", action="store_true", help="Use length-normalized candidate log-likelihood")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {args.model_name} on {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    model.eval()
    
    df = pd.read_csv(args.data_path)
    from v3.scripts.run_aligned_cross_model_patching import get_3way_split
    _, _, test_df = get_3way_split(df)
    
    candidates, va_pairs = generate_81_candidates()
    
    res_df = run_within_model_experiment(model, tokenizer, test_df, candidates, va_pairs, device=device, normalize_length=args.normalize_length)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    res_df.to_csv(args.output_path, index=False)
    print(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    main()
