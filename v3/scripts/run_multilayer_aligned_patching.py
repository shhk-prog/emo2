#!/usr/bin/env python3
"""
v3/scripts/run_multilayer_aligned_patching.py

Multi-layer Simultaneous Aligned Cross-Model Patching with Multivariate Mahalanobis OOD Diagnostics.
Tests:
- Block sizes: 1-Layer (15), 2-Layer ([14, 15]), 4-Layer ([13, 14, 15, 16]), 8-Layer ([11, 12, 13, 14, 15, 16, 17, 18])
- Conditions: Target Baseline, Within-model Positive Control, Raw Cross-model, Aligned Cross-model
- Diagnostics: Ledoit-Wolf Mahalanobis Distance (D_M), Norm Ratio, Cosine Similarity, PCA projection error
- Endpoint: Normalized 2D EMD Recovery and Expected Valence (E[V])
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
import numpy as np
import pandas as pd
import torch
import json
from sklearn.linear_model import Ridge
from scipy.stats import wasserstein_distance
from transformers import AutoModelForCausalLM, AutoTokenizer

from v2.src.likelihood import generate_81_candidates, compute_likelihoods_for_candidates, compute_expected_va
from v3.src.diagnostics import MultivariateActivationDiagnostics

def compute_2d_emd(p1, p2):
    """
    Computes 2D Earth Mover's Distance between two probability distributions over the 9x9 grid.
    Approximated via marginal Wasserstein distances along V and A.
    """
    v_marg1 = p1.sum(axis=1)
    v_marg2 = p2.sum(axis=1)
    a_marg1 = p1.sum(axis=0)
    a_marg2 = p2.sum(axis=0)
    
    grid = np.arange(1, 10)
    wv = wasserstein_distance(grid, grid, v_marg1, v_marg2)
    wa = wasserstein_distance(grid, grid, a_marg1, a_marg2)
    return float(np.sqrt(wv**2 + wa**2))

def get_activation(model, tokenizer, text, layer, comp="mlp"):
    prompt = f"Read the following text and report your affective state.\n\nText: {text}\n\nRespond strictly in JSON format with 'valence' and 'arousal' keys (1-9)."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    target_pos = inputs.input_ids.shape[1] - 1
    
    act_dict = {}
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            act_dict["val"] = output[0][0, target_pos, :].detach().clone()
        else:
            act_dict["val"] = output[0, target_pos, :].detach().clone()
            
    if comp == "mlp":
        handle = model.model.layers[layer].mlp.register_forward_hook(hook)
    else:
        handle = model.model.layers[layer].self_attn.register_forward_hook(hook)
        
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return act_dict["val"]

def extract_layer_activations(model, tokenizer, df, layer, comp="mlp"):
    acts = []
    for _, row in df.iterrows():
        act = get_activation(model, tokenizer, row['text'], layer, comp)
        acts.append(act.cpu().numpy())
    return np.array(acts)

def get_multi_patch_hooks(patch_dict, target_pos, comp="mlp"):
    """
    Registers simultaneous hooks across multiple layers.
    patch_dict: {layer_idx: tensor_val}
    """
    handles = []
    for layer_idx, tensor_val in patch_dict.items():
        def make_hook(val_tensor):
            def hook(module, inputs, output):
                if isinstance(output, tuple):
                    h = output[0]
                    if target_pos < h.shape[1]:
                        h[0, target_pos, :] = val_tensor.to(h.dtype)
                    return (h,) + output[1:]
                else:
                    if target_pos < output.shape[1]:
                        output[0, target_pos, :] = val_tensor.to(output.dtype)
                    return output
            return hook
        return make_hook(tensor_val)

def evaluate_multi_patch(model, tokenizer, prompt, candidates, va_pairs, patch_dict, comp="mlp"):
    """
    Evaluates 81 candidates likelihood under simultaneous multi-layer patch.
    """
    target_pos = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").shape[1] - 1
    handles = []
    
    for layer_idx, val_tensor in patch_dict.items():
        if comp == "mlp":
            target_module = model.model.layers[layer_idx].mlp
        else:
            target_module = model.model.layers[layer_idx].self_attn
            
        def make_hook(val):
            def hook(module, inputs, output):
                if isinstance(output, tuple):
                    h = output[0]
                    if target_pos < h.shape[1]:
                        h[0, target_pos, :] = val.to(h.device).to(h.dtype)
                    return (h,) + output[1:]
                else:
                    if target_pos < output.shape[1]:
                        output[0, target_pos, :] = val.to(output.device).to(output.dtype)
                    return output
            return hook
            
        h = target_module.register_forward_hook(make_hook(val_tensor))
        handles.append(h)
        
    l_patched, _ = compute_likelihoods_for_candidates(model, tokenizer, prompt, candidates, device=model.device)
    e_v, e_a, _, _, probs = compute_expected_va(l_patched, va_pairs)
    p_grid = probs.reshape(9, 9)
    
    for h in handles:
        h.remove()
        
    return e_v, e_a, p_grid

def run_experiment(args):
    print("=== Multi-Layer Aligned Patching & Mahalanobis OOD Experiment ===")
    print(f"Dataset: {args.dataset_path}")
    df = pd.read_csv(args.dataset_path)
    
    train_df = df[df['split'] == 'train']
    dev_df = df[df['split'] == 'dev']
    test_df = df[df['split'] == 'test']
    
    print(f"Splits: Train={train_df['pair_id'].nunique()} pairs, Dev={dev_df['pair_id'].nunique()} pairs, Test={test_df['pair_id'].nunique()} pairs")
    
    print(f"Loading Base: {args.base_model_path}")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.float16, device_map="auto")
    
    print(f"Loading Instruct: {args.instruct_model_path}")
    inst_tokenizer = AutoTokenizer.from_pretrained(args.instruct_model_path)
    inst_model = AutoModelForCausalLM.from_pretrained(args.instruct_model_path, torch_dtype=torch.float16, device_map="auto")
    
    candidates, va_pairs = generate_81_candidates()
    
    # Layer blocks to test
    block_configs = {
        "1-Layer (L15)": [15],
        "2-Layer (L14-15)": [14, 15],
        "4-Layer (L13-16)": [13, 14, 15, 16],
        "8-Layer (L11-18)": [11, 12, 13, 14, 15, 16, 17, 18]
    }
    
    all_target_layers = sorted(list(set(l for layers in block_configs.values() for l in layers)))
    comp = args.patch_comp
    
    # Step 1: Pre-fit Ridge Alignment Mappers & Multivariate Diagnostics for all relevant layers
    print("\n--- Step 1: Fitting Ridge Mappers and Mahalanobis Diagnostics on Dev Split ---")
    mappers = {}
    diagnostics = {}
    
    dev_peak_df = dev_df[dev_df['intensity'] == 'peak']
    dev_neu_df = dev_df[dev_df['condition'] == 'neutral']
    
    for layer in all_target_layers:
        print(f"  Fitting layer {layer}...")
        base_acts = extract_layer_activations(base_model, base_tokenizer, dev_peak_df, layer, comp)
        inst_acts = extract_layer_activations(inst_model, inst_tokenizer, dev_peak_df, layer, comp)
        inst_neu_acts = extract_layer_activations(inst_model, inst_tokenizer, dev_neu_df, layer, comp)
        
        # Ridge Mapper (Base -> Instruct)
        ridge = Ridge(alpha=1.0, fit_intercept=True)
        ridge.fit(base_acts, inst_acts)
        mappers[layer] = ridge
        
        # Diagnostics fitted on natural target in-distribution (Instruct peak & neutral)
        diag = MultivariateActivationDiagnostics(n_components=5)
        diag.fit(np.vstack([inst_acts, inst_neu_acts]))
        diagnostics[layer] = diag

    # Step 2: Evaluate on Held-out Test Split
    print("\n--- Step 2: Evaluating Simultaneous Patching on Test Split ---")
    test_pairs = test_df['pair_id'].unique()
    if args.max_test_pairs > 0:
        test_pairs = test_pairs[:args.max_test_pairs]
        
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    results = []
    
    for pid in test_pairs:
        p_group = test_df[test_df['pair_id'] == pid]
        peak_row = p_group[p_group['intensity'] == 'peak']
        neu_row = p_group[p_group['condition'] == 'neutral']
        if peak_row.empty or neu_row.empty:
            continue
            
        target_text = neu_row.iloc[0]['text']
        source_text = peak_row.iloc[0]['text']
        
        prompt = f"Read the following text and report your affective state.\n\nText: {target_text}\n\nRespond strictly in JSON format with 'valence' and 'arousal' keys (1-9)."
        source_prompt = f"Read the following text and report your affective state.\n\nText: {source_text}\n\nRespond strictly in JSON format with 'valence' and 'arousal' keys (1-9)."
        
        # Baseline Target Distribution (Instruct neutral)
        l_target, _ = compute_likelihoods_for_candidates(inst_model, inst_tokenizer, prompt, candidates, device=inst_model.device)
        ev_target, ea_target, _, _, probs_target = compute_expected_va(l_target, va_pairs)
        p_target = probs_target.reshape(9, 9)
        
        # Source Baseline Distribution (Base peak)
        l_source, _ = compute_likelihoods_for_candidates(base_model, base_tokenizer, source_prompt, candidates, device=base_model.device)
        ev_source, ea_source, _, _, probs_source = compute_expected_va(l_source, va_pairs)
        p_source = probs_source.reshape(9, 9)
        
        baseline_emd = compute_2d_emd(p_target, p_source)
        
        # Pre-extract all activations for this pair across all layers
        base_acts_pair = {}
        inst_source_acts_pair = {}
        aligned_acts_pair = {}
        layer_diagnostics_res = {}
        
        for l in all_target_layers:
            b_act = get_activation(base_model, base_tokenizer, source_text, l, comp)
            i_act = get_activation(inst_model, inst_tokenizer, source_text, l, comp)
            base_acts_pair[l] = b_act
            inst_source_acts_pair[l] = i_act
            
            # Map Base -> Aligned
            b_np = b_act.cpu().numpy().reshape(1, -1)
            aligned_np = mappers[l].predict(b_np)[0]
            aligned_acts_pair[l] = torch.tensor(aligned_np, dtype=torch.float16, device=inst_model.device)
            
            # OOD Diagnostics for layer l
            d_raw = diagnostics[l].compute_diagnostics(b_act.cpu().numpy())
            d_aligned = diagnostics[l].compute_diagnostics(aligned_np)
            layer_diagnostics_res[l] = {
                "raw_mahalanobis": d_raw["mahalanobis_distance"],
                "raw_norm_ratio": d_raw["norm_ratio"],
                "raw_cosine": d_raw["cosine_similarity"],
                "aligned_mahalanobis": d_aligned["mahalanobis_distance"],
                "aligned_norm_ratio": d_aligned["norm_ratio"],
                "aligned_cosine": d_aligned["cosine_similarity"],
                "aligned_pct": d_aligned["mahalanobis_percentile"]
            }

        # Test each block configuration
        pair_res = {
            "pair_id": pid,
            "ev_target": ev_target,
            "ev_source": ev_source,
            "baseline_emd": baseline_emd,
            "layer_diagnostics": layer_diagnostics_res,
            "blocks": {}
        }
        
        for block_name, block_layers in block_configs.items():
            # 1. Within-model Positive Control
            within_dict = {l: inst_source_acts_pair[l] for l in block_layers}
            ev_within, _, p_within = evaluate_multi_patch(inst_model, inst_tokenizer, prompt, candidates, va_pairs, within_dict, comp)
            emd_within = compute_2d_emd(p_within, p_source)
            rec_within = 1.0 - (emd_within / (baseline_emd + 1e-9)) if baseline_emd > 0.01 else 0.0
            
            # 2. Raw Cross-Model Patch
            raw_dict = {l: base_acts_pair[l] for l in block_layers}
            ev_raw, _, p_raw = evaluate_multi_patch(inst_model, inst_tokenizer, prompt, candidates, va_pairs, raw_dict, comp)
            emd_raw = compute_2d_emd(p_raw, p_source)
            rec_raw = 1.0 - (emd_raw / (baseline_emd + 1e-9)) if baseline_emd > 0.01 else 0.0
            
            # 3. Aligned Cross-Model Patch
            aligned_dict = {l: aligned_acts_pair[l] for l in block_layers}
            ev_aligned, _, p_aligned = evaluate_multi_patch(inst_model, inst_tokenizer, prompt, candidates, va_pairs, aligned_dict, comp)
            emd_aligned = compute_2d_emd(p_aligned, p_source)
            rec_aligned = 1.0 - (emd_aligned / (baseline_emd + 1e-9)) if baseline_emd > 0.01 else 0.0
            
            pair_res["blocks"][block_name] = {
                "layers": block_layers,
                "ev_within": ev_within,
                "rec_within": rec_within,
                "ev_raw": ev_raw,
                "rec_raw": rec_raw,
                "ev_aligned": ev_aligned,
                "rec_aligned": rec_aligned
            }
            
        results.append(pair_res)
        print(f"Pair {pid} done. Block 1-L Rec={pair_res['blocks']['1-Layer (L15)']['rec_aligned']:.3f}, Block 8-L Rec={pair_res['blocks']['8-Layer (L11-18)']['rec_aligned']:.3f}")
        
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--instruct_model_path", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset_path", type=str, default="v3/data/aipsy_strict_expanded.csv")
    parser.add_argument("--output_path", type=str, default="v3/results/multilayer_patching_results.json")
    parser.add_argument("--patch_comp", type=str, default="mlp")
    parser.add_argument("--max_test_pairs", type=int, default=20)
    args = parser.parse_args()
    
    run_experiment(args)
