import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
import numpy as np
import pandas as pd
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from v2.src.likelihood import generate_81_candidates, compute_likelihoods_for_candidates, compute_expected_va

def get_3way_split(df, group_col="pair_id"):
    """
    Splits the dataframe into strict 3-way splits ensuring no pair_id leakage.
    Train: For probe learning
    Alignment-dev: For fitting Ridge Alignment mapping (Base -> Instruct)
    Test: Strict held-out for patching causal interventions
    """
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    train_idx, temp_idx = next(gss1.split(df, groups=df[group_col]))
    
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]
    
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    dev_idx, test_idx = next(gss2.split(temp_df, groups=temp_df[group_col]))
    
    dev_df = temp_df.iloc[dev_idx]
    test_df = temp_df.iloc[test_idx]
    
    return train_df, dev_df, test_df

def fit_ridge_alignment(base_activations, instruct_activations, alpha=1.0):
    """
    Fit a Ridge regression to map Base representations to Instruct representations
    using only the Alignment-dev split.
    """
    mapper = Ridge(alpha=alpha, fit_intercept=True)
    mapper.fit(base_activations, instruct_activations)
    return mapper

def get_activation(model, tokenizer, text, layer, comp="mlp"):
    # Dummy hook to extract activation
    prompt = f"Read the following text and report your affective state.\n\nText: {text}\n\nRespond strictly in JSON format with 'valence' and 'arousal' keys (1-9)."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    target_pos = inputs.input_ids.shape[1] - 1
    
    extracted_act = {}
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            extracted_act["val"] = output[0][0, target_pos, :].detach().clone()
        else:
            extracted_act["val"] = output[0, target_pos, :].detach().clone()
    
    if comp == "mlp":
        handle = model.model.layers[layer].mlp.register_forward_hook(hook)
    else:
        handle = model.model.layers[layer].self_attn.register_forward_hook(hook)
        
    with torch.no_grad():
        model(**inputs)
        
    handle.remove()
    return extracted_act["val"]

def get_patch_hook(base_val_tensor, target_pos):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            if target_pos < h.shape[1]:
                h[0, target_pos, :] = base_val_tensor.to(h.dtype)
            return (h,) + output[1:]
        else:
            if target_pos < output.shape[1]:
                output[0, target_pos, :] = base_val_tensor.to(output.dtype)
            return output
    return hook

def get_ev_under_patch(model, tokenizer, prompt, candidates, va_pairs, patch_val, layer, comp):
    target_pos = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").shape[1] - 1
    
    if comp == "mlp":
        target_module = model.model.layers[layer].mlp
    else:
        target_module = model.model.layers[layer].self_attn
        
    hook_handle = target_module.register_forward_hook(get_patch_hook(patch_val, target_pos))
    
    l_int, _ = compute_likelihoods_for_candidates(model, tokenizer, prompt, candidates)
    e_v_int, e_a_int, _, _, _ = compute_expected_va(l_int, va_pairs)
    
    hook_handle.remove()
    return e_v_int

def extract_activations(model, tokenizer, df, layer, comp):
    acts = []
    peak_df = df[df['intensity'] == 'peak']
    for idx, row in peak_df.iterrows():
        act = get_activation(model, tokenizer, row['text'], layer, comp)
        acts.append(act.cpu().numpy())
    return np.array(acts)

def run_aligned_patching_experiment(args):
    print("Running Aligned Cross-Model Patching Experiment...")
    
    df = pd.read_csv(args.dataset_path)
    train_df, dev_df, test_df = get_3way_split(df)
    
    print(f"Loading Base: {args.base_model_path} and Instruct: {args.instruct_model_path}")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.float16, device_map="auto")
    
    instruct_tokenizer = AutoTokenizer.from_pretrained(args.instruct_model_path)
    instruct_model = AutoModelForCausalLM.from_pretrained(args.instruct_model_path, torch_dtype=torch.float16, device_map="auto")
    
    candidates, va_pairs = generate_81_candidates()
    
    layer = args.patch_layer
    comp = args.patch_comp
    
    print(f"Extracting Dev Activations for Alignment (Layer {layer} {comp})...")
    base_acts_dev = extract_activations(base_model, base_tokenizer, dev_df, layer, comp)
    inst_acts_dev = extract_activations(instruct_model, instruct_tokenizer, dev_df, layer, comp)
    
    print("Fitting Ridge Alignment Mapping on Dev Split ONLY...")
    mapper = fit_ridge_alignment(base_acts_dev, inst_acts_dev, alpha=1.0)
    
    print(f"Running Interventions on Held-out Test Split ({len(test_df)} samples)...")
    out_file = args.output_path
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    test_results = []
    with open(out_file, 'w') as f:
        for pair_id, group in test_df.groupby('pair_id'):
            peak_rows = group[group['intensity'] == 'peak']
            neutral_rows = group[group['condition'] == 'neutral']
            if peak_rows.empty or neutral_rows.empty:
                continue
                
            uid = pair_id
            target_context = neutral_rows.iloc[0]['text']
            source_context = peak_rows.iloc[0]['text']
            
            prompt = f"Read the following text and report your affective state.\n\nText: {target_context}\n\nRespond strictly in JSON format with 'valence' and 'arousal' keys (1-9)."
            
            # 1. Baseline Target
            l_base, _ = compute_likelihoods_for_candidates(instruct_model, instruct_tokenizer, prompt, candidates)
            e_v_base, _, _, _, _ = compute_expected_va(l_base, va_pairs)
            
            # Condition A: Within-model Patch (Instruct -> Instruct)
            source_act = get_activation(instruct_model, instruct_tokenizer, source_context, layer, comp)
            ev_within = get_ev_under_patch(instruct_model, instruct_tokenizer, prompt, candidates, va_pairs, source_act, layer, comp)
            
            # Condition B: Raw Cross-model Patch (Base -> Instruct)
            raw_base_act = get_activation(base_model, base_tokenizer, source_context, layer, comp)
            ev_raw = get_ev_under_patch(instruct_model, instruct_tokenizer, prompt, candidates, va_pairs, raw_base_act, layer, comp)
            
            # Condition C: Aligned Cross-model Patch (Base -> Mapper -> Instruct)
            aligned_np = mapper.predict(raw_base_act.cpu().numpy().reshape(1, -1))
            aligned_base_act = torch.tensor(aligned_np[0]).to(instruct_model.device).to(torch.float16)
            ev_aligned = get_ev_under_patch(instruct_model, instruct_tokenizer, prompt, candidates, va_pairs, aligned_base_act, layer, comp)
            
            res = {
                "id": uid,
                "e_v_base": e_v_base,
                "e_v_within": ev_within,
                "e_v_raw": ev_raw,
                "e_v_aligned": ev_aligned
            }
            f.write(json.dumps(res) + "\n")
            test_results.append(res)

    print(f"Finished Aligned Patching. Results saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--instruct_model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="v2/data/processed/aipsy_annotated/train_strict.csv")
    parser.add_argument("--output_path", type=str, default="../results/aligned_patching_results.csv")
    parser.add_argument("--patch_layer", type=int, default=15)
    parser.add_argument("--patch_comp", type=str, default="mlp")
    args = parser.parse_args()
    
    run_aligned_patching_experiment(args)
