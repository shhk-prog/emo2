#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import json
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from v2.src.likelihood import generate_81_candidates, compute_likelihoods_for_candidates, compute_expected_va

def get_capture_hook(cache, name):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            cache[name] = output[0][:, -1, :].detach().clone()
        else:
            cache[name] = output[:, -1, :].detach().clone()
    return hook

def get_patch_hook(base_val_tensor, target_pos):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            if target_pos < h.shape[1]:
                h[:, target_pos, :] = base_val_tensor.to(h.dtype)
            return (h,) + output[1:]
        else:
            if target_pos < output.shape[1]:
                output[:, target_pos, :] = base_val_tensor.to(output.dtype)
            return output
    return hook

def apply_chat_template(tokenizer, text):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Read the following text and report your affective state.\n\nText: {text}\n\nRespond strictly in JSON format with 'valence' and 'arousal' keys (1-9)."}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--target-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--layers", type=int, nargs='+', default=list(range(10, 28)))
    parser.add_argument("--components", type=str, nargs='+', default=["res", "attn", "mlp"])
    parser.add_argument("--data-dir", type=str, default="v2/data/processed/aipsy_annotated")
    parser.add_argument("--out-dir", type=str, default="v2/results/derived/phase7_strict_patching")
    parser.add_argument("--limit", type=int, default=50) # Use 50 for dry run, 0 for all
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    test_df = pd.read_csv(os.path.join(args.data_dir, "test_strict.csv"))
    if args.limit > 0:
        test_df = test_df.head(args.limit)
    test_df = test_df.reset_index(drop=True)
        
    print(f"Loading Instruct model (for tokenizer and target): {args.target_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    inst_model = AutoModelForCausalLM.from_pretrained(args.target_model, torch_dtype=torch.float16, device_map="auto")
    
    print(f"Loading Base model (for strict source extraction): {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16, device_map="auto")
    
    candidates, va_pairs = generate_81_candidates()
    
    # Pre-compute baseline Instruct values (under strict prompt)
    baseline_dict = {}
    print("Computing strict baseline for Instruct model...")
    for idx, row in test_df.iterrows():
        prompt = apply_chat_template(tokenizer, row['text'])
        # compute_likelihoods_for_candidates internally encodes prompt again, we must pass the same to it, or let it do it.
        # compute_likelihoods_for_candidates encodes the string. Since tokenizer is the Instruct tokenizer, it will encode identically.
        likelihoods, _ = compute_likelihoods_for_candidates(inst_model, tokenizer, prompt, candidates)
        if len(likelihoods) == 81:
            E_v, E_a, entropy, p_55, _ = compute_expected_va(likelihoods, va_pairs)
            baseline_dict[row['id']] = {'E_v': E_v, 'E_a': E_a}
    
    summary_results = []
    
    for l in args.layers:
        for comp in args.components:
            print(f"--- Patching Layer {l} Component {comp} ---")
            
            delta_v_list, delta_a_list = [], []
            
            # Identify modules
            if comp == "attn":
                base_module = base_model.model.layers[l].self_attn
                inst_module = inst_model.model.layers[l].self_attn
            elif comp == "mlp":
                base_module = base_model.model.layers[l].mlp
                inst_module = inst_model.model.layers[l].mlp
            elif comp == "res":
                base_module = base_model.model.layers[l]
                inst_module = inst_model.model.layers[l]
            else:
                continue
                
            for idx, row in test_df.iterrows():
                uid = row['id']
                if uid not in baseline_dict: continue
                
                prompt = apply_chat_template(tokenizer, row['text'])
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(base_model.device)
                
                prompt_length = prompt_ids.shape[1]
                target_pos = prompt_length - 1
                
                # 1. Capture Base activation
                base_cache = {}
                base_handle = base_module.register_forward_hook(get_capture_hook(base_cache, "comp"))
                with torch.no_grad():
                    base_model(prompt_ids)
                base_handle.remove()
                
                base_val = base_cache["comp"] # shape: (1, hidden_size)
                
                # 2. Patch Instruct model
                inst_handle = inst_module.register_forward_hook(get_patch_hook(base_val, target_pos))
                likelihoods, _ = compute_likelihoods_for_candidates(inst_model, tokenizer, prompt, candidates)
                inst_handle.remove()
                
                if len(likelihoods) == 81:
                    E_v, E_a, entropy, p_55, _ = compute_expected_va(likelihoods, va_pairs)
                    orig_v = baseline_dict[uid]['E_v']
                    orig_a = baseline_dict[uid]['E_a']
                    
                    delta_v_list.append(E_v - orig_v)
                    delta_a_list.append(E_a - orig_a)
            
            if len(delta_v_list) > 0:
                summary_results.append({
                    "layer": l,
                    "component": comp,
                    "mean_delta_v": np.mean(delta_v_list),
                    "mean_delta_a": np.mean(delta_a_list)
                })
                
    summary_df = pd.DataFrame(summary_results)
    out_csv = os.path.join(args.out_dir, "strict_patching_screening_summary.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"Saved strict screening summary to {out_csv}")

if __name__ == "__main__":
    main()
