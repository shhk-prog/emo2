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
from scipy.stats import wasserstein_distance

def get_capture_hook(cache, name):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            cache[name] = output[0][:, -1, :].detach().clone()
        else:
            cache[name] = output[:, -1, :].detach().clone()
    return hook

def apply_chat_template_val_forcing(tokenizer, text):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Read the following text and report your affective state.\n\nText: {text}\n\nRespond strictly in JSON format with 'valence' and 'arousal' keys (1-9)."}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt += '{\n  "valence": '
    return prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--target-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data-dir", type=str, default="v2/data/processed/aipsy_annotated")
    parser.add_argument("--out-dir", type=str, default="v2/results/derived/phase6_path_patching")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    test_df = pd.read_csv(os.path.join(args.data_dir, "test_strict.csv"))
    if args.limit > 0:
        test_df = test_df.head(args.limit)
    test_df = test_df.reset_index(drop=True)
        
    print(f"Loading Base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16, device_map="auto")
    
    print(f"Loading Instruct model: {args.target_model}")
    inst_model = AutoModelForCausalLM.from_pretrained(args.target_model, torch_dtype=torch.float16, device_map="auto")
    
    # Get token IDs for '1' to '9'
    numeric_tokens = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 10)]
    
    components_to_test = [("res", 10), ("res", 15), ("res", 16), ("mlp", 10), ("attn", 14)]
    
    summary_results = []
    
    for comp_type, layer in components_to_test:
        print(f"--- Path Patching Layer {layer} {comp_type} ---")
        
        # We need to hook Base for the component, Instruct for the component, and Instruct for final layer
        base_cache = {}
        inst_cache = {}
        
        if comp_type == "attn":
            base_module = base_model.model.layers[layer].self_attn
            inst_module = inst_model.model.layers[layer].self_attn
        elif comp_type == "mlp":
            base_module = base_model.model.layers[layer].mlp
            inst_module = inst_model.model.layers[layer].mlp
        elif comp_type == "res":
            base_module = base_model.model.layers[layer]
            inst_module = inst_model.model.layers[layer]
            
        base_handle = base_module.register_forward_hook(get_capture_hook(base_cache, "comp"))
        inst_handle = inst_module.register_forward_hook(get_capture_hook(inst_cache, "comp"))
        
        # Hook for final residual stream
        # Qwen2 has `model.layers[-1]`, so we can just hook the last layer
        num_layers = len(inst_model.model.layers)
        inst_final_handle = inst_model.model.layers[-1].register_forward_hook(get_capture_hook(inst_cache, "final_res"))
        
        # To compute Random control, we need to collect base_states first or do it on the fly by shifting indices.
        # Let's collect all base states for this component first
        all_base_comps = []
        
        for idx, row in test_df.iterrows():
            prompt = apply_chat_template_val_forcing(tokenizer, row['text'])
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(base_model.device)
            with torch.no_grad():
                base_model(prompt_ids)
            all_base_comps.append(base_cache["comp"])
            
        all_base_comps = torch.stack(all_base_comps) # (N, 1, hidden_size)
        np.random.seed(42)
        random_indices = np.random.permutation(len(test_df))
        
        for i, row in test_df.iterrows():
            prompt = apply_chat_template_val_forcing(tokenizer, row['text'])
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(inst_model.device)
            
            with torch.no_grad():
                inst_outputs = inst_model(prompt_ids)
                
            base_c = all_base_comps[i].to(inst_model.device)
            rand_base_c = all_base_comps[random_indices[i]].to(inst_model.device)
            inst_c = inst_cache["comp"]
            inst_final_res = inst_cache["final_res"] # (1, hidden_size)
            
            # Baseline Instruct Logits
            orig_logits = inst_outputs.logits[0, -1, numeric_tokens]
            orig_probs = F.softmax(orig_logits, dim=-1)
            orig_Ev = torch.sum(orig_probs * torch.arange(1, 10, device=orig_probs.device)).item()
            
            # Path Patched (Base)
            patched_res = inst_final_res - inst_c + base_c
            # Forward through norm and lm_head
            patched_norm = inst_model.model.norm(patched_res)
            patched_logits_full = inst_model.lm_head(patched_norm)
            patched_logits = patched_logits_full[0, numeric_tokens]
            patched_probs = F.softmax(patched_logits, dim=-1)
            patched_Ev = torch.sum(patched_probs * torch.arange(1, 10, device=patched_probs.device)).item()
            
            # Path Patched (Random Base)
            rand_res = inst_final_res - inst_c + rand_base_c
            rand_norm = inst_model.model.norm(rand_res)
            rand_logits_full = inst_model.lm_head(rand_norm)
            rand_logits = rand_logits_full[0, numeric_tokens]
            rand_probs = F.softmax(rand_logits, dim=-1)
            rand_Ev = torch.sum(rand_probs * torch.arange(1, 10, device=rand_probs.device)).item()
            
            # Baseline Base Logits (for WD baseline)
            # Since we didn't save final logit of base, let's just forward it
            with torch.no_grad():
                base_outputs = base_model(prompt_ids)
                base_logits = base_outputs.logits[0, -1, numeric_tokens]
                base_probs = F.softmax(base_logits, dim=-1)
                
            # WD to Base
            wd_inst_base = wasserstein_distance(np.arange(1, 10), np.arange(1, 10), u_weights=orig_probs.cpu().detach().numpy(), v_weights=base_probs.cpu().detach().numpy())
            wd_patch_base = wasserstein_distance(np.arange(1, 10), np.arange(1, 10), u_weights=patched_probs.cpu().detach().numpy(), v_weights=base_probs.cpu().detach().numpy())
            wd_rand_base = wasserstein_distance(np.arange(1, 10), np.arange(1, 10), u_weights=rand_probs.cpu().detach().numpy(), v_weights=base_probs.cpu().detach().numpy())
            
            summary_results.append({
                "id": row['id'],
                "component": f"{comp_type}_{layer}",
                "orig_Ev": orig_Ev,
                "patched_Ev": patched_Ev,
                "rand_Ev": rand_Ev,
                "wd_inst_base": wd_inst_base,
                "wd_patch_base": wd_patch_base,
                "wd_rand_base": wd_rand_base,
                "delta_Ev_patch": patched_Ev - orig_Ev,
                "delta_Ev_rand": rand_Ev - orig_Ev,
                "delta_WD_patch": wd_patch_base - wd_inst_base,
                "delta_WD_rand": wd_rand_base - wd_inst_base,
                "orig_prob_5": orig_probs[4].item(),
                "patched_prob_5": patched_probs[4].item(),
                "rand_prob_5": rand_probs[4].item()
            })
            
        base_handle.remove()
        inst_handle.remove()
        inst_final_handle.remove()
        
    summary_df = pd.DataFrame(summary_results)
    out_csv = os.path.join(args.out_dir, "path_patching_results.csv")
    summary_df.to_csv(out_csv, index=False)
    
    # Generate an aggregated view
    agg_df = summary_df.groupby("component").mean()[["delta_Ev_patch", "delta_Ev_rand", "delta_WD_patch", "delta_WD_rand", "orig_prob_5", "patched_prob_5", "rand_prob_5"]]
    agg_csv = os.path.join(args.out_dir, "path_patching_aggregated.csv")
    agg_df.to_csv(agg_csv)
    print(f"Saved results to {out_csv} and {agg_csv}")

if __name__ == "__main__":
    main()
