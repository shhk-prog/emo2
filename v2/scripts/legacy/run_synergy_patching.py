#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import json
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from v2.src.likelihood import generate_81_candidates, compute_likelihoods_for_candidates, compute_expected_va

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

def apply_chat_template(tokenizer, text):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Read the following text and report your affective state.\n\nText: {text}\n\nRespond strictly in JSON format with 'valence' and 'arousal' keys (1-9)."}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--base-states", type=str, default="v2/results/derived/phase4_circuit/Qwen_Qwen2.5-1.5B_module_states.npz")
    parser.add_argument("--patches", type=str, nargs='+', required=True, help="Format: layer_component e.g. 20_mlp 24_attn")
    parser.add_argument("--data-dir", type=str, default="v2/data/processed/aipsy_annotated")
    parser.add_argument("--out-dir", type=str, default="v2/results/derived/phase6_synergy")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    test_df = pd.read_csv(os.path.join(args.data_dir, "test_strict.csv"))
    if args.limit > 0:
        test_df = test_df.head(args.limit)
        
    base_states = dict(np.load(args.base_states, allow_pickle=True))
    for k, v in base_states.items():
        if v.shape == (): base_states[k] = v.item()
        
    print(f"Loading {args.target_model} for Synergy Patching...")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    model = AutoModelForCausalLM.from_pretrained(args.target_model, torch_dtype=torch.float16, device_map="auto")
    
    candidates, va_pairs = generate_81_candidates()
    
    patch_name = "-".join(args.patches)
    out_file = os.path.join(args.out_dir, f"synergy_patching_{patch_name}.jsonl")
    
    with open(out_file, 'w') as f:
        for idx, row in test_df.iterrows():
            uid = row['id']
            
            prompt = apply_chat_template(tokenizer, row['text'])
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
            prompt_length = prompt_ids.shape[1]
            target_pos = prompt_length - 1
            
            hooks = []
            skip = False
            for p in args.patches:
                layer_str, comp = p.split("_")
                l = int(layer_str)
                key = f"{uid}_layer{l}_{comp}"
                
                if key not in base_states:
                    print(f"Missing base state for {key}. Skipping {uid}.")
                    skip = True
                    break
                    
                base_val = torch.tensor(base_states[key]).to(model.device)
                
                if comp == "attn":
                    target_module = model.model.layers[l].self_attn
                elif comp == "mlp":
                    target_module = model.model.layers[l].mlp
                elif comp == "res":
                    target_module = model.model.layers[l]
                else:
                    raise ValueError(f"Unknown component {comp}")
                    
                hooks.append(target_module.register_forward_hook(get_patch_hook(base_val, target_pos)))
                
            if skip:
                for h in hooks: h.remove()
                continue
                
            likelihoods, _ = compute_likelihoods_for_candidates(model, tokenizer, prompt, candidates)
            
            for h in hooks: h.remove()
            
            if len(likelihoods) == 81:
                E_v, E_a, entropy, p_55, _ = compute_expected_va(likelihoods, va_pairs)
                res = {
                    "id": uid,
                    "condition": row.get('condition', 'unknown'),
                    "E_v": E_v,
                    "E_a": E_a,
                    "entropy": entropy,
                    "p_55": p_55
                }
                f.write(json.dumps(res) + "\n")
                
    print(f"Saved synergy patching results to {out_file}")

if __name__ == "__main__":
    main()
