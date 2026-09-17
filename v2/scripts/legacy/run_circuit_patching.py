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
        # output is usually (hidden_states, ...)
        if isinstance(output, tuple):
            h = output[0]
            # h shape: (batch, seq_len, hidden_dim)
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
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--module", type=str, choices=["attn", "mlp"], default="mlp")
    parser.add_argument("--data-dir", type=str, default="v2/data/processed/aipsy")
    parser.add_argument("--out-dir", type=str, default="v2/results/derived/phase4_circuit")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. Load Data
    test_df = pd.read_csv(os.path.join(args.data_dir, "test_strict.csv")).head(args.limit)
    base_states = dict(np.load(args.base_states, allow_pickle=True))
    
    # 2. Load Model
    print(f"Loading {args.target_model} for Activation Patching...")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    model = AutoModelForCausalLM.from_pretrained(args.target_model, torch_dtype=torch.float16, device_map="auto")
    
    candidates, va_pairs = generate_81_candidates()
    
    out_file = os.path.join(args.out_dir, f"patching_layer{args.layer}_{args.module}.jsonl")
    
    with open(out_file, 'w') as f:
        for idx, row in test_df.iterrows():
            uid = row['id']
            key = f"{uid}_layer{args.layer}_{args.module}"
            
            if key not in base_states:
                print(f"Skipping {uid}, base state not found.")
                continue
                
            base_val = torch.tensor(base_states[key]).to(model.device)
            
            prompt = apply_chat_template(tokenizer, row['text'])
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
            prompt_length = prompt_ids.shape[1]
            
            target_pos = prompt_length - 1
            
            # Register hook
            target_module = model.model.layers[args.layer].self_attn if args.module == "attn" else model.model.layers[args.layer].mlp
            hook_handle = target_module.register_forward_hook(get_patch_hook(base_val, target_pos))
            
            # Compute Likelihoods with patch
            likelihoods, _ = compute_likelihoods_for_candidates(model, tokenizer, prompt, candidates)
            hook_handle.remove()
            
            if len(likelihoods) == 81:
                E_v, E_a, entropy, p_55, _ = compute_expected_va(likelihoods, va_pairs)
                res = {
                    "id": uid,
                    "condition": row.get('condition', 'unknown'),
                    "intensity": row.get('intensity', 'unknown'),
                    "E_v": E_v,
                    "E_a": E_a,
                    "p_55": p_55,
                    "likelihoods": likelihoods
                }
                f.write(json.dumps(res) + "\n")
                
    print(f"Saved patching results to {out_file}")

if __name__ == "__main__":
    main()
