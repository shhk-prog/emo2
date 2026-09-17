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

def get_lambda_patch_hook(base_val_tensor, target_pos, lambd):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            if target_pos < h.shape[1]:
                h_inst = h[0, target_pos, :]
                h_base = base_val_tensor.to(h.dtype)
                h[0, target_pos, :] = (1 - lambd) * h_inst + lambd * h_base
            return (h,) + output[1:]
        else:
            if target_pos < output.shape[1]:
                h_inst = output[0, target_pos, :]
                h_base = base_val_tensor.to(output.dtype)
                output[0, target_pos, :] = (1 - lambd) * h_inst + lambd * h_base
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
    parser.add_argument("--layers", type=int, nargs="+", default=[10, 14])
    parser.add_argument("--modules", type=str, nargs="+", default=["mlp", "attn"])
    parser.add_argument("--data-dir", type=str, default="v2/data/processed/aipsy")
    parser.add_argument("--out-dir", type=str, default="v2/results/derived/phase9_advanced_patching")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    test_df = pd.read_csv(os.path.join(args.data_dir, "test_strict.csv")).head(args.limit)
    base_states = dict(np.load(args.base_states, allow_pickle=True))
    
    print(f"Loading {args.target_model} for Lambda Dose-Response Patching...")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    model = AutoModelForCausalLM.from_pretrained(args.target_model, torch_dtype=torch.float16, device_map="auto")
    
    candidates, va_pairs = generate_81_candidates()
    lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for layer, module_name in zip(args.layers, args.modules):
        out_file = os.path.join(args.out_dir, f"lambda_patching_layer{layer}_{module_name}.jsonl")
        print(f"Processing layer {layer} {module_name}...")
        
        with open(out_file, 'w') as f:
            for idx, row in test_df.iterrows():
                uid = row['id']
                key = f"{uid}_layer{layer}_{module_name}"
                
                if key not in base_states:
                    continue
                    
                base_val = torch.tensor(base_states[key]).to(model.device)
                prompt = apply_chat_template(tokenizer, row['text'])
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
                target_pos = prompt_ids.shape[1] - 1
                
                target_module = model.model.layers[layer].self_attn if module_name == "attn" else model.model.layers[layer].mlp
                
                for lambd in lambdas:
                    hook_handle = target_module.register_forward_hook(get_lambda_patch_hook(base_val, target_pos, lambd))
                    likelihoods, _ = compute_likelihoods_for_candidates(model, tokenizer, prompt, candidates)
                    hook_handle.remove()
                    
                    if len(likelihoods) == 81:
                        E_v, E_a, _, _, _ = compute_expected_va(likelihoods, va_pairs)
                        res = {
                            "id": uid,
                            "layer": layer,
                            "module": module_name,
                            "lambda": lambd,
                            "E_v": E_v,
                            "E_a": E_a
                        }
                        f.write(json.dumps(res) + "\n")
                        f.flush()
                        
    print(f"Finished. Saved to {args.out_dir}")

if __name__ == "__main__":
    main()
