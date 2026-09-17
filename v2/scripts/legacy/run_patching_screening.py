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
    parser.add_argument("--layers", type=int, nargs='+', default=list(range(10, 28)))
    parser.add_argument("--components", type=str, nargs='+', default=["res", "attn", "mlp"])
    parser.add_argument("--data-dir", type=str, default="v2/data/processed/aipsy_annotated")
    parser.add_argument("--out-dir", type=str, default="v2/results/derived/phase5_screening")
    parser.add_argument("--limit", type=int, default=50) # Use 50 for fast screening, full for actual run
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    test_df = pd.read_csv(os.path.join(args.data_dir, "test_strict.csv"))
    if args.limit > 0:
        test_df = test_df.head(args.limit)
        
    base_states = dict(np.load(args.base_states, allow_pickle=True))
    for k, v in base_states.items():
        if v.shape == (): base_states[k] = v.item()
        
    # Baseline expected values (unpatched Instruct)
    baseline_df = pd.DataFrame([json.loads(line) for line in open("v2/results/raw/aipsy/Qwen_Qwen2.5-1.5B-Instruct_standard_results.jsonl", 'r')])
    baseline_dict = baseline_df.set_index('id').to_dict('index')
    
    print(f"Loading {args.target_model} for Screening...")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    model = AutoModelForCausalLM.from_pretrained(args.target_model, torch_dtype=torch.float16, device_map="auto")
    
    candidates, va_pairs = generate_81_candidates()
    summary_results = []
    
    for l in args.layers:
        for comp in args.components:
            print(f"--- Patching Layer {l} Component {comp} ---")
            
            delta_v_list, delta_a_list, entropy_list, p55_list = [], [], [], []
            
            for idx, row in test_df.iterrows():
                uid = row['id']
                key = f"{uid}_layer{l}_{comp}"
                
                if key not in base_states or uid not in baseline_dict: continue
                
                base_val = torch.tensor(base_states[key]).to(model.device)
                
                prompt = apply_chat_template(tokenizer, row['text'])
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
                prompt_length = prompt_ids.shape[1]
                target_pos = prompt_length - 1
                
                if comp == "attn":
                    target_module = model.model.layers[l].self_attn
                elif comp == "mlp":
                    target_module = model.model.layers[l].mlp
                elif comp == "res":
                    target_module = model.model.layers[l]
                else:
                    continue
                    
                hook_handle = target_module.register_forward_hook(get_patch_hook(base_val, target_pos))
                
                likelihoods, _ = compute_likelihoods_for_candidates(model, tokenizer, prompt, candidates)
                hook_handle.remove()
                
                if len(likelihoods) == 81:
                    E_v, E_a, entropy, p_55, _ = compute_expected_va(likelihoods, va_pairs)
                    
                    orig_v = baseline_dict[uid].get('E_v', np.nan)
                    orig_a = baseline_dict[uid].get('E_a', np.nan)
                    
                    if not np.isnan(orig_v) and not np.isnan(orig_a):
                        delta_v_list.append(E_v - orig_v)
                        delta_a_list.append(E_a - orig_a)
                        entropy_list.append(entropy)
                        p55_list.append(p_55)
            
            if len(delta_v_list) > 0:
                summary_results.append({
                    "layer": l,
                    "component": comp,
                    "mean_delta_v": np.mean(delta_v_list),
                    "mean_delta_a": np.mean(delta_a_list),
                    "mean_entropy": np.mean(entropy_list),
                    "mean_p55": np.mean(p55_list)
                })
                
    summary_df = pd.DataFrame(summary_results)
    out_csv = os.path.join(args.out_dir, "patching_screening_summary.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"Saved screening summary to {out_csv}")

if __name__ == "__main__":
    main()
