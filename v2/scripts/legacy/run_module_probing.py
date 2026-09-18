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
from v2.scripts.run_steering_and_likelihood import compute_contrastive_directions

def get_module_hook(module_name, layer_idx, hidden_states_dict, uid, prompt_length):
    def hook(module, inputs, output):
        # inputs is a tuple, output is usually a tuple or tensor
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
            
        # h shape: (batch, seq_len, hidden_dim)
        # 最後のプロンプトトークンの出力を記録
        target_pos = prompt_length - 1
        if target_pos < h.shape[1]:
            val = h[0, target_pos, :].cpu().numpy()
            hidden_states_dict[(uid, layer_idx, module_name)] = val
    return hook

def apply_chat_template(tokenizer, text, is_instruct):
    if is_instruct:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Read the following text and report your affective state.\n\nText: {text}\n\nRespond strictly in JSON format with 'valence' and 'arousal' keys (1-9)."}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"Read the following text and report your affective state.\n\nText: {text}\n\nRespond strictly in JSON format with 'valence' and 'arousal' keys (1-9).\n\nOutput:\n"
    return prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--is_instruct", action="store_true")
    parser.add_argument("--limit", type=int, default=50) # Dry run or small batch by default
    parser.add_argument("--data-dir", type=str, default="v2/data/processed/aipsy")
    parser.add_argument("--out-dir", type=str, default="v2/results/derived/phase4_circuit")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    test_df = pd.read_csv(os.path.join(args.data_dir, "test_strict.csv"))
    if args.limit > 0:
        test_df = test_df.head(args.limit)
        
    print(f"Loading {args.model} for Module Probing...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
    
    hidden_states_dict = {}
    
    for idx, row in test_df.iterrows():
        prompt = apply_chat_template(tokenizer, row['text'], args.is_instruct)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
        prompt_length = prompt_ids.shape[1]
        
        hooks = []
        for l in range(model.config.num_hidden_layers):
            # Hook Attention
            h1 = model.model.layers[l].self_attn.register_forward_hook(
                get_module_hook("attn", l, hidden_states_dict, row['id'], prompt_length)
            )
            # Hook MLP
            h2 = model.model.layers[l].mlp.register_forward_hook(
                get_module_hook("mlp", l, hidden_states_dict, row['id'], prompt_length)
            )
            # Hook Residual Stream (Output of the entire layer)
            h3 = model.model.layers[l].register_forward_hook(
                get_module_hook("res", l, hidden_states_dict, row['id'], prompt_length)
            )
            hooks.extend([h1, h2, h3])
            
        with torch.no_grad():
            _ = model(prompt_ids)
            
        for h in hooks:
            h.remove()
            
    # Save the module outputs
    model_name_clean = args.model.replace("/", "_")
    out_file = os.path.join(args.out_dir, f"{model_name_clean}_module_states.npz")
    
    save_dict = {f"{k[0]}_layer{k[1]}_{k[2]}": v for k, v in hidden_states_dict.items()}
    np.savez_compressed(out_file, **save_dict)
    
    print(f"Done. Module states saved to {out_file}")
    print("These states can be dot-producted with d_V to see which module (Attn/MLP) reduces the emotional projection.")

if __name__ == "__main__":
    main()
