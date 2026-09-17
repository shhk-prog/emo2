#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import wasserstein_distance

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
    parser.add_argument("--out-dir", type=str, default="v2/results/derived/phase7_strict_path_patching")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    test_df = pd.read_csv(os.path.join(args.data_dir, "test_strict.csv"))
    if args.limit > 0:
        test_df = test_df.head(args.limit)
    test_df = test_df.reset_index(drop=True)
        
    print(f"Loading Instruct model: {args.target_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    inst_model = AutoModelForCausalLM.from_pretrained(args.target_model, torch_dtype=torch.float16, device_map="auto")

    print(f"Loading Base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16, device_map="auto")
    
    numeric_tokens = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 10)]
    
    results = []
    
    for idx, row in test_df.iterrows():
        prompt = apply_chat_template_val_forcing(tokenizer, row['text'])
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(inst_model.device)
        
        with torch.no_grad():
            base_outputs = base_model(prompt_ids, output_hidden_states=True)
            inst_outputs = inst_model(prompt_ids, output_hidden_states=True)
            
        base_res = base_outputs.hidden_states[-1][:, -1, :] # (1, hidden_size)
        inst_res = inst_outputs.hidden_states[-1][:, -1, :]
        
        # 1. Base Res -> Base Unembedding
        base_normed = base_model.model.norm(base_res)
        base_logits = base_model.lm_head(base_normed)[0, numeric_tokens]
        base_probs = F.softmax(base_logits, dim=-1)
        base_Ev = torch.sum(base_probs * torch.arange(1, 10, device=base_probs.device)).item()
        
        # 2. Inst Res -> Inst Unembedding
        inst_normed = inst_model.model.norm(inst_res)
        inst_logits = inst_model.lm_head(inst_normed)[0, numeric_tokens]
        inst_probs = F.softmax(inst_logits, dim=-1)
        inst_Ev = torch.sum(inst_probs * torch.arange(1, 10, device=inst_probs.device)).item()
        
        # 3. Base Res -> Inst Unembedding (Swap A)
        swapA_normed = inst_model.model.norm(base_res)
        swapA_logits = inst_model.lm_head(swapA_normed)[0, numeric_tokens]
        swapA_probs = F.softmax(swapA_logits, dim=-1)
        swapA_Ev = torch.sum(swapA_probs * torch.arange(1, 10, device=swapA_probs.device)).item()
        
        # 4. Inst Res -> Base Unembedding (Swap B)
        swapB_normed = base_model.model.norm(inst_res)
        swapB_logits = base_model.lm_head(swapB_normed)[0, numeric_tokens]
        swapB_probs = F.softmax(swapB_logits, dim=-1)
        swapB_Ev = torch.sum(swapB_probs * torch.arange(1, 10, device=swapB_probs.device)).item()
        
        results.append({
            "id": row['id'],
            "base_Ev": base_Ev,
            "inst_Ev": inst_Ev,
            "swapA_Ev": swapA_Ev, # Base Res -> Inst Unembedding
            "swapB_Ev": swapB_Ev  # Inst Res -> Base Unembedding
        })
        
    df = pd.DataFrame(results)
    out_csv = os.path.join(args.out_dir, "unembedding_swap_results.csv")
    df.to_csv(out_csv, index=False)
    
    print("\n=== Unembedding Swap Summary ===")
    print(df[["base_Ev", "inst_Ev", "swapA_Ev", "swapB_Ev"]].mean().to_markdown())
    print(f"\nSaved to {out_csv}")

if __name__ == "__main__":
    main()
