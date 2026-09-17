#!/usr/bin/env python3
import os
import json
import hashlib
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from v2.src.likelihood import generate_81_candidates, compute_likelihoods_for_candidates, compute_expected_va

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
    parser.add_argument("--template_type", type=str, default="standard")
    parser.add_argument("--limit", type=int, default=2) # default to small for dry-run
    parser.add_argument("--out_dir", type=str, default="v2/results/raw/aipsy")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load dataset
    splits = ["train_strict", "dev_strict", "test_strict"]
    dfs = []
    for split in splits:
        path = f"v2/data/processed/aipsy/{split}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['split'] = split.replace("_strict", "")
            dfs.append(df)
            
    if not dfs:
        print("No data found!")
        return
        
    full_df = pd.concat(dfs, ignore_index=True)
    if args.limit > 0:
        full_df = full_df.head(args.limit)
        
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
    
    candidates, va_pairs = generate_81_candidates(args.template_type)
    
    results = []
    hidden_states_dict = {} # (id, layer) -> np.array
    
    for idx, row in full_df.iterrows():
        prompt = apply_chat_template(tokenizer, row['text'], args.is_instruct)
        prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:8]
        
        # 1. 尤度計算
        likelihoods, candidate_lengths = compute_likelihoods_for_candidates(
            model, tokenizer, prompt, candidates, device=model.device
        )
        
        E_v, E_a, entropy, p_55, probs = compute_expected_va(likelihoods, va_pairs)
        
        # 2. Hidden states抽出 (プローブ用)
        # プロンプトの最後のトークンの出力を抽出する
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(prompt_ids, output_hidden_states=True)
            # outputs.hidden_states is a tuple of (num_layers + 1) tensors of shape (batch, seq, dim)
            # We want the last token: [0, -1, :]
            for layer_idx, h in enumerate(outputs.hidden_states):
                hidden_states_dict[(row['id'], layer_idx)] = h[0, -1, :].cpu().numpy()
                
        res = {
            "id": row['id'],
            "pair_id": row['pair_id'],
            "condition": row['condition'],
            "intensity": row['intensity'],
            "split": row['split'],
            "model": args.model,
            "is_instruct": args.is_instruct,
            "template_type": args.template_type,
            "prompt_hash": prompt_hash,
            "E_v": float(E_v),
            "E_a": float(E_a),
            "entropy": float(entropy),
            "p_55": float(p_55),
            "likelihoods": likelihoods
        }
        results.append(res)
        print(f"Processed {row['id']} ({row['intensity']}): E_V={E_v:.2f}, E_A={E_a:.2f}")
        
    # Save results
    model_name_clean = args.model.replace("/", "_")
    out_file = os.path.join(args.out_dir, f"{model_name_clean}_{args.template_type}_results.jsonl")
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    # Save hidden states
    hs_file = os.path.join(args.out_dir, f"{model_name_clean}_hidden_states.npz")
    np.savez_compressed(hs_file, **{f"{k[0]}_layer{k[1]}": v for k, v in hidden_states_dict.items()})
    
    print(f"Done. Saved to {out_file} and {hs_file}")

if __name__ == "__main__":
    main()
