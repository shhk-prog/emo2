#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import json
import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from v2.src.likelihood import generate_81_candidates, compute_expected_va

def apply_template_variant(tokenizer, text, variant="json_reverse"):
    if variant == "json_reverse":
        prompt = f"Read the following text and report your affective state.\n\nText: {text}\n\nRespond strictly in JSON format with 'arousal' and 'valence' keys (1-9).\n\nOutput:\n{{"
    elif variant == "natural_language":
        prompt = f"Read the following text and report your affective state.\n\nText: {text}\n\nReport your state in a short sentence like: My valence is X and arousal is Y, where X and Y are numbers from 1 to 9.\n\nOutput:\nMy valence is"
    else:
        prompt = f"Text: {text}\nOutput:\n"
    return prompt

def generate_natural_candidates():
    candidates = []
    for v in range(1, 10):
        for a in range(1, 10):
            # Matches: "My valence is X and arousal is Y"
            candidates.append((v, a, f" {v} and arousal is {a}"))
    return candidates

def compute_variant_likelihoods(model, tokenizer, prompt, candidates, variant="json_reverse"):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    likelihoods = []
    
    with torch.no_grad():
        for v, a, cand_text in candidates:
            cand_ids = tokenizer.encode(cand_text, add_special_tokens=False, return_tensors="pt").to(model.device)
            input_ids = torch.cat([prompt_ids, cand_ids], dim=-1)
            
            outputs = model(input_ids)
            logits = outputs.logits[0]
            
            start_idx = prompt_ids.shape[1] - 1
            log_prob_sum = 0.0
            
            for i in range(cand_ids.shape[1]):
                token_logit = logits[start_idx + i]
                log_probs = torch.nn.functional.log_softmax(token_logit, dim=-1)
                token_id = cand_ids[0, i]
                log_prob_sum += log_probs[token_id].item()
                
            likelihoods.append(log_prob_sum)
            
    return likelihoods

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--variant", type=str, choices=["json_reverse", "natural_language"], default="natural_language")
    parser.add_argument("--data-dir", type=str, default="v2/data/processed/aipsy")
    parser.add_argument("--out-dir", type=str, default="v2/results/derived/phase3b_stress_test")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    test_df = pd.read_csv(os.path.join(args.data_dir, "test_strict.csv")).head(args.limit)
    
    print(f"Loading {args.model} for Output-Gating Test ({args.variant})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
    
    if args.variant == "json_reverse":
        # Simplified for mock
        cands = []
        for v in range(1, 10):
            for a in range(1, 10):
                cands.append((v, a, f'\n  "arousal": {a},\n  "valence": {v}\n}}'))
    else:
        cands = generate_natural_candidates()
        
    out_file = os.path.join(args.out_dir, f"output_gating_{args.variant}.jsonl")
    
    with open(out_file, 'w') as f:
        for idx, row in test_df.iterrows():
            prompt = apply_template_variant(tokenizer, row['text'], args.variant)
            likelihoods = compute_variant_likelihoods(model, tokenizer, prompt, cands, args.variant)
            
            va_pairs = [(c[0], c[1]) for c in cands]
            E_v, E_a, entropy, p_55, _ = compute_expected_va(likelihoods, va_pairs)
            
            res = {
                "id": row['id'],
                "variant": args.variant,
                "E_v": E_v,
                "E_a": E_a,
                "p_55": p_55
            }
            f.write(json.dumps(res) + "\n")
            
    print(f"Saved output gating results to {out_file}")

if __name__ == "__main__":
    main()
