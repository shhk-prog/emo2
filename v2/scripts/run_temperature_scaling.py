#!/usr/bin/env python3
import os
import json
import argparse
import pandas as pd
import numpy as np

def compute_expected_va_with_temp(likelihoods, va_pairs, tau=1.0):
    l_arr = np.array(likelihoods) / tau
    l_max = np.max(l_arr)
    probs = np.exp(l_arr - l_max)
    probs = probs / np.sum(probs)
    
    E_v = 0.0
    E_a = 0.0
    p_55 = 0.0
    
    for p, (v, a) in zip(probs, va_pairs):
        E_v += p * v
        E_a += p * a
        if v == 5 and a == 5:
            p_55 = p
            
    return E_v, E_a, p_55

def generate_va_pairs():
    pairs = []
    for v in range(1, 10):
        for a in range(1, 10):
            pairs.append((v, a))
    return pairs

def process_file(in_file, model_name, is_instruct, taus):
    va_pairs = generate_va_pairs()
    results = []
    
    with open(in_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if 'likelihoods' not in data:
                continue
                
            likelihoods = data['likelihoods']
            
            for tau in taus:
                E_v, E_a, p_55 = compute_expected_va_with_temp(likelihoods, va_pairs, tau)
                results.append({
                    "id": data['id'],
                    "condition": data['condition'],
                    "intensity": data['intensity'],
                    "model": model_name,
                    "is_instruct": is_instruct,
                    "temperature": tau,
                    "E_v": E_v,
                    "E_a": E_a,
                    "p_55": p_55
                })
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-file", type=str, default="v2/results/raw/aipsy/Qwen_Qwen2.5-1.5B_standard_results.jsonl")
    parser.add_argument("--instruct-file", type=str, default="v2/results/raw/aipsy/Qwen_Qwen2.5-1.5B-Instruct_standard_results.jsonl")
    parser.add_argument("--out-dir", type=str, default="v2/results/derived/phase3b_stress_test")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    taus = [0.2, 0.5, 1.0, 1.5, 2.0, 5.0]
    all_results = []
    
    if os.path.exists(args.base_file):
        all_results.extend(process_file(args.base_file, "Base", False, taus))
    
    if os.path.exists(args.instruct_file):
        all_results.extend(process_file(args.instruct_file, "Instruct", True, taus))
        
    df = pd.DataFrame(all_results)
    out_file = os.path.join(args.out_dir, "temperature_scaling_results.csv")
    df.to_csv(out_file, index=False)
    print(f"Saved temperature scaling results to {out_file}")

if __name__ == "__main__":
    main()
