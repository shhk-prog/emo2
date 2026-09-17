#!/usr/bin/env python3
import os
import json
import hashlib
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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

def compute_contrastive_directions(hidden_states, df_train, layer):
    pos_emotions = ['ecstasy', 'admiration', 'amazement']
    neg_emotions = ['rage', 'grief', 'terror', 'loathing']
    high_a_emotions = ['rage', 'terror', 'ecstasy', 'amazement']
    low_a_emotions = ['grief', 'loathing']

    pos_reps, neg_reps = [], []
    high_a_reps, low_a_reps = [], []

    for idx, row in df_train.iterrows():
        uid = row['id']
        key = f"{uid}_layer{layer}"
        if key not in hidden_states:
            continue
        
        rep = hidden_states[key]
        emo = str(row.get('emotion', '')).lower()
        cond = str(row.get('condition', '')).lower()
        
        if emo in pos_emotions: pos_reps.append(rep)
        elif emo in neg_emotions: neg_reps.append(rep)
            
        if emo in high_a_emotions: high_a_reps.append(rep)
        elif emo in low_a_emotions or cond == 'neutral': low_a_reps.append(rep)

    if len(pos_reps) > 0 and len(neg_reps) > 0:
        d_V = np.mean(pos_reps, axis=0) - np.mean(neg_reps, axis=0)
        n = np.linalg.norm(d_V)
        if n > 0: d_V = d_V / n
    else:
        d_V = None
        
    if len(high_a_reps) > 0 and len(low_a_reps) > 0:
        d_A = np.mean(high_a_reps, axis=0) - np.mean(low_a_reps, axis=0)
        n = np.linalg.norm(d_A)
        if n > 0: d_A = d_A / n
    else:
        d_A = None
        
    # Sign Fix and Sigma Computation
    z_V_list = []
    z_A_list = []
    v_h_list = []
    a_h_list = []
    
    for idx, row in df_train.iterrows():
        uid = row['id']
        key = f"{uid}_layer{layer}"
        if key not in hidden_states: continue
        rep = hidden_states[key]
        
        if pd.notna(row.get('V_H')) and pd.notna(row.get('A_H')):
            if d_V is not None: 
                z_V_list.append(np.dot(rep, d_V))
                v_h_list.append(row['V_H'])
            if d_A is not None: 
                z_A_list.append(np.dot(rep, d_A))
                a_h_list.append(row['A_H'])
                
    if d_V is not None and len(z_V_list) > 1:
        from scipy.stats import pearsonr
        r, _ = pearsonr(z_V_list, v_h_list)
        if r < 0:
            d_V = -d_V
            z_V_list = [-z for z in z_V_list]
            
    if d_A is not None and len(z_A_list) > 1:
        from scipy.stats import pearsonr
        r, _ = pearsonr(z_A_list, a_h_list)
        if r < 0:
            d_A = -d_A
            z_A_list = [-z for z in z_A_list]
        
    # Compute std (sigma) of projection scores on train set
    z_V_list = []
    z_A_list = []
    for idx, row in df_train.iterrows():
        uid = row['id']
        key = f"{uid}_layer{layer}"
        if key not in hidden_states: continue
        rep = hidden_states[key]
        if d_V is not None: z_V_list.append(np.dot(rep, d_V))
        if d_A is not None: z_A_list.append(np.dot(rep, d_A))
        
    sigma_V = np.std(z_V_list) if len(z_V_list) > 0 else 1.0
    sigma_A = np.std(z_A_list) if len(z_A_list) > 0 else 1.0
        
    return d_V, d_A, sigma_V, sigma_A

def get_steering_hook(direction, alpha, sigma, prompt_length):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
            
        # h is (batch, seq_len, hidden_size)
        # alpha is in units of sigma. We add alpha * sigma * (direction / ||direction||)
        intervention = alpha * sigma * (direction / np.linalg.norm(direction))
        intervention_tensor = torch.tensor(intervention, dtype=h.dtype, device=h.device)
        
        target_pos = prompt_length - 1
        if target_pos < h.shape[1]:
            h[:, target_pos:, :] += intervention_tensor
            
        if isinstance(output, tuple):
            return (h,) + output[1:]
        else:
            return h
    return hook

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--is_instruct", action="store_true")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--layers", type=int, nargs='+', default=[20, 24, 27])
    parser.add_argument("--alphas", type=float, nargs='+', default=[-3.0, -1.5, 0.0, 1.5, 3.0])
    parser.add_argument("--direction", type=str, choices=["valence", "arousal", "random"], default="valence")
    parser.add_argument("--data-dir", type=str, default="v2/data/processed/aipsy_annotated")
    parser.add_argument("--results-dir", type=str, default="v2/results/raw/aipsy")
    parser.add_argument("--out-dir", type=str, default="v2/results/derived/phase2_steering")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load dataset
    train_df = pd.read_csv(os.path.join(args.data_dir, "train_strict.csv"))
    test_df = pd.read_csv(os.path.join(args.data_dir, "test_strict.csv"))
    
    if args.limit > 0:
        test_df = test_df.head(args.limit)
        
    print(f"Loading hidden states for Contrastive Direction computation...")
    model_name_clean = args.model.replace("/", "_")
    hs_file = os.path.join(args.results_dir, f"{model_name_clean}_hidden_states.npz")
    if not os.path.exists(hs_file):
        print(f"Hidden states file not found: {hs_file}")
        return
        
    hidden_states = dict(np.load(hs_file, allow_pickle=True))
    for k, v in hidden_states.items():
        if v.shape == (): hidden_states[k] = v.item()
        
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
    
    candidates, va_pairs = generate_81_candidates("standard")
    
    out_file = os.path.join(args.out_dir, f"{model_name_clean}_steering_{args.direction}_results.jsonl")
    f_out = open(out_file, "w")
    
    for l in args.layers:
        print(f"\n--- Processing Layer {l} ---")
        d_V, d_A, sigma_V, sigma_A = compute_contrastive_directions(hidden_states, train_df, l)
        
        if args.direction == "valence":
            target_d = d_V
            target_sigma = sigma_V
        elif args.direction == "arousal":
            target_d = d_A
            target_sigma = sigma_A
        elif args.direction == "random":
            # Norm-matched random direction
            np.random.seed(42 + l)
            v = np.random.randn(len(d_V)) if d_V is not None else np.random.randn(hidden_states[list(hidden_states.keys())[0]].shape[-1])
            target_d = v / np.linalg.norm(v)
            target_sigma = sigma_V if d_V is not None else 1.0
            
        if target_d is None:
            print(f"Skipping layer {l} because direction could not be computed.")
            continue
            
        for alpha in args.alphas:
            print(f"  Evaluating alpha={alpha}")
            
            for idx, row in test_df.iterrows():
                prompt = apply_chat_template(tokenizer, row['text'], args.is_instruct)
                prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:8]
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                prompt_length = len(prompt_ids)
                
                # Register steering hook
                hook_handle = model.model.layers[l].register_forward_hook(
                    get_steering_hook(target_d, alpha, target_sigma, prompt_length)
                )
                
                # Compute likelihoods
                likelihoods, _ = compute_likelihoods_for_candidates(
                    model, tokenizer, prompt, candidates, device=model.device
                )
                
                # Remove hook
                hook_handle.remove()
                
                E_v, E_a, entropy, p_55, probs = compute_expected_va(likelihoods, va_pairs)
                
                res = {
                    "id": row['id'],
                    "pair_id": row['pair_id'],
                    "condition": row['condition'],
                    "intensity": row['intensity'],
                    "model": args.model,
                    "model_revision": getattr(model.config, "_commit_hash", "unknown"),
                    "tokenizer_name": tokenizer.name_or_path,
                    "prompt_hash": prompt_hash,
                    "seed": 42, # Fixed seed if we used one for sampling, though likelihood is deterministic
                    "temperature": 1.0,
                    "intervention_layer": l,
                    "intervention_alpha": alpha,
                    "intervention_direction": args.direction,
                    "E_v": float(E_v),
                    "E_a": float(E_a),
                    "entropy": float(entropy),
                    "p_55": float(p_55),
                }
                
                f_out.write(json.dumps(res) + "\n")
                
    f_out.close()
    print(f"\nDone. Results saved to {out_file}")

if __name__ == "__main__":
    main()
