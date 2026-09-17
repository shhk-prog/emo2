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
from scipy.stats import wasserstein_distance, entropy
from scipy.spatial.distance import jensenshannon
try:
    import ot
    HAS_POT = True
except ImportError:
    HAS_POT = False
from tqdm import tqdm
from v2.src.likelihood import generate_81_candidates

# Generate candidates
candidates, va_pairs = generate_81_candidates()
CANDIDATES = [{"text": c, "valence": v, "arousal": a} for c, (v, a) in zip(candidates, va_pairs)]

# Precompute Distance Matrix for 2D EMD
v_vals_np = np.array([c['valence'] for c in CANDIDATES], dtype=float)
a_vals_np = np.array([c['arousal'] for c in CANDIDATES], dtype=float)
coords = np.column_stack((v_vals_np, a_vals_np))
if HAS_POT:
    M_2d = ot.dist(coords, coords, metric='euclidean')
else:
    M_2d = None

def apply_chat_template(tokenizer, text):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Read the following text and report your affective state.\n\nText: {text}\n\nRespond strictly in JSON format with 'valence' and 'arousal' keys (1-9)."}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def get_logits(res, norm, head):
    return head(norm(res))

def compute_cand_score(logits, input_ids, cand_start_idx, tau=1.0):
    # logits shape: (1, seq_len, vocab_size)
    # input_ids shape: (seq_len,)
    # we want the log prob of input_ids[t] given logits at t-1
    log_probs = F.log_softmax(logits[0] / tau, dim=-1)
    
    cand_len = len(input_ids) - cand_start_idx
    score = 0.0
    for i in range(cand_start_idx, len(input_ids)):
        token_id = input_ids[i]
        score += log_probs[i-1, token_id].item()
    return score / cand_len

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
    inst_model = AutoModelForCausalLM.from_pretrained(args.target_model, torch_dtype=torch.bfloat16, device_map="auto")

    print(f"Loading Base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="auto")
    
    conditions = [
        ("BBB", "base", "base", "base"),
        ("III", "inst", "inst", "inst"),
        ("BBI", "base", "base", "inst"),
        ("IIB", "inst", "inst", "base"),
        ("BIB", "base", "inst", "base"),
        ("BII", "base", "inst", "inst"),
        ("IBB", "inst", "base", "base"),
        ("IBI", "inst", "base", "inst")
    ]
    
    results = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        prompt = apply_chat_template(tokenizer, row['text'])
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        
        # We need to collect scores for all 81 candidates for each condition
        cond_scores = {c[0]: [] for c in conditions}
        
        for cand_dict in CANDIDATES:
            cand = cand_dict['text']
            cand_ids = tokenizer.encode(cand, add_special_tokens=False)
            input_ids = prompt_ids + cand_ids
            input_tensor = torch.tensor([input_ids], device=base_model.device)
            cand_start_idx = len(prompt_ids)
            
            with torch.no_grad():
                base_outputs = base_model(input_tensor, output_hidden_states=True)
                inst_outputs = inst_model(input_tensor, output_hidden_states=True)
                
            base_res = base_outputs.hidden_states[-1]
            inst_res = inst_outputs.hidden_states[-1]
            
            # Compute logits for each condition
            for c_name, res_src, norm_src, head_src in conditions:
                res = base_res if res_src == "base" else inst_res
                norm = base_model.model.norm if norm_src == "base" else inst_model.model.norm
                head = base_model.lm_head if head_src == "base" else inst_model.lm_head
                
                logits = get_logits(res, norm, head)
                score = compute_cand_score(logits, input_ids, cand_start_idx, tau=1.0)
                cond_scores[c_name].append(score)
        
        # Now compute metrics for each condition
        row_res = {"id": row['id']}
        
        # Base distribution for WD reference
        base_probs = np.exp(np.array(cond_scores["BBB"]))
        base_probs = base_probs / np.sum(base_probs)
        
        for c_name in cond_scores:
            scores = np.array(cond_scores[c_name])
            probs = np.exp(scores)
            probs = probs / np.sum(probs)
            
            E_V = sum(probs[i] * CANDIDATES[i]['valence'] for i in range(81))
            E_A = sum(probs[i] * CANDIDATES[i]['arousal'] for i in range(81))
            
            idx_5_5 = [i for i, c in enumerate(CANDIDATES) if c['valence']==5 and c['arousal']==5][0]
            p_5_5 = probs[idx_5_5]
            
            ent = entropy(probs)
            
            # WD computation (1D on Valence)
            v_vals = [c['valence'] for c in CANDIDATES]
            wd = wasserstein_distance(v_vals, v_vals, u_weights=probs, v_weights=base_probs)
            
            # JSD computation (base is 2)
            jsd = jensenshannon(probs, base_probs) ** 2  # Jensen-Shannon Divergence
            
            # 2D EMD computation
            emd2d = ot.emd2(probs, base_probs, M_2d) if HAS_POT else 0.0
            
            row_res[f"{c_name}_Ev"] = E_V
            row_res[f"{c_name}_Ea"] = E_A
            row_res[f"{c_name}_p55"] = p_5_5
            row_res[f"{c_name}_ent"] = ent
            row_res[f"{c_name}_wd"] = wd
            row_res[f"{c_name}_jsd"] = jsd
            row_res[f"{c_name}_emd2d"] = emd2d
            
        results.append(row_res)
        
    df = pd.DataFrame(results)
    out_csv = os.path.join(args.out_dir, "unembedding_norm_swap_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved to {out_csv}")
    
    # Print Summary
    print("\n=== 8-Condition Unembedding Swap Summary ===")
    summary_cols = []
    for c_name, res_src, norm_src, head_src in conditions:
        mean_ev = df[f"{c_name}_Ev"].mean()
        mean_ea = df[f"{c_name}_Ea"].mean()
        mean_p55 = df[f"{c_name}_p55"].mean()
        mean_ent = df[f"{c_name}_ent"].mean()
        mean_wd = df[f"{c_name}_wd"].mean()
        mean_jsd = df[f"{c_name}_jsd"].mean()
        mean_emd2d = df[f"{c_name}_emd2d"].mean()
        summary_cols.append({
            "Condition": c_name,
            "Res": res_src,
            "Norm": norm_src,
            "Head": head_src,
            "E[V]": mean_ev,
            "E[A]": mean_ea,
            "WD to BBB": mean_wd,
            "JSD to BBB": mean_jsd,
            "2D EMD": mean_emd2d
        })
    sum_df = pd.DataFrame(summary_cols)
    try:
        print(sum_df.to_markdown(index=False))
    except:
        print(sum_df)

if __name__ == "__main__":
    main()
