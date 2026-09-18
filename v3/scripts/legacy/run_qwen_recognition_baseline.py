#!/usr/bin/env python3
"""
Evaluate Qwen2.5-1.5B-Instruct on:
1. EmoBank: Recognition VA and Self-Report (Post) VA.
2. AIPsy-Affect: Recognition VA and Self-Report (Post) VA across conditions.

Computes both:
- Greedy output text (JSON parsed)
- Continuous Sequence-Likelihood expected values E[V], E[A] across 81 candidates
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import re
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.stats import pearsonr
from transformers import AutoModelForCausalLM, AutoTokenizer

from v2.src.likelihood import generate_81_candidates, compute_expected_va

def parse_json_va(text):
    """Try parsing {"valence": V, "arousal": A} from generated text."""
    try:
        # Find JSON substring
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            v = int(data.get("valence", data.get("Valence", 5)))
            a = int(data.get("arousal", data.get("Arousal", 5)))
            return v, a, True
    except Exception:
        pass
    return 5, 5, False

def compute_likelihoods_batched(model, tokenizer, prompt, candidates, device="cuda"):
    """
    Batched computation of log-likelihoods using canonical joint tokenization.
    """
    from affective_empathy_eval.likelihood import compute_sequence_likelihoods_for_candidates
    log_ll, _ = compute_sequence_likelihoods_for_candidates(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        candidates=candidates,
        device=device,
        batch_size=81,
        normalize_length=False,
    )
    return log_ll.tolist()


def evaluate_emobank(model, tokenizer, device, candidates, va_pairs, stimuli_path="v1/data/processed/stimuli.csv", limit=None):
    print("\n=======================================================")
    print("Evaluating Qwen2.5-1.5B-Instruct on EmoBank Benchmark")
    print("=======================================================")
    df = pd.read_csv(stimuli_path)
    if limit:
        df = df.head(limit)
    print(f"Loaded {len(df)} EmoBank stimuli.")
    
    rec_results = []
    post_results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="EmoBank"):
        text = row['text']
        human_v = row['V'] # 1-5
        human_a = row['A'] # 1-5
        human_v_scaled = row['V_scaled'] # [-1, 1]
        human_a_scaled = row['A_scaled'] # [-1, 1]
        
        # 1. Recognition Task Prompt
        rec_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Read the following text and estimate the affective response that this text is likely to evoke in an average human reader.\n\nText: {text}\n\nRespond strictly in JSON format with 'valence' and 'arousal' keys (1-9)."}
        ]
        rec_prompt = tokenizer.apply_chat_template(rec_messages, tokenize=False, add_generation_prompt=True)
        
        # 2. Post Self-report Task Prompt
        post_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Read the following text and report your affective state.\n\nText: {text}\n\nRespond strictly in JSON format with 'valence' and 'arousal' keys (1-9)."}
        ]
        post_prompt = tokenizer.apply_chat_template(post_messages, tokenize=False, add_generation_prompt=True)
        
        # Evaluate Recognition
        rec_ll = compute_likelihoods_batched(model, tokenizer, rec_prompt, candidates, device=device)
        rec_ev, rec_ea, rec_ent, rec_p55, _ = compute_expected_va(rec_ll, va_pairs)
        
        # Greedy generation for Recognition
        rec_inp = tokenizer(rec_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_out = model.generate(**rec_inp, max_new_tokens=20, do_sample=False)
        rec_gen_text = tokenizer.decode(gen_out[0][rec_inp.input_ids.shape[1]:], skip_special_tokens=True)
        rec_gv, rec_ga, rec_valid = parse_json_va(rec_gen_text)
        
        # Evaluate Post
        post_ll = compute_likelihoods_batched(model, tokenizer, post_prompt, candidates, device=device)
        post_ev, post_ea, post_ent, post_p55, _ = compute_expected_va(post_ll, va_pairs)
        
        # Greedy generation for Post
        post_inp = tokenizer(post_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_out_post = model.generate(**post_inp, max_new_tokens=20, do_sample=False)
        post_gen_text = tokenizer.decode(gen_out_post[0][post_inp.input_ids.shape[1]:], skip_special_tokens=True)
        post_gv, post_ga, post_valid = parse_json_va(post_gen_text)
        
        rec_results.append({
            "stimulus_id": row['stimulus_id'],
            "text": text,
            "human_v": human_v,
            "human_a": human_a,
            "human_v_scaled": human_v_scaled,
            "human_a_scaled": human_a_scaled,
            "rec_ev": rec_ev,
            "rec_ea": rec_ea,
            "rec_gv": rec_gv,
            "rec_ga": rec_ga,
            "rec_gen_text": rec_gen_text.strip(),
            "rec_p55": rec_p55,
            "post_ev": post_ev,
            "post_ea": post_ea,
            "post_gv": post_gv,
            "post_ga": post_ga,
            "post_gen_text": post_gen_text.strip(),
            "post_p55": post_p55
        })
        
    res_df = pd.DataFrame(rec_results)
    
    # Compute Correlations
    r_v_ev, p_v_ev = pearsonr(res_df['human_v'], res_df['rec_ev'])
    r_a_ev, p_a_ev = pearsonr(res_df['human_a'], res_df['rec_ea'])
    r_v_gv, p_v_gv = pearsonr(res_df['human_v'], res_df['rec_gv'])
    r_a_ga, p_a_ga = pearsonr(res_df['human_a'], res_df['rec_ga'])
    
    # Self-report Neutralization check
    post_exact_55_pct = ((res_df['post_gv'] == 5) & (res_df['post_ga'] == 5)).mean() * 100.0
    post_mean_p55 = res_df['post_p55'].mean() * 100.0
    
    print("\n--- EmoBank Results Summary ---")
    print(f"Sample Size: N = {len(res_df)}")
    print(f"Recognition Continuous (E[V], E[A]):")
    print(f"  Valence  r = {r_v_ev:.4f} (p = {p_v_ev:.2e})")
    print(f"  Arousal  r = {r_a_ev:.4f} (p = {p_a_ev:.2e})")
    print(f"Recognition Discrete Greedy (parsed):")
    print(f"  Valence  r = {r_v_gv:.4f} (p = {p_v_gv:.2e})")
    print(f"  Arousal  r = {r_a_ga:.4f} (p = {p_a_ga:.2e})")
    print(f"Post Self-Report:")
    print(f"  Greedy Exact (5, 5) Neutral Rate: {post_exact_55_pct:.2f}%")
    print(f"  Continuous Mean p(5, 5): {post_mean_p55:.2f}%")
    print(f"  Mean E_post[V]: {res_df['post_ev'].mean():.3f} (std: {res_df['post_ev'].std():.3f})")
    print(f"  Mean E_post[A]: {res_df['post_ea'].mean():.3f} (std: {res_df['post_ea'].std():.3f})")
    
    return res_df, {
        "r_v_ev": r_v_ev, "r_a_ev": r_a_ev,
        "r_v_gv": r_v_gv, "r_a_ga": r_a_ga,
        "post_exact_55_pct": post_exact_55_pct,
        "post_mean_p55": post_mean_p55
    }

def evaluate_aipsy(model, tokenizer, device, candidates, va_pairs, aipsy_dir="v2/data/processed/aipsy", limit=None):
    print("\n=======================================================")
    print("Evaluating Qwen2.5-1.5B-Instruct on AIPsy-Affect Cohort")
    print("=======================================================")
    dfs = []
    for split in ["train_strict", "dev_strict", "test_strict"]:
        p = os.path.join(aipsy_dir, f"{split}.csv")
        if os.path.exists(p):
            sub = pd.read_csv(p)
            sub['split'] = split
            dfs.append(sub)
    if not dfs:
        raise FileNotFoundError("AIPsy-Affect files not found.")
    df = pd.concat(dfs, ignore_index=True)
    if limit:
        df = df.head(limit)
    print(f"Loaded {len(df)} AIPsy-Affect stimuli.")
    
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="AIPsy-Affect"):
        text = row['text']
        cond = row['condition'] # affective vs neutral
        intensity = row.get('intensity', 'none')
        
        # 1. Recognition Prompt
        rec_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Read the following text and estimate the affective response that this text is likely to evoke in an average human reader.\n\nText: {text}\n\nRespond strictly in JSON format with 'valence' and 'arousal' keys (1-9)."}
        ]
        rec_prompt = tokenizer.apply_chat_template(rec_messages, tokenize=False, add_generation_prompt=True)
        
        # 2. Self-Report Prompt
        post_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Read the following text and report your affective state.\n\nText: {text}\n\nRespond strictly in JSON format with 'valence' and 'arousal' keys (1-9)."}
        ]
        post_prompt = tokenizer.apply_chat_template(post_messages, tokenize=False, add_generation_prompt=True)
        
        # Evaluate Recognition
        rec_ll = compute_likelihoods_batched(model, tokenizer, rec_prompt, candidates, device=device)
        rec_ev, rec_ea, rec_ent, rec_p55, _ = compute_expected_va(rec_ll, va_pairs)
        
        # Greedy Recognition
        rec_inp = tokenizer(rec_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_out = model.generate(**rec_inp, max_new_tokens=20, do_sample=False)
        rec_gen = tokenizer.decode(gen_out[0][rec_inp.input_ids.shape[1]:], skip_special_tokens=True)
        rec_gv, rec_ga, _ = parse_json_va(rec_gen)
        
        # Evaluate Post
        post_ll = compute_likelihoods_batched(model, tokenizer, post_prompt, candidates, device=device)
        post_ev, post_ea, post_ent, post_p55, _ = compute_expected_va(post_ll, va_pairs)
        
        # Greedy Post
        post_inp = tokenizer(post_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_out_post = model.generate(**post_inp, max_new_tokens=20, do_sample=False)
        post_gen = tokenizer.decode(gen_out_post[0][post_inp.input_ids.shape[1]:], skip_special_tokens=True)
        post_gv, post_ga, _ = parse_json_va(post_gen)
        
        results.append({
            "id": row['id'],
            "pair_id": row['pair_id'],
            "condition": cond,
            "intensity": intensity,
            "text": text,
            "rec_ev": rec_ev,
            "rec_ea": rec_ea,
            "rec_gv": rec_gv,
            "rec_ga": rec_ga,
            "rec_gen": rec_gen.strip(),
            "post_ev": post_ev,
            "post_ea": post_ea,
            "post_gv": post_gv,
            "post_ga": post_ga,
            "post_gen": post_gen.strip(),
            "post_p55": post_p55
        })
        
    res_df = pd.DataFrame(results)
    
    # Analysis by condition
    aff_df = res_df[res_df['condition'] == 'affective']
    neu_df = res_df[res_df['condition'] == 'neutral']
    peak_df = res_df[res_df['intensity'] == 'peak']
    
    print("\n--- AIPsy-Affect Results Summary ---")
    print(f"Total Samples: {len(res_df)} (Affective: {len(aff_df)}, Neutral: {len(neu_df)})")
    print("\n[Recognition E_rec[V]]")
    print(f"  Affective Peak:     mean = {peak_df['rec_ev'].mean():.3f} (std: {peak_df['rec_ev'].std():.3f})")
    print(f"  Affective All:      mean = {aff_df['rec_ev'].mean():.3f} (std: {aff_df['rec_ev'].std():.3f})")
    print(f"  Neutral:            mean = {neu_df['rec_ev'].mean():.3f} (std: {neu_df['rec_ev'].std():.3f})")
    diff_rec = abs(aff_df['rec_ev'].mean() - neu_df['rec_ev'].mean())
    print(f"  |Affective - Neutral| Recognition Difference: {diff_rec:.3f}")
    
    print("\n[Self-Report E_post[V]]")
    print(f"  Affective Peak:     mean = {peak_df['post_ev'].mean():.3f} (std: {peak_df['post_ev'].std():.3f})")
    print(f"  Affective All:      mean = {aff_df['post_ev'].mean():.3f} (std: {aff_df['post_ev'].std():.3f})")
    print(f"  Neutral:            mean = {neu_df['post_ev'].mean():.3f} (std: {neu_df['post_ev'].std():.3f})")
    diff_post = abs(aff_df['post_ev'].mean() - neu_df['post_ev'].mean())
    print(f"  |Affective - Neutral| Self-Report Difference:  {diff_post:.3f}")
    
    post_neutral_rate = ((res_df['post_gv'] == 5) & (res_df['post_ga'] == 5)).mean() * 100.0
    print(f"\nGreedy Self-Report Exact (5, 5) Neutral Rate: {post_neutral_rate:.2f}%")
    
    return res_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--emobank-path", type=str, default="v1/data/processed/stimuli.csv")
    parser.add_argument("--aipsy-dir", type=str, default="v2/data/processed/aipsy")
    parser.add_argument("--out-dir", type=str, default="v3/results/recognition_baseline")
    parser.add_argument("--limit-emobank", type=int, default=None)
    parser.add_argument("--limit-aipsy", type=int, default=None)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Loading Model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
    
    candidates, va_pairs = generate_81_candidates("standard")
    
    # 1. EmoBank
    emobank_df, emobank_summary = evaluate_emobank(
        model, tokenizer, device, candidates, va_pairs, 
        stimuli_path=args.emobank_path, limit=args.limit_emobank
    )
    emobank_df.to_csv(os.path.join(args.out_dir, "qwen_emobank_recognition_post.csv"), index=False)
    with open(os.path.join(args.out_dir, "qwen_emobank_summary.json"), "w") as f:
        json.dump(emobank_summary, f, indent=2)
        
    # 2. AIPsy-Affect
    aipsy_df = evaluate_aipsy(
        model, tokenizer, device, candidates, va_pairs, 
        aipsy_dir=args.aipsy_dir, limit=args.limit_aipsy
    )
    aipsy_df.to_csv(os.path.join(args.out_dir, "qwen_aipsy_recognition_post.csv"), index=False)
    
    print("\nAll evaluations complete! Results saved to:", args.out_dir)

if __name__ == "__main__":
    main()
