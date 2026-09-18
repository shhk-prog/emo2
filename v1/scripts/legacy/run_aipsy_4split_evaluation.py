#!/usr/bin/env python3
"""
Evaluate LLMs on AIPsy-Affect 4-split stimuli (N=480) across:
  - Task 1: Reader-Response Prediction (R)
  - Task 2: Self-Report (S)
  - Task 3: Writer-State Estimation (W)

Computes expected values E[V], E[A], E[D] and greedy argmax predictions over
all 729 joint candidate triplets (V, A, D) in {1..9}^3.
"""

import os
import json
import argparse
import itertools
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def build_candidates():
    candidates = []
    triplets = []
    for v, a, d in itertools.product(range(1, 10), range(1, 10), range(1, 10)):
        candidates.append(f'{{"valence": {v}, "arousal": {a}, "dominance": {d}}}')
        triplets.append((v, a, d))
    return candidates, triplets

def construct_prompt(tokenizer, text, task_type, is_instruct=False):
    if task_type == "writer":
        instruction = "Read the following text and estimate the affective state of the writer who wrote it."
    elif task_type == "reader":
        instruction = "Read the following text and estimate the affective response that this text is likely to evoke in an average human reader."
    elif task_type == "self":
        instruction = "Read the following text and report your affective state."
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    user_content = (
        f"{instruction}\n\n"
        f"Text: {text}\n\n"
        f"Respond strictly in JSON format with 'valence', 'arousal', and 'dominance' keys (integers from 1 to 9)."
    )

    if is_instruct:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content}
        ]
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            messages = [{"role": "user", "content": user_content}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Base model continuation format
        prompt = (
            f"Task: Evaluate emotional Valence, Arousal, and Dominance (1-9).\n\n"
            f"{instruction}\n\n"
            f"Text: {text}\n\n"
            f"Output:\n"
        )
    return prompt

def compute_likelihoods_batched(model, tokenizer, prompt, candidates, device="cuda", sub_batch_size=243):
    """
    Computes likelihood of all candidates using Joint Tokenization of (prompt + candidate)
    and teacher-forced scoring on candidate token positions using canonical likelihood module.
    """
    from affective_empathy_eval.likelihood import compute_sequence_likelihoods_for_candidates
    _, probs = compute_sequence_likelihoods_for_candidates(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        candidates=candidates,
        device=device,
        batch_size=sub_batch_size,
        normalize_length=False,
    )
    return probs


def evaluate_aipsy_stimuli(model, tokenizer, device, candidates, vad_triplets,
                           stimuli_path="v1/data/processed/aipsy_4split_all.csv",
                           is_instruct=False, limit=0):
    df = pd.read_csv(stimuli_path)
    if limit > 0:
        df = df.head(limit)
        
    print(f"Starting AIPsy 4-Split evaluation on {len(df)} stimuli...")
    
    tasks = ["writer", "reader", "self"]
    results = []
    
    for idx, row in df.iterrows():
        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"[{idx+1}/{len(df)}] Evaluating stimulus id: {row['id']} (split: {row['split']}, emotion: {row.get('emotion')})")
            
        text = row["text"]
        row_dict = row.to_dict()
        
        for task in tasks:
            prompt = construct_prompt(tokenizer, text, task, is_instruct=is_instruct)
            probs = compute_likelihoods_batched(model, tokenizer, prompt, candidates, device=device)
            
            # Expected values
            ev = sum(p * t[0] for p, t in zip(probs, vad_triplets))
            ea = sum(p * t[1] for p, t in zip(probs, vad_triplets))
            ed = sum(p * t[2] for p, t in zip(probs, vad_triplets))
            
            # Greedy argmax
            argmax_idx = int(np.argmax(probs))
            best_v, best_a, best_d = vad_triplets[argmax_idx]
            
            # Probability of exact (5,5,5) neutral
            neutral_idx = vad_triplets.index((5, 5, 5))
            p555 = probs[neutral_idx]
            
            prefix = task[0] # 'w', 'r', 's'
            row_dict[f"{prefix}_ev"] = float(ev)
            row_dict[f"{prefix}_ea"] = float(ea)
            row_dict[f"{prefix}_ed"] = float(ed)
            row_dict[f"{prefix}_gv"] = int(best_v)
            row_dict[f"{prefix}_ga"] = int(best_a)
            row_dict[f"{prefix}_gd"] = int(best_d)
            row_dict[f"{prefix}_p555"] = float(p555)
            
        results.append(row_dict)
        
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on AIPsy-Affect 4-split dataset.")
    parser.add_argument("--model", type=str, required=True, help="Model path or HF name")
    parser.add_argument("--tag", type=str, required=True, help="Short tag for the model")
    parser.add_argument("--is-instruct", action="store_true", help="Whether to apply chat template")
    parser.add_argument("--stimuli-path", type=str, default="v1/data/processed/aipsy_4split_all.csv")
    parser.add_argument("--out-dir", type=str, default="v1/results/aipsy_4split_eval")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for dry-run")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Evaluating Model: {args.model} (Tag: {args.tag}, Instruct: {args.is_instruct})")
    print(f"Stimuli Path: {args.stimuli_path} (Device: {args.device})")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if args.device == "cuda" else None,
        trust_remote_code=True
    )
    model.eval()
    
    candidates, vad_triplets = build_candidates()
    print(f"Generated {len(candidates)} VAD candidate triplets in {{1..9}}^3.")
    
    res_df = evaluate_aipsy_stimuli(
        model, tokenizer, args.device, candidates, vad_triplets,
        stimuli_path=args.stimuli_path, is_instruct=args.is_instruct, limit=args.limit
    )
    
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"{args.tag}_aipsy_4split.csv")
    res_df.to_csv(out_csv, index=False)
    
    print("\n" + "=" * 60)
    print(f"SUCCESS: Saved {len(res_df)} evaluation results to {out_csv}")
    print("=" * 60)

if __name__ == "__main__":
    main()
