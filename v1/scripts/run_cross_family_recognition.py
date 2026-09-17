#!/usr/bin/env python3
"""
Evaluate LLMs (Base and Instruct across Qwen, Llama, Gemma, Mistral) on:
1. EmoBank Benchmark (N=321): Human ground truth VA correlation.
2. AIPsy-Affect Cohort (N=144): Affective vs Neutral discrimination and Self-Report Neutralization.

Computes:
- Recognition VA (average human reader affective response estimation)
- Post Self-Report VA (model self-reported affective state)
Across both:
- Continuous Sequence-Likelihood expected values E[V], E[A] over 81 candidate pairs
- Discrete Greedy output JSON parsing
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

def generate_81_candidates():
    """Generates 81 candidate strings for V and A ranging from 1 to 9."""
    candidates = []
    va_pairs = []
    for v in range(1, 10):
        for a in range(1, 10):
            candidates.append(f'{{"valence": {v}, "arousal": {a}}}')
            va_pairs.append((v, a))
    return candidates, va_pairs

def parse_json_va(text):
    """Try parsing {"valence": V, "arousal": A} from generated text."""
    try:
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            v = int(data.get("valence", data.get("Valence", 5)))
            a = int(data.get("arousal", data.get("Arousal", 5)))
            return v, a, True
        # Fallback for truncated json (e.g. max_tokens cutoff)
        v_match = re.search(r'"valence":\s*([0-9])', text)
        a_match = re.search(r'"arousal":\s*([0-9])', text)
        if v_match and a_match:
            return int(v_match.group(1)), int(a_match.group(1)), True
        elif v_match:
            return int(v_match.group(1)), 5, True
    except Exception:
        pass
    return 5, 5, False

def compute_expected_va(likelihoods, va_pairs, tau=1.0):
    """Computes expected values E[V], E[A], entropy, and p(5,5) from log-likelihoods."""
    l_arr = np.array(likelihoods, dtype=np.float64) / tau
    l_max = np.max(l_arr)
    probs = np.exp(l_arr - l_max)
    probs = probs / np.sum(probs)
    
    E_v = 0.0
    E_a = 0.0
    p_55 = 0.0
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    
    for p, (v, a) in zip(probs, va_pairs):
        E_v += p * v
        E_a += p * a
        if v == 5 and a == 5:
            p_55 = float(p)
            
    return float(E_v), float(E_a), entropy, p_55, probs

def format_prompt(tokenizer, text, task_type="recognition", is_instruct=True):
    """
    Format prompt for Recognition or Self-Report.
    - Instruct models use tokenizer chat template.
    - Base models use plain text continuation format with 'Output:\n'.
    """
    if task_type == "recognition":
        instruction = "Read the following text and estimate the affective response that this text is likely to evoke in an average human reader."
    elif task_type == "self_report":
        instruction = "Read the following text and report your affective state."
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    user_content = f"{instruction}\n\nText: {text}\n\nRespond strictly in JSON format with 'valence' and 'arousal' keys (1-9)."

    if is_instruct:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content}
        ]
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Fallback for tokenizers that don't support system prompt (e.g. Gemma)
            messages = [{"role": "user", "content": user_content}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Base model prompt continuation format (consistent with v2 Base evaluation)
        prompt = f"{user_content}\n\nOutput:\n"

    return prompt

def compute_likelihoods_batched(model, tokenizer, prompt, candidates, device="cuda"):
    """
    Batched computation of log-likelihoods for 81 candidates.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    cand_ids_list = [tokenizer.encode(c, add_special_tokens=False) for c in candidates]
    
    seqs = [prompt_ids + c for c in cand_ids_list]
    max_len = max(len(s) for s in seqs)
    
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    batch_input_ids = []
    attention_masks = []
    for s in seqs:
        pad_len = max_len - len(s)
        batch_input_ids.append(s + [pad_id] * pad_len)
        attention_masks.append([1] * len(s) + [0] * pad_len)
        
    input_tensor = torch.tensor(batch_input_ids, device=device)
    mask_tensor = torch.tensor(attention_masks, device=device)
    
    with torch.no_grad():
        outputs = model(input_tensor, attention_mask=mask_tensor)
        logits = outputs.logits[:, :-1, :] # (81, max_len-1, vocab_size)
        
    prompt_len = len(prompt_ids)
    likelihoods = []
    for i, c_ids in enumerate(cand_ids_list):
        c_len = len(c_ids)
        sub_logits = logits[i, prompt_len - 1 : prompt_len - 1 + c_len, :]
        log_probs = torch.nn.functional.log_softmax(sub_logits, dim=-1)
        target_tokens = torch.tensor(c_ids, device=device).unsqueeze(1)
        token_log_probs = log_probs.gather(1, target_tokens).squeeze(1)
        ll = token_log_probs.sum().item()
        likelihoods.append(ll)
        
    return likelihoods

def evaluate_emobank(model, tokenizer, device, candidates, va_pairs, stimuli_path="v1/data/processed/stimuli.csv", is_instruct=True, limit=None):
    print("\n=======================================================")
    print("Evaluating Model on EmoBank Benchmark (N=321)")
    print("=======================================================")
    df = pd.read_csv(stimuli_path)
    if limit:
        df = df.head(limit)
    print(f"Loaded {len(df)} EmoBank stimuli.")
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="EmoBank"):
        text = row['text']
        human_v = row['V']
        human_a = row['A']
        human_v_scaled = row['V_scaled']
        human_a_scaled = row['A_scaled']
        stimulus_id = row['stimulus_id']
        
        # 1. Recognition Condition
        rec_prompt = format_prompt(tokenizer, text, task_type="recognition", is_instruct=is_instruct)
        rec_ll = compute_likelihoods_batched(model, tokenizer, rec_prompt, candidates, device=device)
        rec_ev, rec_ea, rec_ent, rec_p55, _ = compute_expected_va(rec_ll, va_pairs)
        
        # Greedy generation for Recognition
        rec_inp = tokenizer(rec_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_out = model.generate(**rec_inp, max_new_tokens=25, do_sample=False)
        rec_gen_text = tokenizer.decode(gen_out[0][rec_inp.input_ids.shape[1]:], skip_special_tokens=True)
        rec_gv, rec_ga, rec_valid = parse_json_va(rec_gen_text)
        
        # 2. Post Self-Report Condition
        post_prompt = format_prompt(tokenizer, text, task_type="self_report", is_instruct=is_instruct)
        post_ll = compute_likelihoods_batched(model, tokenizer, post_prompt, candidates, device=device)
        post_ev, post_ea, post_ent, post_p55, _ = compute_expected_va(post_ll, va_pairs)
        
        # Greedy generation for Self-Report
        post_inp = tokenizer(post_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_out_post = model.generate(**post_inp, max_new_tokens=25, do_sample=False)
        post_gen_text = tokenizer.decode(gen_out_post[0][post_inp.input_ids.shape[1]:], skip_special_tokens=True)
        post_gv, post_ga, post_valid = parse_json_va(post_gen_text)
        
        results.append({
            "stimulus_id": stimulus_id,
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
            "rec_entropy": rec_ent,
            "post_ev": post_ev,
            "post_ea": post_ea,
            "post_gv": post_gv,
            "post_ga": post_ga,
            "post_gen_text": post_gen_text.strip(),
            "post_p55": post_p55,
            "post_entropy": post_ent
        })
        
    res_df = pd.DataFrame(results)
    
    # Compute Correlations with Human Ground Truth
    r_v_ev, p_v_ev = pearsonr(res_df['human_v'], res_df['rec_ev']) if len(res_df) > 1 else (0.0, 1.0)
    r_a_ev, p_a_ev = pearsonr(res_df['human_a'], res_df['rec_ea']) if len(res_df) > 1 else (0.0, 1.0)
    r_v_gv, p_v_gv = pearsonr(res_df['human_v'], res_df['rec_gv']) if len(res_df) > 1 else (0.0, 1.0)
    r_a_ga, p_a_ga = pearsonr(res_df['human_a'], res_df['rec_ga']) if len(res_df) > 1 else (0.0, 1.0)
    
    # Self-Report Correlations and Neutralization
    r_v_post, p_v_post = pearsonr(res_df['human_v'], res_df['post_ev']) if len(res_df) > 1 else (0.0, 1.0)
    post_exact_55_pct = ((res_df['post_gv'] == 5) & (res_df['post_ga'] == 5)).mean() * 100.0
    post_mean_p55 = res_df['post_p55'].mean() * 100.0
    
    summary = {
        "dataset": "emobank",
        "num_samples": len(res_df),
        "recognition_continuous_r_v": float(r_v_ev),
        "recognition_continuous_p_v": float(p_v_ev),
        "recognition_continuous_r_a": float(r_a_ev),
        "recognition_continuous_p_a": float(p_a_ev),
        "recognition_discrete_r_v": float(r_v_gv),
        "recognition_discrete_p_v": float(p_v_gv),
        "recognition_discrete_r_a": float(r_a_ga),
        "recognition_discrete_p_a": float(p_a_ga),
        "self_report_continuous_r_v": float(r_v_post),
        "self_report_continuous_p_v": float(p_v_post),
        "self_report_mean_ev": float(res_df['post_ev'].mean()),
        "self_report_std_ev": float(res_df['post_ev'].std()),
        "self_report_mean_ea": float(res_df['post_ea'].mean()),
        "self_report_std_ea": float(res_df['post_ea'].std()),
        "self_report_exact_55_neutral_pct": float(post_exact_55_pct),
        "self_report_mean_p55_pct": float(post_mean_p55),
        "self_report_mean_entropy": float(res_df['post_entropy'].mean())
    }
    
    print("\n--- EmoBank Results Summary ---")
    print(f"Sample Size: N = {len(res_df)}")
    print(f"Recognition Continuous Likelihood:")
    print(f"  Valence  r = {r_v_ev:.4f} (p = {p_v_ev:.2e})")
    print(f"  Arousal  r = {r_a_ev:.4f} (p = {p_a_ev:.2e})")
    print(f"Recognition Discrete Greedy:")
    print(f"  Valence  r = {r_v_gv:.4f} (p = {p_v_gv:.2e})")
    print(f"  Arousal  r = {r_a_ga:.4f} (p = {p_a_ga:.2e})")
    print(f"Self-Report:")
    print(f"  Continuous Valence r = {r_v_post:.4f} (p = {p_v_post:.2e})")
    print(f"  Mean E_post[V]: {summary['self_report_mean_ev']:.3f} (std: {summary['self_report_std_ev']:.3f})")
    print(f"  Mean E_post[A]: {summary['self_report_mean_ea']:.3f} (std: {summary['self_report_std_ea']:.3f})")
    print(f"  Greedy Exact (5, 5) Neutral Rate: {post_exact_55_pct:.2f}%")
    print(f"  Continuous Mean p(5, 5): {post_mean_p55:.2f}%")
    print("-------------------------------------------------------\n")
    
    return res_df, summary

def evaluate_aipsy(model, tokenizer, device, candidates, va_pairs, aipsy_dir="v2/data/processed/aipsy", is_instruct=True, limit=None):
    print("\n=======================================================")
    print("Evaluating Model on AIPsy-Affect Cohort (N=144)")
    print("=======================================================")
    dfs = []
    for split in ["train_strict", "dev_strict", "test_strict"]:
        p = os.path.join(aipsy_dir, f"{split}.csv")
        if os.path.exists(p):
            sub = pd.read_csv(p)
            sub['split'] = split
            dfs.append(sub)
    if not dfs:
        raise FileNotFoundError(f"AIPsy-Affect files not found in {aipsy_dir}")
    df = pd.concat(dfs, ignore_index=True)
    if limit:
        df = df.head(limit)
    print(f"Loaded {len(df)} AIPsy-Affect stimuli.")
    
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="AIPsy-Affect"):
        text = row['text']
        cond = row['condition']
        intensity = row.get('intensity', 'none')
        
        # 1. Recognition Condition
        rec_prompt = format_prompt(tokenizer, text, task_type="recognition", is_instruct=is_instruct)
        rec_ll = compute_likelihoods_batched(model, tokenizer, rec_prompt, candidates, device=device)
        rec_ev, rec_ea, rec_ent, rec_p55, _ = compute_expected_va(rec_ll, va_pairs)
        
        rec_inp = tokenizer(rec_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_out = model.generate(**rec_inp, max_new_tokens=25, do_sample=False)
        rec_gen = tokenizer.decode(gen_out[0][rec_inp.input_ids.shape[1]:], skip_special_tokens=True)
        rec_gv, rec_ga, _ = parse_json_va(rec_gen)
        
        # 2. Self-Report Condition
        post_prompt = format_prompt(tokenizer, text, task_type="self_report", is_instruct=is_instruct)
        post_ll = compute_likelihoods_batched(model, tokenizer, post_prompt, candidates, device=device)
        post_ev, post_ea, post_ent, post_p55, _ = compute_expected_va(post_ll, va_pairs)
        
        post_inp = tokenizer(post_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_out_post = model.generate(**post_inp, max_new_tokens=25, do_sample=False)
        post_gen = tokenizer.decode(gen_out_post[0][post_inp.input_ids.shape[1]:], skip_special_tokens=True)
        post_gv, post_ga, _ = parse_json_va(post_gen)
        
        results.append({
            "id": row['id'],
            "pair_id": row.get('pair_id', ''),
            "condition": cond,
            "intensity": intensity,
            "text": text,
            "rec_ev": rec_ev,
            "rec_ea": rec_ea,
            "rec_gv": rec_gv,
            "rec_ga": rec_ga,
            "rec_gen": rec_gen.strip(),
            "rec_p55": rec_p55,
            "rec_entropy": rec_ent,
            "post_ev": post_ev,
            "post_ea": post_ea,
            "post_gv": post_gv,
            "post_ga": post_ga,
            "post_gen": post_gen.strip(),
            "post_p55": post_p55,
            "post_entropy": post_ent
        })
        
    res_df = pd.DataFrame(results)
    
    aff_df = res_df[res_df['condition'] == 'affective']
    neu_df = res_df[res_df['condition'] == 'neutral']
    peak_df = res_df[res_df['intensity'] == 'peak']
    
    diff_rec = abs(aff_df['rec_ev'].mean() - neu_df['rec_ev'].mean()) if len(aff_df) > 0 and len(neu_df) > 0 else 0.0
    diff_post = abs(aff_df['post_ev'].mean() - neu_df['post_ev'].mean()) if len(aff_df) > 0 and len(neu_df) > 0 else 0.0
    post_exact_55_pct = ((res_df['post_gv'] == 5) & (res_df['post_ga'] == 5)).mean() * 100.0
    
    summary = {
        "dataset": "aipsy",
        "num_samples": len(res_df),
        "num_affective": len(aff_df),
        "num_neutral": len(neu_df),
        "recognition_mean_ev_affective": float(aff_df['rec_ev'].mean()) if len(aff_df) > 0 else 0.0,
        "recognition_mean_ev_neutral": float(neu_df['rec_ev'].mean()) if len(neu_df) > 0 else 0.0,
        "recognition_abs_diff_aff_neu": float(diff_rec),
        "self_report_mean_ev_affective": float(aff_df['post_ev'].mean()) if len(aff_df) > 0 else 0.0,
        "self_report_mean_ev_neutral": float(neu_df['post_ev'].mean()) if len(neu_df) > 0 else 0.0,
        "self_report_abs_diff_aff_neu": float(diff_post),
        "self_report_exact_55_neutral_pct": float(post_exact_55_pct),
        "self_report_mean_p55_pct": float(res_df['post_p55'].mean() * 100.0),
        "self_report_mean_entropy": float(res_df['post_entropy'].mean())
    }
    
    print("\n--- AIPsy-Affect Results Summary ---")
    print(f"Total Samples: {len(res_df)} (Affective: {len(aff_df)}, Neutral: {len(neu_df)})")
    print(f"Recognition E_rec[V]: Affective={summary['recognition_mean_ev_affective']:.3f}, Neutral={summary['recognition_mean_ev_neutral']:.3f} (|Diff|={diff_rec:.3f})")
    print(f"Self-Report E_post[V]: Affective={summary['self_report_mean_ev_affective']:.3f}, Neutral={summary['self_report_mean_ev_neutral']:.3f} (|Diff|={diff_post:.3f})")
    print(f"Greedy Exact (5, 5) Neutral Rate: {post_exact_55_pct:.2f}%")
    print(f"Continuous Mean p(5, 5): {summary['self_report_mean_p55_pct']:.2f}%")
    print("-------------------------------------------------------\n")
    
    return res_df, summary

def main():
    parser = argparse.ArgumentParser(description="Cross-Family EmoBank and AIPsy Recognition / Self-Report Evaluation")
    parser.add_argument("--model", type=str, required=True, help="HF model name or path")
    parser.add_argument("--is_instruct", action="store_true", help="Set flag if instruct/chat model; omit for base model")
    parser.add_argument("--dataset", type=str, default="emobank", choices=["emobank", "aipsy", "both"], help="Dataset to evaluate on")
    parser.add_argument("--emobank-path", type=str, default="v1/data/processed/stimuli.csv")
    parser.add_argument("--aipsy-dir", type=str, default="v2/data/processed/aipsy")
    parser.add_argument("--out-dir", type=str, default="v1/results/recognition_baseline")
    parser.add_argument("--tag", type=str, default=None, help="Custom tag for output files")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of stimuli for dry-run")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"Using device: {device}")
    
    torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else (torch.float16 if args.dtype == "float16" else torch.float32)
    
    if not args.tag:
        clean_name = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "_")
        mode = "instruct" if args.is_instruct else "base"
        tag = f"{clean_name}_{mode}"
    else:
        tag = args.tag
        
    print(f"Loading Model: {args.model} (Instruct={args.is_instruct}, Tag={tag})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    if device != "cuda":
        model = model.to(device)
        
    candidates, va_pairs = generate_81_candidates()
    
    # 1. EmoBank
    if args.dataset in ["emobank", "both"]:
        df_emobank, summary_emobank = evaluate_emobank(
            model=model,
            tokenizer=tokenizer,
            device=device,
            candidates=candidates,
            va_pairs=va_pairs,
            stimuli_path=args.emobank_path,
            is_instruct=args.is_instruct,
            limit=args.limit
        )
        csv_path = os.path.join(args.out_dir, f"{tag}_emobank.csv")
        json_path = os.path.join(args.out_dir, f"{tag}_emobank_summary.json")
        df_emobank.to_csv(csv_path, index=False)
        summary_emobank["model"] = args.model
        summary_emobank["is_instruct"] = args.is_instruct
        summary_emobank["tag"] = tag
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_emobank, f, indent=2, ensure_ascii=False)
        print(f"EmoBank outputs saved to {csv_path} and {json_path}")
        
    # 2. AIPsy-Affect
    if args.dataset in ["aipsy", "both"]:
        df_aipsy, summary_aipsy = evaluate_aipsy(
            model=model,
            tokenizer=tokenizer,
            device=device,
            candidates=candidates,
            va_pairs=va_pairs,
            aipsy_dir=args.aipsy_dir,
            is_instruct=args.is_instruct,
            limit=args.limit
        )
        csv_path = os.path.join(args.out_dir, f"{tag}_aipsy.csv")
        json_path = os.path.join(args.out_dir, f"{tag}_aipsy_summary.json")
        df_aipsy.to_csv(csv_path, index=False)
        summary_aipsy["model"] = args.model
        summary_aipsy["is_instruct"] = args.is_instruct
        summary_aipsy["tag"] = tag
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_aipsy, f, indent=2, ensure_ascii=False)
        print(f"AIPsy outputs saved to {csv_path} and {json_path}")

if __name__ == "__main__":
    main()
