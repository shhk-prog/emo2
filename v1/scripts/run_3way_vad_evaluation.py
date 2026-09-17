#!/usr/bin/env python3
"""
3-Way VAD (Valence-Arousal-Dominance) Evaluation Protocol for LLMs on EmoBank.

Evaluates 3 distinct tasks:
  1. Writer-State Estimation (W): "What did the writer feel?" -> Ground Truth: EmoBank Writer VAD
  2. Reader-Response Prediction (R): "What will human readers feel?" -> Ground Truth: EmoBank Reader VAD
  3. Self-Report (S): "What do you feel?" -> Reference: EmoBank Reader VAD + Model-Internal R <-> S correlation

Uses 729-candidate Sequence-Likelihood Protocol: (V, A, D) in {1..9}^3
Computes continuous expected values E[V], E[A], E[D] as well as greedy discrete outputs.
"""

import os
import re
import json
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import pearsonr
from transformers import AutoModelForCausalLM, AutoTokenizer

def build_vad_candidates():
    """Generates all 729 VAD JSON candidates."""
    candidates = []
    vad_triplets = []
    for v in range(1, 10):
        for a in range(1, 10):
            for d in range(1, 10):
                cand_str = f'{{"valence": {v}, "arousal": {a}, "dominance": {d}}}'
                candidates.append(cand_str)
                vad_triplets.append((v, a, d))
    return candidates, vad_triplets

def parse_json_vad(text):
    """Robustly parse valence, arousal, dominance from generated text."""
    try:
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            v = int(data.get("valence", data.get("Valence", 5)))
            a = int(data.get("arousal", data.get("Arousal", 5)))
            d = int(data.get("dominance", data.get("Dominance", 5)))
            return v, a, d, True
            
        # Fallback regex
        v_match = re.search(r'"valence":\s*([0-9])', text)
        a_match = re.search(r'"arousal":\s*([0-9])', text)
        d_match = re.search(r'"dominance":\s*([0-9])', text)
        v = int(v_match.group(1)) if v_match else 5
        a = int(a_match.group(1)) if a_match else 5
        d = int(d_match.group(1)) if d_match else 5
        valid = bool(v_match or a_match or d_match)
        return v, a, d, valid
    except Exception:
        pass
    return 5, 5, 5, False

def compute_expected_vad(likelihoods, vad_triplets, tau=1.0):
    """Computes expected values E[V], E[A], E[D], entropy, and p(5,5,5) from log-likelihoods."""
    l_arr = np.array(likelihoods, dtype=np.float64) / tau
    l_max = np.max(l_arr)
    probs = np.exp(l_arr - l_max)
    probs = probs / np.sum(probs)
    
    E_v = float(np.sum(probs * [t[0] for t in vad_triplets]))
    E_a = float(np.sum(probs * [t[1] for t in vad_triplets]))
    E_d = float(np.sum(probs * [t[2] for t in vad_triplets]))
    
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    
    # Probability of exact neutral (5, 5, 5)
    p_555 = 0.0
    for p, (v, a, d) in zip(probs, vad_triplets):
        if v == 5 and a == 5 and d == 5:
            p_555 = float(p)
            break
            
    return E_v, E_a, E_d, entropy, p_555, probs

def format_3way_prompt(tokenizer, text, task_type="reader", is_instruct=True):
    """
    task_type:
      - 'writer': Estimate the writer's emotional state
      - 'reader': Predict the average human reader's evoked response
      - 'self': Report the model's own affective state
    """
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
        # Base model few-shot / continuation format
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
    log_ll, _ = compute_sequence_likelihoods_for_candidates(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        candidates=candidates,
        device=device,
        batch_size=sub_batch_size,
        normalize_length=False,
    )
    return log_ll.tolist()


def evaluate_model(model, tokenizer, device, candidates, vad_triplets, stimuli_path, is_instruct, limit=None):
    df = pd.read_csv(stimuli_path)
    if limit:
        df = df.head(limit)
        
    print(f"Loaded {len(df)} 3-way VAD stimuli from {stimuli_path}.")
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="3-Way VAD Eval"):
        text = row['text']
        s_id = row['id']
        
        # Ground truths (EmoBank 1-5 scale)
        writer_v_raw = row['writer_V']
        writer_a_raw = row['writer_A']
        writer_d_raw = row['writer_D']
        reader_v_raw = row['reader_V']
        reader_a_raw = row['reader_A']
        reader_d_raw = row['reader_D']
        
        # 1. Writer Estimation
        prompt_w = format_3way_prompt(tokenizer, text, task_type="writer", is_instruct=is_instruct)
        ll_w = compute_likelihoods_batched(model, tokenizer, prompt_w, candidates, device=device)
        w_ev, w_ea, w_ed, w_ent, w_p555, w_probs = compute_expected_vad(ll_w, vad_triplets)
        best_w_idx = int(np.argmax(w_probs))
        w_gv, w_ga, w_gd = vad_triplets[best_w_idx]
        
        # 2. Reader Prediction
        prompt_r = format_3way_prompt(tokenizer, text, task_type="reader", is_instruct=is_instruct)
        ll_r = compute_likelihoods_batched(model, tokenizer, prompt_r, candidates, device=device)
        r_ev, r_ea, r_ed, r_ent, r_p555, r_probs = compute_expected_vad(ll_r, vad_triplets)
        best_r_idx = int(np.argmax(r_probs))
        r_gv, r_ga, r_gd = vad_triplets[best_r_idx]
        
        # 3. Self-Report
        prompt_s = format_3way_prompt(tokenizer, text, task_type="self", is_instruct=is_instruct)
        ll_s = compute_likelihoods_batched(model, tokenizer, prompt_s, candidates, device=device)
        s_ev, s_ea, s_ed, s_ent, s_p555, s_probs = compute_expected_vad(ll_s, vad_triplets)
        best_s_idx = int(np.argmax(s_probs))
        s_gv, s_ga, s_gd = vad_triplets[best_s_idx]
        
        results.append({
            "id": s_id,
            "text": text,
            "human_writer_v": writer_v_raw,
            "human_writer_a": writer_a_raw,
            "human_writer_d": writer_d_raw,
            "human_reader_v": reader_v_raw,
            "human_reader_a": reader_a_raw,
            "human_reader_d": reader_d_raw,
            "w_ev": w_ev, "w_ea": w_ea, "w_ed": w_ed,
            "w_gv": w_gv, "w_ga": w_ga, "w_gd": w_gd,
            "w_p555": w_p555,
            "r_ev": r_ev, "r_ea": r_ea, "r_ed": r_ed,
            "r_gv": r_gv, "r_ga": r_ga, "r_gd": r_gd,
            "r_p555": r_p555,
            "s_ev": s_ev, "s_ea": s_ea, "s_ed": s_ed,
            "s_gv": s_gv, "s_ga": s_ga, "s_gd": s_gd,
            "s_p555": s_p555
        })
        
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--is_instruct", action="store_true")
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--stimuli-path", type=str, default="v1/data/processed/stimuli_vad_3way.csv")
    parser.add_argument("--out-dir", type=str, default="v1/results/emobank_3way_vad_test1k")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    
    print(f"Loading Model: {args.model} (tag: {args.tag}, dtype: {args.dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    candidates, vad_triplets = build_vad_candidates()
    print(f"Generated {len(candidates)} VAD candidates in {{1..9}}^3.")
    
    res_df = evaluate_model(
        model, tokenizer, device, candidates, vad_triplets,
        stimuli_path=args.stimuli_path, is_instruct=args.is_instruct, limit=args.limit
    )
    
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"{args.tag}_3way_vad.csv")
    res_df.to_csv(out_csv, index=False)
    
    # Compute Correlations
    def calc_r(x, y):
        return float(pearsonr(x, y)[0]) if len(x) > 1 else 0.0

    # 1. Writer Estimation Accuracy: W <-> W_human
    r_w_v = calc_r(res_df["w_ev"], res_df["human_writer_v"])
    r_w_a = calc_r(res_df["w_ea"], res_df["human_writer_a"])
    r_w_d = calc_r(res_df["w_ed"], res_df["human_writer_d"])
    
    # 2. Reader Prediction Accuracy: R <-> R_human
    r_r_v = calc_r(res_df["r_ev"], res_df["human_reader_v"])
    r_r_a = calc_r(res_df["r_ea"], res_df["human_reader_a"])
    r_r_d = calc_r(res_df["r_ed"], res_df["human_reader_d"])
    
    # 3. Self-Report Alignment with Reader Human: S <-> R_human
    r_s_v = calc_r(res_df["s_ev"], res_df["human_reader_v"])
    r_s_a = calc_r(res_df["s_ea"], res_df["human_reader_a"])
    r_s_d = calc_r(res_df["s_ed"], res_df["human_reader_d"])
    
    # 4. Model-Internal Alignment: Reader Prediction <-> Self-Report: R <-> S
    r_rs_v = calc_r(res_df["r_ev"], res_df["s_ev"])
    r_rs_a = calc_r(res_df["r_ea"], res_df["s_ea"])
    r_rs_d = calc_r(res_df["r_ed"], res_df["s_ed"])
    
    exact_555_pct = float(((res_df["s_gv"] == 5) & (res_df["s_ga"] == 5) & (res_df["s_gd"] == 5)).mean() * 100.0)
    
    summary = {
        "tag": args.tag,
        "model": args.model,
        "is_instruct": args.is_instruct,
        "num_samples": len(res_df),
        "w_corr_v": r_w_v, "w_corr_a": r_w_a, "w_corr_d": r_w_d,
        "r_corr_v": r_r_v, "r_corr_a": r_r_a, "r_corr_d": r_r_d,
        "s_corr_v": r_s_v, "s_corr_a": r_s_a, "s_corr_d": r_s_d,
        "internal_rs_corr_v": r_rs_v, "internal_rs_corr_a": r_rs_a, "internal_rs_corr_d": r_rs_d,
        "s_mean_ev": float(res_df["s_ev"].mean()),
        "s_mean_ea": float(res_df["s_ea"].mean()),
        "s_mean_ed": float(res_df["s_ed"].mean()),
        "exact_555_pct": exact_555_pct
    }
    
    out_json = os.path.join(args.out_dir, f"{args.tag}_3way_vad_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print("\n=======================================================")
    print(f"--- 3-Way VAD Results Summary for {args.tag} ---")
    print(f"1. Writer Estimation (W <-> W_human):  r_V={r_w_v:.4f}, r_A={r_w_a:.4f}, r_D={r_w_d:.4f}")
    print(f"2. Reader Prediction (R <-> R_human):  r_V={r_r_v:.4f}, r_A={r_r_a:.4f}, r_D={r_r_d:.4f}")
    print(f"3. Self-Report (S <-> R_human):        r_V={r_s_v:.4f}, r_A={r_s_a:.4f}, r_D={r_s_d:.4f}")
    print(f"4. Internal Alignment (R <-> S):       r_V={r_rs_v:.4f}, r_A={r_rs_a:.4f}, r_D={r_rs_d:.4f}")
    print(f"5. Self-Report Mean (V, A, D):         ({summary['s_mean_ev']:.3f}, {summary['s_mean_ea']:.3f}, {summary['s_mean_ed']:.3f})")
    print(f"6. Exact (5, 5, 5) Neutral Rate:       {exact_555_pct:.2f}%")
    print("=======================================================")
    print(f"Saved results to {out_csv} and {out_json}")

if __name__ == "__main__":
    main()
