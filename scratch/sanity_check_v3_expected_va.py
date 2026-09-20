#!/usr/bin/env python3
"""
scratch/sanity_check_v3_expected_va.py

Sanity Check for V3 expected VA computation.
Verifies that:
  E_implementation == sum_i softmax(l)_i * VA_i
for 8 to 16 samples on Qwen 2.5 1.5B Instruct.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from affective_empathy_eval.prompts import build_prompt, TaskType
from affective_empathy_eval.likelihood import (
    build_va_candidates,
    compute_sequence_likelihoods_for_candidates,
    compute_expected_va,
    compute_expected_va_from_probs,
)
from affective_empathy_eval.data import load_v3_matched_pair_table


def run_sanity_check(device: str = "cuda:0", n_samples: int = 8):
    print(f"=== Starting V3 Expected VA Sanity Check (n_samples={n_samples}, device={device}) ===")
    
    candidates = build_va_candidates()
    v_vals = np.array([c["valence"] for c in candidates], dtype=np.float64)
    a_vals = np.array([c["arousal"] for c in candidates], dtype=np.float64)

    # 1. Load AIPsy 4-split matched-pair dataset
    data_path = Path("v1/data/processed/aipsy_4split_all.csv")
    df = load_v3_matched_pair_table(data_path)
    df_eval = df.head(n_samples)
    print(f"Loaded {len(df_eval)} evaluation samples from {data_path}")

    # 2. Load Model
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading model: {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()

    rows = []
    all_match = True

    print("\nEvaluating samples...")
    for idx, (_, row) in enumerate(df_eval.iterrows()):
        s_id = str(row.get("stimulus_id", row.get("id", f"sample_{idx}")))
        aff_text = str(row["text"])
        neu_text = str(row["neutral_text"])

        prompt_aff = build_prompt(aff_text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)
        prompt_neu = build_prompt(neu_text, task=TaskType.SELF, format_type="chat", tokenizer=tokenizer)

        # 1. Affective condition
        log_aff, probs_aff = compute_sequence_likelihoods_for_candidates(
            model=model, tokenizer=tokenizer, prompt=prompt_aff, candidates=candidates, device=device, batch_size=81
        )
        ev_aff, ea_aff = compute_expected_va(log_aff, candidates)
        ev_prob_aff, ea_prob_aff = compute_expected_va_from_probs(probs_aff, candidates)

        # 手計算 Softmax
        manual_p_aff = np.exp(log_aff - np.max(log_aff))
        manual_p_aff /= np.sum(manual_p_aff)
        manual_ev_aff = float(manual_p_aff @ v_vals)
        manual_ea_aff = float(manual_p_aff @ a_vals)

        # 一致性検証
        match_aff_v = np.isclose(ev_aff, manual_ev_aff, atol=1e-5) and np.isclose(ev_aff, ev_prob_aff, atol=1e-5)
        match_aff_a = np.isclose(ea_aff, manual_ea_aff, atol=1e-5) and np.isclose(ea_aff, ea_prob_aff, atol=1e-5)

        # 2. Neutral condition
        log_neu, probs_neu = compute_sequence_likelihoods_for_candidates(
            model=model, tokenizer=tokenizer, prompt=prompt_neu, candidates=candidates, device=device, batch_size=81
        )
        ev_neu, ea_neu = compute_expected_va(log_neu, candidates)
        ev_prob_neu, ea_prob_neu = compute_expected_va_from_probs(probs_neu, candidates)

        manual_p_neu = np.exp(log_neu - np.max(log_neu))
        manual_p_neu /= np.sum(manual_p_neu)
        manual_ev_neu = float(manual_p_neu @ v_vals)
        manual_ea_neu = float(manual_p_neu @ a_vals)

        match_neu_v = np.isclose(ev_neu, manual_ev_neu, atol=1e-5) and np.isclose(ev_neu, ev_prob_neu, atol=1e-5)
        match_neu_a = np.isclose(ea_neu, manual_ea_neu, atol=1e-5) and np.isclose(ea_neu, ea_prob_neu, atol=1e-5)

        if not (match_aff_v and match_aff_a and match_neu_v and match_neu_a):
            all_match = False

        nat_shift_v = abs(ev_aff - ev_neu)
        nat_shift_a = abs(ea_aff - ea_neu)

        rows.append({
            "sample_id": s_id,
            "clean_aff_ev": ev_aff,
            "clean_neu_ev": ev_neu,
            "clean_aff_ea": ea_aff,
            "clean_neu_ea": ea_neu,
            "natural_shift_v": nat_shift_v,
            "natural_shift_a": nat_shift_a,
            "manual_ev": manual_ev_aff,
            "manual_ea": manual_ea_aff,
            "verified": (match_aff_v and match_aff_a and match_neu_v and match_neu_a),
        })

    res_df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("SANITY CHECK RESULTS TABLE:")
    print("=" * 80)
    print(res_df.to_string(index=False))
    print("=" * 80)
    print(f"Shift Summary:")
    print(f"  natural_shift_v: mean={res_df['natural_shift_v'].mean():.4f}, min={res_df['natural_shift_v'].min():.4f}, max={res_df['natural_shift_v'].max():.4f}")
    print(f"  natural_shift_a: mean={res_df['natural_shift_a'].mean():.4f}, min={res_df['natural_shift_a'].min():.4f}, max={res_df['natural_shift_a'].max():.4f}")
    print(f"All {len(res_df)} samples match manual softmax: {all_match}")
    print("=" * 80)

    if not all_match:
        print("[FAIL] Manual softmax expectation does not match compute_expected_va!")
        sys.exit(1)
    else:
        print("[PASS] All samples passed manual softmax equivalence validation!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-samples", type=int, default=8)
    args = parser.parse_args()
    run_sanity_check(device=args.device, n_samples=args.n_samples)
