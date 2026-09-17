"""
v3/scripts/run_generation_time_causal_sweep.py

Purpose:
Performs a comprehensive layer-by-layer sweep across all 28 layers for
Response-Generation Position causal activation patching:
  Intervention Point: Last token of prefix '{"valence": '
                      (The state immediately predicting the first numeric token)
  Components: MLP output, Projected Attention output, Residual stream (Block output)
  Metric: 2D Joint Optimal Transport (OT) Recovery towards Peak with small-denominator safeguard,
          plus 1D Valence W1 and Marginal W1 sum.
  Inference constraint: use_cache=False strictly enforced.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import pearsonr, spearmanr
from v3.src.ot_utils import (
    compute_joint_ot_2d,
    compute_marginal_wasserstein_sum,
    compute_1d_wasserstein,
    compute_joint_ot_recovery
)
from v3.src.model_utils import (
    get_model_layers,
    get_component_module,
    get_patch_hook,
    format_base_prompt,
    build_generation_prefix_inputs
)
from v3.scripts.run_aligned_cross_model_patching import get_3way_split

PREFIX_STR = '{"valence": '

def generate_81_suffixes():
    """
    Generates 81 candidate suffixes for the JSON format starting after '{"valence": '
    Example: '1, "arousal": 1}'
    """
    candidates = []
    va_pairs = []
    for v in range(1, 10):
        for a in range(1, 10):
            cand_str = f'{v}, "arousal": {a}}}'
            candidates.append(cand_str)
            va_pairs.append((v, a))
    return candidates, va_pairs

def compute_conditional_candidate_logprobs(model, tokenizer, base_prompt, suffixes, target_pos, hook_fn=None, hook_module=None, device="cuda", normalize_length=False):
    """
    Computes conditional log-likelihood for all 81 suffixes in a SINGLE batched forward pass.
    If hook_fn and hook_module are provided, registers forward hook during evaluation.
    use_cache=False is strictly enforced.
    """
    prefix_prompt = base_prompt + PREFIX_STR
    # Joint tokenization for all candidate suffixes
    from affective_empathy_eval.likelihood import prepare_joint_sequence_with_boundary
    full_seqs = []
    suffix_lens = []
    cand_start_indices = []
    for s in suffixes:
        full_ids, c_start = prepare_joint_sequence_with_boundary(
            prompt=prefix_prompt,
            candidate=s,
            tokenizer=tokenizer,
            delimiter="",
            require_strict_prefix=False,
        )
        full_seqs.append(full_ids)
        cand_start_indices.append(c_start)
        suffix_lens.append(len(full_ids) - c_start)

    max_len = max(len(s) for s in full_seqs)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    padded_ids = []
    attention_masks = []

    for s in full_seqs:
        s_len = len(s)
        pad_len = max_len - s_len
        padded_ids.append(s + [pad_id] * pad_len)
        attention_masks.append([1] * s_len + [0] * pad_len)

    input_tensor = torch.tensor(padded_ids, dtype=torch.long, device=device)
    attn_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)

    
    handle = None
    if hook_fn is not None and hook_module is not None:
        handle = hook_module.register_forward_hook(hook_fn)
        
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_tensor, attention_mask=attn_mask, use_cache=False)
            logits = outputs.logits # (batch_size, max_len, vocab_size)
            log_probs = torch.log_softmax(logits, dim=-1)
            
            cand_lps = []
            for b_idx, (s_len, c_start) in enumerate(zip(suffix_lens, cand_start_indices)):
                target_tokens = input_tensor[b_idx, c_start : c_start + s_len]
                # Predictions come from c_start - 1 to c_start + s_len - 2
                step_log_p = log_probs[b_idx, c_start - 1 : c_start + s_len - 1, :]
                selected_p = step_log_p.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
                lp = selected_p.sum().item()
                if normalize_length and s_len > 0:
                    lp = lp / s_len
                cand_lps.append(lp)

    finally:
        if handle is not None:
            handle.remove()
            
    # Softmax over 81 candidates
    cand_lps = np.array(cand_lps)
    max_lp = np.max(cand_lps)
    probs = np.exp(cand_lps - max_lp)
    probs = probs / np.sum(probs)
    
    prob_matrix = probs.reshape((9, 9))
    return prob_matrix

def extract_generation_time_activation(model, tokenizer, prompt_text, layer, comp="mlp", device="cuda"):
    """
    Extracts the hidden activation at the generation-time position (tail of PREFIX_STR).
    """
    inputs, target_pos, _ = build_generation_prefix_inputs(tokenizer, prompt_text, PREFIX_STR, device=device)
    target_mod = get_component_module(model, layer, comp)
    
    extracted = {}
    def hook(m, inp, out):
        val = out[0] if isinstance(out, tuple) else out
        extracted["val"] = val[0, target_pos, :].detach().clone()
        
    handle = target_mod.register_forward_hook(hook)
    with torch.no_grad():
        model(**inputs, use_cache=False)
    handle.remove()
    return extracted["val"], target_pos

def evaluate_generation_causal_effect(model, tokenizer, valid_pairs, layer, comp, suffixes, base_cache=None, device="cuda", normalize_length=False, eps_rec=0.05):
    """
    Evaluates generation-time causal patching for a given layer and component across pairs.
    """
    target_mod = get_component_module(model, layer, comp)
    
    ot_recoveries = []
    ot_displacements = []
    marg_recoveries = []
    w1_v_recoveries = []
    
    for pid, peak_row, neutral_row in valid_pairs:
        peak_prompt = format_base_prompt(peak_row['text'])
        neut_prompt = format_base_prompt(neutral_row['text'])
        
        _, target_pos, _ = build_generation_prefix_inputs(tokenizer, neut_prompt, PREFIX_STR, device=device)
        
        # 1 & 2: Base distributions from cache
        if base_cache is not None and pid in base_cache:
            p_peak = base_cache[pid]["p_peak"]
            p_neut = base_cache[pid]["p_neut"]
        else:
            p_peak = compute_conditional_candidate_logprobs(
                model, tokenizer, peak_prompt, suffixes, target_pos,
                hook_fn=None, hook_module=None, device=device, normalize_length=normalize_length
            )
            p_neut = compute_conditional_candidate_logprobs(
                model, tokenizer, neut_prompt, suffixes, target_pos,
                hook_fn=None, hook_module=None, device=device, normalize_length=normalize_length
            )
            
        # 3. Extract Peak activation (Source)
        source_tensor, _ = extract_generation_time_activation(model, tokenizer, peak_prompt, layer, comp=comp, device=device)
        
        # 4. Patch into Neutral run (Target)
        patch_hook = get_patch_hook(source_tensor, target_pos)
        p_patch = compute_conditional_candidate_logprobs(
            model, tokenizer, neut_prompt, suffixes, target_pos,
            hook_fn=patch_hook, hook_module=target_mod, device=device, normalize_length=normalize_length
        )
        
        # 5. Compute Joint OT Recovery with safeguard
        rec_ot, d_patch_peak, d_neut_peak, is_valid = compute_joint_ot_recovery(
            p_patch, p_peak, p_neut, eps_rec=eps_rec
        )
        ot_displacements.append(d_patch_peak)
        if is_valid and rec_ot is not None:
            ot_recoveries.append(rec_ot)
            
        # 6. Secondary metrics
        d_marg_np = compute_marginal_wasserstein_sum(p_neut, p_peak)
        d_marg_pp = compute_marginal_wasserstein_sum(p_patch, p_peak)
        if d_marg_np >= eps_rec:
            marg_recoveries.append(1.0 - (d_marg_pp / d_marg_np))
            
        p_v_peak = np.sum(p_peak, axis=1)
        p_v_neut = np.sum(p_neut, axis=1)
        p_v_patch = np.sum(p_patch, axis=1)
        d_w1_np = compute_1d_wasserstein(p_v_neut, p_v_peak)
        d_w1_pp = compute_1d_wasserstein(p_v_patch, p_v_peak)
        if d_w1_np >= eps_rec:
            w1_v_recoveries.append(1.0 - (d_w1_pp / d_w1_np))
            
    mean_ot_rec = float(np.mean(ot_recoveries)) * 100.0 if len(ot_recoveries) > 0 else 0.0
    median_ot_rec = float(np.median(ot_recoveries)) * 100.0 if len(ot_recoveries) > 0 else 0.0
    mean_disp = float(np.mean(ot_displacements)) if len(ot_displacements) > 0 else 0.0
    mean_marg_rec = float(np.mean(marg_recoveries)) * 100.0 if len(marg_recoveries) > 0 else 0.0
    mean_w1_rec = float(np.mean(w1_v_recoveries)) * 100.0 if len(w1_v_recoveries) > 0 else 0.0
    valid_pair_count = len(ot_recoveries)
    
    return {
        "mean_ot_recovery": mean_ot_rec,
        "median_ot_recovery": median_ot_rec,
        "mean_ot_displacement": mean_disp,
        "mean_marg_recovery": mean_marg_rec,
        "mean_w1_v_recovery": mean_w1_rec,
        "valid_pairs": valid_pair_count
    }

def main():
    parser = argparse.ArgumentParser(description="Generation-Time Causal Localization Sweep")
    parser.add_argument("--data_path", type=str, default="v3/data/aipsy_strict_expanded.csv")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output_path", type=str, default="v3/results/generation_time_causal_sweep.csv")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--normalize_length", action="store_true", default=True)
    parser.add_argument("--eps_rec", type=float, default=0.05)
    parser.add_argument("--max_pairs", type=int, default=15, help="Pairs per layer for efficiency")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model_name} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16 if "cuda" in args.device else torch.float32
    )
    model.to(args.device)
    model.eval()
    
    # 3-way split
    df = pd.read_csv(args.data_path)
    train_df, val_df, test_df = get_3way_split(df)
    test_peaks = test_df[test_df['intensity'] == 'peak'].copy()
    test_neutrals = test_df[test_df['intensity'] == 'neutral'].copy()
    
    valid_pairs = []
    for pid in test_peaks['pair_id'].unique():
        p_row = test_peaks[test_peaks['pair_id'] == pid]
        n_row = test_neutrals[test_neutrals['pair_id'] == pid]
        if len(p_row) > 0 and len(n_row) > 0:
            valid_pairs.append((pid, p_row.iloc[0], n_row.iloc[0]))
            
    eval_pairs = valid_pairs[:args.max_pairs]
    print(f"Found {len(valid_pairs)} test pairs, using {len(eval_pairs)} pairs for generation sweep.")
    
    suffixes, _ = generate_81_suffixes()
    layers = get_model_layers(model)
    num_layers = len(layers)
    print(f"Total layers: {num_layers}")
    
    # Pre-compute baseline Peak and Neutral distributions once for all pairs (using batching)
    print("Pre-computing baseline Peak and Neutral generation-time candidate distributions (batched)...")
    base_cache = {}
    for pid, peak_row, neut_row in eval_pairs:
        peak_prompt = format_base_prompt(peak_row['text'])
        neut_prompt = format_base_prompt(neut_row['text'])
        
        _, p_target_pos, _ = build_generation_prefix_inputs(tokenizer, peak_prompt, PREFIX_STR, device=args.device)
        _, n_target_pos, _ = build_generation_prefix_inputs(tokenizer, neut_prompt, PREFIX_STR, device=args.device)
        
        p_peak = compute_conditional_candidate_logprobs(
            model, tokenizer, peak_prompt, suffixes, p_target_pos,
            hook_fn=None, hook_module=None, device=args.device, normalize_length=args.normalize_length
        )
        p_neut = compute_conditional_candidate_logprobs(
            model, tokenizer, neut_prompt, suffixes, n_target_pos,
            hook_fn=None, hook_module=None, device=args.device, normalize_length=args.normalize_length
        )
        base_cache[pid] = {
            "p_peak": p_peak,
            "p_neut": p_neut
        }
    print("Baseline caching completed! Starting layer sweep...")
    
    records = []
    
    for l in range(num_layers):
        print(f"\n>>> Running Generation-Time Sweep for Layer {l}/{num_layers - 1} <<<")
        for comp in ["mlp", "attn", "resid"]:
            res = evaluate_generation_causal_effect(
                model, tokenizer, eval_pairs, l, comp, suffixes,
                base_cache=base_cache,
                device=args.device, normalize_length=args.normalize_length, eps_rec=args.eps_rec
            )
            print(f"Layer {l} | {comp.upper()}: Joint OT Rec={res['mean_ot_recovery']:.2f}% (Med: {res['median_ot_recovery']:.2f}%), Disp={res['mean_ot_displacement']:.4f}, Valid Pairs={res['valid_pairs']}")
            
            record = {
                "layer": l,
                "component": comp,
                "mean_ot_recovery": res["mean_ot_recovery"],
                "median_ot_recovery": res["median_ot_recovery"],
                "mean_ot_displacement": res["mean_ot_displacement"],
                "mean_marg_recovery": res["mean_marg_recovery"],
                "mean_w1_v_recovery": res["mean_w1_v_recovery"],
                "valid_pairs": res["valid_pairs"],
                "model_name": args.model_name
            }
            records.append(record)
            
        # Intermediate save
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        pd.DataFrame(records).to_csv(args.output_path, index=False)
        
    print(f"\nAll layers completed! Final results saved to {args.output_path}")

if __name__ == "__main__":
    main()
