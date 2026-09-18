"""
v3/scripts/run_focused_39pairs_sweep.py

Evaluates the 6 key representative layers across all 39 complete held-out test pairs:
  Layers: [10, 14, 15, 18, 20, 24]
  Components: ['mlp', 'attn', 'resid']
  Timing: Both Prompt-Time and Generation-Time Joint OT Recovery

Output:
  v3/results/focused_causal_sweep_39pairs.csv
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import spearmanr

from v2.src.likelihood import generate_81_candidates, compute_expected_va
from v3.src.batch_likelihood import compute_likelihoods_batched
from v3.src.ot_utils import compute_joint_ot_2d, compute_joint_ot_recovery
from v3.src.model_utils import (
    get_model_layers,
    get_component_module,
    get_patch_hook,
    format_base_prompt
)
from v3.scripts.run_aligned_cross_model_patching import get_3way_split
from v3.scripts.run_generation_time_causal_sweep import (
    PREFIX_STR,
    build_generation_prefix_inputs,
    compute_conditional_candidate_logprobs,
    extract_generation_time_activation
)

TARGET_LAYERS = [10, 14, 15, 18, 20, 24]
COMPONENTS = ["mlp", "attn", "resid"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data_path", type=str, default="v3/data/aipsy_strict_expanded.csv")
    parser.add_argument("--output_path", type=str, default="v3/results/focused_causal_sweep_39pairs.csv")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eps_rec", type=float, default=0.05)
    args = parser.parse_args()

    print(f"Loading {args.model_name} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if "cuda" in args.device else torch.float32,
        device_map=args.device if "cuda" in args.device else None
    )
    model.eval()

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

    print(f"Total available test pairs: {len(valid_pairs)} (Using ALL pairs for focused sweep)")

    candidates, va_pairs = generate_81_candidates()
    suffixes = [c.replace('{"valence": ', '') for c in candidates]

    # Pre-cache prompt-time baseline distributions
    print("\nPre-computing Prompt-Time baseline distributions for all 39 pairs...")
    base_cache_prompt = {}
    for pid, peak_row, neut_row in valid_pairs:
        p_prompt = format_base_prompt(peak_row['text'])
        n_prompt = format_base_prompt(neut_row['text'])
        lp, _ = compute_likelihoods_batched(model, tokenizer, p_prompt, candidates, device=args.device, normalize_length=True)
        _, _, _, _, pp_flat = compute_expected_va(lp, va_pairs)
        ln, _ = compute_likelihoods_batched(model, tokenizer, n_prompt, candidates, device=args.device, normalize_length=True)
        _, _, _, _, pn_flat = compute_expected_va(ln, va_pairs)
        base_cache_prompt[pid] = {
            "p_peak": pp_flat.reshape((9, 9)),
            "p_neut": pn_flat.reshape((9, 9))
        }

    # Pre-cache generation-time baseline distributions
    print("Pre-computing Generation-Time baseline distributions for all 39 pairs...")
    base_cache_gen = {}
    for pid, peak_row, neut_row in valid_pairs:
        p_prompt = format_base_prompt(peak_row['text'])
        n_prompt = format_base_prompt(neut_row['text'])
        _, p_tpos, _ = build_generation_prefix_inputs(tokenizer, p_prompt, PREFIX_STR, device=args.device)
        _, n_tpos, _ = build_generation_prefix_inputs(tokenizer, n_prompt, PREFIX_STR, device=args.device)
        pp_mat = compute_conditional_candidate_logprobs(model, tokenizer, p_prompt, suffixes, p_tpos, device=args.device)
        pn_mat = compute_conditional_candidate_logprobs(model, tokenizer, n_prompt, suffixes, n_tpos, device=args.device)
        base_cache_gen[pid] = {
            "p_peak": pp_mat,
            "p_neut": pn_mat,
            "n_tpos": n_tpos
        }

    records = []
    pair_level_dict = {pid: {} for pid, _, _ in valid_pairs}

    print("\n--- Starting Focused Evaluation Across Key Layers ---")
    for l in TARGET_LAYERS:
        for comp in COMPONENTS:
            target_mod = get_component_module(model, l, comp)
            
            # --- Prompt-Time Evaluation ---
            prompt_recs = []
            for pid, peak_row, neut_row in valid_pairs:
                p_prompt = format_base_prompt(peak_row['text'])
                n_prompt = format_base_prompt(neut_row['text'])
                
                # Source activation at prompt tail
                inputs = tokenizer(p_prompt, return_tensors="pt").to(args.device)
                src_pos = inputs.input_ids.shape[1] - 1
                act_container = {}
                def hook_src(m, inp, out):
                    val = out[0] if isinstance(out, tuple) else out
                    act_container["val"] = val[0, src_pos, :].detach().clone()
                h_src = target_mod.register_forward_hook(hook_src)
                with torch.no_grad():
                    model(**inputs, use_cache=False)
                h_src.remove()
                
                # Patch into neutral prompt
                inputs_tgt = tokenizer(n_prompt, return_tensors="pt").to(args.device)
                tgt_pos = inputs_tgt.input_ids.shape[1] - 1
                p_hook = get_patch_hook(act_container["val"], tgt_pos)
                h_tgt = target_mod.register_forward_hook(p_hook)
                lp_patch, _ = compute_likelihoods_batched(model, tokenizer, n_prompt, candidates, device=args.device, normalize_length=True)
                h_tgt.remove()
                _, _, _, _, p_patch_flat = compute_expected_va(lp_patch, va_pairs)
                
                r_ot, _, _, is_valid = compute_joint_ot_recovery(
                    p_patch_flat.reshape((9, 9)),
                    base_cache_prompt[pid]["p_peak"],
                    base_cache_prompt[pid]["p_neut"],
                    eps_rec=args.eps_rec
                )
                if is_valid and r_ot is not None:
                    prompt_recs.append(r_ot)
                    pair_level_dict[pid][f"prompt_rec_L{l}_{comp}"] = r_ot * 100.0
                else:
                    pair_level_dict[pid][f"prompt_rec_L{l}_{comp}"] = None
                    
            mean_p_rec = float(np.mean(prompt_recs)) * 100.0 if len(prompt_recs) > 0 else 0.0
            med_p_rec = float(np.median(prompt_recs)) * 100.0 if len(prompt_recs) > 0 else 0.0
            
            # --- Generation-Time Evaluation ---
            gen_recs = []
            for pid, peak_row, neut_row in valid_pairs:
                p_prompt = format_base_prompt(peak_row['text'])
                n_prompt = format_base_prompt(neut_row['text'])
                tpos = base_cache_gen[pid]["n_tpos"]
                
                src_val, _ = extract_generation_time_activation(model, tokenizer, p_prompt, l, comp=comp, device=args.device)
                p_hook = get_patch_hook(src_val, tpos)
                p_mat = compute_conditional_candidate_logprobs(
                    model, tokenizer, n_prompt, suffixes, tpos,
                    hook_fn=p_hook, hook_module=target_mod, device=args.device
                )
                r_ot, _, _, is_valid = compute_joint_ot_recovery(
                    p_mat, base_cache_gen[pid]["p_peak"], base_cache_gen[pid]["p_neut"], eps_rec=args.eps_rec
                )
                if is_valid and r_ot is not None:
                    gen_recs.append(r_ot)
                    pair_level_dict[pid][f"gen_rec_L{l}_{comp}"] = r_ot * 100.0
                else:
                    pair_level_dict[pid][f"gen_rec_L{l}_{comp}"] = None
                    
            mean_g_rec = float(np.mean(gen_recs)) * 100.0 if len(gen_recs) > 0 else 0.0
            med_g_rec = float(np.median(gen_recs)) * 100.0 if len(gen_recs) > 0 else 0.0
            q25_g = float(np.percentile(gen_recs, 25)) * 100.0 if len(gen_recs) > 0 else 0.0
            q75_g = float(np.percentile(gen_recs, 75)) * 100.0 if len(gen_recs) > 0 else 0.0
            pos_frac_g = float(np.mean(np.array(gen_recs) > 0.001)) * 100.0 if len(gen_recs) > 0 else 0.0

            print(f"Layer {l:2d} | {comp.upper():5s} (N={len(valid_pairs)} pairs) => Prompt Mean: {mean_p_rec:5.2f}% (Med: {med_p_rec:5.2f}%) | Gen Mean: {mean_g_rec:5.2f}% (Med: {med_g_rec:5.2f}%, IQR: [{q25_g:.1f}, {q75_g:.1f}]%, Pos: {pos_frac_g:.1f}%)")

            records.append({
                "layer": l,
                "component": comp,
                "total_pairs": len(valid_pairs),
                "valid_pairs_prompt": len(prompt_recs),
                "prompt_mean_ot_recovery": mean_p_rec,
                "prompt_median_ot_recovery": med_p_rec,
                "valid_pairs_gen": len(gen_recs),
                "gen_mean_ot_recovery": mean_g_rec,
                "gen_median_ot_recovery": med_g_rec,
                "gen_iqr_25": q25_g,
                "gen_iqr_75": q75_g,
                "gen_positive_fraction": pos_frac_g
            })

            os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
            pd.DataFrame(records).to_csv(args.output_path, index=False)

    # Save complete pair-level table
    pair_rows = []
    for pid in [p[0] for p in valid_pairs]:
        row = {"pair_id": pid}
        row.update(pair_level_dict[pid])
        # Add primary peak-site columns for convenience
        row["l15_mlp_recovery"] = pair_level_dict[pid].get("gen_rec_L15_mlp", 0.0)
        row["l24_resid_recovery"] = pair_level_dict[pid].get("gen_rec_L24_resid", 0.0)
        if row["l15_mlp_recovery"] is not None and row["l24_resid_recovery"] is not None:
            row["delta_g"] = row["l24_resid_recovery"] - row["l15_mlp_recovery"]
        pair_rows.append(row)

    pair_df = pd.DataFrame(pair_rows)
    pair_output_path = args.output_path.replace(".csv", "_pair_level.csv")
    pair_df.to_csv(pair_output_path, index=False)
    print(f"Saved complete pair-level results to {pair_output_path}")

    print(f"\nFocused 39-pair evaluation completed! Saved summary to {args.output_path}")

if __name__ == "__main__":
    main()
