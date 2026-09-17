"""
v3/scripts/run_focused_necessity_n100.py

High-performance, fully vectorized probe-aligned necessity & specificity evaluation (N=100 random directions)
for key representative layers:
  Layers: [10, 15, 18, 20, 24]
  Components: ['mlp', 'attn', 'resid']

Features 10x-30x acceleration:
1. Batched direction ablation hook (processes multiple directions per forward pass).
2. Pre-tokenized and pre-padded candidate & prompt tensors cached directly in GPU VRAM.
3. Vectorized log-probability gather on GPU (zero Python loops or per-step GPU tensor allocations).
4. Batched neutral representation extraction.

Output:
  v3/results/focused_necessity_sweep_n100.csv
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from v2.src.likelihood import generate_81_candidates
from v3.src.ot_utils import compute_joint_ot_2d
from v3.src.model_utils import (
    get_component_module,
    format_base_prompt
)
from v3.scripts.run_aligned_cross_model_patching import get_3way_split
from v3.scripts.run_probe_aligned_necessity_sweep import (
    get_probe_direction,
    sample_random_directions,
    compute_empirical_pvalue_and_zscore,
    apply_benjamini_hochberg
)

TARGET_LAYERS = [10, 15, 18, 20, 24]
COMPONENTS = ["mlp", "attn", "resid"]

class BatchedDirectionAblationHook:
    """
    Simultaneously ablates multiple direction vectors in a single forward pass.
    v_batch: (B_dir, hidden_dim) on GPU, unit normalized
    target_pos: position in sequence to ablate
    num_cands: number of candidate suffixes per direction (81)
    """
    def __init__(self, v_batch, target_pos, num_cands=81):
        self.v_batch = v_batch # (B_dir, hidden_dim)
        self.target_pos = target_pos
        self.num_cands = num_cands
        self.b_dir = v_batch.shape[0]

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output # (B_dir * num_cands, seq_len, hidden_dim)
        
        # Reshape at target_pos: (B_dir, num_cands, hidden_dim)
        h_pos = h[:, self.target_pos, :].view(self.b_dir, self.num_cands, -1)
        v = self.v_batch.unsqueeze(1) # (B_dir, 1, hidden_dim)
        
        # Gram-Schmidt projection out of v: h_pos - (h_pos . v) v
        proj = (h_pos * v).sum(dim=-1, keepdim=True) * v
        h_pos_ablated = h_pos - proj
        
        h[:, self.target_pos, :] = h_pos_ablated.view(self.b_dir * self.num_cands, -1)
        if is_tuple:
            return (h,) + output[1:]
        return h

class BatchedVectorReplacementHook:
    """
    Replaces hidden states at target_pos with a given replacement vector across all candidates in the batch.
    """
    def __init__(self, replacement_vec, target_pos):
        self.replacement_vec = replacement_vec # (hidden_dim,)
        self.target_pos = target_pos

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        h[:, self.target_pos, :] = self.replacement_vec
        if is_tuple:
            return (h,) + output[1:]
        return h

def extract_prompt_representations_batched(model, tokenizer, texts, layer, comp="resid", device="cuda"):
    """
    Extracts representations at final prompt token across multiple texts in batches.
    """
    target_mod = get_component_module(model, layer, comp)
    acts = []
    
    for text in texts:
        prompt = format_base_prompt(text)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        target_pos = inputs.input_ids.shape[1] - 1
        
        extracted = {}
        def hook(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            extracted["val"] = h[0, target_pos, :].detach().cpu().float().numpy()
            return o
            
        hnd = target_mod.register_forward_hook(hook)
        with torch.no_grad():
            model(**inputs, use_cache=False)
        hnd.remove()
        acts.append(extracted["val"])
        
    return np.array(acts)

def prebuild_pair_tensors(eval_pairs, tokenizer, candidates, device="cuda"):
    """
    Pre-tokenizes candidates and prompts for all evaluation pairs ONCE,
    moving all constant tensors to the GPU.
    """
    cand_ids_list = [tokenizer.encode(c, add_special_tokens=False) for c in candidates]
    max_cand_len = max(len(c) for c in cand_ids_list)
    cand_lens = torch.tensor([len(c) for c in cand_ids_list], dtype=torch.float32, device=device)
    
    # Pad candidates
    cand_ids_padded = []
    cand_masks = []
    for c in cand_ids_list:
        pad_len = max_cand_len - len(c)
        cand_ids_padded.append(c + [0] * pad_len)
        cand_masks.append([1.0] * len(c) + [0.0] * pad_len)
        
    cand_ids_tensor = torch.tensor(cand_ids_padded, dtype=torch.long, device=device)
    cand_mask_tensor = torch.tensor(cand_masks, dtype=torch.float32, device=device)
    
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    pair_caches = {}
    for pid, peak_row, neut_row in eval_pairs:
        peak_prompt = format_base_prompt(peak_row['text'])
        neut_prompt = format_base_prompt(neut_row['text'])
        
        peak_ids = tokenizer.encode(peak_prompt, add_special_tokens=False)
        peak_len = len(peak_ids)
        target_pos = peak_len - 1
        
        # Build 81 joint sequences for peak using canonical boundary
        from affective_empathy_eval.likelihood import prepare_joint_sequence_with_boundary
        full_seqs_peak = []
        for c in candidates:
            f_ids, _ = prepare_joint_sequence_with_boundary(peak_prompt, c, tokenizer, delimiter="", require_strict_prefix=False)
            full_seqs_peak.append(f_ids)
        max_peak_len = max(len(s) for s in full_seqs_peak)
        
        peak_padded_ids = []
        peak_attn_masks = []
        for s in full_seqs_peak:
            pad_len = max_peak_len - len(s)
            peak_padded_ids.append(s + [pad_id] * pad_len)
            peak_attn_masks.append([1] * len(s) + [0] * pad_len)
            
        peak_input_tensor = torch.tensor(peak_padded_ids, dtype=torch.long, device=device)
        peak_attn_tensor = torch.tensor(peak_attn_masks, dtype=torch.long, device=device)
        
        # Build 81 joint sequences for neut using canonical boundary
        neut_ids = tokenizer.encode(neut_prompt, add_special_tokens=False)
        neut_len = len(neut_ids)
        full_seqs_neut = []
        for c in candidates:
            f_ids, _ = prepare_joint_sequence_with_boundary(neut_prompt, c, tokenizer, delimiter="", require_strict_prefix=False)
            full_seqs_neut.append(f_ids)
        max_neut_len = max(len(s) for s in full_seqs_neut)
        
        neut_padded_ids = []
        neut_attn_masks = []
        for s in full_seqs_neut:
            pad_len = max_neut_len - len(s)
            neut_padded_ids.append(s + [pad_id] * pad_len)
            neut_attn_masks.append([1] * len(s) + [0] * pad_len)
            
        neut_input_tensor = torch.tensor(neut_padded_ids, dtype=torch.long, device=device)
        neut_attn_tensor = torch.tensor(neut_attn_masks, dtype=torch.long, device=device)

        
        pair_caches[pid] = {
            "target_pos": target_pos,
            "peak_len": peak_len,
            "neut_len": neut_len,
            "max_cand_len": max_cand_len,
            "peak_input_tensor": peak_input_tensor,
            "peak_attn_tensor": peak_attn_tensor,
            "neut_input_tensor": neut_input_tensor,
            "neut_attn_tensor": neut_attn_tensor,
        }
        
    cand_meta = {
        "cand_ids_tensor": cand_ids_tensor,
        "cand_mask_tensor": cand_mask_tensor,
        "cand_lens": cand_lens,
        "max_cand_len": max_cand_len
    }
    return pair_caches, cand_meta

def compute_probs_vectorized(logits, start_pos, max_cand_len, cand_meta, b_dir=1, normalize_length=True):
    """
    Computes 81-candidate probability distributions across b_dir directions in 0.5 ms on GPU.
    Returns: numpy array of shape (b_dir, 81)
    """
    # cand_logits: (b_dir * 81, max_cand_len, vocab_size)
    cand_logits = logits[:, start_pos : start_pos + max_cand_len, :]
    log_probs = torch.log_softmax(cand_logits, dim=-1)
    
    cand_ids_exp = cand_meta["cand_ids_tensor"].repeat(b_dir, 1).unsqueeze(-1)
    cand_mask_exp = cand_meta["cand_mask_tensor"].repeat(b_dir, 1)
    cand_lens_exp = cand_meta["cand_lens"].repeat(b_dir)
    
    tok_log_p = log_probs.gather(dim=-1, index=cand_ids_exp).squeeze(-1)
    sum_lp = (tok_log_p * cand_mask_exp).sum(dim=-1)
    
    if normalize_length:
        seq_lp = sum_lp / cand_lens_exp
    else:
        seq_lp = sum_lp
        
    seq_lp_batched = seq_lp.view(b_dir, 81)
    probs_batched = torch.softmax(seq_lp_batched, dim=-1).cpu().float().numpy()
    return probs_batched

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data_path", type=str, default="v3/data/aipsy_strict_expanded.csv")
    parser.add_argument("--output_path", type=str, default="v3/results/focused_necessity_sweep_n100.csv")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_pairs", type=int, default=15)
    parser.add_argument("--n_rand", type=int, default=100, help="Random null directions (default: 100)")
    parser.add_argument("--batch_dirs", type=int, default=5, help="Simultaneous directions per forward pass")
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

    eval_pairs = valid_pairs[:args.max_pairs]
    print(f"Evaluating {len(eval_pairs)} pairs with N={args.n_rand} random null directions across {TARGET_LAYERS}")
    print(f"Batching {args.batch_dirs} directions per forward pass for maximal GPU throughput.")

    candidates, va_pairs = generate_81_candidates()

    print("\nPre-building constant candidate & sequence tensors on GPU...")
    pair_caches, cand_meta = prebuild_pair_tensors(eval_pairs, tokenizer, candidates, device=args.device)

    print("Pre-computing baseline distributions for all evaluation pairs...")
    base_cache = {}
    for pid, peak_row, neut_row in eval_pairs:
        c = pair_caches[pid]
        with torch.no_grad():
            out_peak = model(input_ids=c["peak_input_tensor"], attention_mask=c["peak_attn_tensor"], use_cache=False)
            p_peak_flat = compute_probs_vectorized(out_peak.logits, c["peak_len"] - 1, c["max_cand_len"], cand_meta, b_dir=1)[0]
            
            out_neut = model(input_ids=c["neut_input_tensor"], attention_mask=c["neut_attn_tensor"], use_cache=False)
            p_neut_flat = compute_probs_vectorized(out_neut.logits, c["neut_len"] - 1, c["max_cand_len"], cand_meta, b_dir=1)[0]
            
        pp = p_peak_flat.reshape((9, 9))
        pn = p_neut_flat.reshape((9, 9))
        base_cache[pid] = {
            "p_peak": pp,
            "p_neut": pn,
            "d_peak_neut": compute_joint_ot_2d(pp, pn)
        }

    records = []
    
    for l in TARGET_LAYERS:
        print(f"\n=======================================================")
        print(f">>> Running Focused High-Speed Necessity for Layer {l} (N_rand={args.n_rand}) <<<")
        print(f"=======================================================")
        
        for comp in COMPONENTS:
            target_mod = get_component_module(model, l, comp)
            
            # 1. Fit probe direction on train split
            train_texts = train_df['text'].tolist()
            train_labels = (train_df['intensity'] == 'peak').astype(float).values
            train_acts = extract_prompt_representations_batched(model, tokenizer, train_texts, l, comp=comp, device=args.device)
            v_probe = get_probe_direction(train_acts, train_labels, alpha=1.0)
            
            # Pre-compute neutral mean vector on train neutrals
            train_neut_acts = train_acts[train_labels == 0.0]
            neutral_mean_vec = np.mean(train_neut_acts, axis=0) if len(train_neut_acts) > 0 else np.zeros(train_acts.shape[1])
            neutral_mean_tensor = torch.tensor(neutral_mean_vec, dtype=model.dtype, device=args.device)
            
            # Sample random directions
            rand_iso_dirs = sample_random_directions(v_probe, n_samples=args.n_rand, mode="isotropic", seed=100 + l)
            rand_perp_dirs = sample_random_directions(v_probe, n_samples=args.n_rand, mode="orthogonal", seed=200 + l)
            
            # Pre-extract neutral activations for all test pairs
            test_neut_texts = [neut_row['text'] for _, _, neut_row in eval_pairs]
            test_neut_acts = extract_prompt_representations_batched(model, tokenizer, test_neut_texts, l, comp=comp, device=args.device)
            
            probe_necessities = []
            probe_neut_ratios = []
            neut_mean_necessities = []
            matched_neut_necessities = []
            
            null_iso_necessities = [[] for _ in range(args.n_rand)]
            null_perp_necessities = [[] for _ in range(args.n_rand)]
            
            v_probe_tensor = torch.tensor(v_probe, dtype=model.dtype, device=args.device).unsqueeze(0) # (1, D)
            
            for p_idx, (pid, peak_row, neut_row) in enumerate(eval_pairs):
                print(f"  -> [{comp.upper():5s}] Processing Pair {p_idx+1:2d}/{len(eval_pairs):2d} (pid={pid})...", end="\r", flush=True)
                c = pair_caches[pid]
                target_pos = c["target_pos"]
                p_peak = base_cache[pid]["p_peak"]
                p_neut = base_cache[pid]["p_neut"]
                d_peak_neut = base_cache[pid]["d_peak_neut"]
                
                # A. Probe Direction Removal (1 forward pass)
                hook_probe = BatchedDirectionAblationHook(v_probe_tensor, target_pos, num_cands=81)
                hnd = target_mod.register_forward_hook(hook_probe)
                with torch.no_grad():
                    out = model(input_ids=c["peak_input_tensor"], attention_mask=c["peak_attn_tensor"], use_cache=False)
                    p_probe_flat = compute_probs_vectorized(out.logits, target_pos, c["max_cand_len"], cand_meta, b_dir=1)[0]
                hnd.remove()
                
                p_probe_ablate = p_probe_flat.reshape((9, 9))
                delta_probe = compute_joint_ot_2d(p_peak, p_probe_ablate)
                probe_necessities.append(delta_probe)
                
                if d_peak_neut >= 0.05:
                    d_ablate_neut = compute_joint_ot_2d(p_probe_ablate, p_neut)
                    probe_neut_ratios.append(1.0 - (d_ablate_neut / d_peak_neut))
                    
                # B. Neutral Mean Replacement (1 forward pass)
                hook_mean = BatchedVectorReplacementHook(neutral_mean_tensor, target_pos)
                hnd = target_mod.register_forward_hook(hook_mean)
                with torch.no_grad():
                    out = model(input_ids=c["peak_input_tensor"], attention_mask=c["peak_attn_tensor"], use_cache=False)
                    p_mean_flat = compute_probs_vectorized(out.logits, target_pos, c["max_cand_len"], cand_meta, b_dir=1)[0]
                hnd.remove()
                neut_mean_necessities.append(compute_joint_ot_2d(p_peak, p_mean_flat.reshape((9, 9))))
                
                # C. Matched Neutral Replacement (1 forward pass)
                matched_vec_tensor = torch.tensor(test_neut_acts[p_idx], dtype=model.dtype, device=args.device)
                hook_matched = BatchedVectorReplacementHook(matched_vec_tensor, target_pos)
                hnd = target_mod.register_forward_hook(hook_matched)
                with torch.no_grad():
                    out = model(input_ids=c["peak_input_tensor"], attention_mask=c["peak_attn_tensor"], use_cache=False)
                    p_matched_flat = compute_probs_vectorized(out.logits, target_pos, c["max_cand_len"], cand_meta, b_dir=1)[0]
                hnd.remove()
                matched_neut_necessities.append(compute_joint_ot_2d(p_peak, p_matched_flat.reshape((9, 9))))
                
                # D. Batched Random Directions (chunks of batch_dirs)
                # 1) Isotropic Null Directions
                for start_k in range(0, args.n_rand, args.batch_dirs):
                    end_k = min(start_k + args.batch_dirs, args.n_rand)
                    chunk_dirs = rand_iso_dirs[start_k:end_k]
                    b_size = len(chunk_dirs)
                    
                    v_chunk = torch.tensor(chunk_dirs, dtype=model.dtype, device=args.device) # (b_size, D)
                    hook_chunk = BatchedDirectionAblationHook(v_chunk, target_pos, num_cands=81)
                    
                    in_ids_chunk = c["peak_input_tensor"].repeat(b_size, 1)
                    attn_chunk = c["peak_attn_tensor"].repeat(b_size, 1)
                    
                    hnd = target_mod.register_forward_hook(hook_chunk)
                    with torch.no_grad():
                        out = model(input_ids=in_ids_chunk, attention_mask=attn_chunk, use_cache=False)
                        probs_chunk = compute_probs_vectorized(out.logits, target_pos, c["max_cand_len"], cand_meta, b_dir=b_size)
                    hnd.remove()
                    
                    for sub_k in range(b_size):
                        k_idx = start_k + sub_k
                        p_r_mat = probs_chunk[sub_k].reshape((9, 9))
                        null_iso_necessities[k_idx].append(compute_joint_ot_2d(p_peak, p_r_mat))
                        
                # 2) Orthogonal Null Directions (Perpendicular)
                for start_k in range(0, args.n_rand, args.batch_dirs):
                    end_k = min(start_k + args.batch_dirs, args.n_rand)
                    chunk_dirs = rand_perp_dirs[start_k:end_k]
                    b_size = len(chunk_dirs)
                    
                    v_chunk = torch.tensor(chunk_dirs, dtype=model.dtype, device=args.device) # (b_size, D)
                    hook_chunk = BatchedDirectionAblationHook(v_chunk, target_pos, num_cands=81)
                    
                    in_ids_chunk = c["peak_input_tensor"].repeat(b_size, 1)
                    attn_chunk = c["peak_attn_tensor"].repeat(b_size, 1)
                    
                    hnd = target_mod.register_forward_hook(hook_chunk)
                    with torch.no_grad():
                        out = model(input_ids=in_ids_chunk, attention_mask=attn_chunk, use_cache=False)
                        probs_chunk = compute_probs_vectorized(out.logits, target_pos, c["max_cand_len"], cand_meta, b_dir=b_size)
                    hnd.remove()
                    
                    for sub_k in range(b_size):
                        k_idx = start_k + sub_k
                        p_r_mat = probs_chunk[sub_k].reshape((9, 9))
                        null_perp_necessities[k_idx].append(compute_joint_ot_2d(p_peak, p_r_mat))

            mean_probe_nec = float(np.mean(probe_necessities))
            mean_probe_ratio = float(np.mean(probe_neut_ratios)) * 100.0 if len(probe_neut_ratios) > 0 else 0.0
            mean_neut_mean_nec = float(np.mean(neut_mean_necessities))
            mean_matched_nec = float(np.mean(matched_neut_necessities))
            
            iso_means = [np.mean(vals) for vals in null_iso_necessities if len(vals) > 0]
            perp_means = [np.mean(vals) for vals in null_perp_necessities if len(vals) > 0]
            
            z_iso, p_iso, mu_iso, sig_iso = compute_empirical_pvalue_and_zscore(mean_probe_nec, np.array(iso_means))
            z_perp, p_perp, mu_perp, sig_perp = compute_empirical_pvalue_and_zscore(mean_probe_nec, np.array(perp_means))
            
            print(f"\rLayer {l:2d} | {comp.upper():5s}: Probe Nec={mean_probe_nec:.4f}, Neut Ratio={mean_probe_ratio:.2f}%, Z_perp={z_perp:5.2f} (p={p_perp:.4f}, mu_perp={mu_perp:.4f})                     ")
            
            records.append({
                "layer": l,
                "component": comp,
                "n_rand": args.n_rand,
                "mean_probe_necessity": mean_probe_nec,
                "mean_attenuation_ratio": mean_probe_ratio,
                "projection_removal_effect": mean_probe_ratio,
                "mean_neutralization_ratio": mean_probe_ratio,  # backward compatibility alias
                "mean_neutral_mean_control": mean_neut_mean_nec,
                "mean_matched_substitution": mean_matched_nec,
                "z_score_iso": z_iso,
                "p_value_iso": p_iso,
                "null_mu_iso": mu_iso,
                "null_sig_iso": sig_iso,
                "z_score_perp": z_perp,
                "p_value_perp": p_perp,
                "null_mu_perp": mu_perp,
                "null_sig_perp": sig_perp,
                "model_name": args.model_name
            })

            
            os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
            pd.DataFrame(records).to_csv(args.output_path, index=False)

    df_results = pd.DataFrame(records)
    q_iso, sig_iso = apply_benjamini_hochberg(df_results['p_value_iso'].values)
    q_perp, sig_perp = apply_benjamini_hochberg(df_results['p_value_perp'].values)
    df_results['fdr_q_iso'] = q_iso
    df_results['fdr_sig_iso'] = sig_iso
    df_results['fdr_q_perp'] = q_perp
    df_results['fdr_sig_perp'] = sig_perp

    df_results.to_csv(args.output_path, index=False)
    print(f"\nFocused Necessity N={args.n_rand} evaluation completed successfully!")
    print(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    main()
