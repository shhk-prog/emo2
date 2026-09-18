"""
v3/scripts/run_probe_aligned_necessity_sweep.py

Purpose:
Performs a comprehensive layer-by-layer sweep across all 28 layers for
Probe-Aligned Local Necessity and Specificity Controls:
  Intervention Point: Last prompt token (Prompt-time representation)
  Interventions (4 distinct conditions):
    1. probe_direction_removal: h <- h - (h . v_probe) * v_probe (Primary necessity)
    2. random_direction_removal:
       - R_iso: Isotropic random unit vectors
       - R_perp: Probe-orthogonal random unit vectors
    3. neutral_mean_replacement: h <- mean(h_neutral) (Neutralization control)
    4. matched_neutral_replacement: h <- h_neutral (Matched substitution control)
  Metrics:
    - Primary: Joint OT Absolute Effect Delta_necessity = OT(P_peak, P_ablate)
    - Secondary: Neutralization Ratio with safeguard (eps=0.05), EV Shift
    - Specificity: Z-score (with eps_sigma guard), empirical p-value (pseudo-count +1),
                   and Benjamini-Hochberg FDR across pre-specified hypothesis families.
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
from sklearn.linear_model import Ridge
from v2.src.likelihood import generate_81_candidates, compute_expected_va
from v3.src.batch_likelihood import compute_likelihoods_batched
from v3.src.ot_utils import compute_joint_ot_2d, compute_marginal_wasserstein_sum
from v3.src.model_utils import (
    get_model_layers,
    get_component_module,
    get_direction_ablation_hook,
    get_replacement_hook,
    format_base_prompt
)
from v3.scripts.run_aligned_cross_model_patching import get_3way_split

def extract_prompt_representations(model, tokenizer, texts, layer, comp="mlp", device="cuda"):
    """
    Extracts final token representations across texts.
    """
    target_mod = get_component_module(model, layer, comp)
    acts = []
    for text in texts:
        prompt = format_base_prompt(text)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        target_pos = inputs.input_ids.shape[1] - 1
        
        extracted = {}
        def hook(m, inp, out):
            val = out[0] if isinstance(out, tuple) else out
            extracted["val"] = val[0, target_pos, :].detach().float().cpu().numpy()
            
        handle = target_mod.register_forward_hook(hook)
        with torch.no_grad():
            model(**inputs, use_cache=False)
        handle.remove()
        acts.append(extracted["val"])
    return np.array(acts)

def get_probe_direction(train_acts, train_labels, alpha=1.0):
    """
    Fits Ridge regression on train split and extracts the normalized unit direction vector.
    """
    probe = Ridge(alpha=alpha)
    probe.fit(train_acts, train_labels)
    w = probe.coef_.flatten()
    norm_w = np.linalg.norm(w)
    if norm_w > 1e-12:
        v_unit = w / norm_w
    else:
        v_unit = np.zeros_like(w)
    return v_unit

def sample_random_directions(v_probe, n_samples=50, mode="isotropic", seed=42):
    """
    Samples n_samples random unit vectors.
    mode='isotropic': uniform on unit sphere
    mode='orthogonal': Gram-Schmidt orthogonalized against v_probe
    """
    rng = np.random.RandomState(seed)
    dim = len(v_probe)
    rand_dirs = []
    
    v_norm = v_probe / (np.linalg.norm(v_probe) + 1e-12)
    
    for _ in range(n_samples):
        vec = rng.randn(dim)
        if mode == "orthogonal":
            # Gram-Schmidt: remove projection on v_probe
            proj = np.dot(vec, v_norm)
            vec = vec - proj * v_norm
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            rand_dirs.append(vec / norm)
        else:
            rand_dirs.append(vec)
    return np.array(rand_dirs)

def compute_empirical_pvalue_and_zscore(delta_probe, delta_rands, eps_sigma=1e-6):
    """
    Computes Z-score with sigma safeguard and empirical p-value with pseudo-count +1.
    """
    mu_rand = float(np.mean(delta_rands))
    sigma_rand = float(np.std(delta_rands))
    effective_sigma = max(sigma_rand, eps_sigma)
    z_score = (delta_probe - mu_rand) / effective_sigma
    
    # Pseudo-count +1 empirical p-value
    n_rand = len(delta_rands)
    count_geq = np.sum(delta_rands >= delta_probe)
    p_val = float(1 + count_geq) / float(n_rand + 1)
    
    return z_score, p_val, mu_rand, sigma_rand

def evaluate_necessity_for_layer(
    model, tokenizer, valid_pairs, train_df, layer, comp, candidates, va_pairs,
    base_cache=None, n_rand=20,
    device="cuda", normalize_length=True, eps=0.05
):
    target_mod = get_component_module(model, layer, comp)
    
    # 1. Fit probe direction on train split
    train_texts = train_df['text'].tolist()
    train_labels = (train_df['intensity'] == 'peak').astype(float).values
    train_acts = extract_prompt_representations(model, tokenizer, train_texts, layer, comp=comp, device=device)
    v_probe = get_probe_direction(train_acts, train_labels, alpha=1.0)
    
    # Pre-compute neutral mean vector on train neutrals
    train_neut_acts = train_acts[train_labels == 0.0]
    neutral_mean_vec = np.mean(train_neut_acts, axis=0) if len(train_neut_acts) > 0 else np.zeros(train_acts.shape[1])
    
    # Sample random directions for null specificity distributions
    n_samples = n_rand
    rand_iso_dirs = sample_random_directions(v_probe, n_samples=n_samples, mode="isotropic", seed=100 + layer)
    rand_perp_dirs = sample_random_directions(v_probe, n_samples=n_samples, mode="orthogonal", seed=200 + layer)
    
    probe_necessities = []
    probe_neut_ratios = []
    neut_mean_necessities = []
    matched_neut_necessities = []
    
    null_iso_necessities = [[] for _ in range(n_samples)]
    null_perp_necessities = [[] for _ in range(n_samples)]
    
    for idx, (pid, peak_row, neut_row) in enumerate(valid_pairs):
        peak_prompt = format_base_prompt(peak_row['text'])
        neut_prompt = format_base_prompt(neut_row['text'])
        
        # Use cached baseline distributions if available, otherwise compute with batched likelihood
        if base_cache is not None and pid in base_cache:
            p_peak = base_cache[pid]["p_peak"]
            p_neut = base_cache[pid]["p_neut"]
            d_peak_neut = base_cache[pid]["d_peak_neut"]
        else:
            l_peak, _ = compute_likelihoods_batched(model, tokenizer, peak_prompt, candidates, device=device, normalize_length=normalize_length)
            _, _, _, _, p_peak_flat = compute_expected_va(l_peak, va_pairs)
            p_peak = p_peak_flat.reshape((9, 9))
            
            l_neut, _ = compute_likelihoods_batched(model, tokenizer, neut_prompt, candidates, device=device, normalize_length=normalize_length)
            _, _, _, _, p_neut_flat = compute_expected_va(l_neut, va_pairs)
            p_neut = p_neut_flat.reshape((9, 9))
            d_peak_neut = compute_joint_ot_2d(p_peak, p_neut)
        
        peak_inputs = tokenizer(peak_prompt, return_tensors="pt").to(device)
        target_pos = peak_inputs.input_ids.shape[1] - 1
        
        # A. Probe Direction Removal (1 single batched forward pass!)
        hook_probe = get_direction_ablation_hook(v_probe, target_pos)
        handle = target_mod.register_forward_hook(hook_probe)
        l_probe, _ = compute_likelihoods_batched(model, tokenizer, peak_prompt, candidates, device=device, normalize_length=normalize_length)
        handle.remove()
        _, _, _, _, p_ablate_flat = compute_expected_va(l_probe, va_pairs)
        p_probe_ablate = p_ablate_flat.reshape((9, 9))
        
        delta_probe = compute_joint_ot_2d(p_peak, p_probe_ablate)
        probe_necessities.append(delta_probe)
        
        if d_peak_neut >= eps:
            d_ablate_neut = compute_joint_ot_2d(p_probe_ablate, p_neut)
            probe_neut_ratios.append(1.0 - (d_ablate_neut / d_peak_neut))
            
        # B. Neutral Mean Replacement (1 single batched forward pass!)
        hook_mean = get_replacement_hook(neutral_mean_vec, target_pos)
        handle = target_mod.register_forward_hook(hook_mean)
        l_mean, _ = compute_likelihoods_batched(model, tokenizer, peak_prompt, candidates, device=device, normalize_length=normalize_length)
        handle.remove()
        _, _, _, _, p_mean_flat = compute_expected_va(l_mean, va_pairs)
        neut_mean_necessities.append(compute_joint_ot_2d(p_peak, p_mean_flat.reshape((9, 9))))
        
        # C. Matched Neutral Replacement (1 single batched forward pass!)
        neut_act = extract_prompt_representations(model, tokenizer, [neut_row['text']], layer, comp=comp, device=device)[0]
        hook_matched = get_replacement_hook(neut_act, target_pos)
        handle = target_mod.register_forward_hook(hook_matched)
        l_matched, _ = compute_likelihoods_batched(model, tokenizer, peak_prompt, candidates, device=device, normalize_length=normalize_length)
        handle.remove()
        _, _, _, _, p_matched_flat = compute_expected_va(l_matched, va_pairs)
        matched_neut_necessities.append(compute_joint_ot_2d(p_peak, p_matched_flat.reshape((9, 9))))
        
        # D. Random Null Directions (each is 1 batched forward pass!)
        for k in range(n_samples):
            h_iso = get_direction_ablation_hook(rand_iso_dirs[k], target_pos)
            hnd = target_mod.register_forward_hook(h_iso)
            l_r_iso, _ = compute_likelihoods_batched(model, tokenizer, peak_prompt, candidates, device=device, normalize_length=normalize_length)
            hnd.remove()
            _, _, _, _, p_r_flat = compute_expected_va(l_r_iso, va_pairs)
            null_iso_necessities[k].append(compute_joint_ot_2d(p_peak, p_r_flat.reshape((9, 9))))
            
            h_perp = get_direction_ablation_hook(rand_perp_dirs[k], target_pos)
            hnd = target_mod.register_forward_hook(h_perp)
            l_r_perp, _ = compute_likelihoods_batched(model, tokenizer, peak_prompt, candidates, device=device, normalize_length=normalize_length)
            hnd.remove()
            _, _, _, _, p_rp_flat = compute_expected_va(l_r_perp, va_pairs)
            null_perp_necessities[k].append(compute_joint_ot_2d(p_peak, p_rp_flat.reshape((9, 9))))
            
    mean_probe_nec = float(np.mean(probe_necessities))
    mean_probe_ratio = float(np.mean(probe_neut_ratios)) * 100.0 if len(probe_neut_ratios) > 0 else 0.0
    mean_neut_mean_nec = float(np.mean(neut_mean_necessities))
    mean_matched_nec = float(np.mean(matched_neut_necessities))
    
    # Compute mean across pairs for each random direction
    iso_means = [np.mean(vals) for vals in null_iso_necessities if len(vals) > 0]
    perp_means = [np.mean(vals) for vals in null_perp_necessities if len(vals) > 0]
    
    z_iso, p_iso, mu_iso, sig_iso = compute_empirical_pvalue_and_zscore(mean_probe_nec, np.array(iso_means))
    z_perp, p_perp, mu_perp, sig_perp = compute_empirical_pvalue_and_zscore(mean_probe_nec, np.array(perp_means))
    
    return {
        "mean_probe_necessity": mean_probe_nec,
        "mean_neutralization_ratio": mean_probe_ratio,
        "mean_neutral_mean_control": mean_neut_mean_nec,
        "mean_matched_substitution": mean_matched_nec,
        "z_score_iso": z_iso,
        "p_value_iso": p_iso,
        "null_mu_iso": mu_iso,
        "null_sig_iso": sig_iso,
        "z_score_perp": z_perp,
        "p_value_perp": p_perp,
        "null_mu_perp": mu_perp,
        "null_sig_perp": sig_perp
    }

def apply_benjamini_hochberg(p_values, q_threshold=0.05):
    """
    Applies Benjamini-Hochberg FDR correction.
    """
    p_arr = np.asarray(p_values)
    n = len(p_arr)
    sorted_indices = np.argsort(p_arr)
    sorted_p = p_arr[sorted_indices]
    
    q_values = np.zeros(n)
    q_values[-1] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        q_values[i] = min(q_values[i + 1], sorted_p[i] * n / (i + 1))
        
    fdr_significant = q_values <= q_threshold
    orig_q = np.zeros(n)
    orig_q[sorted_indices] = q_values
    orig_sig = np.zeros(n, dtype=bool)
    orig_sig[sorted_indices] = fdr_significant
    return orig_q, orig_sig

def main():
    parser = argparse.ArgumentParser(description="Probe-Aligned Necessity & Specificity Controls Sweep")
    parser.add_argument("--data_path", type=str, default="v3/data/aipsy_strict_expanded.csv")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output_path", type=str, default="v3/results/probe_aligned_necessity_sweep.csv")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--normalize_length", action="store_true", default=True)
    parser.add_argument("--max_pairs", type=int, default=15)
    parser.add_argument("--n_rand", type=int, default=20, help="Random null directions per layer (default: 20)")
    args = parser.parse_args()
    
    print(f"Loading {args.model_name} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16 if "cuda" in args.device else torch.float32
    )
    model.to(args.device)
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
    print(f"Using {len(eval_pairs)} pairs for necessity evaluation.")
    
    candidates, va_pairs = generate_81_candidates()
    layers = get_model_layers(model)
    num_layers = len(layers)
    
    # Pre-compute baseline Peak and Neutral distributions once for all pairs (using batching)
    print("Pre-computing baseline Peak and Neutral candidate distributions (batched)...")
    base_cache = {}
    for pid, peak_row, neut_row in eval_pairs:
        peak_prompt = format_base_prompt(peak_row['text'])
        neut_prompt = format_base_prompt(neut_row['text'])
        
        l_p, _ = compute_likelihoods_batched(model, tokenizer, peak_prompt, candidates, device=args.device, normalize_length=args.normalize_length)
        _, _, _, _, p_p_flat = compute_expected_va(l_p, va_pairs)
        p_p = p_p_flat.reshape((9, 9))
        
        l_n, _ = compute_likelihoods_batched(model, tokenizer, neut_prompt, candidates, device=args.device, normalize_length=args.normalize_length)
        _, _, _, _, p_n_flat = compute_expected_va(l_n, va_pairs)
        p_n = p_n_flat.reshape((9, 9))
        
        base_cache[pid] = {
            "p_peak": p_p,
            "p_neut": p_n,
            "d_peak_neut": compute_joint_ot_2d(p_p, p_n)
        }
    print("Baseline caching completed! Starting layer sweep...")
    
    records = []
    
    for l in range(num_layers):
        print(f"\n>>> Running Necessity Sweep for Layer {l}/{num_layers - 1} <<<")
        for comp in ["mlp", "attn", "resid"]:
            res = evaluate_necessity_for_layer(
                model, tokenizer, eval_pairs, train_df, l, comp, candidates, va_pairs,
                base_cache=base_cache, n_rand=args.n_rand,
                device=args.device, normalize_length=args.normalize_length
            )
            print(f"Layer {l} | {comp.upper()}: Probe Nec={res['mean_probe_necessity']:.4f}, Neut Ratio={res['mean_neutralization_ratio']:.2f}%, Z_perp={res['z_score_perp']:.2f} (p={res['p_value_perp']:.4f}), Matched Sub={res['mean_matched_substitution']:.4f}")
            
            rec = {
                "layer": l,
                "component": comp,
                **res,
                "model_name": args.model_name
            }
            records.append(rec)
            
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        pd.DataFrame(records).to_csv(args.output_path, index=False)
        
    df_results = pd.DataFrame(records)
        
    # Apply BH-FDR across separate 84-test families
    q_iso, sig_iso = apply_benjamini_hochberg(df_results['p_value_iso'].values)
    q_perp, sig_perp = apply_benjamini_hochberg(df_results['p_value_perp'].values)
    df_results['fdr_q_iso'] = q_iso
    df_results['fdr_sig_iso'] = sig_iso
    df_results['fdr_q_perp'] = q_perp
    df_results['fdr_sig_perp'] = sig_perp
    
    # Also apply joint 168-test BH-FDR
    all_p = np.concatenate([df_results['p_value_iso'].values, df_results['p_value_perp'].values])
    q_joint, sig_joint = apply_benjamini_hochberg(all_p)
    n = len(df_results)
    df_results['fdr_q_joint_168_iso'] = q_joint[:n]
    df_results['fdr_q_joint_168_perp'] = q_joint[n:]
    df_results['fdr_sig_joint_168'] = sig_joint[:n] | sig_joint[n:]
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    df_results.to_csv(args.output_path, index=False)
    print(f"\nNecessity sweep complete with BH-FDR correction! Saved to {args.output_path}")

if __name__ == "__main__":
    main()
