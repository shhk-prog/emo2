"""
v3/scripts/run_causal_localization_sweep.py

Purpose:
Performs a comprehensive layer-by-layer sweep across all 28 layers (Layer 0 to 27)
for Qwen2.5-1.5B-Instruct (or LLaMA architectures) testing the core dissociation:
  argmax_l Decodability_l != argmax_l Causal_Influence_l
  rho(Decodability, Causal_Influence) ~= 0

Components evaluated at each layer:
  1. 'mlp': MLP submodule output (prior to residual addition)
  2. 'attn': Projected attention-module output at final prompt token (prior to residual addition)
  3. 'resid': TransformerBlock output / Residual stream (post residual addition)

Metrics:
  - Decodability (D_l): Held-out linear probe R^2 for Peak-vs-Neutral indicator
  - Prompt-time Sufficiency (S_l): Joint 2D Optimal Transport (OT) Recovery towards Peak with safeguard,
                                  plus Marginal Wasserstein Sum recovery and absolute OT displacement.
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
from scipy.stats import pearsonr, spearmanr
from v2.src.likelihood import generate_81_candidates, compute_expected_va
from v3.src.batch_likelihood import compute_likelihoods_batched
from v3.src.ot_utils import (
    compute_joint_ot_2d,
    compute_marginal_wasserstein_sum,
    compute_joint_ot_recovery
)
from v3.src.model_utils import (
    get_model_layers,
    get_component_module,
    get_patch_hook,
    format_base_prompt
)
from v3.scripts.run_aligned_cross_model_patching import get_3way_split

def extract_layer_representations(model, tokenizer, texts, layer, comp="mlp", device="cuda"):
    """
    Extracts final token activation for each text at (layer, comp).
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

def fit_and_eval_probe(train_acts, train_labels, test_acts, test_labels, alpha=1.0):
    probe = Ridge(alpha=alpha)
    probe.fit(train_acts, train_labels)
    preds = probe.predict(test_acts)
    
    ss_res = np.sum((test_labels - preds) ** 2)
    ss_tot = np.sum((test_labels - np.mean(test_labels)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-6 else 0.0
    return float(r2)

def evaluate_layer_causal_effect(
    model, tokenizer, valid_pairs, layer, comp, candidates, va_pairs,
    base_cache=None, device="cuda", normalize_length=True, eps_rec=0.05
):
    """
    Evaluates prompt-time causal patching for a given layer and component.
    """
    target_mod = get_component_module(model, layer, comp)
    
    ot_recoveries = []
    ot_displacements = []
    marg_recoveries = []
    
    for pid, peak_row, neutral_row in valid_pairs:
        peak_prompt = format_base_prompt(peak_row['text'])
        neutral_prompt = format_base_prompt(neutral_row['text'])
        
        # 1 & 2: Use cached baseline distributions
        if base_cache is not None and pid in base_cache:
            p_peak = base_cache[pid]["p_peak"]
            p_neut = base_cache[pid]["p_neut"]
        else:
            l_peak, _ = compute_likelihoods_batched(model, tokenizer, peak_prompt, candidates, device=device, normalize_length=normalize_length)
            _, _, _, _, p_peak_flat = compute_expected_va(l_peak, va_pairs)
            p_peak = p_peak_flat.reshape(9, 9)
            
            l_neut, _ = compute_likelihoods_batched(model, tokenizer, neutral_prompt, candidates, device=device, normalize_length=normalize_length)
            _, _, _, _, p_neut_flat = compute_expected_va(l_neut, va_pairs)
            p_neut = p_neut_flat.reshape(9, 9)
        
        # 3. Extract Peak activation at final prompt token
        extracted_act = {}
        def extract_hook(m, inp, out):
            val = out[0] if isinstance(out, tuple) else out
            extracted_act["val"] = val[0, -1, :].detach().clone()
            
        handle = target_mod.register_forward_hook(extract_hook)
        peak_inputs = tokenizer(peak_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**peak_inputs, use_cache=False)
        handle.remove()
        
        # 4. Patch into Neutral forward pass (1 single batched forward pass!)
        neutral_inputs = tokenizer(neutral_prompt, return_tensors="pt").to(device)
        neutral_last_pos = neutral_inputs.input_ids.shape[1] - 1
        
        patch_handle = target_mod.register_forward_hook(
            get_patch_hook(extracted_act["val"], neutral_last_pos)
        )
        l_patch, _ = compute_likelihoods_batched(model, tokenizer, neutral_prompt, candidates, device=device, normalize_length=normalize_length)
        _, _, _, _, p_patch_flat = compute_expected_va(l_patch, va_pairs)
        patch_handle.remove()
        p_patch = p_patch_flat.reshape(9, 9)
        
        # 5. Joint OT Recovery with safeguard
        rec_ot, d_patch_peak, d_neut_peak, is_valid = compute_joint_ot_recovery(
            p_patch, p_peak, p_neut, eps_rec=eps_rec
        )
        ot_displacements.append(d_patch_peak)
        if is_valid and rec_ot is not None:
            ot_recoveries.append(rec_ot)
            
        # 6. Marginal Wasserstein Sum recovery
        d_marg_np = compute_marginal_wasserstein_sum(p_neut, p_peak)
        d_marg_pp = compute_marginal_wasserstein_sum(p_patch, p_peak)
        if d_marg_np >= eps_rec:
            marg_recoveries.append(1.0 - (d_marg_pp / d_marg_np))
            
    mean_ot_rec = float(np.mean(ot_recoveries)) * 100.0 if len(ot_recoveries) > 0 else 0.0
    median_ot_rec = float(np.median(ot_recoveries)) * 100.0 if len(ot_recoveries) > 0 else 0.0
    mean_disp = float(np.mean(ot_displacements)) if len(ot_displacements) > 0 else 0.0
    mean_marg_rec = float(np.mean(marg_recoveries)) * 100.0 if len(marg_recoveries) > 0 else 0.0
    valid_pair_count = len(ot_recoveries)
    
    return {
        "mean_ot_recovery": mean_ot_rec,
        "median_ot_recovery": median_ot_rec,
        "mean_ot_displacement": mean_disp,
        "mean_marg_recovery": mean_marg_rec,
        "valid_pairs": valid_pair_count
    }

def main():
    parser = argparse.ArgumentParser(description="Full Layer Causal Localization Sweep with Joint OT")
    parser.add_argument("--data_path", type=str, default="v3/data/aipsy_strict_expanded.csv")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output_path", type=str, default="v3/results/causal_localization_sweep_joint_ot.csv")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--normalize_length", action="store_true", default=True)
    parser.add_argument("--max_pairs", type=int, default=15)
    parser.add_argument("--eps_rec", type=float, default=0.05)
    args = parser.parse_args()
    
    print(f"Loading {args.model_name} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16 if "cuda" in args.device else torch.float32,
        device_map="auto" if args.device == "cuda" else None
    )
    model.eval()
    
    df = pd.read_csv(args.data_path)
    train_df, val_df, test_df = get_3way_split(df)
    train_texts = train_df['text'].tolist()
    train_labels = (train_df['intensity'] == 'peak').astype(float).values
    test_texts = test_df['text'].tolist()
    test_labels = (test_df['intensity'] == 'peak').astype(float).values
    
    test_peaks = test_df[test_df['intensity'] == 'peak'].copy()
    test_neutrals = test_df[test_df['intensity'] == 'neutral'].copy()
    
    valid_pairs = []
    for pid in test_peaks['pair_id'].unique():
        p_row = test_peaks[test_peaks['pair_id'] == pid]
        n_row = test_neutrals[test_neutrals['pair_id'] == pid]
        if len(p_row) > 0 and len(n_row) > 0:
            valid_pairs.append((pid, p_row.iloc[0], n_row.iloc[0]))
            
    eval_pairs = valid_pairs[:args.max_pairs]
    print(f"Found {len(valid_pairs)} test pairs, evaluating {len(eval_pairs)} pairs across all layers.")
    
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
            "p_neut": p_n
        }
    print("Baseline caching completed! Starting layer sweep...")
    
    records = []
    
    for l in range(num_layers):
        print(f"\n>>> Running Layer {l}/{num_layers - 1} <<<")
        for comp in ["mlp", "attn", "resid"]:
            # 1. Decodability (Probe R^2)
            train_acts = extract_layer_representations(model, tokenizer, train_texts, l, comp=comp, device=args.device)
            test_acts = extract_layer_representations(model, tokenizer, test_texts, l, comp=comp, device=args.device)
            r2 = fit_and_eval_probe(train_acts, train_labels, test_acts, test_labels, alpha=1.0)
            
            # 2. Causal Recovery (Prompt-time Joint OT)
            res = evaluate_layer_causal_effect(
                model, tokenizer, eval_pairs, l, comp, candidates, va_pairs,
                base_cache=base_cache,
                device=args.device, normalize_length=args.normalize_length, eps_rec=args.eps_rec
            )
            print(f"Layer {l} | {comp.upper()}: Probe R^2={r2:.4f}, Joint OT Rec={res['mean_ot_recovery']:.2f}% (Med: {res['median_ot_recovery']:.2f}%), Disp={res['mean_ot_displacement']:.4f}")
            
            rec = {
                "layer": l,
                "component": comp,
                "probe_r2": r2,
                **res,
                "model_name": args.model_name
            }
            records.append(rec)
            
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        pd.DataFrame(records).to_csv(args.output_path, index=False)
        
    df_results = pd.DataFrame(records)
    print("\n--- Full Layer Sweep Summary ---")
    for comp in ["mlp", "attn", "resid"]:
        cdf = df_results[df_results['component'] == comp]
        corr_s, p_s = spearmanr(cdf['probe_r2'], cdf['mean_ot_recovery'])
        print(f"Component {comp.upper()}: max Probe R^2={cdf['probe_r2'].max():.4f}, max OT Rec={cdf['mean_ot_recovery'].max():.2f}%, Spearman rho={corr_s:.4f} (p={p_s:.4f})")
        
    print(f"\nCompleted! Saved to {args.output_path}")

if __name__ == "__main__":
    main()
