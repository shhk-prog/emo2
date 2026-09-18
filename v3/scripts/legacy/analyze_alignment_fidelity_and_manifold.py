"""
analyze_alignment_fidelity_and_manifold.py

Purpose:
Addresses Reviewer Critiques 2, 3, and 4:
1. Full-State Reconstruction (Exp B):
   - R^2_activation (dimension-averaged)
   - Linear CKA (Centered Kernel Alignment)
   - Pair Retrieval Top-1 Accuracy (Nearest Neighbor)
2. Empirical Manifold Test (Exp C):
   - Empirical reference distribution of Mahalanobis distance D_M(h_I) for natural Instruct
   - Percentile rank of Aligned Base activations within the natural Instruct distribution
   - Two-sample linear classifier (Natural Instruct vs Aligned Base) test AUC
   - Cosine controls (Matched vs Unmatched same-valence vs Unmatched diff-valence vs Natural-Natural)
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from v3.scripts.run_aligned_cross_model_patching import get_3way_split

def format_prompt(text):
    return f"Read the following text and report your affective state.\n\nText: {text}\n\nRespond strictly in JSON format with 'valence' and 'arousal' keys (1-9)."

def linear_cka(X, Y):
    """
    Computes Linear CKA between matrix X (n x d) and Y (n x d).
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    dot_X = X @ X.T
    dot_Y = Y @ Y.T
    
    # HSIC
    hsic_xy = np.sum(dot_X * dot_Y)
    hsic_xx = np.sum(dot_X * dot_X)
    hsic_yy = np.sum(dot_Y * dot_Y)
    
    if hsic_xx * hsic_yy == 0:
        return 0.0
    return hsic_xy / (np.sqrt(hsic_xx) * np.sqrt(hsic_yy))

def compute_retrieval_accuracy(aligned_acts, true_acts):
    """
    Computes Top-1 retrieval accuracy using cosine similarity.
    For each aligned_acts[i], finds argmax_j cos(aligned_acts[i], true_acts[j]).
    Success if argmax_j == i.
    """
    # Normalize vectors
    norm_aligned = aligned_acts / (np.linalg.norm(aligned_acts, axis=1, keepdims=True) + 1e-9)
    norm_true = true_acts / (np.linalg.norm(true_acts, axis=1, keepdims=True) + 1e-9)
    
    sim_matrix = norm_aligned @ norm_true.T # (n, n)
    top1_preds = np.argmax(sim_matrix, axis=1)
    correct = np.sum(top1_preds == np.arange(len(aligned_acts)))
    return correct / len(aligned_acts)

def extract_layer_activations(model, tokenizer, texts, layer, comp="mlp", device="cuda"):
    """
    Extracts last token activation for a list of texts at specified layer.
    """
    acts = []
    for text in texts:
        prompt = f"Read the following text and report your affective state.\n\nText: {text}\n\nRespond strictly in JSON format with 'valence' and 'arousal' keys (1-9)."
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        target_pos = inputs.input_ids.shape[1] - 1
        
        extracted = {}
        def hook(m, inp, out):
            val = out[0] if isinstance(out, tuple) else out
            extracted["val"] = val[0, target_pos, :].detach().cpu().numpy()
            
        if comp == "mlp":
            handle = model.model.layers[layer].mlp.register_forward_hook(hook)
        else:
            handle = model.model.layers[layer].register_forward_hook(hook)
            
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        acts.append(extracted["val"])
        
    return np.array(acts)

def run_alignment_and_manifold_diagnostics(
    base_model, instruct_model, tokenizer,
    dev_df, test_df,
    layer=15, comp="mlp", alpha=1.0, device="cuda"
):
    print(f"\n=======================================================")
    print(f"Running Alignment Fidelity & Manifold Analysis (Layer {layer}, {comp})")
    print(f"=======================================================")
    
    # 1. Extract Dev Activations
    print("Extracting Alignment-dev activations...")
    dev_texts = dev_df['text'].tolist()
    h_B_dev = extract_layer_activations(base_model, tokenizer, dev_texts, layer, comp, device)
    h_I_dev = extract_layer_activations(instruct_model, tokenizer, dev_texts, layer, comp, device)
    
    # Fit Ridge Alignment on Dev
    print(f"Fitting Ridge alignment (alpha={alpha}) on Dev (N={len(dev_df)})...")
    mapper = Ridge(alpha=alpha, fit_intercept=True)
    mapper.fit(h_B_dev, h_I_dev)
    
    # Estimate Instruct Covariance on Dev for Mahalanobis
    lw = LedoitWolf()
    lw.fit(h_I_dev)
    mu_I = lw.location_
    precision_I = lw.precision_
    
    # 2. Extract Held-Out Test Activations
    print("Extracting Held-out test activations...")
    test_texts = test_df['text'].tolist()
    h_B_test = extract_layer_activations(base_model, tokenizer, test_texts, layer, comp, device)
    h_I_test = extract_layer_activations(instruct_model, tokenizer, test_texts, layer, comp, device)
    
    # Predict Aligned Instruct Activations on Test
    h_aligned_test = mapper.predict(h_B_test)
    
    # ---------------------------------------------------------
    # Experiment B: Full-State Reconstruction Analysis
    # ---------------------------------------------------------
    # 1. Dimension-averaged R^2_activation
    numer = np.sum((h_aligned_test - h_I_test) ** 2, axis=0) # per dimension
    denom = np.sum((h_I_test - np.mean(h_I_test, axis=0)) ** 2, axis=0)
    valid_dims = denom > 1e-6
    r2_per_dim = 1.0 - (numer[valid_dims] / denom[valid_dims])
    r2_activation_mean = np.mean(r2_per_dim)
    r2_activation_median = np.median(r2_per_dim)
    
    # 2. Linear CKA
    cka_score = linear_cka(h_aligned_test, h_I_test)
    
    # 3. Retrieval Accuracy
    retrieval_acc = compute_retrieval_accuracy(h_aligned_test, h_I_test)
    
    print("\n--- Experiment B: Full-State Reconstruction Results ---")
    print(f"R^2_activation (Mean across dimensions):   {r2_activation_mean:.4f}")
    print(f"R^2_activation (Median across dimensions): {r2_activation_median:.4f}")
    print(f"Linear CKA:                                {cka_score:.4f}")
    print(f"Pair Retrieval Top-1 Accuracy:            {retrieval_acc * 100:.2f}%")
    
    # ---------------------------------------------------------
    # Experiment C: Empirical Manifold Test
    # ---------------------------------------------------------
    def calc_dm(acts):
        diff = acts - mu_I
        dist_sq = np.sum((diff @ precision_I) * diff, axis=1)
        return np.sqrt(np.maximum(dist_sq, 0))
        
    dm_natural_instruct = calc_dm(h_I_test)
    dm_aligned_base = calc_dm(h_aligned_test)
    dm_raw_base = calc_dm(h_B_test)
    
    # Percentiles of natural instruct
    pcts = [5, 25, 50, 75, 95]
    nat_pct_vals = np.percentile(dm_natural_instruct, pcts)
    
    # Percentile ranks of aligned base
    aligned_median_dm = np.median(dm_aligned_base)
    rank_in_natural = np.mean(dm_natural_instruct < aligned_median_dm) * 100
    
    print("\n--- Experiment C: Empirical Manifold Test Results ---")
    print("Natural Instruct D_M Percentiles:")
    for p, v in zip(pcts, nat_pct_vals):
        print(f"  {p}th: {v:.2f}")
    print(f"Aligned Base Median D_M: {aligned_median_dm:.2f} (Rank in Natural Instruct: {rank_in_natural:.1f}%)")
    print(f"Raw Base Median D_M:     {np.median(dm_raw_base):.2f}")
    
    # Two-Sample Classifier: Can a linear classifier distinguish Natural Instruct from Aligned Base?
    X_cls = np.vstack([h_I_test, h_aligned_test])
    y_cls = np.array([0] * len(h_I_test) + [1] * len(h_aligned_test))
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    for train_idx, val_idx in skf.split(X_cls, y_cls):
        clf = LogisticRegression(C=1.0, max_iter=1000)
        clf.fit(X_cls[train_idx], y_cls[train_idx])
        preds = clf.predict_proba(X_cls[val_idx])[:, 1]
        auc_scores.append(roc_auc_score(y_cls[val_idx], preds))
    two_sample_auc = np.mean(auc_scores)
    print(f"Two-sample Classifier (Natural vs Aligned) 5-fold AUC: {two_sample_auc:.4f} (Ideal indistinguishable = 0.50)")
    
    # Cosine Controls
    def cosine_sim(a, b):
        norm_a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        norm_b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        return np.sum(norm_a * norm_b, axis=1)
        
    matched_cos = cosine_sim(h_aligned_test, h_I_test)
    
    # Unmatched pairs
    shuffled_idx = np.random.RandomState(42).permutation(len(h_I_test))
    unmatched_cos = cosine_sim(h_aligned_test, h_I_test[shuffled_idx])
    
    # Natural Instruct - Instruct pairs
    shuffled_idx2 = np.random.RandomState(99).permutation(len(h_I_test))
    natural_pair_cos = cosine_sim(h_I_test, h_I_test[shuffled_idx2])
    
    print("\nCosine Similarity Controls:")
    print(f"  Matched Aligned Base -> Instruct:      {np.mean(matched_cos):.4f}")
    print(f"  Unmatched Aligned Base -> Instruct:    {np.mean(unmatched_cos):.4f}")
    print(f"  Natural Instruct - Instruct Pairs:     {np.mean(natural_pair_cos):.4f}")
    
    summary = {
        "layer": layer,
        "comp": comp,
        "r2_activation_mean": r2_activation_mean,
        "r2_activation_median": r2_activation_median,
        "linear_cka": cka_score,
        "retrieval_accuracy_top1": retrieval_acc,
        "dm_natural_median": np.median(dm_natural_instruct),
        "dm_aligned_median": aligned_median_dm,
        "dm_aligned_rank_in_natural_pct": rank_in_natural,
        "two_sample_auc": two_sample_auc,
        "matched_cosine_mean": np.mean(matched_cos),
        "unmatched_cosine_mean": np.mean(unmatched_cos),
        "natural_pair_cosine_mean": np.mean(natural_pair_cos)
    }
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--instruct_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data_path", type=str, default="v3/data/aipsy_strict_expanded.csv")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--comp", type=str, default="mlp")
    parser.add_argument("--output_path", type=str, default="v3/results/alignment_fidelity_manifold_results.json")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.instruct_model)
    
    print(f"Loading Base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    base_model.eval()
    
    print(f"Loading Instruct model: {args.instruct_model}")
    instruct_model = AutoModelForCausalLM.from_pretrained(
        args.instruct_model,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    instruct_model.eval()
    
    df = pd.read_csv(args.data_path)
    from v3.scripts.run_aligned_cross_model_patching import get_3way_split
    _, dev_df, test_df = get_3way_split(df)
    
    res = run_alignment_and_manifold_diagnostics(
        base_model, instruct_model, tokenizer,
        dev_df, test_df,
        layer=args.layer, comp=args.comp, device=device
    )
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    def json_serializer(o):
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.integer, np.int32, np.int64)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
        
    with open(args.output_path, "w") as f:
        json.dump(res, f, indent=2, default=json_serializer)
    print(f"\nDiagnostics completed. Results saved to {args.output_path}")

if __name__ == "__main__":
    main()
