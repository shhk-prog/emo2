"""
run_ridge_alpha_sweep.py

Purpose:
Sweeps Ridge regularization alpha over [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
to test how regularization strength affects:
1. High-dimensional activation reconstruction (R^2_activation, Linear CKA, Retrieval Accuracy)
2. Manifold distance (Median Mahalanobis D_M relative to Natural Instruct median ~39.66)
3. Two-sample Linear Classifier AUC (Distinguishability from Natural Instruct)
4. Matched vs Unmatched Cosine Similarity
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
from v3.scripts.analyze_alignment_fidelity_and_manifold import (
    extract_layer_activations, linear_cka, compute_retrieval_accuracy
)

def run_alpha_sweep(base_model, instruct_model, tokenizer, dev_df, test_df, layer=15, comp="mlp", device="cuda"):
    alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    
    print("Extracting Alignment-dev activations...")
    dev_texts = dev_df['text'].tolist()
    h_B_dev = extract_layer_activations(base_model, tokenizer, dev_texts, layer, comp, device)
    h_I_dev = extract_layer_activations(instruct_model, tokenizer, dev_texts, layer, comp, device)
    
    print("Extracting Held-out test activations...")
    test_texts = test_df['text'].tolist()
    h_B_test = extract_layer_activations(base_model, tokenizer, test_texts, layer, comp, device)
    h_I_test = extract_layer_activations(instruct_model, tokenizer, test_texts, layer, comp, device)
    
    # Estimate Instruct Covariance on Dev for Mahalanobis
    lw = LedoitWolf()
    lw.fit(h_I_dev)
    mu_I = lw.location_
    precision_I = lw.precision_
    
    def calc_dm(acts):
        diff = acts - mu_I
        dist_sq = np.sum((diff @ precision_I) * diff, axis=1)
        return np.sqrt(np.maximum(dist_sq, 0))
        
    dm_natural_instruct = calc_dm(h_I_test)
    nat_median_dm = float(np.median(dm_natural_instruct))
    print(f"Natural Instruct Median D_M: {nat_median_dm:.2f}")
    
    results = []
    
    for alpha in alphas:
        print(f"\n--- Testing Ridge alpha = {alpha} ---")
        mapper = Ridge(alpha=alpha, fit_intercept=True)
        mapper.fit(h_B_dev, h_I_dev)
        
        h_aligned_test = mapper.predict(h_B_test)
        
        # 1. R^2 activation
        numer = np.sum((h_aligned_test - h_I_test) ** 2, axis=0)
        denom = np.sum((h_I_test - np.mean(h_I_test, axis=0)) ** 2, axis=0)
        valid_dims = denom > 1e-6
        r2_mean = float(np.mean(1.0 - (numer[valid_dims] / denom[valid_dims])))
        
        # 2. Linear CKA
        cka_score = float(linear_cka(h_aligned_test, h_I_test))
        
        # 3. Retrieval Accuracy
        retrieval_acc = float(compute_retrieval_accuracy(h_aligned_test, h_I_test))
        
        # 4. Mahalanobis D_M
        dm_aligned = calc_dm(h_aligned_test)
        aligned_median_dm = float(np.median(dm_aligned))
        
        # 5. Two-sample classifier AUC
        X_cls = np.vstack([h_I_test, h_aligned_test])
        y_cls = np.array([0] * len(h_I_test) + [1] * len(h_aligned_test))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        auc_scores = []
        for train_idx, val_idx in skf.split(X_cls, y_cls):
            clf = LogisticRegression(C=1.0, max_iter=1000)
            clf.fit(X_cls[train_idx], y_cls[train_idx])
            preds = clf.predict_proba(X_cls[val_idx])[:, 1]
            auc_scores.append(roc_auc_score(y_cls[val_idx], preds))
        two_sample_auc = float(np.mean(auc_scores))
        
        # 6. Cosine similarities
        norm_aligned = h_aligned_test / (np.linalg.norm(h_aligned_test, axis=1, keepdims=True) + 1e-9)
        norm_true = h_I_test / (np.linalg.norm(h_I_test, axis=1, keepdims=True) + 1e-9)
        matched_cos = float(np.mean(np.sum(norm_aligned * norm_true, axis=1)))
        
        shuffled_idx = np.random.RandomState(42).permutation(len(h_I_test))
        unmatched_cos = float(np.mean(np.sum(norm_aligned * norm_true[shuffled_idx], axis=1)))
        
        print(f"alpha={alpha:1.0e} | R^2={r2_mean:.3f} | CKA={cka_score:.3f} | Top1={retrieval_acc*100:.1f}% | Median D_M={aligned_median_dm:.2f} (Nat={nat_median_dm:.2f}) | AUC={two_sample_auc:.3f} | Cos={matched_cos:.4f}")
        
        results.append({
            "alpha": alpha,
            "r2_activation_mean": r2_mean,
            "linear_cka": cka_score,
            "retrieval_accuracy_top1": retrieval_acc,
            "aligned_median_dm": aligned_median_dm,
            "natural_median_dm": nat_median_dm,
            "two_sample_auc": two_sample_auc,
            "matched_cosine": matched_cos,
            "unmatched_cosine": unmatched_cos
        })
        
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--instruct_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data_path", type=str, default="v3/data/aipsy_strict_expanded.csv")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--comp", type=str, default="mlp")
    parser.add_argument("--output_path", type=str, default="v3/results/ridge_alpha_sweep_results.csv")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.instruct_model)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float16 if device == "cuda" else torch.float32, device_map="auto" if device == "cuda" else None
    ).eval()
    instruct_model = AutoModelForCausalLM.from_pretrained(
        args.instruct_model, dtype=torch.float16 if device == "cuda" else torch.float32, device_map="auto" if device == "cuda" else None
    ).eval()
    
    df = pd.read_csv(args.data_path)
    _, dev_df, test_df = get_3way_split(df)
    
    res_df = run_alpha_sweep(base_model, instruct_model, tokenizer, dev_df, test_df, layer=args.layer, comp=args.comp, device=device)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    res_df.to_csv(args.output_path, index=False)
    print(f"\nRidge alpha sweep completed. Results saved to {args.output_path}")

if __name__ == "__main__":
    main()
