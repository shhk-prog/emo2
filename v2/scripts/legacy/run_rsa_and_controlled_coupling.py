#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
import pandas as pd
import numpy as np
import scipy.spatial.distance as dist
from scipy.stats import pearsonr
import statsmodels.api as sm
import statsmodels.formula.api as smf
from v2.scripts.run_confirmatory_analysis import load_hidden_states
import json

def calculate_rsa(df_meta, hs_data, df_results, layer, d_V, d_A):
    # Filter matching IDs
    uids = []
    H_V, H_A = [], []
    I_V, I_A = [], []
    S_V, S_A = [], []
    
    res_dict = {row['id']: row for _, row in df_results.iterrows()}
    
    for _, row in df_meta.iterrows():
        uid = row['id']
        key = f"{uid}_layer{layer}"
        if key in hs_data and pd.notna(row.get('V_H')) and uid in res_dict:
            uids.append(uid)
            H_V.append(row['V_H'])
            H_A.append(row['A_H'])
            
            h = hs_data[key]
            I_V.append(np.dot(h, d_V))
            I_A.append(np.dot(h, d_A))
            
            S_V.append(res_dict[uid]['E_v'])
            S_A.append(res_dict[uid]['E_a'])
            
    if len(uids) < 2: return None
    
    # Create distance matrices
    H_mat = np.column_stack((H_V, H_A))
    I_mat = np.column_stack((I_V, I_A))
    S_mat = np.column_stack((S_V, S_A))
    
    D_H = dist.pdist(H_mat, metric='euclidean')
    D_I = dist.pdist(I_mat, metric='euclidean')
    D_S = dist.pdist(S_mat, metric='euclidean')
    
    rsa_HI, _ = pearsonr(D_H, D_I)
    rsa_HS, _ = pearsonr(D_H, D_S)
    rsa_IS, _ = pearsonr(D_I, D_S)
    
    return {
        "layer": layer,
        "N": len(uids),
        "RSA_HI": rsa_HI,
        "RSA_HS": rsa_HS,
        "RSA_IS": rsa_IS
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", type=str, default="v2/data/processed/aipsy_annotated/test_strict.csv")
    parser.add_argument("--test-hs", type=str, default="v2/results/raw/aipsy/Qwen_Qwen2.5-1.5B-Instruct_hidden_states.npz")
    parser.add_argument("--test-results", type=str, default="v2/results/raw/aipsy/Qwen_Qwen2.5-1.5B-Instruct_standard_results.jsonl")
    parser.add_argument("--out-dir", type=str, default="v2/results/derived/phase1.5_confirmatory")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    df_test = pd.read_csv(args.test_data)
    hs_test = load_hidden_states(args.test_hs)
    df_results = pd.DataFrame([json.loads(l) for l in open(args.test_results, 'r')])
    
    # Use simple random projection for d_V/d_A placeholder if we don't have contrastive dir here
    # In full script, we would load the true directions from Train set
    hidden_dim = list(hs_test.values())[0].shape[0]
    np.random.seed(42)
    d_V = np.random.randn(hidden_dim)
    d_A = np.random.randn(hidden_dim)
    
    rsa_results = []
    for l in range(28):
        res = calculate_rsa(df_test, hs_test, df_results, l, d_V, d_A)
        if res: rsa_results.append(res)
        
    df_rsa = pd.DataFrame(rsa_results)
    out_path = os.path.join(args.out_dir, "rsa_analysis.csv")
    df_rsa.to_csv(out_path, index=False)
    print(f"Saved RSA results to {out_path}")
    
    # Controlled Regression
    print("Running controlled regression...")
    merged = pd.merge(df_test, df_results, on='id')
    merged['word_count'] = merged['text'].apply(lambda x: len(str(x).split()))
    merged['z_V'] = merged['id'].apply(lambda uid: np.dot(hs_test.get(f"{uid}_layer24", np.zeros(hidden_dim)), d_V))
    
    # Example formula: E_v ~ z_V + V_H + word_count
    model = smf.ols('E_v ~ z_V + V_H + word_count', data=merged).fit()
    with open(os.path.join(args.out_dir, "controlled_regression.txt"), "w") as f:
        f.write(model.summary().as_text())
    print("Saved controlled regression results.")

if __name__ == "__main__":
    main()
