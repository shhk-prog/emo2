#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr

def load_hidden_states(npz_path):
    d = dict(np.load(npz_path, allow_pickle=True))
    for k, v in d.items():
        if v.shape == (): d[k] = v.item()
    return d

def compute_hic_hbc(df_train, hs_train, df_test, hs_test, df_results_test, layers, alpha=1.0):
    results = []
    
    # 評価結果（E[V_self], E[A_self]）の辞書化
    res_dict = {row['id']: row for _, row in df_results_test.iterrows()}
    
    for layer in layers:
        # Train Ridge Probe
        X_train, y_v_train, y_a_train = [], [], []
        for _, row in df_train.iterrows():
            uid = row['id']
            key = f"{uid}_layer{layer}"
            if key in hs_train and pd.notna(row.get('V_H')):
                X_train.append(hs_train[key])
                y_v_train.append(row['V_H'])
                y_a_train.append(row['A_H'])
                
        if len(X_train) == 0:
            print(f"No training data for layer {layer}")
            continue
            
        X_train = np.array(X_train)
        
        probe_v = Ridge(alpha=alpha)
        probe_v.fit(X_train, y_v_train)
        
        probe_a = Ridge(alpha=alpha)
        probe_a.fit(X_train, y_a_train)
        
        # Test & Evaluate HIC, HBC
        z_v_test, v_h_test, e_v_test = [], [], []
        for _, row in df_test.iterrows():
            uid = row['id']
            key = f"{uid}_layer{layer}"
            if key in hs_test and pd.notna(row.get('V_H')) and uid in res_dict:
                h = hs_test[key]
                z_v = probe_v.predict([h])[0]
                
                z_v_test.append(z_v)
                v_h_test.append(row['V_H'])
                e_v_test.append(res_dict[uid]['E_v'])
                
        if len(z_v_test) > 1:
            hic_v, _ = pearsonr(z_v_test, v_h_test)
            hbc_v, _ = pearsonr(z_v_test, e_v_test)
            
            # Slope and Covariance
            hic_slope = np.polyfit(z_v_test, v_h_test, 1)[0]
            hbc_slope = np.polyfit(z_v_test, e_v_test, 1)[0]
            
            hic_cov = np.cov(z_v_test, v_h_test)[0, 1]
            hbc_cov = np.cov(z_v_test, e_v_test)[0, 1]
            
            results.append({
                "layer": layer,
                "dimension": "valence",
                "HIC_pearson": hic_v,
                "HBC_pearson": hbc_v,
                "HIC_slope": hic_slope,
                "HBC_slope": hbc_slope,
                "HIC_cov": hic_cov,
                "HBC_cov": hbc_cov,
                "Dissociation_Gap_Pearson": hic_v - hbc_v,
                "Dissociation_Gap_Slope": hic_slope - hbc_slope
            })
            
        # Arousal Evaluation
        z_a_test, a_h_test, e_a_test = [], [], []
        for _, row in df_test.iterrows():
            uid = row['id']
            key = f"{uid}_layer{layer}"
            if key in hs_test and pd.notna(row.get('A_H')) and uid in res_dict:
                h = hs_test[key]
                z_a = probe_a.predict([h])[0]
                
                z_a_test.append(z_a)
                a_h_test.append(row['A_H'])
                e_a_test.append(res_dict[uid]['E_a'])
                
        if len(z_a_test) > 1:
            hic_a, _ = pearsonr(z_a_test, a_h_test)
            hbc_a, _ = pearsonr(z_a_test, e_a_test)
            
            hic_slope_a = np.polyfit(z_a_test, a_h_test, 1)[0]
            hbc_slope_a = np.polyfit(z_a_test, e_a_test, 1)[0]
            
            hic_cov_a = np.cov(z_a_test, a_h_test)[0, 1]
            hbc_cov_a = np.cov(z_a_test, e_a_test)[0, 1]
            
            results.append({
                "layer": layer,
                "dimension": "arousal",
                "HIC_pearson": hic_a,
                "HBC_pearson": hbc_a,
                "HIC_slope": hic_slope_a,
                "HBC_slope": hbc_slope_a,
                "HIC_cov": hic_cov_a,
                "HBC_cov": hbc_cov_a,
                "Dissociation_Gap_Pearson": hic_a - hbc_a,
                "Dissociation_Gap_Slope": hic_slope_a - hbc_slope_a
            })
            
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    # EmoBank等で差し替え可能なようにtrain/testを分離して指定可能にする
    parser.add_argument("--train-data", type=str, default="v2/data/processed/aipsy_annotated/train_strict.csv")
    parser.add_argument("--train-hs", type=str, default="v2/results/raw/aipsy/Qwen_Qwen2.5-1.5B_hidden_states.npz")
    
    parser.add_argument("--test-data", type=str, default="v2/data/processed/aipsy_annotated/test_strict.csv")
    parser.add_argument("--test-hs", type=str, default="v2/results/raw/aipsy/Qwen_Qwen2.5-1.5B_hidden_states.npz")
    parser.add_argument("--test-results", type=str, default="v2/results/raw/aipsy/Qwen_Qwen2.5-1.5B_standard_results.jsonl")
    
    parser.add_argument("--layers", type=int, nargs='+', default=list(range(28)))
    parser.add_argument("--out-dir", type=str, default="v2/results/derived/phase1.5_confirmatory")
    parser.add_argument("--out-prefix", type=str, default="Qwen_Qwen2.5-1.5B")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    df_train = pd.read_csv(args.train_data)
    df_test = pd.read_csv(args.test_data)
    
    hs_train = load_hidden_states(args.train_hs)
    hs_test = load_hidden_states(args.test_hs)
    
    # Load E[V_self] results
    df_results_test = pd.DataFrame([json.loads(l) for l in open(args.test_results, 'r')])
    
    df_out = compute_hic_hbc(df_train, hs_train, df_test, hs_test, df_results_test, args.layers)
    
    out_file = os.path.join(args.out_dir, f"{args.out_prefix}_confirmatory_results.csv")
    df_out.to_csv(out_file, index=False)
    print(f"Saved confirmatory results to {out_file}")

if __name__ == "__main__":
    import json
    main()
