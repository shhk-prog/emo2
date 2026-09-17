#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.linalg import orthogonal_procrustes
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

def load_jsonl(path: str) -> pd.DataFrame:
    records = []
    with open(path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            records.append(json.loads(line))
    return pd.DataFrame(records)

def compute_contrastive_directions(rep_dict, df_train):
    pos_emotions = ['ecstasy', 'admiration', 'amazement']
    neg_emotions = ['rage', 'grief', 'terror', 'loathing']
    high_a_emotions = ['rage', 'terror', 'ecstasy', 'amazement']
    low_a_emotions = ['grief', 'loathing']

    pos_reps, neg_reps = [], []
    high_a_reps, low_a_reps = [], []

    for idx, row in df_train.iterrows():
        uid = row['id']
        if uid not in rep_dict:
            continue
        
        rep = rep_dict[uid]
        emo = str(row.get('emotion', '')).lower()
        cond = str(row.get('condition', '')).lower()
        
        if emo in pos_emotions: pos_reps.append(rep)
        elif emo in neg_emotions: neg_reps.append(rep)
            
        if emo in high_a_emotions: high_a_reps.append(rep)
        elif emo in low_a_emotions or cond == 'neutral': low_a_reps.append(rep)

    if len(pos_reps) > 0 and len(neg_reps) > 0:
        d_V = np.mean(pos_reps, axis=0) - np.mean(neg_reps, axis=0)
        n = np.linalg.norm(d_V)
        if n > 0: d_V = d_V / n
    else:
        d_V = np.zeros(list(rep_dict.values())[0].shape)
        
    if len(high_a_reps) > 0 and len(low_a_reps) > 0:
        d_A = np.mean(high_a_reps, axis=0) - np.mean(low_a_reps, axis=0)
        n = np.linalg.norm(d_A)
        if n > 0: d_A = d_A / n
    else:
        d_A = np.zeros(list(rep_dict.values())[0].shape)
        
    return d_V, d_A

def extract_matrices(hidden_states_base, hidden_states_inst, jsonl_dict, df, layer):
    X, Y, uids, E_v, E_a = [], [], [], [], []
    
    for idx, row in df.iterrows():
        uid = row['id']
        key = f"{uid}_layer{layer}"
        
        if key in hidden_states_base and key in hidden_states_inst and uid in jsonl_dict:
            E_v_val = jsonl_dict[uid].get('E_v', np.nan)
            E_a_val = jsonl_dict[uid].get('E_a', np.nan)
            if not np.isnan(E_v_val) and not np.isnan(E_a_val):
                X.append(hidden_states_base[key])
                Y.append(hidden_states_inst[key])
                E_v.append(E_v_val)
                E_a.append(E_a_val)
                uids.append(uid)
                
    return np.array(X), np.array(Y), uids, np.array(E_v), np.array(E_a)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="v2/data/processed/aipsy")
    parser.add_argument("--results-dir", type=str, default="v2/results/raw/aipsy")
    parser.add_argument("--out-dir", type=str, default="v2/results/derived/phase2_cross_decoding")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("Loading datasets...")
    train_df = pd.read_csv(os.path.join(args.data_dir, "train_strict.csv"))
    # Evaluate on dev + test (or just test). We'll use dev+test to maximize statistical power for IBC.
    dev_df = pd.read_csv(os.path.join(args.data_dir, "dev_strict.csv"))
    test_df = pd.read_csv(os.path.join(args.data_dir, "test_strict.csv"))
    eval_df = pd.concat([dev_df, test_df], ignore_index=True)
    
    print("Loading Base and Instruct hidden states...")
    base_npz = os.path.join(args.results_dir, "Qwen_Qwen2.5-1.5B_hidden_states.npz")
    inst_npz = os.path.join(args.results_dir, "Qwen_Qwen2.5-1.5B-Instruct_hidden_states.npz")
    
    base_hs = dict(np.load(base_npz, allow_pickle=True))
    inst_hs = dict(np.load(inst_npz, allow_pickle=True))
    for k, v in base_hs.items():
        if v.shape == (): base_hs[k] = v.item()
    for k, v in inst_hs.items():
        if v.shape == (): inst_hs[k] = v.item()
        
    # We use Instruct standard results for evaluating the Cross-IBC
    inst_jsonl_path = os.path.join(args.results_dir, "Qwen_Qwen2.5-1.5B-Instruct_standard_results.jsonl")
    jsonl_df = load_jsonl(inst_jsonl_path)
    jsonl_dict = jsonl_df.set_index('id').to_dict(orient='index')
    
    layers_set = set(int(k.split('_layer')[-1]) for k in base_hs.keys() if '_layer' in k)
    layers = sorted(list(layers_set))
    
    results = []
    
    print("Running cross-model decoding...")
    for l in layers:
        # Extract train and eval matrices
        X_train, Y_train, train_uids, _, _ = extract_matrices(base_hs, inst_hs, jsonl_dict, train_df, l)
        X_eval, Y_eval, eval_uids, Ev_eval, Ea_eval = extract_matrices(base_hs, inst_hs, jsonl_dict, eval_df, l)
        
        X_train = X_train.astype(np.float32)
        Y_train = Y_train.astype(np.float32)
        X_eval = X_eval.astype(np.float32)
        Y_eval = Y_eval.astype(np.float32)
        
        if len(X_train) < 2 or len(X_eval) < 2:
            continue
            
        # Get instruct directions on train
        # Rebuild a temporary dict mapping uid -> representation for Y_train
        Y_train_dict = {uid: Y_train[i] for i, uid in enumerate(train_uids)}
        d_V_inst, d_A_inst = compute_contrastive_directions(Y_train_dict, train_df)
        
        # Original Instruct IBC on eval (Upper bound)
        z_V_inst = Y_eval @ d_V_inst
        z_A_inst = Y_eval @ d_A_inst
        ibc_V_inst, _ = pearsonr(z_V_inst, Ev_eval)
        ibc_A_inst, _ = pearsonr(z_A_inst, Ea_eval)
        
        # 1. Direct transfer (no alignment)
        Y_hat_direct = X_eval
        r2_direct = r2_score(Y_eval, Y_hat_direct, multioutput='variance_weighted')
        z_V_direct = Y_hat_direct @ d_V_inst
        z_A_direct = Y_hat_direct @ d_A_inst
        ibc_V_direct, _ = pearsonr(z_V_direct, Ev_eval)
        ibc_A_direct, _ = pearsonr(z_A_direct, Ea_eval)
        
        # 2. Orthogonal Procrustes
        # Scipy orthogonal_procrustes finds R that best maps A to B minimizing ||A R - B||_F
        R, _ = orthogonal_procrustes(X_train, Y_train)
        Y_hat_ortho = X_eval @ R
        r2_ortho = r2_score(Y_eval, Y_hat_ortho, multioutput='variance_weighted')
        z_V_ortho = Y_hat_ortho @ d_V_inst
        z_A_ortho = Y_hat_ortho @ d_A_inst
        ibc_V_ortho, _ = pearsonr(z_V_ortho, Ev_eval)
        ibc_A_ortho, _ = pearsonr(z_A_ortho, Ea_eval)
        
        # 3. Regularized linear (Ridge)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, Y_train)
        Y_hat_ridge = ridge.predict(X_eval)
        r2_ridge = r2_score(Y_eval, Y_hat_ridge, multioutput='variance_weighted')
        z_V_ridge = Y_hat_ridge @ d_V_inst
        z_A_ridge = Y_hat_ridge @ d_A_inst
        ibc_V_ridge, _ = pearsonr(z_V_ridge, Ev_eval)
        ibc_A_ridge, _ = pearsonr(z_A_ridge, Ea_eval)
        
        results.append({
            'layer': l,
            'r2_direct': r2_direct,
            'r2_ortho': r2_ortho,
            'r2_ridge': r2_ridge,
            'ibc_V_target_inst': ibc_V_inst,
            'ibc_V_direct': ibc_V_direct,
            'ibc_V_ortho': ibc_V_ortho,
            'ibc_V_ridge': ibc_V_ridge,
            'ibc_A_target_inst': ibc_A_inst,
            'ibc_A_direct': ibc_A_direct,
            'ibc_A_ortho': ibc_A_ortho,
            'ibc_A_ridge': ibc_A_ridge,
        })
        
    res_df = pd.DataFrame(results)
    out_csv = os.path.join(args.out_dir, "cross_decoding_results.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved cross-model decoding results to {out_csv}")
    
if __name__ == "__main__":
    main()
