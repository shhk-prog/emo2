#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

def load_jsonl(path: str) -> pd.DataFrame:
    records = []
    with open(path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            records.append(json.loads(line))
    return pd.DataFrame(records)

def build_contrastive_directions(df_train: pd.DataFrame, hidden_states: dict) -> dict:
    """
    Constructs contrastive directions (d_V, d_A) for each layer using train set hidden states.
    """
    # positive vs negative
    pos_emotions = ['ecstasy', 'admiration', 'amazement']
    neg_emotions = ['rage', 'grief', 'terror', 'loathing']
    
    # high-arousal vs low-arousal/neutral
    high_a_emotions = ['rage', 'terror', 'ecstasy', 'amazement']
    low_a_emotions = ['grief', 'loathing']
    
    # Determine unique layers from the keys
    layers_set = set()
    for k in hidden_states.keys():
        if '_layer' in k:
            layers_set.add(int(k.split('_layer')[-1]))
    layers = sorted(list(layers_set))
    directions = {}
    
    for l in layers:
        # Collect representations
        pos_reps = []
        neg_reps = []
        high_a_reps = []
        low_a_reps = []
        
        for idx, row in df_train.iterrows():
            uid = row['id']
            key = f"{uid}_layer{l}"
            if key not in hidden_states:
                continue
            
            rep = hidden_states[key]
            emo = str(row.get('emotion', '')).lower()
            cond = str(row.get('condition', '')).lower()
            
            # Valence
            if emo in pos_emotions:
                pos_reps.append(rep)
            elif emo in neg_emotions:
                neg_reps.append(rep)
                
            # Arousal
            if emo in high_a_emotions:
                high_a_reps.append(rep)
            elif emo in low_a_emotions or cond == 'neutral':
                low_a_reps.append(rep)
                
        if len(pos_reps) == 0 or len(neg_reps) == 0:
            print(f"Warning: Missing positive or negative samples in train set. Cannot compute d_V for layer {l}")
            d_V = np.zeros(rep.shape)
        else:
            d_V = np.mean(pos_reps, axis=0) - np.mean(neg_reps, axis=0)
            norm_V = np.linalg.norm(d_V)
            if norm_V > 0:
                d_V = d_V / norm_V
                
        if len(high_a_reps) == 0 or len(low_a_reps) == 0:
            print(f"Warning: Missing high or low arousal samples in train set. Cannot compute d_A for layer {l}")
            d_A = np.zeros(rep.shape)
        else:
            d_A = np.mean(high_a_reps, axis=0) - np.mean(low_a_reps, axis=0)
            norm_A = np.linalg.norm(d_A)
            if norm_A > 0:
                d_A = d_A / norm_A
                
        directions[l] = {'d_V': d_V, 'd_A': d_A}
        
    return directions

def evaluate_ibc_and_projection(df: pd.DataFrame, hidden_states: dict, jsonl_df: pd.DataFrame, directions: dict):
    """
    Calculate projection scores (z_V, z_A) and correlate with expected self-report E_v, E_a (IBC).
    """
    jsonl_dict = jsonl_df.set_index('id').to_dict(orient='index')
    
    results = []
    dose_responses = []
    layers = sorted(list(directions.keys()))
    
    for l in layers:
        d_V = directions[l]['d_V']
        d_A = directions[l]['d_A']
        
        z_V_list, z_A_list, E_V_list, E_A_list = [], [], [], []
        # For dose-response
        dr_V = {'peak': [], 'moderate': [], 'none': []}
        dr_A = {'peak': [], 'moderate': [], 'none': []}
        
        for idx, row in df.iterrows():
            uid = row['id']
            key = f"{uid}_layer{l}"
            if key not in hidden_states or uid not in jsonl_dict:
                continue
                
            rep = hidden_states[key]
            z_V = np.dot(d_V, rep)
            z_A = np.dot(d_A, rep)
            
            E_V = jsonl_dict[uid].get('E_v', np.nan)
            E_A = jsonl_dict[uid].get('E_a', np.nan)
            
            intensity = str(row.get('intensity', '')).lower()
            if intensity in dr_V:
                dr_V[intensity].append(z_V)
                dr_A[intensity].append(z_A)
            
            if not np.isnan(E_V) and not np.isnan(E_A):
                z_V_list.append(z_V)
                z_A_list.append(z_A)
                E_V_list.append(E_V)
                E_A_list.append(E_A)
                
        if len(z_V_list) > 1:
            ibc_V, _ = pearsonr(z_V_list, E_V_list)
            ibc_A, _ = pearsonr(z_A_list, E_A_list)
        else:
            ibc_V, ibc_A = 0.0, 0.0
            
        results.append({
            'layer': l,
            'IBC_V': ibc_V,
            'IBC_A': ibc_A
        })
        
        dose_responses.append({
            'layer': l,
            'z_V_peak_mean': np.mean(dr_V['peak']) if dr_V['peak'] else np.nan,
            'z_V_moderate_mean': np.mean(dr_V['moderate']) if dr_V['moderate'] else np.nan,
            'z_V_neutral_mean': np.mean(dr_V['none']) if dr_V['none'] else np.nan,
            'z_A_peak_mean': np.mean(dr_A['peak']) if dr_A['peak'] else np.nan,
            'z_A_moderate_mean': np.mean(dr_A['moderate']) if dr_A['moderate'] else np.nan,
            'z_A_neutral_mean': np.mean(dr_A['none']) if dr_A['none'] else np.nan,
        })
        
    return pd.DataFrame(results), pd.DataFrame(dose_responses)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="v2/data/processed/aipsy")
    parser.add_argument("--results-dir", type=str, default="v2/results/raw/aipsy")
    parser.add_argument("--out-dir", type=str, default="v2/results/derived/phase1_preliminary")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("Loading datasets...")
    train_df = pd.read_csv(os.path.join(args.data_dir, "train_strict.csv"))
    dev_df = pd.read_csv(os.path.join(args.data_dir, "dev_strict.csv"))
    test_df = pd.read_csv(os.path.join(args.data_dir, "test_strict.csv"))
    
    # We will compute IBC on the entire strict subset (or test set)
    full_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)
    
    models = [
        ("Qwen_Qwen2.5-1.5B", "Base"),
        ("Qwen_Qwen2.5-1.5B-Instruct", "Instruct")
    ]
    templates = ["standard", "reversed"]
    
    all_results = []
    all_dose_responses = []
    
    for model_prefix, model_type in models:
        npz_path = os.path.join(args.results_dir, f"{model_prefix}_hidden_states.npz")
        if not os.path.exists(npz_path):
            print(f"File not found: {npz_path}")
            continue
            
        print(f"\nLoading hidden states for {model_type} ({model_prefix})...")
        hidden_states = dict(np.load(npz_path, allow_pickle=True))
        # Ensure it's a dict of dicts instead of 0-d arrays of dicts if structured that way
        for k, v in hidden_states.items():
            if v.shape == ():
                hidden_states[k] = v.item()
                
        print(f"Building contrastive directions for {model_type}...")
        directions = build_contrastive_directions(train_df, hidden_states)
        
        for tmpl in templates:
            jsonl_path = os.path.join(args.results_dir, f"{model_prefix}_{tmpl}_results.jsonl")
            if not os.path.exists(jsonl_path):
                print(f"File not found: {jsonl_path}")
                continue
                
            print(f"Evaluating {model_type} - {tmpl} template...")
            jsonl_df = load_jsonl(jsonl_path)
            
            res_df, dose_df = evaluate_ibc_and_projection(full_df, hidden_states, jsonl_df, directions)
            res_df['model_type'] = model_type
            res_df['template'] = tmpl
            dose_df['model_type'] = model_type
            dose_df['template'] = tmpl
            
            all_results.append(res_df)
            all_dose_responses.append(dose_df)
            
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_dose = pd.concat(all_dose_responses, ignore_index=True)
        
        out_csv = os.path.join(args.out_dir, "ibc_results.csv")
        final_df.to_csv(out_csv, index=False)
        dose_csv = os.path.join(args.out_dir, "dose_response_results.csv")
        final_dose.to_csv(dose_csv, index=False)
        print(f"\nSaved IBC results to {out_csv}")
        print(f"Saved Dose-Response results to {dose_csv}")
        
        # Calculate Delta IBC for standard template
        base_std = final_df[(final_df['model_type'] == 'Base') & (final_df['template'] == 'standard')]
        inst_std = final_df[(final_df['model_type'] == 'Instruct') & (final_df['template'] == 'standard')]
        
        if not base_std.empty and not inst_std.empty:
            merged = pd.merge(base_std, inst_std, on='layer', suffixes=('_base', '_inst'))
            merged['Delta_IBC_V'] = merged['IBC_V_base'] - merged['IBC_V_inst']
            merged['Delta_IBC_A'] = merged['IBC_A_base'] - merged['IBC_A_inst']
            delta_csv = os.path.join(args.out_dir, "delta_ibc_standard.csv")
            merged[['layer', 'Delta_IBC_V', 'Delta_IBC_A', 'IBC_V_base', 'IBC_V_inst']].to_csv(delta_csv, index=False)
            print(f"Saved Delta IBC (Standard) to {delta_csv}")
            
if __name__ == "__main__":
    main()
