#!/usr/bin/env python3
import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List

from affective_empathy_eval.manifests import ManifestManager
from affective_empathy_eval.probing import LayerProber, run_rsa_analysis, run_shuffled_baseline_probing

def main():
    parser = argparse.ArgumentParser(description="Run Layerwise Probing and RSA analysis on main experiment representations.")
    parser.add_argument("--run-id", type=str, required=True, help="Run ID of the main experiment.")
    parser.add_argument("--condition", type=str, default="post_reported_va", help="Condition to extract representations from.")
    parser.add_argument("--position", type=str, default="stimulus_last_token", help="Token position to probe.")
    args = parser.parse_args()

    matches = glob.glob(os.path.join("results/raw/*", args.run_id))
    if not matches:
        print(f"Error: Run directory for {args.run_id} not found.")
        return
    run_dir = matches[0]
    phase = os.path.basename(os.path.dirname(run_dir))

    analysis_dataset_path = os.path.join("results/derived", phase, args.run_id, "analysis_dataset.csv")
    if not os.path.exists(analysis_dataset_path):
        print(f"Error: Missing analysis dataset at {analysis_dataset_path}. Run run_analysis.py first.")
        return
    df_analysis = pd.read_csv(analysis_dataset_path)

    reps_dir = os.path.join(run_dir, "representations")
    print(f"Analyzing representations from {reps_dir} using ManifestManager...")
    manager = ManifestManager(reps_dir)
    df_manifest = manager.load_all_as_dataframe()
    
    if df_manifest.empty:
        print("Error: No manifests found in representation directory.")
        return
        
    out_dir = os.path.join("results/derived", phase, args.run_id, "probing")
    os.makedirs(out_dir, exist_ok=True)

    # Filter manifest to target condition and position
    df_m_filtered = df_manifest[
        (df_manifest['condition'] == args.condition) & 
        (df_manifest['extraction_position'] == args.position)
    ].copy()

    if df_m_filtered.empty:
        print(f"Error: No representations found for condition={args.condition} and position={args.position}.")
        return

    models = df_m_filtered['model_name'].unique()
    print(f"Found models: {models}")

    for model_name in models:
        print(f"\n--- Running Probing for Model: {model_name} ---")
        df_m_model = df_m_filtered[df_m_filtered['model_name'] == model_name].copy()
        
        # Filter analysis to the specific model
        df_a_model = df_analysis[df_analysis['model_id'] == model_name].copy()
        
        # Merge on stimulus_id (assuming 1 rep or taking first match)
        df_merged = df_m_model.merge(df_a_model, on='stimulus_id', how='inner')
        
        # For simplicity, if there are multiple repetitions per stimulus, group by stimulus_id and take first to avoid duplication
        df_merged = df_merged.groupby('stimulus_id').first().reset_index()
        
        if df_merged.empty:
            print(f"Warning: No matching derived data for model {model_name}. Skipping.")
            continue
            
        print(f"Matched {len(df_merged)} unique stimuli samples for probing.")
        
        # Prepare targets (drop NaNs)
        df_merged = df_merged.dropna(subset=['V_scaled', 'A_scaled', 'parsed_valence', 'parsed_arousal'])
        if df_merged.empty:
            print("Warning: All samples dropped due to missing target values.")
            continue
            
        # Target Variables
        target_human_v = df_merged['V_scaled'].values
        target_human_a = df_merged['A_scaled'].values
        target_post_v = df_merged['parsed_valence'].values
        target_post_a = df_merged['parsed_arousal'].values
        target_delta_v = df_merged['delta_V'].values
        target_delta_a = df_merged['delta_A'].values
        
        target_vectors = {
            "human_VA": np.column_stack([target_human_v, target_human_a]),
            "reported_post_VA": np.column_stack([target_post_v, target_post_a]),
            "delta_VA": np.column_stack([target_delta_v, target_delta_a])
        }
        
        # Load tensors per layer
        layers = sorted(df_merged['layer'].unique())
        layer_vectors = {}
        layer_probing_results = []
        
        group_ids = df_merged['stimulus_id'].factorize()[0]
        prober = LayerProber(alpha=1.0, cv=5, seed=42, use_pca=True, n_components=min(50, len(df_merged)//2))

        for layer in layers:
            df_layer = df_merged[df_merged['layer'] == layer]
            
            X_layer = []
            for path in df_layer['tensor_path']:
                if not os.path.isabs(path):
                    if not os.path.exists(path) and os.path.exists(os.path.join(reps_dir, os.path.basename(path))):
                        path = os.path.join(reps_dir, os.path.basename(path))
                tensor = np.load(path)
                X_layer.append(tensor)
                
            X_layer = np.array(X_layer)
            if X_layer.ndim == 3:
                X_layer = X_layer.squeeze(1) # [batch, hidden_dim]
                
            layer_vectors[layer] = X_layer
            
            # Evaluate Human Valence Probing
            res_v = prober.evaluate_probing(X_layer, target_human_v, group_ids)
            shuffled_res = run_shuffled_baseline_probing(X_layer, target_human_v, group_ids, prober, n_permutations=3)

            layer_probing_results.append({
                "model": model_name,
                "layer": layer,
                "target": "human_valence",
                "r2": res_v["r2"],
                "rmse": res_v["rmse"],
                "mae": res_v["mae"],
                "pearson_r": res_v["pearson_r"],
                "spearman_r": res_v["spearman_r"],
                "shuffled_mean_r2": shuffled_res["mean_shuffled_r2"]
            })
            
            # Evaluate Reported Post Valence Probing
            res_post_v = prober.evaluate_probing(X_layer, target_post_v, group_ids)
            layer_probing_results.append({
                "model": model_name,
                "layer": layer,
                "target": "reported_post_valence",
                "r2": res_post_v["r2"],
                "rmse": res_post_v["rmse"],
                "mae": res_post_v["mae"],
                "pearson_r": res_post_v["pearson_r"],
                "spearman_r": res_post_v["spearman_r"],
                "shuffled_mean_r2": 0.0
            })

        df_probing = pd.DataFrame(layer_probing_results)
        safe_model_name = model_name.replace("/", "_")
        probing_out_path = os.path.join(out_dir, f"probing_summary_{safe_model_name}.csv")
        df_probing.to_csv(probing_out_path, index=False)
        print(f"  -> Saved probing metrics to {probing_out_path}")

        # RSA Analysis
        try:
            rsa_dict = run_rsa_analysis(layer_vectors, target_vectors, metric='euclidean')
            rsa_rows = []
            for target_name, layer_res in rsa_dict.items():
                for layer, scores in layer_res.items():
                    rsa_rows.append({
                        "model": model_name,
                        "target": target_name,
                        "layer": layer,
                        "spearman": scores["spearman"],
                        "pearson": scores["pearson"]
                    })
            df_rsa = pd.DataFrame(rsa_rows)
            rsa_out_path = os.path.join(out_dir, f"rsa_summary_{safe_model_name}.csv")
            df_rsa.to_csv(rsa_out_path, index=False)
            print(f"  -> Saved RSA metrics to {rsa_out_path}")
        except Exception as e:
            print(f"  -> RSA analysis failed for {model_name}: {e}")
            
    print(f"Probing script completed successfully. Results saved to {out_dir}/")

if __name__ == "__main__":
    main()
