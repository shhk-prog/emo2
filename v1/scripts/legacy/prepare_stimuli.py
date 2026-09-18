#!/usr/bin/env python3
import os
import yaml
import pandas as pd
import argparse
from affective_empathy_eval.data import load_emobank, stratify_stimuli

def main():
    parser = argparse.ArgumentParser(description="Prepare stimuli for the experiment.")
    parser.add_argument("--config", type=str, default="v1/configs/experiment.yaml", help="Path to the config file.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    data_path = "v1/data/raw/emobank.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run scripts/download_data.py first.")
        return
        
    v_col = config["sampling"].get("v_col", "V")
    a_col = config["sampling"].get("a_col", "A")
    df = load_emobank(data_path, v_col=v_col, a_col=a_col)
    
    n_per_cell = config["sampling"].get("n_per_cell", 50)
    seed = config.get("seed", 42)
    
    sampled, report = stratify_stimuli(df, n_per_cell=n_per_cell, seed=seed)
    
    # Calculate some metadata for the report
    import hashlib
    with open(data_path, "rb") as bf:
        dataset_hash = hashlib.sha256(bf.read()).hexdigest()[:8]
        
    report['sampling_seed'] = seed
    report['filter_version'] = "v1"
    report['dataset_hash'] = dataset_hash
    
    # Add stimulus_id using the original id
    if 'id' in sampled.columns:
        sampled['stimulus_id'] = 'emobank_' + sampled['id'].astype(str)
    else:
        sampled['stimulus_id'] = ['emobank_' + str(i) for i in range(len(sampled))]
    
    out_path = "v1/data/processed/stimuli.csv"
    sampled.to_csv(out_path, index=False)
    
    report_path = "v1/data/processed/stimulus_sampling_report.csv"
    report.to_csv(report_path, index=False)
    
    # Print warnings for shortfalls
    shortfalls = report[report['shortfall_n'] > 0]
    if not shortfalls.empty:
        print("\nWARNING: Some cells had fewer candidates than the target samples_per_cell.")
        print(shortfalls[['v_cell', 'a_cell', 'candidate_n', 'target_n', 'shortfall_n']].to_string(index=False))
        print()
    
    print(f"Sampled {len(sampled)} stimuli and saved to {out_path}")
    print(f"Sampling report saved to {report_path}")

if __name__ == "__main__":
    main()
