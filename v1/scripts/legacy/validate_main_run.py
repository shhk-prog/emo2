#!/usr/bin/env python3
import os
import argparse
import pandas as pd
from affective_empathy_eval.manifests import ManifestManager

def main():
    parser = argparse.ArgumentParser(description="Validate extraction manifests against response logs.")
    parser.add_argument("--run-id", type=str, required=True, help="Run ID of the main experiment.")
    args = parser.parse_args()

    import glob
    matches = glob.glob(os.path.join("results/raw/*", args.run_id))
    if not matches:
        print(f"Error: Run directory not found: {args.run_id}")
        return
    run_dir = matches[0]

    responses_path = os.path.join(run_dir, "responses.jsonl")
    reps_dir = os.path.join(run_dir, "representations")

    if not os.path.exists(responses_path):
        print("Validation failed: responses.jsonl missing.")
        return

    print("Loading responses.jsonl...")
    try:
        df_responses = pd.read_json(responses_path, lines=True)
    except Exception as e:
        print(f"Failed to read responses: {e}")
        return

    print("Loading manifests...")
    manager = ManifestManager(reps_dir)
    df_manifest = manager.load_all_as_dataframe()

    if df_manifest.empty:
        print("Validation failed: No manifests found in representations directory.")
        return

    print(f"Loaded {len(df_responses)} responses and {len(df_manifest)} manifest entries.")
    
    # Check if all request_ids in responses have at least one manifest
    # (Since request_id is prefix of tensor_path in our naming convention)
    manifest_files = df_manifest['tensor_path'].apply(os.path.basename)
    
    missing_reqs = 0
    for req_id in df_responses['request_id']:
        has_match = manifest_files.str.startswith(req_id).any()
        if not has_match:
            missing_reqs += 1
            
    if missing_reqs > 0:
        print(f"Validation Error: {missing_reqs} requests have no corresponding tensor files.")
    else:
        print("Validation Passed: All requests have corresponding extraction manifests.")
        
if __name__ == "__main__":
    main()
