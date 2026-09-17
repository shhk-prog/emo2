#!/usr/bin/env python3
import os
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Summarize results across analysis outputs.")
    parser.add_argument("--run-id", type=str, required=True, help="Run ID of the main experiment.")
    args = parser.parse_args()

    import glob
    matches = glob.glob(os.path.join("results/raw/*", args.run_id))
    if not matches:
        print(f"Error: Run directory not found for {args.run_id}")
        return
    run_dir = matches[0]
    phase = os.path.basename(os.path.dirname(run_dir))
    
    out_dir = os.path.join("results/derived", phase, args.run_id)
    if not os.path.exists(out_dir):
        print(f"Error: Derived results directory not found: {out_dir}")
        return

    # Check for analysis_dataset
    dataset_path = os.path.join(out_dir, "analysis_dataset.csv")
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        print("=== Behavioral Analysis Summary ===")
        if 'R' in df.columns:
            non_react = (df['R'] == 0.0).sum()
            print(f"Total Samples: {len(df)}")
            print(f"Non-reactivity rate (R=0): {non_react / len(df):.2%}")
        if 'ADA' in df.columns:
            print(f"Mean ADA (Anchor Direction Alignment): {df['ADA'].mean():.3f}")

    # Check for probing summary
    probing_path = os.path.join(out_dir, "probing_summary.csv")
    if os.path.exists(probing_path):
        print("\n=== Probing Summary ===")
        df_probe = pd.read_csv(probing_path)
        best_layer = df_probe.loc[df_probe['r2'].idxmax()]
        print(f"Best predicting layer: {best_layer['layer']} (Target: {best_layer['target']}, R2={best_layer['r2']:.3f})")

    # Check for intervention summary
    interv_path = os.path.join(out_dir, "causal_intervention_summary.csv")
    if os.path.exists(interv_path):
        print("\n=== Intervention Summary ===")
        df_interv = pd.read_csv(interv_path)
        print(df_interv[['intervention', 'recovery']].to_string(index=False))

if __name__ == "__main__":
    main()
