#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd

def main():
    out_dir_screening = "v2/results/derived/phase7_strict_patching"
    out_dir_path = "v2/results/derived/phase7_strict_path_patching"
    
    screening_csv = os.path.join(out_dir_screening, "strict_patching_screening_summary.csv")
    path_csv = os.path.join(out_dir_path, "strict_path_patching_aggregated.csv")
    swap_csv = os.path.join(out_dir_path, "unembedding_swap_results.csv")
    
    print("=== Strict Patching Screening Summary ===")
    if os.path.exists(screening_csv):
        df_scr = pd.read_csv(screening_csv)
        print(df_scr.sort_values("mean_delta_v", ascending=False).head(10).to_markdown())
    else:
        print(f"Not found: {screening_csv}")
        
    print("\n=== Strict Path Patching Summary ===")
    if os.path.exists(path_csv):
        df_path = pd.read_csv(path_csv)
        print(df_path.to_markdown())
    else:
        print(f"Not found: {path_csv}")
        
    print("\n=== Unembedding Swap Summary ===")
    if os.path.exists(swap_csv):
        df_swap = pd.read_csv(swap_csv)
        print(df_swap[["base_Ev", "inst_Ev", "swapA_Ev", "swapB_Ev"]].mean().to_markdown())
    else:
        print(f"Not found: {swap_csv}")

if __name__ == "__main__":
    main()
