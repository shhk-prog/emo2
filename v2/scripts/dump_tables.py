import pandas as pd
import json

def main():
    print("=== Table 3: Patching Screening (Top components) ===")
    try:
        df = pd.read_csv("v2/results/derived/phase7_strict_patching/strict_patching_screening_summary.csv")
        print(df.sort_values(by="delta_WD_V").head(10).to_markdown())
    except:
        print("Not found")

    print("\n=== Table 4: Causal Scrubbing ===")
    try:
        df = pd.read_csv("v2/results/derived/phase8_causal_scrubbing/strict_causal_scrubbing_results.csv")
        print(df.groupby('component')[['delta_Ev_l27', 'delta_WD_l27']].mean().to_markdown())
    except:
        print("Not found")
        
    print("\n=== Table 5: Unembedding Norm Swap ===")
    try:
        df = pd.read_csv("v2/results/derived/phase7_strict_path_patching/unembedding_norm_swap_results.csv")
        print(df.to_markdown())
    except:
        print("Not found")

if __name__ == "__main__":
    main()
