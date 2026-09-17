import os
import json
import argparse
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

def load_all_results(results_dir):
    dfs = []
    for f in os.listdir(results_dir):
        if f.endswith("_results.jsonl"):
            path = os.path.join(results_dir, f)
            data = []
            with open(path, 'r') as fp:
                for line in fp:
                    data.append(json.loads(line))
            df = pd.DataFrame(data)
            dfs.append(df)
            
    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(dfs, ignore_index=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="v2/results/raw/aipsy")
    parser.add_argument("--out_file", type=str, default="v2/results/derived/mixed_effects_summary.txt")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    
    print("Loading results...")
    df = load_all_results(args.results_dir)
    
    if df.empty:
        print("No results found in", args.results_dir)
        return
        
    # Preprocess
    # Map intensity to numeric: neutral=0, moderate=1, peak=2
    intensity_map = {"neutral": 0, "moderate": 1, "peak": 2}
    df['intensity_num'] = df['intensity'].map(intensity_map)
    df['is_instruct_num'] = df['is_instruct'].astype(int)
    
    # Create model family grouping (e.g., "Qwen", "Llama", "Gemma")
    def get_family(model_name):
        model_name = model_name.lower()
        if 'qwen' in model_name: return 'Qwen'
        if 'llama' in model_name: return 'Llama'
        if 'gemma' in model_name: return 'Gemma'
        return 'Other'
    
    df['family'] = df['model'].apply(get_family)
    
    # Analyze Valence
    print("Fitting Mixed-Effects Model for Expected Valence...")
    
    # Equation: E_v ~ intensity_num * is_instruct_num + (1 | pair_id) + (1 | model)
    # Since statsmodels mixedlm natively supports 1 group level, we use pair_id as group,
    # and add 'model' as a variance component.
    
    formula = "E_v ~ intensity_num * is_instruct_num"
    vcf = {"model": "0 + C(model)"}
    
    try:
        model = smf.mixedlm(formula, df, groups=df["pair_id"], vc_formula=vcf)
        result = model.fit(method='lbfgs')
        print(result.summary())
        
        with open(args.out_file, "w") as f:
            f.write("=== Mixed-Effects Model for Expected Valence (Across Models) ===\n")
            f.write(result.summary().as_text())
            f.write("\n\n")
            
    except Exception as e:
        print(f"Error fitting mixedlm: {e}")

if __name__ == "__main__":
    main()
