#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import time

def main():
    parser = argparse.ArgumentParser(description="Acquire continuous V/A annotations (proxy)")
    parser.add_argument("--data-dir", type=str, default="v2/data/processed/aipsy")
    parser.add_argument("--out-dir", type=str, default="v2/data/processed/aipsy_annotated")
    parser.add_argument("--dummy", action="store_true", help="Generate dummy random annotations instead of calling API")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    splits = ["train_strict.csv", "test_strict.csv"]
    for split_file in splits:
        in_path = os.path.join(args.data_dir, split_file)
        if not os.path.exists(in_path):
            continue
            
        df = pd.read_csv(in_path)
        
        # LLM API Call placeholder
        if args.dummy:
            print(f"Generating dummy annotations for {split_file} (N={len(df)})")
            # Generate dummy continuous values around the prototype if available, else random
            def dummy_v(row):
                if pd.isna(row.get('emotion')): return np.random.uniform(1, 9)
                emo = row['emotion'].lower()
                if emo in ['ecstasy', 'admiration', 'amazement']: return np.clip(np.random.normal(7.5, 1.0), 1, 9)
                if emo in ['rage', 'grief', 'terror', 'loathing']: return np.clip(np.random.normal(2.5, 1.0), 1, 9)
                return np.random.uniform(1, 9)
                
            def dummy_a(row):
                if pd.isna(row.get('emotion')): return np.random.uniform(1, 9)
                emo = row['emotion'].lower()
                if emo in ['rage', 'terror', 'ecstasy', 'amazement']: return np.clip(np.random.normal(7.5, 1.0), 1, 9)
                if emo in ['grief', 'loathing']: return np.clip(np.random.normal(3.5, 1.0), 1, 9)
                return np.random.uniform(1, 9)
                
            df['V_H'] = df.apply(dummy_v, axis=1)
            df['A_H'] = df.apply(dummy_a, axis=1)
        else:
            print("Real API call is not implemented yet. Please use --dummy for now.")
            return
            
        out_path = os.path.join(args.out_dir, split_file)
        df.to_csv(out_path, index=False)
        print(f"Saved annotated data to {out_path}")

if __name__ == "__main__":
    main()
