#!/usr/bin/env python3
import os
import pandas as pd
from affective_empathy_eval.data import load_aipsy_affect, split_aipsy_affect

def main():
    print("Loading AIPsy-Affect dataset...")
    df = load_aipsy_affect()
    print(f"Loaded {len(df)} rows.")
    
    # Preprocessing to ensure we have a 'pair_id' and correct conditions
    # AIPsy-Affect usually has columns like 'emotion' and 'condition' (affective vs neutral)
    # The 'id' might be something like 'anger_affective_1', we need to create a unified 'pair_id'
    print(df.head())
    
    # Check what columns we actually have
    cols = list(df.columns)
    print("Columns:", cols)
    
    if "pair_id" not in df.columns:
        if "id" in cols:
            df["pair_id"] = df["id"].apply(lambda x: str(x).replace("_affective", "").replace("_neutral", ""))
        else:
            print("Warning: could not infer pair_id. Here is a sample:")
            print(df.head(2).to_dict())
            return
            
    train_df, dev_df, test_df = split_aipsy_affect(df)
    print(f"Splits: Train={len(train_df)}, Dev={len(dev_df)}, Test={len(test_df)}")
    
    out_dir = "v1/data/processed/aipsy"
    os.makedirs(out_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    dev_df.to_csv(os.path.join(out_dir, "dev.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    print(f"Saved splits to {out_dir}")

if __name__ == "__main__":
    main()
