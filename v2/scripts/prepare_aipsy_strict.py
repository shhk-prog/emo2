#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from datasets import load_dataset

def main():
    print("Loading AIPsy-Affect datasets from HF...")
    
    # load splits
    df_clinical = load_dataset('keidolabs/aipsy-affect', split='clinical').to_pandas()
    df_clinical['condition'] = 'affective'
    df_clinical['intensity'] = 'peak'
    
    df_moderate = load_dataset('keidolabs/aipsy-affect', split='moderate').to_pandas()
    df_moderate['condition'] = 'affective'
    df_moderate['intensity'] = 'moderate'
    
    df_neutral = load_dataset('keidolabs/aipsy-affect', split='neutral').to_pandas()
    df_neutral['condition'] = 'neutral'
    df_neutral['intensity'] = 'none'
    
    full_df = pd.concat([df_clinical, df_moderate, df_neutral], ignore_index=True)
    print(f"Loaded {len(full_df)} rows in total.")
    
    # Create a mapping from peak ID to neutral ID (which is the matched_control_id)
    peak_id_to_control = df_clinical.set_index('id')['matched_control_id'].to_dict()
    
    # pair_id の再構築
    def get_pair_id(row):
        if row['condition'] == 'neutral':
            return row['id']
        elif row['condition'] == 'affective' and row['intensity'] == 'peak':
            return row['matched_control_id']
        elif row['condition'] == 'affective' and row['intensity'] == 'moderate':
            # Bm-rage-d1-v1 -> B-rage-d1-v1
            peak_id = str(row['id']).replace('Bm-', 'B-', 1)
            return peak_id_to_control.get(peak_id, None)
        return None
            
    full_df['pair_id'] = full_df.apply(get_pair_id, axis=1)
    # Drop rows without pair_id (should be none, but just in case)
    full_df = full_df.dropna(subset=['pair_id'])
    
    print("\nIntensity distribution:")
    print(pd.crosstab(full_df['condition'], full_df['intensity']))
    
    # pair_id ごとに抽出
    pairs = full_df.groupby('pair_id')
    print(f"\nTotal unique pair_ids: {len(pairs)}")
    
    strict_pairs = []
    for pid, group in pairs:
        intensities = set(group['intensity'].unique())
        if 'moderate' in intensities:
            print(f"PID: {pid} has intensities {intensities}")
        if 'peak' in intensities and 'moderate' in intensities and 'none' in intensities:
            strict_pairs.append(pid)
            
    print(f"Number of strict matched pair_ids (has neutral, moderate, peak): {len(strict_pairs)}")
    
    strict_df = full_df[full_df['pair_id'].isin(strict_pairs)].copy()
    print(f"Strict matched subset has {len(strict_df)} rows.")
    
    # 分割して保存 (Group Split by pair_id)
    np.random.seed(42)
    shuffled_pairs = np.random.permutation(strict_pairs)
    n = len(shuffled_pairs)
    train_pairs = shuffled_pairs[:int(0.6*n)]
    dev_pairs = shuffled_pairs[int(0.6*n):int(0.8*n)]
    test_pairs = shuffled_pairs[int(0.8*n):]
    
    out_dir = "v2/data/processed/aipsy"
    os.makedirs(out_dir, exist_ok=True)
    
    train_df = strict_df[strict_df['pair_id'].isin(train_pairs)]
    dev_df = strict_df[strict_df['pair_id'].isin(dev_pairs)]
    test_df = strict_df[strict_df['pair_id'].isin(test_pairs)]
    
    train_df.to_csv(os.path.join(out_dir, "train_strict.csv"), index=False)
    dev_df.to_csv(os.path.join(out_dir, "dev_strict.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test_strict.csv"), index=False)
    
    print(f"Saved strict splits to {out_dir}")
    print(f"Train: {len(train_df)} rows ({len(train_pairs)} pairs)")
    print(f"Dev: {len(dev_df)} rows ({len(dev_pairs)} pairs)")
    print(f"Test: {len(test_df)} rows ({len(test_pairs)} pairs)")

if __name__ == "__main__":
    main()
