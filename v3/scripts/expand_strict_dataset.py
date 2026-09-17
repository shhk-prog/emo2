#!/usr/bin/env python3
"""
v3/scripts/expand_strict_dataset.py

Expands the strict minimal-pair AIPsy-Affect dataset to 50-100 pairs (150-300 samples)
by searching across all available AIPsy processed datasets and raw arrow/csv files,
extracting complete triplets (peak, moderate, neutral), and validating word counts and discourse structure.
Ensures strict 3-way GroupShuffleSplit (Train 40%, Alignment-dev 30%, Held-out Test 30%).
"""

import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

def find_all_aipsy_data():
    records = []
    
    # Check v1 processed
    for path in glob.glob("v1/data/processed/aipsy/*.csv"):
        try:
            df = pd.read_csv(path)
            records.append(df)
            print(f"Loaded {len(df)} rows from {path}")
        except Exception as e:
            print(f"Error reading {path}: {e}")
            
    # Check v2 processed
    for path in glob.glob("v2/data/processed/aipsy_annotated/*.csv"):
        try:
            df = pd.read_csv(path)
            records.append(df)
            print(f"Loaded {len(df)} rows from {path}")
        except Exception as e:
            print(f"Error reading {path}: {e}")
            
    # Check raw datasets if pyarrow / datasets available
    arrow_files = glob.glob("v1/data/raw/keidolabs___aipsy-affect/**/*.arrow", recursive=True)
    for af in arrow_files:
        try:
            import pyarrow as pa
            import pyarrow.feather as feather
            table = feather.read_table(af)
            adf = table.to_pandas()
            records.append(adf)
            print(f"Loaded {len(adf)} rows from {af}")
        except Exception as e:
            pass

    if not records:
        raise RuntimeError("No AIPsy data found!")
        
    combined = pd.concat(records, ignore_index=True)
    # Deduplicate by text or id if available
    if 'id' in combined.columns and 'text' in combined.columns:
        combined = combined.drop_duplicates(subset=['id', 'text'])
    elif 'text' in combined.columns:
        combined = combined.drop_duplicates(subset=['text'])
        
    print(f"Total unified unique rows: {len(combined)}")
    return combined

def extract_and_verify_triplets(df):
    """
    Identifies clean triplets:
    For a given baseline pair/scenario:
    - Peak affective
    - Moderate affective
    - Matched Neutral
    """
    # Standardize columns
    # Need: id, pair_id, condition, intensity, text, word_count, emotion
    print("Columns present:", list(df.columns))
    
    # Normalize condition & intensity
    df['condition'] = df['condition'].fillna('').astype(str).str.lower()
    df['intensity'] = df['intensity'].fillna('').astype(str).str.lower()
    
    # Infer pair_id if missing or inconsistent
    # E.g. B-rage-d1-v1 -> matched_control_id N-rage-d1-c1
    # Often pair_id connects them, or matched_control_id connects peak/moderate to neutral
    if 'pair_id' not in df.columns or df['pair_id'].nunique() < 10:
        if 'matched_control_id' in df.columns:
            df['pair_id'] = df['matched_control_id'].fillna(df['id'])
            
    # If pair_id is e.g. N-rage-d1-c1, neutral text might have id == N-rage-d1-c1
    # Let's see how neutral items are identified
    # Some neutral items have condition == 'neutral' and id like 'N-...'
    print("Conditions count:\n", df['condition'].value_counts())
    print("Intensity count:\n", df['intensity'].value_counts())
    
    def get_pair_key(row):
        uid = str(row.get('id', '')).strip()
        mid = str(row.get('matched_control_id', '')).strip()
        pid = str(row.get('pair_id', '')).strip()
        
        # In AIPsy, the neutral counterpart is always prefixed with 'N-'
        # Affective has matched_control_id='N-...', Neutral has id='N-...'
        # Moderate has pair_id='N-...'
        if uid.startswith('N-'):
            return uid
        if mid.startswith('N-'):
            return mid
        if pid.startswith('N-'):
            return pid
            
        # Fallback: sort non-empty identifiers deterministically
        cands = sorted([c for c in [uid, mid, pid] if c and c not in ['nan', 'None']])
        return cands[0] if cands else uid
        
    df['pair_key'] = df.apply(get_pair_key, axis=1)
    print(f"Unique pair keys: {df['pair_key'].nunique()}")
    
    valid_pairs = []
    
    for pkey, group in df.groupby('pair_key'):
        peak_rows = group[(group['condition'] == 'affective') & (group['intensity'] == 'peak')]
        mod_rows = group[(group['condition'] == 'affective') & (group['intensity'] == 'moderate')]
        neu_rows = group[(group['condition'] == 'neutral') | (group['intensity'].isin(['none', 'neutral'])) | (group['id'].str.startswith('N-'))]
        
        if len(peak_rows) >= 1 and len(neu_rows) >= 1:
            p_row = peak_rows.iloc[0].copy()
            n_row = neu_rows.iloc[0].copy()
            
            p_row['pair_id'] = pkey
            p_row['condition'] = 'affective'
            p_row['intensity'] = 'peak'
            
            n_row['pair_id'] = pkey
            n_row['condition'] = 'neutral'
            n_row['intensity'] = 'neutral'
            
            if len(mod_rows) >= 1:
                m_row = mod_rows.iloc[0].copy()
                m_row['pair_id'] = pkey
                m_row['condition'] = 'affective'
                m_row['intensity'] = 'moderate'
                valid_pairs.extend([p_row.to_dict(), m_row.to_dict(), n_row.to_dict()])
            else:
                valid_pairs.extend([p_row.to_dict(), n_row.to_dict()])
                
    if not valid_pairs:
        raise RuntimeError("No matched pairs found! Check dataset matching criteria.")
        
    expanded_df = pd.DataFrame(valid_pairs)
    print(f"Extracted {expanded_df['pair_id'].nunique()} clean pairs ({len(expanded_df)} total rows).")
    
    # Calculate word counts if missing
    expanded_df['word_count'] = expanded_df['text'].apply(lambda x: len(str(x).split()))
    
    # Filter for quality: 40 <= word_count <= 250
    expanded_df = expanded_df[(expanded_df['word_count'] >= 40) & (expanded_df['word_count'] <= 250)]
    
    # Keep only pairs where both peak and neutral exist
    pair_counts = expanded_df.groupby('pair_id')['intensity'].unique()
    complete_pids = [pid for pid, ints in pair_counts.items() if 'peak' in ints and 'neutral' in ints]
    expanded_df = expanded_df[expanded_df['pair_id'].isin(complete_pids)].copy()
    
    print(f"After word count filtering: {expanded_df['pair_id'].nunique()} complete pairs ({len(expanded_df)} rows).")
    return expanded_df

def apply_strict_split(df, group_col="pair_id"):
    """
    Strict 3-Way GroupShuffleSplit ensuring no pair_id leakage across splits:
    - Train: 40%
    - Alignment-Dev: 30%
    - Held-out Test: 30%
    """
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.6, random_state=42)
    train_idx, temp_idx = next(gss1.split(df, groups=df[group_col]))
    
    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()
    
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    dev_idx, test_idx = next(gss2.split(temp_df, groups=temp_df[group_col]))
    
    dev_df = temp_df.iloc[dev_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()
    
    train_df['split'] = 'train'
    dev_df['split'] = 'dev'
    test_df['split'] = 'test'
    
    final_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)
    
    print(f"Final Splits by Pairs:")
    print(f"  Train: {train_df[group_col].nunique()} pairs ({len(train_df)} rows)")
    print(f"  Dev:   {dev_df[group_col].nunique()} pairs ({len(dev_df)} rows)")
    print(f"  Test:  {test_df[group_col].nunique()} pairs ({len(test_df)} rows)")
    
    # Assert zero leakage
    train_set = set(train_df[group_col].unique())
    dev_set = set(dev_df[group_col].unique())
    test_set = set(test_df[group_col].unique())
    
    assert len(train_set & dev_set) == 0, "Leakage between train and dev!"
    assert len(train_set & test_set) == 0, "Leakage between train and test!"
    assert len(dev_set & test_set) == 0, "Leakage between dev and test!"
    print("Zero-leakage verification passed successfully!")
    
    return final_df

def main():
    os.makedirs("v3/data", exist_ok=True)
    raw_df = find_all_aipsy_data()
    clean_triplets = extract_and_verify_triplets(raw_df)
    final_dataset = apply_strict_split(clean_triplets)
    
    out_path = "v3/data/aipsy_strict_expanded.csv"
    final_dataset.to_csv(out_path, index=False)
    print(f"Saved expanded dataset to {out_path}")

if __name__ == "__main__":
    main()
