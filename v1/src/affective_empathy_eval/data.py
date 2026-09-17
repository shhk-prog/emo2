import pandas as pd
import numpy as np

def scale_vad(raw_value: float) -> float:
    """Scales EmoBank 5-point rating to approximately [-1, 1]."""
    return (raw_value - 3) / 2

def load_emobank(filepath: str, v_col: str = "V", a_col: str = "A") -> pd.DataFrame:
    """Loads EmoBank dataset and applies V/A scaling using explicit column names."""
    df = pd.read_csv(filepath)
    
    if v_col not in df.columns or a_col not in df.columns:
        raise ValueError(f"Columns {v_col} and/or {a_col} not found in {filepath}. Available: {df.columns}")
        
    df['V_scaled'] = df[v_col].apply(scale_vad)
    df['A_scaled'] = df[a_col].apply(scale_vad)
    
    return df

def stratify_stimuli(df: pd.DataFrame, cells_v: int = 3, cells_a: int = 3, n_per_cell: int = 50, seed: int = 42):
    """Stratifies the stimuli into a VA grid and samples n_per_cell.
    Returns: (sampled_df, report_df)
    """
    df_filtered = df.copy()
    
    v_bins = np.linspace(-1, 1, cells_v + 1)
    a_bins = np.linspace(-1, 1, cells_a + 1)
    
    df_filtered['v_cell'] = pd.cut(df_filtered['V_scaled'], bins=v_bins, labels=False, include_lowest=True)
    df_filtered['a_cell'] = pd.cut(df_filtered['A_scaled'], bins=a_bins, labels=False, include_lowest=True)
    
    report_rows = []
    sampled_blocks = []
    
    for (v_cell, a_cell), group in df_filtered.groupby(['v_cell', 'a_cell']):
        candidate_n = len(group)
        sampled_n = min(candidate_n, n_per_cell)
        shortfall_n = n_per_cell - sampled_n
        
        report_rows.append({
            'v_cell': v_cell,
            'a_cell': a_cell,
            'candidate_n': candidate_n,
            'sampled_n': sampled_n,
            'target_n': n_per_cell,
            'shortfall_n': shortfall_n
        })
        
        if sampled_n > 0:
            sampled_blocks.append(group.sample(sampled_n, random_state=seed))
            
    sampled_df = pd.concat(sampled_blocks, ignore_index=True) if sampled_blocks else pd.DataFrame()
    report_df = pd.DataFrame(report_rows)
    
    return sampled_df, report_df

def load_aipsy_affect(cache_dir: str = "data/raw") -> pd.DataFrame:
    """
    Loads the AIPsy-Affect dataset from HuggingFace Hub or local cache.
    The dataset provides clinical vignettes with and without emotion keywords/content,
    arranged in minimal pairs.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("The 'datasets' package is required to load AIPsy-Affect. Please install it.")
        
    ds_clinical = load_dataset("keidolabs/aipsy-affect", split="clinical", cache_dir=cache_dir)
    ds_neutral = load_dataset("keidolabs/aipsy-affect", split="neutral", cache_dir=cache_dir)
    
    df_clinical = ds_clinical.to_pandas()
    df_clinical["condition"] = "affective"
    
    df_neutral = ds_neutral.to_pandas()
    df_neutral["condition"] = "neutral"
    
    df = pd.concat([df_clinical, df_neutral], ignore_index=True)
    return df

def split_aipsy_affect(df: pd.DataFrame, train_ratio: float = 0.6, dev_ratio: float = 0.2, seed: int = 42):
    """
    Splits the AIPsy-Affect dataset into train/dev/test strictly by Minimal Pair ID
    to prevent data leakage between affective and neutral conditions.
    """
    if "pair_id" not in df.columns:
        # Fallback if the dataset structure differs, create a pseudo pair ID if possible
        # Usually AIPsy-Affect has something like 'pair_id' or we can infer it
        if "id" in df.columns:
            # Assuming IDs have a structure that indicates the pair, e.g., pair1_affective, pair1_neutral
            df["pair_id"] = df["id"].apply(lambda x: str(x).rsplit("_", 1)[0] if "_" in str(x) else str(x))
        else:
            raise ValueError("Dataset does not contain a 'pair_id' or 'id' column for minimal pair splitting.")
            
    unique_pairs = np.array(df["pair_id"].unique().tolist())
    np.random.seed(seed)
    np.random.shuffle(unique_pairs)
    
    n_pairs = len(unique_pairs)
    n_train = int(n_pairs * train_ratio)
    n_dev = int(n_pairs * dev_ratio)
    
    train_pairs = unique_pairs[:n_train]
    dev_pairs = unique_pairs[n_train:n_train + n_dev]
    test_pairs = unique_pairs[n_train + n_dev:]
    
    train_df = df[df["pair_id"].isin(train_pairs)].copy()
    dev_df = df[df["pair_id"].isin(dev_pairs)].copy()
    test_df = df[df["pair_id"].isin(test_pairs)].copy()
    
    return train_df, dev_df, test_df

