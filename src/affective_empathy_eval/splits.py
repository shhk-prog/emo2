import numpy as np
import pandas as pd
from typing import List, Tuple, Generator, Optional
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

def create_dev_test_split(df: pd.DataFrame, 
                          group_col: str = 'stimulus_id', 
                          test_size: float = 0.2, 
                          random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into development (dev) and test sets, ensuring that no 
    stimulus_id (group) leaks across the split.
    """
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in the DataFrame.")

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    dummy_X = np.zeros(len(df))
    groups = df[group_col].values

    train_idx, test_idx = next(gss.split(dummy_X, y=None, groups=groups))
    dev_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    return dev_df, test_df

def get_grouped_kfold_splits(df: pd.DataFrame, 
                             n_splits: int = 5, 
                             group_col: str = 'stimulus_id') -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generates cross-validation splits using GroupKFold to ensure stimuli are not 
    split across training and validation sets.
    """
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in the DataFrame.")
        
    gkf = GroupKFold(n_splits=n_splits)
    dummy_X = np.zeros(len(df))
    groups = df[group_col].values
    
    for train_idx, val_idx in gkf.split(dummy_X, y=None, groups=groups):
        yield train_idx, val_idx
