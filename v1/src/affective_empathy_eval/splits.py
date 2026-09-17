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
    
    Args:
        df: Input DataFrame.
        group_col: The column name representing the groups (e.g., stimulus_id).
        test_size: Proportion of the dataset to include in the test split.
        random_state: Random state for reproducibility.
        
    Returns:
        A tuple of (dev_df, test_df).
    """
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in the DataFrame.")

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    # We don't actually need X and y for the split index calculation
    # Just need an array of the right length
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
    
    Args:
        df: Input DataFrame (e.g., the dev set).
        n_splits: Number of folds.
        group_col: Column used for grouping.
        
    Yields:
        Tuple of (train_indices, val_indices).
    """
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in the DataFrame.")
        
    gkf = GroupKFold(n_splits=n_splits)
    
    dummy_X = np.zeros(len(df))
    groups = df[group_col].values
    
    for train_idx, val_idx in gkf.split(dummy_X, y=None, groups=groups):
        yield train_idx, val_idx
