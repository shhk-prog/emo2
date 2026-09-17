import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Basic emotional words list for Lexical-emotion detection
EMOTION_LEXICON = {
    "happy", "sad", "angry", "fear", "scared", "afraid", "joy", "grief",
    "delighted", "depressed", "furious", "terrified", "miserable", "ecstatic",
    "love", "hate", "disgust", "hope", "anxious", "sorrow", "pain", "pleasure"
}

def is_lexical_emotion(text: str) -> bool:
    """Checks if text contains explicit emotion words."""
    words = set(text.lower().replace(".", "").replace(",", "").split())
    return len(words.intersection(EMOTION_LEXICON)) > 0

def filter_control_subsets(stimuli_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Categorizes stimuli into control subsets:
    - original: full set
    - lexical_emotion: contains explicit emotion words
    - contextual_emotion: emotion conveyed without explicit emotion words
    - neutral: near-zero human VA origin (|V_scaled| <= 0.2 and |A_scaled| <= 0.2)
    """
    df = stimuli_df.copy()
    
    if 'text' in df.columns:
        df['has_explicit_emotion'] = df['text'].apply(is_lexical_emotion)
    else:
        df['has_explicit_emotion'] = False

    if 'V_scaled' in df.columns and 'A_scaled' in df.columns:
        is_neutral = (df['V_scaled'].abs() <= 0.2) & (df['A_scaled'].abs() <= 0.2)
    else:
        is_neutral = pd.Series([False] * len(df))

    return {
        "original": df,
        "lexical_emotion": df[df['has_explicit_emotion']],
        "contextual_emotion": df[~df['has_explicit_emotion'] & ~is_neutral],
        "neutral": df[is_neutral]
    }

def generate_shuffled_control_dataset(stimuli_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Generates a Label-shuffled control dataset where stimulus texts are randomly permuted 
    relative to their human VA annotation labels.
    """
    df_shuffled = stimuli_df.copy()
    rng = np.random.default_rng(seed)
    
    shuffled_indices = rng.permutation(len(df_shuffled))
    if 'V_scaled' in df_shuffled.columns:
        df_shuffled['V_scaled'] = df_shuffled['V_scaled'].values[shuffled_indices]
    if 'A_scaled' in df_shuffled.columns:
        df_shuffled['A_scaled'] = df_shuffled['A_scaled'].values[shuffled_indices]
        
    df_shuffled['is_label_shuffled'] = True
    return df_shuffled

def calculate_empirical_p_value(true_stat: float, shuffled_stats: List[float], alternative: str = 'greater') -> float:
    """
    Calculates the empirical p-value from a permutation test.
    
    Args:
        true_stat: The test statistic from the true (unshuffled) data.
        shuffled_stats: A list of test statistics from the shuffled datasets.
        alternative: 'greater', 'less', or 'two-sided'.
        
    Returns:
        The empirical p-value.
    """
    shuffled_stats_arr = np.array(shuffled_stats)
    n = len(shuffled_stats_arr)
    
    if alternative == 'greater':
        count = np.sum(shuffled_stats_arr >= true_stat)
    elif alternative == 'less':
        count = np.sum(shuffled_stats_arr <= true_stat)
    elif alternative == 'two-sided':
        true_abs = np.abs(true_stat)
        count = np.sum(np.abs(shuffled_stats_arr) >= true_abs)
    else:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")
        
    # pseudo-count formulation (count + 1) / (N + 1)
    return float((count + 1) / (n + 1))
