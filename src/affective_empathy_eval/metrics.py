import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple

def calculate_reactivity_vector(base_v: float, base_a: float, post_v: float, post_a: float) -> Tuple[float, float, float]:
    """
    Calculates the reaction vector (delta_V, delta_A) and magnitude R from normalized [-1, 1] states.
    """
    delta_v = post_v - base_v
    delta_a = post_a - base_a
    magnitude_r = float(np.sqrt(delta_v**2 + delta_a**2))
    return delta_v, delta_a, magnitude_r

def calculate_anchor_direction_alignment(
    delta_v: float, delta_a: float, human_v: float, human_a: float
) -> Tuple[Optional[float], Optional[str]]:
    """
    Calculates Anchor Direction Alignment (ADA): cos(human_vector, delta_LLM_vector).
    Returns (ADA, exclusion_reason). If human vector norm or delta norm is 0, returns (None, 'zero_norm').
    """
    human_norm = float(np.sqrt(human_v**2 + human_a**2))
    delta_norm = float(np.sqrt(delta_v**2 + delta_a**2))
    
    if human_norm < 1e-6 or delta_norm < 1e-6:
        return None, "zero_norm"
        
    dot_product = (human_v * delta_v) + (human_a * delta_a)
    ada = float(dot_product / (human_norm * delta_norm))
    return ada, None

def calculate_post_distance(post_v: float, post_a: float, human_v: float, human_a: float) -> float:
    """Calculates Euclidean distance between post-reported VA state and human stimulus anchor."""
    return float(np.sqrt((post_v - human_v)**2 + (post_a - human_a)**2))

def calculate_stimulus_gain(magnitude_r: float, human_v: float, human_a: float, eps: float = 1e-5) -> float:
    """Calculates stimulus-to-response gain G = R / (||h|| + eps)."""
    human_norm = float(np.sqrt(human_v**2 + human_a**2))
    return float(magnitude_r / (human_norm + eps))

def compute_rsa_similarity(matrix_a: np.ndarray, matrix_b: np.ndarray) -> Dict[str, float]:
    """
    Computes Representational Similarity Analysis (RSA) correlation between two distance matrices.
    Uses upper triangular elements excluding diagonal.
    """
    from scipy.stats import spearmanr, pearsonr
    
    n = matrix_a.shape[0]
    if n != matrix_b.shape[0] or n < 3:
        return {"spearman": np.nan, "pearson": np.nan}
        
    triu_idx = np.triu_indices(n, k=1)
    vec_a = matrix_a[triu_idx]
    vec_b = matrix_b[triu_idx]
    
    if np.std(vec_a) == 0 or np.std(vec_b) == 0:
        pearson_corr = np.nan
        spearman_corr = np.nan
    else:
        spearman_corr, _ = spearmanr(vec_a, vec_b)
        pearson_corr, _ = pearsonr(vec_a, vec_b)
    
    return {
        "spearman": float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
        "pearson": float(pearson_corr) if not np.isnan(pearson_corr) else 0.0
    }

def calculate_euclidean_recovery(
    e_source_v: float, e_source_a: float,
    e_target_v: float, e_target_a: float,
    e_patch_v: float, e_patch_a: float,
    threshold: float = 1e-3
) -> Optional[float]:
    """
    Calculates the Euclidean distance-based normalized Recovery:
    Recovery = 1 - (||E_patch - E_source||_2 / ||E_target - E_source||_2)
    """
    dist_target_source = float(np.sqrt((e_target_v - e_source_v)**2 + (e_target_a - e_source_a)**2))
    
    if dist_target_source < threshold:
        return None
        
    dist_patch_source = float(np.sqrt((e_patch_v - e_source_v)**2 + (e_patch_a - e_source_a)**2))
    
    recovery = 1.0 - (dist_patch_source / dist_target_source)
    return recovery
