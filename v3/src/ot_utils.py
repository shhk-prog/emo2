"""
v3/src/ot_utils.py

Mathematical Core:
Strict 2D Joint Optimal Transport (OT) Solver and Wasserstein Distance Metrics
for the 9x9 Valence-Arousal grid.

Primary Solver: POT (Python Optimal Transport) ot.emd2
Ground Cost: Manhattan distance on the 9x9 grid: c((v,a), (v',a')) = |v - v'| + |a - a'|
"""

import numpy as np
try:
    import ot
    _HAS_POT = True
except ImportError:
    ot = None
    _HAS_POT = False
from scipy.stats import wasserstein_distance

# Pre-compute and cache the static 81x81 Manhattan ground metric matrix
def _build_manhattan_cost_matrix():
    # Grid coordinates: 81 points ordered by (v, a) for v in 1..9, a in 1..9
    coords = []
    for v in range(1, 10):
        for a in range(1, 10):
            coords.append((v, a))
    coords = np.array(coords, dtype=np.float64) # shape: (81, 2)
    
    # Pairwise Manhattan distance: |v1 - v2| + |a1 - a2|
    # coords[:, None, :] shape (81, 1, 2), coords[None, :, :] shape (1, 81, 2)
    diff = np.abs(coords[:, None, :] - coords[None, :, :])
    cost_matrix = np.sum(diff, axis=-1) # shape: (81, 81)
    return cost_matrix

MANHATTAN_COST_81 = _build_manhattan_cost_matrix()
COORDS_1D = np.arange(1, 10, dtype=np.float64)

def compute_joint_ot_2d(p, q, cost_matrix=MANHATTAN_COST_81):
    """
    Computes the exact 2D Joint Optimal Transport (Earth Mover's Distance)
    between two 9x9 probability distributions P and Q using Manhattan ground cost.
    
    Args:
        p: np.ndarray of shape (9, 9) or (81,)
        q: np.ndarray of shape (9, 9) or (81,)
        cost_matrix: (81, 81) ground cost matrix
        
    Returns:
        float: Exact Optimal Transport distance >= 0.0
    """
    p_flat = np.asarray(p, dtype=np.float64).flatten()
    q_flat = np.asarray(q, dtype=np.float64).flatten()
    
    sum_p = np.sum(p_flat)
    sum_q = np.sum(q_flat)
    
    if sum_p <= 0.0 or sum_q <= 0.0:
        return 0.0
        
    p_norm = p_flat / sum_p
    q_norm = q_flat / sum_q
    
    # Fast check for exact equality
    if np.allclose(p_norm, q_norm, atol=1e-12):
        return 0.0
        
    if not _HAS_POT:
        raise ImportError("POT (Python Optimal Transport) is required for compute_joint_ot_2d. Install it via `pip install POT`.")
    ot_dist = ot.emd2(p_norm, q_norm, cost_matrix)
    return float(ot_dist)

def compute_marginal_wasserstein_sum(p, q):
    """
    Computes the sum of 1D marginal Wasserstein distances (W1_V + W1_A).
    Retained as a secondary / backwards-compatibility metric.
    
    Note: As demonstrated in theoretical tests, this metric can be 0.0 even when
    the joint distributions differ (if marginals match).
    """
    p_mat = np.asarray(p, dtype=np.float64).reshape((9, 9))
    q_mat = np.asarray(q, dtype=np.float64).reshape((9, 9))
    
    # Marginals
    p_v = np.sum(p_mat, axis=1) # shape (9,)
    p_a = np.sum(p_mat, axis=0)
    q_v = np.sum(q_mat, axis=1)
    q_a = np.sum(q_mat, axis=0)
    
    sum_pv = np.sum(p_v)
    sum_pa = np.sum(p_a)
    sum_qv = np.sum(q_v)
    sum_qa = np.sum(q_a)
    
    if sum_pv > 0: p_v /= sum_pv
    if sum_pa > 0: p_a /= sum_pa
    if sum_qv > 0: q_v /= sum_qv
    if sum_qa > 0: q_a /= sum_qa
    
    w1_v = wasserstein_distance(COORDS_1D, COORDS_1D, p_v, q_v)
    w1_a = wasserstein_distance(COORDS_1D, COORDS_1D, p_a, q_a)
    return float(w1_v + w1_a)

def compute_1d_wasserstein(pv, qv):
    """
    Computes 1D Wasserstein distance between 1D probability distributions
    over {1, 2, ..., 9}.
    """
    pv = np.asarray(pv, dtype=np.float64).flatten()
    qv = np.asarray(qv, dtype=np.float64).flatten()
    sum_pv = np.sum(pv)
    sum_qv = np.sum(qv)
    if sum_pv > 0: pv /= sum_pv
    if sum_qv > 0: qv /= sum_qv
    return float(wasserstein_distance(COORDS_1D, COORDS_1D, pv, qv))

def compute_joint_ot_recovery(p_patch, p_peak, p_neut, eps_rec=0.05):
    """
    Computes true Recovery percentage towards Peak distribution:
      Recovery = 1.0 - (OT(P_patch, P_peak) / OT(P_neut, P_peak))
      
    Includes small-denominator safeguard:
      Only returns valid recovery if OT(P_neut, P_peak) >= eps_rec.
      Also returns absolute displacement Delta_patch = OT(P_patch, P_peak).
      
    Returns:
        recovery_ratio: float or None (if denominator < eps_rec)
        delta_patch: float
        delta_neut_peak: float
        is_valid: bool
    """
    d_patch_peak = compute_joint_ot_2d(p_patch, p_peak)
    d_neut_peak = compute_joint_ot_2d(p_neut, p_peak)
    
    if d_neut_peak >= eps_rec:
        recovery = 1.0 - (d_patch_peak / d_neut_peak)
        return float(recovery), float(d_patch_peak), float(d_neut_peak), True
    else:
        return None, float(d_patch_peak), float(d_neut_peak), False
