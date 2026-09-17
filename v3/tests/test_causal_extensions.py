"""
v3/tests/test_causal_extensions.py

Unit tests for causal extension modules:
1. Optimal Transport mathematical invariants, symmetry, triangle inequality
2. Counterexample verification: Marginal W1 sum == 0.0 while Joint OT == 8.0
3. True recovery assertions on synthetic distributions with eps safeguard
4. Direction removal geometric orthogonality and norm conservation
5. Random null direction sampling and Gram-Schmidt orthogonality
6. Benjamini-Hochberg FDR correction validity
"""

import pytest
pytest.importorskip("ot")
import numpy as np
import torch

from v3.src.ot_utils import (
    compute_joint_ot_2d,
    compute_marginal_wasserstein_sum,
    compute_1d_wasserstein,
    compute_joint_ot_recovery
)
from v3.src.model_utils import (
    get_patch_hook,
    get_direction_ablation_hook,
    get_replacement_hook
)
from v3.scripts.run_probe_aligned_necessity_sweep import (
    sample_random_directions,
    compute_empirical_pvalue_and_zscore,
    apply_benjamini_hochberg
)

def test_ot_invariants_and_symmetry():
    # Identical distributions have zero distance
    p = np.zeros((9, 9))
    p[4, 4] = 1.0 # Center
    assert compute_joint_ot_2d(p, p) < 1e-6
    
    # Random distributions for symmetry & triangle inequality
    rng = np.random.RandomState(42)
    p = rng.dirichlet(np.ones(81)).reshape((9, 9))
    q = rng.dirichlet(np.ones(81)).reshape((9, 9))
    r = rng.dirichlet(np.ones(81)).reshape((9, 9))
    
    ot_pq = compute_joint_ot_2d(p, q)
    ot_qp = compute_joint_ot_2d(q, p)
    assert abs(ot_pq - ot_qp) < 1e-6, f"Symmetry violated: {ot_pq} vs {ot_qp}"
    
    # Triangle inequality: OT(P, R) <= OT(P, Q) + OT(Q, R)
    ot_pr = compute_joint_ot_2d(p, r)
    ot_qr = compute_joint_ot_2d(q, r)
    assert ot_pr <= ot_pq + ot_qr + 1e-5, f"Triangle inequality violated: {ot_pr} > {ot_pq} + {ot_qr}"

def test_counterexample_marginal_vs_joint_ot():
    """
    CRITICAL TEST:
    P has mass at (1,1) and (9,9).
    Q has mass at (1,9) and (9,1).
    Both have identical marginals (0.5 at 1, 0.5 at 9 for both V and A).
    Marginal Wasserstein sum MUST be 0.0.
    Joint OT (Manhattan ground cost) MUST be strictly 8.0.
    """
    p = np.zeros((9, 9))
    p[0, 0] = 0.5 # (v=1, a=1)
    p[8, 8] = 0.5 # (v=9, a=9)
    
    q = np.zeros((9, 9))
    q[0, 8] = 0.5 # (v=1, a=9)
    q[8, 0] = 0.5 # (v=9, a=1)
    
    marg_sum = compute_marginal_wasserstein_sum(p, q)
    joint_ot = compute_joint_ot_2d(p, q)
    
    assert abs(marg_sum) < 1e-6, f"Expected marginal sum 0.0, got {marg_sum}"
    assert abs(joint_ot - 8.0) < 1e-5, f"Expected joint OT 8.0, got {joint_ot}"
    print(f"Counterexample verified: Marginal={marg_sum:.4f}, Joint={joint_ot:.4f}")

def test_recovery_synthetic_distributions():
    # Peak at (9, 9), Neutral at (5, 5)
    p_peak = np.zeros((9, 9))
    p_peak[8, 8] = 1.0
    
    p_neut = np.zeros((9, 9))
    p_neut[4, 4] = 1.0
    
    # 1. Patch == Neutral -> Recovery == 0.0
    rec_zero, d_patch, d_np, is_valid = compute_joint_ot_recovery(p_neut, p_peak, p_neut, eps_rec=0.05)
    assert is_valid
    assert abs(rec_zero) < 1e-5, f"Expected Recovery 0.0, got {rec_zero}"
    
    # 2. Patch == Peak -> Recovery == 1.0
    rec_one, d_patch, d_np, is_valid = compute_joint_ot_recovery(p_peak, p_peak, p_neut, eps_rec=0.05)
    assert is_valid
    assert abs(rec_one - 1.0) < 1e-5, f"Expected Recovery 1.0, got {rec_one}"
    
    # 3. Small denominator safeguard
    # Peak and Neutral almost identical
    p_close = np.copy(p_neut)
    rec_invalid, _, _, is_valid = compute_joint_ot_recovery(p_close, p_close, p_neut, eps_rec=0.05)
    assert not is_valid
    assert rec_invalid is None

def test_direction_removal_geometric_properties():
    dim = 64
    rng = np.random.RandomState(42)
    v_raw = rng.randn(dim)
    v_unit = v_raw / np.linalg.norm(v_raw)
    
    h = rng.randn(dim)
    
    # Manual projection removal
    proj = np.dot(h, v_unit)
    h_prime = h - proj * v_unit
    
    # Orthogonality: (h') . v == 0
    dot_prod = np.dot(h_prime, v_unit)
    assert abs(dot_prod) < 1e-6, f"Orthogonality violated: {dot_prod}"
    
    # Norm conservation: ||h - h'|| == |h . v|
    diff_norm = np.linalg.norm(h - h_prime)
    assert abs(diff_norm - abs(proj)) < 1e-6, f"Norm relation violated: {diff_norm} vs {abs(proj)}"
    
    # Hook implementation verification
    h_tensor = torch.tensor(h, dtype=torch.float32).unsqueeze(0).unsqueeze(1) # (1, 1, dim)
    hook = get_direction_ablation_hook(v_unit, target_pos=0)
    
    out_tensor = hook(None, None, h_tensor.clone())
    out_np = out_tensor[0, 0, :].numpy()
    assert np.allclose(out_np, h_prime, atol=1e-5)

def test_random_direction_sampling_orthogonality():
    dim = 32
    v_probe = np.zeros(dim)
    v_probe[0] = 1.0
    
    # Orthogonal directions
    ortho_dirs = sample_random_directions(v_probe, n_samples=20, mode="orthogonal", seed=123)
    for vec in ortho_dirs:
        # Check unit norm
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5
        # Check orthogonality with probe
        dot = np.dot(vec, v_probe)
        assert abs(dot) < 1e-6, f"Non-orthogonal direction sampled: {dot}"

def test_benjamini_hochberg_fdr():
    # 5 hypotheses: 2 strong, 1 moderate, 2 null
    p_vals = np.array([0.001, 0.005, 0.04, 0.50, 0.80])
    q_vals, sig = apply_benjamini_hochberg(p_vals, q_threshold=0.05)
    
    # q-values must be non-decreasing with respect to sorted p-values
    assert np.all(q_vals[:-1] <= q_vals[1:] + 1e-12)
    # The first two should easily be significant at q < 0.05
    assert sig[0] and sig[1]
    # The last two should be insignificant
    assert not sig[3] and not sig[4]

if __name__ == "__main__":
    pytest.main(["-v", __file__])
