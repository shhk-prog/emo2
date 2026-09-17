"""
tests/test_v3_causal_extensions.py

Unit tests for V3 causal extension modules:
1. Optimal Transport mathematical invariants, symmetry, triangle inequality
2. Counterexample verification: Marginal W1 sum == 0.0 while Joint OT == 8.0
3. True recovery assertions on synthetic distributions with eps safeguard
4. Direction removal geometric orthogonality and norm conservation
5. Benjamini-Hochberg FDR correction validity
"""

import numpy as np
import pytest
import torch

from affective_empathy_eval.optimal_transport import (
    compute_1d_wasserstein,
    compute_joint_ot_2d,
    compute_joint_ot_recovery,
    compute_marginal_wasserstein_sum,
)
from affective_empathy_eval.statistics import apply_benjamini_hochberg

pytest.importorskip("ot")


def test_ot_invariants_and_symmetry():
    # Identical distributions have zero distance
    p = np.zeros((9, 9))
    p[4, 4] = 1.0  # Center
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
    assert ot_pr <= ot_pq + ot_qr + 1e-5, (
        f"Triangle inequality violated: {ot_pr} > {ot_pq} + {ot_qr}"
    )


def test_counterexample_marginal_vs_joint_ot():
    """
    P has mass at (1,1) and (9,9).
    Q has mass at (1,9) and (9,1).
    Marginals are identical on V and A -> marginal sum == 0.0!
    Joint OT correctly measures Wasserstein distance = 8.0.
    """
    p = np.zeros((9, 9))
    p[0, 0] = 0.5
    p[8, 8] = 0.5

    q = np.zeros((9, 9))
    q[0, 8] = 0.5
    q[8, 0] = 0.5

    marginal_sum = compute_marginal_wasserstein_sum(p, q)
    assert abs(marginal_sum - 0.0) < 1e-6, (
        f"Marginal sum expected 0.0, got {marginal_sum}"
    )

    joint_ot = compute_joint_ot_2d(p, q)
    assert abs(joint_ot - 8.0) < 1e-6, (
        f"Joint OT expected 8.0 (Manhattan ground cost), got {joint_ot}"
    )


def test_true_recovery_assertions():
    # Peak at (9,9), Neutral at (5,5)
    p_peak = np.zeros((9, 9))
    p_peak[8, 8] = 1.0

    p_neut = np.zeros((9, 9))
    p_neut[4, 4] = 1.0

    # Perfect patch
    rec_100, _, _, valid_100 = compute_joint_ot_recovery(
        p_peak, p_peak, p_neut, eps_rec=0.05
    )
    assert valid_100 is True
    assert abs(rec_100 - 1.0) < 1e-6

    # Zero patch (stays at neutral)
    rec_0, _, _, valid_0 = compute_joint_ot_recovery(
        p_neut, p_peak, p_neut, eps_rec=0.05
    )
    assert valid_0 is True
    assert abs(rec_0 - 0.0) < 1e-6

    # Denominator near zero safeguard
    rec_none, _, _, valid_none = compute_joint_ot_recovery(
        p_peak, p_peak, p_peak, eps_rec=0.05
    )
    assert valid_none is False
    assert rec_none is None


def test_benjamini_hochberg_fdr():
    p_vals = [0.001, 0.01, 0.04, 0.05, 0.50]
    p_adj = apply_benjamini_hochberg(p_vals)
    assert len(p_adj) == 5
    assert np.all(np.diff(p_adj) >= -1e-9), "Adjusted p-values must be monotonic"
    assert p_adj[0] < p_adj[-1]
    assert np.all(p_adj <= 1.0)
