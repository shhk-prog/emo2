import numpy as np
import pandas as pd
import pytest
from scipy import stats

from affective_empathy_eval.statistics import (
    compute_bootstrap_ci,
    compute_bivariate_bootstrap_ci,
    compute_paired_cohen_dz,
    generate_derangement,
    apply_fdr_correction,
    fit_sample_level_lmm,
)


def test_paired_cohen_dz():
    # When differences are constant 2.0 with zero variance
    x1 = np.array([5.0, 6.0, 7.0])
    x2 = np.array([3.0, 4.0, 5.0])
    dz = compute_paired_cohen_dz(x1, x2, ddof=1)
    assert np.isnan(dz)  # std is 0 -> effect size is undefined (NaN)

    # With realistic differences
    x1 = np.array([5.0, 7.0, 6.0, 8.0, 9.0])
    x2 = np.array([3.0, 4.0, 4.0, 5.0, 6.0])
    # diffs: [2, 3, 2, 3, 3], mean = 2.6, s = sqrt(0.3)
    dz = compute_paired_cohen_dz(x1, x2, ddof=1)
    diffs = x1 - x2
    expected_dz = float(np.mean(diffs) / np.std(diffs, ddof=1))
    assert pytest.approx(dz, rel=1e-5) == expected_dz
    assert dz > 0.0


def test_generate_derangement():
    rng = np.random.default_rng(123)
    for n in [2, 5, 20, 100]:
        deranged = generate_derangement(n, rng)
        assert len(deranged) == n
        # Check permutation property
        assert set(deranged) == set(range(n))
        # Check fixed-point free property: deranged[i] != i for all i
        for i in range(n):
            assert deranged[i] != i

    # Edge cases
    assert list(generate_derangement(0, rng)) == []
    assert list(generate_derangement(1, rng)) == [0]


def test_bivariate_bootstrap_ci():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.2, 1.9, 3.1, 3.8, 5.2])
    
    # Correlation statistic
    def corr_stat(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return float(stats.pearsonr(a, b)[0])

    ci_low, ci_high = compute_bivariate_bootstrap_ci(x, y, stat_fn=corr_stat, n_bootstraps=500, ci=0.95, seed=42)
    assert -1.0 <= ci_low <= ci_high <= 1.0
    assert ci_low > 0.8  # Strong positive correlation


def test_e6_pivoted_rm_anova_equivalence():
    """Verify that pivoted repeated-measures difference correctly aligns pair_ids regardless of row ordering."""
    pairs = [f"p_{i}" for i in range(10)]
    records = []
    for p in pairs:
        records.append({"pair_id": p, "task": "Reader", "site_type": "ReaderSite", "outcome": 0.8})
        records.append({"pair_id": p, "task": "Self", "site_type": "ReaderSite", "outcome": 0.2})
        records.append({"pair_id": p, "task": "Reader", "site_type": "SelfSite", "outcome": 0.3})
        records.append({"pair_id": p, "task": "Self", "site_type": "SelfSite", "outcome": 0.9})
    
    df = pd.DataFrame(records)
    # Shuffle dataframe completely
    df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    piv = df_shuffled.pivot(index="pair_id", columns=["task", "site_type"], values="outcome").dropna()
    diff_rs = (piv[("Reader", "ReaderSite")] - piv[("Self", "ReaderSite")]).values
    diff_ss = (piv[("Reader", "SelfSite")] - piv[("Self", "SelfSite")]).values

    # ReaderSite impact is +0.6 for all pairs; SelfSite impact is -0.6 for all pairs
    assert np.allclose(diff_rs, 0.6)
    assert np.allclose(diff_ss, -0.6)
    
    # Crossover condition
    mean_r_rs = piv[("Reader", "ReaderSite")].mean()
    mean_s_rs = piv[("Self", "ReaderSite")].mean()
    mean_r_ss = piv[("Reader", "SelfSite")].mean()
    mean_s_ss = piv[("Self", "SelfSite")].mean()
    assert (mean_r_rs > mean_s_rs) and (mean_s_ss > mean_r_ss)
