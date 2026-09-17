import pytest
import numpy as np
from affective_empathy_eval.probing import LayerProber, run_rsa_analysis, run_shuffled_baseline_probing

def test_layer_prober():
    prober = LayerProber(alpha=1.0, cv=3, seed=42, use_pca=False)
    rng = np.random.default_rng(42)
    
    y = rng.uniform(-1, 1, size=(30,))
    X = np.outer(y, np.ones(10)) + rng.normal(0, 0.1, size=(30, 10))
    group_ids = np.repeat(np.arange(15), 2)
    
    res = prober.evaluate_probing(X, y, group_ids)
    assert res["r2"] > 0.5
    assert res["pearson_r"] > 0.7

def test_shuffled_baseline_probing():
    prober = LayerProber(alpha=1.0, cv=3, seed=42, use_pca=False)
    rng = np.random.default_rng(42)
    
    y = rng.uniform(-1, 1, size=(30,))
    X = rng.normal(0, 1.0, size=(30, 10))
    group_ids = np.repeat(np.arange(15), 2)
    
    shuffled_res = run_shuffled_baseline_probing(X, y, group_ids, prober, n_permutations=3)
    assert "mean_shuffled_r2" in shuffled_res
