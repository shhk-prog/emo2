import pytest
import numpy as np
from affective_empathy_eval.metrics import (
    calculate_reactivity_vector,
    calculate_anchor_direction_alignment,
    calculate_post_distance,
    calculate_stimulus_gain,
    compute_rsa_similarity
)

def test_calculate_reactivity_vector():
    delta_v, delta_a, magnitude_r = calculate_reactivity_vector(0.0, 0.0, 0.5, 0.5)
    assert delta_v == 0.5
    assert delta_a == 0.5
    assert pytest.approx(magnitude_r) == np.sqrt(0.5)

def test_calculate_anchor_direction_alignment_valid():
    ada, reason = calculate_anchor_direction_alignment(0.5, 0.5, 0.5, 0.5)
    assert reason is None
    assert pytest.approx(ada) == 1.0

def test_calculate_anchor_direction_alignment_zero_norm():
    ada, reason = calculate_anchor_direction_alignment(0.0, 0.0, 0.5, 0.5)
    assert ada is None
    assert reason == "zero_norm"

def test_calculate_stimulus_gain():
    gain = calculate_stimulus_gain(magnitude_r=1.0, human_v=0.6, human_a=0.8)
    # human norm is sqrt(0.36 + 0.64) = 1.0
    assert pytest.approx(gain, rel=1e-3) == 1.0

def test_compute_rsa_similarity():
    mat_a = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
    mat_b = np.array([[0, 2, 4], [2, 0, 2], [4, 2, 0]], dtype=float)
    res = compute_rsa_similarity(mat_a, mat_b)
    assert pytest.approx(res["spearman"]) == 1.0
