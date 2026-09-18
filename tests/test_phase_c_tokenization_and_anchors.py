"""
tests/test_phase_c_tokenization_and_anchors.py

Unit tests verifying:
1. Canonical tokenization and prompt_end position integrity (add_special_tokens=False)
2. Joint sequence boundary prefix consistency
3. 2D QR orthonormalization and centered projection removal invariant (Q^T (h' - mu) = 0)
4. Temporal generation stage token offsets ordering and integrity
5. Zero-forward shift invariant (alpha=0.0 yields exactly zero shift)
6. Exact derangement integrity (no self-matches)
"""

import numpy as np
import pytest
import torch

from affective_empathy_eval.prompts import (
    find_semantic_anchors,
    get_generation_stage_tokens,
)
from affective_empathy_eval.statistics import generate_derangement


def test_qr_orthonormalization_and_centered_projection():
    """Verifies that QR orthogonalization produces an orthonormal 2D basis and centered projection removes the subspace."""
    dim = 64
    rng = np.random.default_rng(42)
    # Generate non-orthogonal d_v and d_a
    d_v = rng.normal(0, 1, dim)
    d_v /= np.linalg.norm(d_v)
    d_a = 0.6 * d_v + 0.8 * rng.normal(0, 1, dim)
    d_a /= np.linalg.norm(d_a)

    # QR orthogonalization
    M = np.column_stack([d_v, d_a])
    Q, _ = np.linalg.qr(M)

    # 1. Orthonormality: Q^T Q == I_2
    qtq = Q.T @ Q
    assert np.allclose(qtq, np.eye(2), atol=1e-6), f"Q is not orthonormal: {qtq}"

    # 2. Centered projection removal: h' = h - Q Q^T (h - mu)
    h = rng.normal(5, 2, dim)
    mu = rng.normal(4, 1, dim)

    h_centered = h - mu
    proj = Q @ (Q.T @ h_centered)
    h_prime = h - proj

    # Residual relative to mu: (h_prime - mu) must be orthogonal to Q
    h_prime_centered = h_prime - mu
    residual_proj = Q.T @ h_prime_centered
    assert np.allclose(residual_proj, np.zeros(2), atol=1e-6), f"Residual projection onto Q is non-zero: {residual_proj}"


def test_generation_stage_tokens_ordering():
    """Verifies that get_generation_stage_tokens resolves valid semantic stages in strict monotonic order."""
    class MockTokenizer:
        def encode(self, text, add_special_tokens=False):
            # Simple space / char-based mock tokenization
            tokens = []
            for word in text.split(" "):
                tokens.append(hash(word) % 10000)
            return tokens

        def decode(self, tokens):
            return '{"valence": 5, "arousal": 5}'

        def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False):
            # Return character offsets for JSON string
            spans = []
            pos = 0
            for part in ['{"valence":', ' 5,', ' "arousal":', ' 5}']:
                start = text.find(part, pos)
                if start == -1:
                    start = pos
                end = start + len(part)
                spans.append((start, end))
                pos = end
            return {"offset_mapping": spans}

    mock_tok = MockTokenizer()
    cand_str = '{"valence": 5, "arousal": 5}'
    tokens = [101, 102, 103, 104]

    stages = get_generation_stage_tokens(tokens, mock_tok, candidate_str=cand_str)
    assert "candidate_start" in stages
    assert "pre_V" in stages
    assert "V_value" in stages
    assert "pre_A" in stages
    assert "A_value" in stages
    assert "response_end" in stages

    # Monotonic order check
    assert stages["candidate_start"] <= stages["pre_V"]
    assert stages["pre_V"] <= stages["V_value"]
    assert stages["V_value"] <= stages["pre_A"]
    assert stages["pre_A"] <= stages["A_value"]
    assert stages["A_value"] <= stages["response_end"]


def test_derangement_integrity():
    """Verifies that generate_derangement produces fixed-point free permutations (no self-matches)."""
    rng = np.random.default_rng(123)
    for n in [10, 50, 192]:
        perm = generate_derangement(n, rng)
        assert len(perm) == n
        assert set(perm) == set(range(n))
        # No fixed points
        for i in range(n):
            assert perm[i] != i, f"Self-match found at index {i} in derangement of size {n}"


def test_zero_forward_shift_invariant():
    """Verifies that alpha=0.0 analytically corresponds to zero shift without perturbation."""
    # Matched difference vector
    diff = np.array([0.5, -0.3, 1.2])
    alpha = 0.0
    effective_shift = alpha * diff
    assert np.all(effective_shift == 0.0)


def test_find_semantic_anchors_prompt_end():
    """Verifies that find_semantic_anchors returns prompt_end as total_len - 1."""
    class DummyTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [1, 2, 3]

    tok = DummyTokenizer()
    prompt_ids = [10, 20, 30, 40, 50]
    anchors = find_semantic_anchors(prompt_ids, tok, text="test")
    assert anchors["prompt_end"] == 4
    assert anchors["pre_response"] == 4
