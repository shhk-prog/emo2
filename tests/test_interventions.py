import numpy as np
import pytest

from affective_empathy_eval.interventions import (
    compute_causal_leverage,
    compute_orthonormal_subspace,
    estimate_interventional_slope,
    extract_conditional_directions,
    generate_control_directions,
)


def test_extract_conditional_directions():
    rng = np.random.default_rng(42)
    N = 100
    D = 16

    # 人為的な相関データを作成
    # V と A を相関させる
    V = rng.uniform(1.0, 9.0, size=N)
    A = 0.5 * V + rng.normal(0.0, 1.0, size=N)

    # 潜在真方向
    true_dv = rng.standard_normal(D)
    true_dv /= np.linalg.norm(true_dv)
    true_da = rng.standard_normal(D)
    true_da /= np.linalg.norm(true_da)

    # 活性化行列 H: V 方向と A 方向の線形結合 + ノイズ
    H = np.outer(V, true_dv) * 2.0 + np.outer(A, true_da) * 1.5 + rng.normal(0.0, 0.1, size=(N, D))

    # 条件付き抽出
    dv_est, da_est = extract_conditional_directions(H, V, A)

    # ノルムが1であること
    assert pytest.approx(np.linalg.norm(dv_est), abs=1e-5) == 1.0
    assert pytest.approx(np.linalg.norm(da_est), abs=1e-5) == 1.0

    # 真の方向とのコサイン類似度が高いこと
    cos_v = abs(np.dot(dv_est, true_dv))
    cos_a = abs(np.dot(da_est, true_da))
    assert cos_v > 0.9
    assert cos_a > 0.9


def test_compute_orthonormal_subspace():
    D = 32
    rng = np.random.default_rng(123)
    v1 = rng.standard_normal(D)
    v1 /= np.linalg.norm(v1)
    v2 = v1 * 0.8 + rng.standard_normal(D) * 0.2
    v2 /= np.linalg.norm(v2)

    Q, P = compute_orthonormal_subspace(v1, v2)

    # Q は (D, 2)
    assert Q.shape == (D, 2)
    # Q^T Q = I_2
    qtq = Q.T @ Q
    assert np.allclose(qtq, np.eye(2), atol=1e-6)

    # 射影行列 P = Q Q^T は対称行列かつべき等行列 (P^2 = P)
    assert np.allclose(P, P.T, atol=1e-6)
    assert np.allclose(P @ P, P, atol=1e-6)


def test_generate_control_directions():
    D = 20
    target_d = np.ones(D)
    norm_target = np.linalg.norm(target_d)

    d_rand, d_perp = generate_control_directions(target_d, seed=42)

    # ノルムが等しいこと
    assert pytest.approx(np.linalg.norm(d_rand), abs=1e-5) == norm_target
    assert pytest.approx(np.linalg.norm(d_perp), abs=1e-5) == norm_target

    # d_perp は target_d と直交していること
    dot_prod = np.dot(d_perp, target_d)
    assert pytest.approx(dot_prod, abs=1e-6) == 0.0


def test_estimate_interventional_slope():
    # 線形関係: y = 0.75 * z + 0.1
    z_list = [-1.0, -0.5, 0.0, 0.5, 1.0]
    y_list = [0.75 * z + 0.1 for z in z_list]

    slope = estimate_interventional_slope(z_list, y_list)
    assert pytest.approx(slope, abs=1e-5) == 0.75

    # 変化なし
    y_const = [3.0, 3.0, 3.0, 3.0, 3.0]
    assert pytest.approx(estimate_interventional_slope(z_list, y_const), abs=1e-5) == 0.0


def test_compute_causal_leverage():
    c_abs, c_dir = compute_causal_leverage(expected_patched=6.5, expected_baseline=5.0)
    assert pytest.approx(c_abs, abs=1e-5) == 1.5
    assert pytest.approx(c_dir, abs=1e-5) == 1.5


def test_apply_centered_projection_removal():
    from affective_empathy_eval.interventions import (
        apply_centered_projection_removal_1d,
        apply_centered_projection_removal_subspace,
    )
    D = 16
    rng = np.random.default_rng(42)
    h = rng.standard_normal(D)
    mu_neu = rng.standard_normal(D)
    d = rng.standard_normal(D)
    d /= np.linalg.norm(d)

    # 1D removal
    h_prime = apply_centered_projection_removal_1d(h, mu_neu, d)
    # (h_prime - mu_neu) は d と直交するはず
    dot = np.dot(h_prime - mu_neu, d)
    assert pytest.approx(dot, abs=1e-6) == 0.0

    # Subspace removal
    v2 = rng.standard_normal(D)
    v2 /= np.linalg.norm(v2)
    Q, P = compute_orthonormal_subspace(d, v2)
    h_prime_sub = apply_centered_projection_removal_subspace(h, mu_neu, P)
    # P @ (h_prime_sub - mu_neu) == 0
    proj = P @ (h_prime_sub - mu_neu)
    assert np.allclose(proj, np.zeros(D), atol=1e-6)


    c_abs2, c_dir2 = compute_causal_leverage(expected_patched=3.0, expected_baseline=5.0)
    assert pytest.approx(c_abs2, abs=1e-5) == 2.0
    assert pytest.approx(c_dir2, abs=1e-5) == -2.0


def test_conditional_directions_hybrid_access():
    from affective_empathy_eval.interventions import ConditionalDirections

    d_v = np.array([1.0, 0.0, 0.0])
    d_a = np.array([0.0, 1.0, 0.0])
    res = ConditionalDirections(d_v, d_a)

    # 1. Tuple unpacking
    u_v, u_a = res
    assert np.array_equal(u_v, d_v)
    assert np.array_equal(u_a, d_a)

    # 2. Integer indexing
    assert np.array_equal(res[0], d_v)
    assert np.array_equal(res[1], d_a)

    # 3. String dictionary-style indexing
    assert np.array_equal(res["direction_v"], d_v)
    assert np.array_equal(res["direction_a"], d_a)
    assert np.array_equal(res["d_v"], d_v)
    assert np.array_equal(res["d_a"], d_a)

    # 4. Property access
    assert np.array_equal(res.direction_v, d_v)
    assert np.array_equal(res.direction_a, d_a)

    # 5. Dict conversion & keys
    d_dict = dict(res.items())
    assert "direction_v" in d_dict
    assert "direction_a" in d_dict
    assert res.get("direction_v") is not None
    assert res.get("nonexistent", "default") == "default"
