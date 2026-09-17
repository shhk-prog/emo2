import numpy as np
import pytest

from affective_empathy_eval.geometry import (
    compute_center_of_mass,
    compute_dissociation_metrics,
    compute_relative_depth,
    eval_held_out_cross_decoding,
    eval_held_out_procrustes,
    train_and_eval_held_out_probe,
)


def test_held_out_probe():
    rng = np.random.default_rng(42)
    N_train, N_test, D = 80, 20, 10

    W = rng.standard_normal(D)
    H_train = rng.standard_normal((N_train, D))
    y_train = H_train @ W + rng.normal(0, 0.1, N_train)

    H_test = rng.standard_normal((N_test, D))
    y_test = H_test @ W + rng.normal(0, 0.1, N_test)

    r2 = train_and_eval_held_out_probe(H_train, y_train, H_test, y_test)
    assert r2 > 0.85


def test_held_out_cross_decoding():
    rng = np.random.default_rng(42)
    N_train, N_test, D = 60, 20, 8

    W = rng.standard_normal(D)
    H_train = rng.standard_normal((N_train, D))
    y_train = H_train @ W

    # Target 表現空間が Source と同一方向を持つ場合
    H_test = rng.standard_normal((N_test, D))
    y_test = H_test @ W

    r2 = eval_held_out_cross_decoding(H_train, y_train, H_test, y_test)
    assert r2 > 0.9


def test_held_out_procrustes():
    rng = np.random.default_rng(42)
    N_train, N_test, D = 50, 20, 5

    H_train_1 = rng.standard_normal((N_train, D))
    # 直交回転行列 R を作成
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    H_train_2 = H_train_1 @ Q

    H_test_1 = rng.standard_normal((N_test, D))
    H_test_2 = H_test_1 @ Q

    disparity = eval_held_out_procrustes(H_train_1, H_train_2, H_test_1, H_test_2)
    assert pytest.approx(disparity, abs=1e-5) == 0.0


def test_relative_depth_and_center_of_mass():
    # 28層モデル (0..27)
    assert compute_relative_depth(0, 28) == 0.0
    assert pytest.approx(compute_relative_depth(27, 28), abs=1e-5) == 1.0

    # 重み非負化重心
    # profile: [0.0, 1.0, 0.0] -> 重心は中央 (0.5)
    com = compute_center_of_mass([0.0, 1.0, 0.0])
    assert pytest.approx(com, abs=1e-5) == 0.5

    # 負値を含むプロファイル: [-0.5, 0.0, 2.0]
    # max(M, 0) -> [0, 0, 2] -> 重心は末尾 (1.0)
    com_neg = compute_center_of_mass([-0.5, 0.0, 2.0])
    assert pytest.approx(com_neg, abs=1e-5) == 1.0


def test_dissociation_metrics():
    # Decodability は中間層 (index 1 / depth 0.5) で最大
    d_prof = [0.2, 0.9, 0.3]
    # Causal leverage は後期層 (index 2 / depth 1.0) で最大
    c_prof = [0.1, 0.2, 0.8]

    metrics = compute_dissociation_metrics(d_prof, c_prof)
    assert pytest.approx(metrics["d_d_star"], abs=1e-4) == 0.5
    assert pytest.approx(metrics["d_c_star"], abs=1e-4) == 1.0
    assert pytest.approx(metrics["delta_d_star"], abs=1e-4) == 0.5
    assert metrics["delta_bar_d"] > 0.0
