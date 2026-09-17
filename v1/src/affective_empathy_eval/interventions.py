"""
介入・方向抽出・QR直交化・傾きγ推定モジュール
"""

import numpy as np


def extract_conditional_directions(
    H: np.ndarray,
    V: np.ndarray,
    A: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    外部刺激情動ラベル (V, A) を用いた重回帰により、他軸を条件付き統制した Valence 方向 d_V と Arousal 方向 d_A を抽出
    H: (N, D) - 活性化行列
    V: (N,) - 外部Valenceラベル
    A: (N,) - 外部Arousalラベル

    モデル: H = beta_0 + beta_V * V + beta_A * A + epsilon
    各次元 d について重回帰を実行（または行列一括最小二乗法）
    """
    N, _ = H.shape
    # デザイン行列: X = [1, V, A] (N, 3)
    X = np.column_stack([np.ones(N), V, A])

    # 最小二乗解: (X^T X)^{-1} X^T H -> (3, D)
    # beta[0] = 切片, beta[1] = beta_V, beta[2] = beta_A
    beta, _, _, _ = np.linalg.lstsq(X, H, rcond=None)

    beta_v = beta[1]  # (D,)
    beta_a = beta[2]  # (D,)

    # 単位ベクトル化
    norm_v = np.linalg.norm(beta_v)
    norm_a = np.linalg.norm(beta_a)

    d_v = beta_v / (norm_v + 1e-12)
    d_a = beta_a / (norm_a + 1e-12)

    return d_v, d_a


def compute_orthonormal_subspace(
    d_v: np.ndarray,
    d_a: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    基底行列 B = [d_v, d_a] に対し QR 分解を適用して直交正規基底 Q (D, 2) を構成
    Q^T Q = I_2
    射影行列: P = Q Q^T
    """
    B = np.column_stack([d_v, d_a])  # (D, 2)
    Q, _ = np.linalg.qr(B)  # Q is (D, 2)
    P = Q @ Q.T  # (D, D)
    return Q, P


def generate_control_directions(
    target_d: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ターゲット方向と同じノルムを持つ「ランダム方向」および「直交方向」を生成
    """
    rng = np.random.default_rng(seed)
    D = len(target_d)
    norm = np.linalg.norm(target_d)

    # 1. ランダム方向
    d_random = rng.standard_normal(D)
    d_random = (d_random / np.linalg.norm(d_random)) * norm

    # 2. 直交方向 (Gram-Schmidt で target_d 成分を除去)
    v_raw = rng.standard_normal(D)
    proj = (np.dot(v_raw, target_d) / (norm**2 + 1e-12)) * target_d
    d_perp = v_raw - proj
    d_perp = (d_perp / np.linalg.norm(d_perp)) * norm

    return d_random, d_perp


def estimate_interventional_slope(
    delta_z_list: list[float],
    delta_report_list: list[float],
) -> float:
    """
    複数強度 alpha 介入における (delta_z, delta_report) から
    Interventional Slope gamma = partial Delta Report / partial Delta z を線形回帰で推定
    """
    z = np.asarray(delta_z_list, dtype=np.float64)
    y = np.asarray(delta_report_list, dtype=np.float64)

    if len(z) < 2 or np.all(z == z[0]):
        return 0.0

    # slope = Cov(z, y) / Var(z)
    z_mean = np.mean(z)
    y_mean = np.mean(y)
    numerator = np.sum((z - z_mean) * (y - y_mean))
    denominator = np.sum((z - z_mean) ** 2)

    if denominator < 1e-12:
        return 0.0

    return float(numerator / denominator)


def compute_causal_leverage(
    expected_patched: float,
    expected_baseline: float,
) -> tuple[float, float]:
    """
    因果変位量 C を算出
    戻り値: (C_absolute, C_directional)
    """
    c_dir = expected_patched - expected_baseline
    c_abs = abs(c_dir)
    return c_abs, c_dir


def apply_centered_projection_removal_1d(
    h: np.ndarray,
    mu_neu: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    """
    Centered Projection Removal (1D):
    h' = h - [(h - mu_neu)^T d] d
    外部刺激情動方向 d 成分を中心化後に除去する。
    """
    h_centered = h - mu_neu
    proj = np.dot(h_centered, d) * d
    return h - proj


def apply_centered_projection_removal_subspace(
    h: np.ndarray,
    mu_neu: np.ndarray,
    P: np.ndarray,
) -> np.ndarray:
    """
    Centered Projection Removal (Subspace):
    h' = h - P (h - mu_neu)   (P = Q Q^T)
    QR直交基底から構築された 2D アフェクティブ部分空間成分を除去する。
    """
    h_centered = h - mu_neu
    proj = h_centered @ P.T
    return h - proj

