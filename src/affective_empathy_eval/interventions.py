"""
介入・方向抽出・QR直交化・傾きγ推定モジュール
"""

from typing import Any
import numpy as np


class ConditionalDirections(tuple):
    """
    Tuple of (d_v, d_a) that also supports dictionary-style and attribute access.
    Enables:
      - d_v, d_a = extract_conditional_directions(...)
      - res["direction_v"], res["direction_a"]
      - res["d_v"], res["d_a"]
      - res.direction_v, res.direction_a
    """
    def __new__(cls, d_v: np.ndarray, d_a: np.ndarray):
        return super().__new__(cls, (d_v, d_a))

    @property
    def direction_v(self) -> np.ndarray:
        return self[0]

    @property
    def direction_a(self) -> np.ndarray:
        return self[1]

    @property
    def d_v(self) -> np.ndarray:
        return self[0]

    @property
    def d_a(self) -> np.ndarray:
        return self[1]

    def __getitem__(self, item):
        if isinstance(item, str):
            key = item.lower()
            if key in ("direction_v", "d_v", "v"):
                return self[0]
            elif key in ("direction_a", "d_a", "a"):
                return self[1]
            raise KeyError(f"Invalid key '{item}'. Available keys: 'direction_v', 'direction_a'")
        return super().__getitem__(item)

    def get(self, item, default=None):
        try:
            return self[item]
        except KeyError:
            return default

    def keys(self):
        return ["direction_v", "direction_a"]

    def values(self):
        return [self[0], self[1]]

    def items(self):
        return [("direction_v", self[0]), ("direction_a", self[1])]


def extract_conditional_directions(
    H: np.ndarray,
    V: np.ndarray,
    A: np.ndarray,
    method: str = "lstsq",
    alpha: float = 1.0,
    **kwargs,
) -> ConditionalDirections:
    """
    外部刺激情動ラベル (V, A) または内部予測値を用いた重回帰（最小二乗または Ridge）により、
    他軸を条件付き統制した Valence 方向 d_V と Arousal 方向 d_A を抽出
    H: (N, D) - 活性化行列
    V: (N,) - Valenceラベル/スコア
    A: (N,) - Arousalラベル/スコア
    """
    N, D = H.shape
    if method == "ridge":
        from sklearn.linear_model import Ridge
        X = np.column_stack([V, A])
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(X, H)
        beta_v = ridge.coef_[:, 0]
        beta_a = ridge.coef_[:, 1]
    else:
        # デザイン行列: X = [1, V, A] (N, 3)
        X = np.column_stack([np.ones(N), V, A])
        beta, _, _, _ = np.linalg.lstsq(X, H, rcond=None)
        beta_v = beta[1]  # (D,)
        beta_a = beta[2]  # (D,)

    norm_v = np.linalg.norm(beta_v)
    norm_a = np.linalg.norm(beta_a)

    d_v = beta_v / (norm_v + 1e-12)
    d_a = beta_a / (norm_a + 1e-12)

    return ConditionalDirections(d_v, d_a)


def compute_orthonormal_subspace(
    *directions: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    基底行列 B に対し QR 分解を適用して直交正規基底 Q (D, K) を構成
    引数は (d_v, d_a) または ([d_v, d_a]) のどちらの呼び出しでも柔軟に受け付ける。
    Q^T Q = I_K
    射影行列: P = Q Q^T
    """
    if len(directions) == 1 and isinstance(directions[0], (list, tuple)):
        dir_list = list(directions[0])
    else:
        dir_list = list(directions)

    if not dir_list:
        raise ValueError("At least one direction vector required for compute_orthonormal_subspace.")

    B = np.column_stack(dir_list)  # (D, K)
    U, S, _ = np.linalg.svd(B, full_matrices=False)

    tol = max(B.shape) * np.finfo(float).eps * S[0]
    keep = S > tol

    if not np.any(keep):
        raise ValueError("Affect subspace has rank 0.")

    Q = U[:, keep]
    P = Q @ Q.T
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

