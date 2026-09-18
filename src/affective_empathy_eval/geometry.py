"""
幾何・アラインメント・相対深度・重心・ピーク解離計算モジュール
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


def train_and_eval_held_out_probe(
    H_train: np.ndarray,
    y_train: np.ndarray,
    H_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = 1.0,
) -> float:
    """
    Train split で Ridge プローブを学習し、完全に独立な Held-out test split で R^2 (Decodability) を評価
    """
    model = Ridge(alpha=alpha)
    model.fit(H_train, y_train)
    preds = model.predict(H_test)
    r2 = r2_score(y_test, preds)
    return float(r2)


def eval_held_out_cross_decoding(
    H_train_source: np.ndarray,
    y_train: np.ndarray,
    H_test_target: np.ndarray,
    y_test: np.ndarray,
    alpha: float = 1.0,
) -> float:
    """
    Source タスク表現でプローブを学習し、Target タスク表現で評価する Cross-decoding
    """
    model = Ridge(alpha=alpha)
    model.fit(H_train_source, y_train)
    preds = model.predict(H_test_target)
    return float(r2_score(y_test, preds))


def eval_held_out_procrustes(
    H_train_1: np.ndarray,
    H_train_2: np.ndarray,
    H_test_1: np.ndarray,
    H_test_2: np.ndarray,
    pca_dim: int | None = 64,
) -> float:
    """
    Train split で Procrustes 直交回転行列 R を求め、Held-out test split に適用した残差平方和（不一致度）を評価
    不一致度が低いほど幾何が整合。
    高次元・低サンプル時の rank-deficiency と null space 回転不定性を回避するため、
    pca_dim が指定され D > pca_dim の場合は train split で共通 PCA を行い次元削減後に Procrustes を計算。
    """
    # センタリング
    mu1 = np.mean(H_train_1, axis=0)
    mu2 = np.mean(H_train_2, axis=0)
    X1 = H_train_1 - mu1
    X2 = H_train_2 - mu2

    D = X1.shape[1]
    V_k = None
    if pca_dim is not None and D > pca_dim:
        k = min(pca_dim, X1.shape[0] - 1, X2.shape[0] - 1)
        if 0 < k < D:
            from sklearn.decomposition import PCA

            # Train の両表現を結合して共通主成分基底を fit
            pca = PCA(n_components=k, random_state=42)
            pca.fit(np.vstack([X1, X2]))
            V_k = pca.components_.T  # (D, k)
            X1 = X1 @ V_k
            X2 = X2 @ V_k

    # SVD による最適直交回転行列 R
    import torch

    tX1 = torch.from_numpy(X1).float()
    tX2 = torch.from_numpy(X2).float()
    M = torch.matmul(tX1.t(), tX2)
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    R = torch.matmul(U, Vh).numpy().astype(H_train_1.dtype)

    # Test split に適用
    X_test1 = H_test_1 - mu1
    X_test2 = H_test_2 - mu2
    if V_k is not None:
        X_test1 = X_test1 @ V_k
        X_test2 = X_test2 @ V_k

    X_test1_rotated = X_test1 @ R

    # 正規化フロベニウス残差
    diff = np.linalg.norm(X_test1_rotated - X_test2, ord="fro")
    denom = np.linalg.norm(X_test2, ord="fro") + 1e-12
    normalized_disparity = float(diff / denom)
    return normalized_disparity


def compute_rsa_correlation(
    H1: np.ndarray,
    H2: np.ndarray,
    metric: str = "cosine",
) -> float:
    """
    Representational Similarity Analysis (RSA): 2つの表現空間のRDMs (Representational Dissimilarity Matrices) の相関
    """
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr

    rdm1 = pdist(H1, metric=metric)
    rdm2 = pdist(H2, metric=metric)
    rho, _ = spearmanr(rdm1, rdm2)
    return float(rho)


def compute_relative_depth(layer_idx: int, num_layers: int) -> float:
    """相対計算深度 d = layer_idx / (num_layers - 1)"""
    if num_layers <= 1:
        return 0.0
    return float(layer_idx / (num_layers - 1))


def get_block_hidden_state(hidden_states: tuple | list, layer_idx: int):
    """
    Transformer block l (0 <= layer_idx < num_layers) の出力を取得する。
    Hugging Face の outputs.hidden_states において:
      hidden_states[0] = embedding output
      hidden_states[l + 1] = Transformer block l output (0 <= l < L)
    hidden_state_index と transformer_layer_index を混同しないための標準アクセサ。
    """
    assert 0 <= layer_idx < len(hidden_states) - 1, (
        f"layer_idx {layer_idx} out of range for hidden_states of length {len(hidden_states)}. "
        f"Valid block index is 0 <= layer_idx < {len(hidden_states) - 1}."
    )
    return hidden_states[layer_idx + 1]



def compute_center_of_mass(
    profile: list[float],
    relative_depths: list[float] | None = None,
) -> float:
    """
    重み非負化重心（Center of Mass）:
    bar_d = sum(d_l * max(M_l, 0)) / sum(max(M_l, 0))
    """
    L = len(profile)
    if relative_depths is None:
        depths = np.linspace(0.0, 1.0, L)
    else:
        depths = np.asarray(relative_depths, dtype=np.float64)

    vals = np.asarray(profile, dtype=np.float64)
    weights = np.maximum(vals, 0.0)
    total_w = np.sum(weights)

    if total_w < 1e-12:
        # 重みがすべて0なら中央値を返す
        return float(np.mean(depths))

    return float(np.sum(depths * weights) / total_w)


def compute_peak_depth(
    profile: list[float],
    relative_depths: list[float] | None = None,
) -> float:
    """
    ピーク相対深度 d* = argmax_d M(d)
    """
    L = len(profile)
    if relative_depths is None:
        depths = np.linspace(0.0, 1.0, L)
    else:
        depths = np.asarray(relative_depths, dtype=np.float64)

    idx = int(np.argmax(profile))
    return float(depths[idx])


def compute_dissociation_metrics(
    decodability_profile: list[float],
    causal_profile: list[float],
    relative_depths: list[float] | None = None,
) -> dict[str, float]:
    """
    ピークおよび重心の解離量を一括算出
    Delta d* = d_C* - d_D*
    Delta bar_d = bar_d_C - bar_d_D
    """
    d_d_star = compute_peak_depth(decodability_profile, relative_depths)
    d_c_star = compute_peak_depth(causal_profile, relative_depths)

    bar_d_d = compute_center_of_mass(decodability_profile, relative_depths)
    bar_d_c = compute_center_of_mass(causal_profile, relative_depths)

    return {
        "d_d_star": d_d_star,
        "d_c_star": d_c_star,
        "delta_d_star": d_c_star - d_d_star,
        "bar_d_d": bar_d_d,
        "bar_d_c": bar_d_c,
        "delta_bar_d": bar_d_c - bar_d_d,
        # v3 script compatibility aliases
        "d_peak_D": d_d_star,
        "d_peak_C": d_c_star,
        "delta_d_peak": d_c_star - d_d_star,
        "d_center_D": bar_d_d,
        "d_center_C": bar_d_c,
        "delta_d_center": bar_d_c - bar_d_d,
    }


def compute_layer_dissociation(
    relative_depths: list[float],
    decodability_profile: list[float],
    causal_profile: list[float],
) -> dict[str, float]:
    """
    相対深度、デコーダビリティプロファイル、因果効果プロファイルを受け取り解離量を算出
    """
    return compute_dissociation_metrics(
        decodability_profile=decodability_profile,
        causal_profile=causal_profile,
        relative_depths=relative_depths,
    )

