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


def compute_decodability_peak(
    profile: list[float] | np.ndarray,
    relative_depths: list[float] | np.ndarray | None = None,
) -> float:
    """
    デコーダビリティピーク相対深度 d_D*
    全層 R^2 <= 0 の場合は NaN（「最も悪くない失敗層」の誤採択を防止）
    """
    vals = np.asarray(profile, dtype=np.float64)
    L = len(vals)
    if relative_depths is None:
        depths = np.linspace(0.0, 1.0, L, dtype=np.float64)
    else:
        depths = np.asarray(relative_depths, dtype=np.float64)

    valid = np.isfinite(vals) & np.isfinite(depths)
    if not np.any(valid):
        return float("nan")

    vals_v = vals[valid]
    depths_v = depths[valid]

    if np.max(vals_v) <= 0.0:
        return float("nan")

    return float(depths_v[np.argmax(vals_v)])


def compute_decodability_center_of_mass(
    profile: list[float] | np.ndarray,
    relative_depths: list[float] | np.ndarray | None = None,
) -> float:
    """
    デコーダビリティ重心 bar_d_D
    正のデコード質量 sum(max(R^2, 0)) <= 0 の場合は NaN
    """
    vals = np.asarray(profile, dtype=np.float64)
    L = len(vals)
    if relative_depths is None:
        depths = np.linspace(0.0, 1.0, L, dtype=np.float64)
    else:
        depths = np.asarray(relative_depths, dtype=np.float64)

    valid = np.isfinite(vals) & np.isfinite(depths)
    if not np.any(valid):
        return float("nan")

    vals_v = vals[valid]
    depths_v = depths[valid]

    positive = np.maximum(vals_v, 0.0)
    total = np.sum(positive)
    if total <= 0.0:
        return float("nan")

    return float(np.sum(depths_v * positive) / total)


def compute_causal_peak_from_net(
    c_net: list[float] | np.ndarray,
    relative_depths: list[float] | np.ndarray | None = None,
) -> float:
    """
    Net 因果効果ピーク相対深度 d_C*
    全層 C_net <= 0 の場合は NaN（コントロールを上回る正の介入効果なし）
    """
    vals = np.asarray(c_net, dtype=np.float64)
    L = len(vals)
    if relative_depths is None:
        depths = np.linspace(0.0, 1.0, L, dtype=np.float64)
    else:
        depths = np.asarray(relative_depths, dtype=np.float64)

    valid = np.isfinite(vals) & np.isfinite(depths)
    if not np.any(valid):
        return float("nan")

    vals_v = vals[valid]
    depths_v = depths[valid]

    positive = np.maximum(vals_v, 0.0)
    if np.max(positive) <= 0.0:
        return float("nan")

    return float(depths_v[np.argmax(positive)])


def compute_causal_center_of_mass_from_net(
    c_net: list[float] | np.ndarray,
    relative_depths: list[float] | np.ndarray | None = None,
) -> float:
    """
    Net 因果効果重心 bar_d_C
    正の因果質量 sum(max(C_net, 0)) <= 0 の場合は NaN
    """
    vals = np.asarray(c_net, dtype=np.float64)
    L = len(vals)
    if relative_depths is None:
        depths = np.linspace(0.0, 1.0, L, dtype=np.float64)
    else:
        depths = np.asarray(relative_depths, dtype=np.float64)

    valid = np.isfinite(vals) & np.isfinite(depths)
    if not np.any(valid):
        return float("nan")

    vals_v = vals[valid]
    depths_v = depths[valid]

    positive = np.maximum(vals_v, 0.0)
    total = np.sum(positive)
    if total <= 0.0:
        return float("nan")

    return float(np.sum(depths_v * positive) / total)


def compute_net_causal_dissociation_metrics(
    decodability_profile: list[float] | np.ndarray,
    net_causal_profile: list[float] | np.ndarray,
    relative_depths: list[float] | np.ndarray | None = None,
) -> dict[str, float | bool]:
    """
    Net 因果効果とデコーダビリティの解離量を算出（完全双対 positive ガード）。
    有意な decodability peak (R^2 > 0) と positive causal peak (C_net > 0) の両方が
    存在する場合にのみ Delta d* を定義。
    """
    d_d = compute_decodability_peak(decodability_profile, relative_depths)
    d_c = compute_causal_peak_from_net(net_causal_profile, relative_depths)
    bar_dd = compute_decodability_center_of_mass(decodability_profile, relative_depths)
    bar_dc = compute_causal_center_of_mass_from_net(net_causal_profile, relative_depths)

    fin = lambda x: bool(np.isfinite(x)) if x is not None else False
    delta_d_star = float(d_c - d_d) if fin(d_c) and fin(d_d) else float("nan")
    delta_bar_d = float(bar_dc - bar_dd) if fin(bar_dc) and fin(bar_dd) else float("nan")

    return {
        "d_d_star": d_d,
        "d_c_star": d_c,
        "delta_d_star": delta_d_star,
        "bar_d_d": bar_dd,
        "bar_d_c": bar_dc,
        "delta_bar_d": delta_bar_d,
        "no_positive_net_causal_peak": not fin(d_c),
        "no_positive_decodability_peak": not fin(d_d),
    }


