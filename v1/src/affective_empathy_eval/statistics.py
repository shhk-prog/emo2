"""
統計解析・LMM・Bootstrap CI・FDR・Cluster Permutation モジュール
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def fit_sample_level_lmm(
    df: pd.DataFrame,
    formula: str,
    groups: str = "pair",
    re_formula: str | None = None,
) -> dict[str, Any]:
    """
    Sample-level 線形混合効果モデル (Mixed Linear Model) のフィッティング
    特異フィット（Singular fit）や収束エラーが発生した場合は自動的に Random Intercept モデルへフォールバック
    """
    import statsmodels.formula.api as smf

    model_res = None
    fit_success = False
    used_re_formula = re_formula

    # 1. 指定された re_formula (Random Slope) でトライ
    if re_formula is not None:
        try:
            model = smf.mixedlm(formula, df, groups=df[groups], re_formula=re_formula)
            fit = model.fit(method=["lbfgs", "cg"], maxiter=500)
            if fit.converged and not fit.cov_re.isna().any().any():
                model_res = fit
                fit_success = True
        except (ValueError, np.linalg.LinAlgError):
            fit_success = False

    # 2. フォールバック: 単純な Random Intercept モデル (1 | pair)
    if not fit_success:
        used_re_formula = None
        model = smf.mixedlm(formula, df, groups=df[groups])
        model_res = model.fit(method=["lbfgs", "cg"], maxiter=500)

    summary_table = model_res.summary().tables[1]
    params = model_res.params.to_dict()
    pvalues = model_res.pvalues.to_dict()
    conf_int = model_res.conf_int().to_dict()

    return {
        "model": model_res,
        "converged": bool(model_res.converged),
        "used_re_formula": used_re_formula,
        "params": params,
        "pvalues": pvalues,
        "conf_int": conf_int,
        "summary_table": summary_table,
    }


def compute_bootstrap_ci(
    data: list[float] | np.ndarray,
    statistic_fn: Any = np.mean,
    n_boot: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Bootstrap リサンプリングによる統計量の点推定値および (1 - alpha) 信頼区間を算出
    戻り値: (point_estimate, ci_lower, ci_upper)
    """
    arr = np.asarray(data, dtype=np.float64)
    point_est = float(statistic_fn(arr))

    rng = np.random.default_rng(seed)
    n = len(arr)
    boot_stats = []

    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_stats.append(statistic_fn(sample))

    alpha = 1.0 - ci_level
    lower = float(np.percentile(boot_stats, 100 * (alpha / 2.0)))
    upper = float(np.percentile(boot_stats, 100 * (1.0 - alpha / 2.0)))

    return point_est, lower, upper


def paired_family_comparison(
    scores_inst: list[float],
    scores_base: list[float],
) -> dict[str, Any]:
    """
    4モデルファミリー等でのペア比較（符号順位検定または差分要約）
    """
    diffs = np.array(scores_inst) - np.array(scores_base)
    mean_diff = float(np.mean(diffs))
    median_diff = float(np.median(diffs))

    # 4サンプルの場合は Wilcoxon 検定または置換検定
    try:
        _, pval = wilcoxon(diffs)
        p_value = float(pval)
    except (ValueError, np.linalg.LinAlgError):
        p_value = 1.0

    return {
        "mean_diff": mean_diff,
        "median_diff": median_diff,
        "diffs": diffs.tolist(),
        "p_value": p_value,
        "all_positive": bool(np.all(diffs > 0)),
        "all_negative": bool(np.all(diffs < 0)),
    }


def apply_fdr_correction(
    p_values: list[float] | np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg 法による FDR (False Discovery Rate) 多重比較補正
    戻り値: (reject_mask, q_values)
    """
    pvals = np.asarray(p_values, dtype=np.float64)
    valid_mask = ~np.isnan(pvals)
    q_vals = np.ones_like(pvals)
    reject = np.zeros_like(pvals, dtype=bool)

    if not np.any(valid_mask):
        return reject, q_vals

    try:
        from statsmodels.stats.multitest import multipletests
        rej, q, _, _ = multipletests(pvals[valid_mask], alpha=alpha, method="fdr_bh")
        reject[valid_mask] = rej
        q_vals[valid_mask] = q
    except ImportError:
        # Pure numpy による Benjamini-Hochberg 実装
        valid_p = pvals[valid_mask]
        n = len(valid_p)
        order = np.argsort(valid_p)
        sorted_p = valid_p[order]

        # q_i = min_{j >= i} (sorted_p_j * n / (j + 1))
        factors = n / np.arange(1, n + 1)
        q_raw = sorted_p * factors
        # 逆順累積最小値
        q_sorted = np.minimum.accumulate(q_raw[::-1])[::-1]
        q_sorted = np.clip(q_sorted, 0.0, 1.0)

        q_orig = np.zeros(n)
        q_orig[order] = q_sorted

        reject[valid_mask] = q_orig < alpha
        q_vals[valid_mask] = q_orig

    return reject, q_vals


def cluster_based_permutation_test(
    matrix_real: np.ndarray,  # (Layers, Stages)
    matrix_perm_null: np.ndarray,  # (N_perm, Layers, Stages)
    cluster_threshold: float = 0.05,
    p_threshold: float = 0.05,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    時空間グリッド (Layer x Stage) に対する Cluster-based Permutation Test
    連続するセルで構成される有意クラスタ（Island）を検出し、多重比較を補正
    """
    from scipy.ndimage import label

    # 1. 閾値を超えるセルのマスク
    # matrix_real は例えば p値行列（p < cluster_threshold）
    binary_map = matrix_real < cluster_threshold
    labeled_map, num_features = label(binary_map)

    # 2. 各クラスタの統計量（クラスタサイズまたはクラスタ内スコア和）
    cluster_results = []
    significant_mask = np.zeros_like(binary_map, dtype=bool)

    if num_features == 0:
        return significant_mask, cluster_results

    # 各クラスタのサイズ
    cluster_sizes = [np.sum(labeled_map == i) for i in range(1, num_features + 1)]

    # 3. Null分布における最大クラスタサイズの分布を算出
    n_perm = matrix_perm_null.shape[0]
    max_null_cluster_sizes = []
    for p_idx in range(n_perm):
        null_map = matrix_perm_null[p_idx] < cluster_threshold
        null_labeled, null_num = label(null_map)
        if null_num > 0:
            null_sizes = [np.sum(null_labeled == j) for j in range(1, null_num + 1)]
            max_null_cluster_sizes.append(max(null_sizes))
        else:
            max_null_cluster_sizes.append(0)

    max_null_cluster_sizes = np.array(max_null_cluster_sizes)

    # 4. クラスタ p値の算出
    for i, c_size in enumerate(cluster_sizes):
        cluster_id = i + 1
        p_val = float(np.mean(max_null_cluster_sizes >= c_size))
        is_sig = p_val < p_threshold
        if is_sig:
            significant_mask[labeled_map == cluster_id] = True

        cluster_results.append({
            "cluster_id": cluster_id,
            "size": int(c_size),
            "p_value": p_val,
            "significant": is_sig,
        })

    return significant_mask, cluster_results


class BootstrapCI(tuple):
    """Tuple (ci_lower, ci_upper) with point_estimate attribute."""
    def __new__(cls, point_est: float, lower: float, upper: float):
        instance = super().__new__(cls, (lower, upper))
        instance.point_estimate = point_est
        instance.ci_lower = lower
        instance.ci_upper = upper
        return instance


def compute_bivariate_bootstrap_ci(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
    statistic_fn: Any = None,
    stat_fn: Any = None,
    n_boot: int = 1000,
    n_bootstraps: int | None = None,
    ci_level: float = 0.95,
    ci: float | None = None,
    seed: int = 42,
) -> BootstrapCI:
    """
    ペア (x_i, y_i) の Bootstrap リサンプリングによる統計量の点推定値および (1 - alpha) 信頼区間を算出
    戻り値: BootstrapCI(point_estimate, ci_lower, ci_upper) - 2要素の (lower, upper) としてもアンパック可能
    """
    fn = stat_fn if stat_fn is not None else statistic_fn
    if fn is None:
        raise ValueError("Either statistic_fn or stat_fn must be provided.")
    n_b = n_bootstraps if n_bootstraps is not None else n_boot
    target_ci = ci if ci is not None else ci_level

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if len(x_arr) != len(y_arr) or len(x_arr) < 2:
        return BootstrapCI(np.nan, np.nan, np.nan)

    point_est = float(fn(x_arr, y_arr))

    rng = np.random.default_rng(seed)
    n = len(x_arr)
    boot_stats = []

    for _ in range(n_b):
        indices = rng.choice(n, size=n, replace=True)
        stat = fn(x_arr[indices], y_arr[indices])
        if not np.isnan(stat):
            boot_stats.append(stat)

    if len(boot_stats) == 0:
        return BootstrapCI(point_est, np.nan, np.nan)

    alpha = 1.0 - target_ci
    lower = float(np.percentile(boot_stats, 100 * (alpha / 2.0)))
    upper = float(np.percentile(boot_stats, 100 * (1.0 - alpha / 2.0)))

    return BootstrapCI(point_est, lower, upper)


def compute_paired_cohen_dz(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray | None = None,
    ddof: int = 1,
) -> float:
    """
    対応のあるペアの差分 Delta に対する paired effect size d_z = mean(Delta) / s_Delta (ddof=1)
    x, y が両方渡された場合は diff = x - y として計算。x のみの場合は x を差分とみなす。
    """
    if y is not None:
        d_arr = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    else:
        d_arr = np.asarray(x, dtype=np.float64)

    if len(d_arr) < 2:
        return 0.0
    s_delta = float(np.std(d_arr, ddof=ddof))
    if s_delta < 1e-9:
        return 0.0
    return float(np.mean(d_arr) / s_delta)


def generate_derangement(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    固定点（自己一致 perm[i] == i）を持たない完全撹乱順列 (Derangement) を生成
    """
    if n <= 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)
    if rng is None:
        rng = np.random.default_rng(42)

    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm

