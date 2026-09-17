import numpy as np
import pandas as pd
import pytest

from affective_empathy_eval.statistics import (
    apply_fdr_correction,
    cluster_based_permutation_test,
    compute_bootstrap_ci,
    fit_sample_level_lmm,
    paired_family_comparison,
)


def test_bootstrap_ci():
    data = [10.0, 11.0, 9.0, 10.5, 9.5]
    mean_val, ci_low, ci_high = compute_bootstrap_ci(data, n_boot=500, ci_level=0.95, seed=42)
    assert pytest.approx(mean_val, abs=1e-4) == 10.0
    assert ci_low < mean_val < ci_high


def test_paired_family_comparison():
    scores_inst = [0.8, 0.75, 0.9, 0.85]
    scores_base = [0.5, 0.45, 0.6, 0.55]
    res = paired_family_comparison(scores_inst, scores_base)
    assert res["all_positive"] is True
    assert res["mean_diff"] > 0.0


def test_fdr_correction():
    # 有意なp値と非有意なp値の混在
    pvals = [0.001, 0.002, 0.04, 0.6, 0.8]
    reject, qvals = apply_fdr_correction(pvals, alpha=0.05)
    assert bool(reject[0])
    assert bool(reject[1])
    assert not bool(reject[4])
    assert len(qvals) == 5


def test_cluster_permutation():
    # 5層 x 4ステージの行列
    matrix_real = np.ones((5, 4))
    # 特定領域 (層 2..3, ステージ 1..2) に有意なシグナルを注入
    matrix_real[2:4, 1:3] = 0.001

    # ヌル分布 (ランダム)
    rng = np.random.default_rng(42)
    null_matrices = rng.uniform(0.0, 1.0, size=(100, 5, 4))

    sig_mask, clusters = cluster_based_permutation_test(
        matrix_real=matrix_real,
        matrix_perm_null=null_matrices,
        cluster_threshold=0.05,
        p_threshold=0.05,
    )
    assert len(clusters) > 0
    # 注入した領域が有意クラスタとして検出されること
    assert bool(sig_mask[2, 1]) or bool(sig_mask[2, 2])


def test_sample_level_lmm():
    # ダミーデータ生成
    rng = np.random.default_rng(42)
    n_pairs = 30
    records = []
    for pair_id in range(n_pairs):
        pair_effect = rng.normal(0, 0.5)
        for align in ["base", "instruct"]:
            align_effect = 0.8 if align == "instruct" else 0.0
            for task in ["reader", "self"]:
                task_effect = 0.4 if task == "self" else 0.0
                y = 5.0 + align_effect + task_effect + pair_effect + rng.normal(0, 0.2)
                records.append({
                    "pair": f"pair_{pair_id}",
                    "alignment": align,
                    "task": task,
                    "y": y,
                })
    df = pd.DataFrame(records)

    res = fit_sample_level_lmm(df, formula="y ~ alignment + task", groups="pair")
    assert res["converged"] is True
    assert "alignment[T.instruct]" in res["params"]
    assert res["params"]["alignment[T.instruct]"] > 0.5
