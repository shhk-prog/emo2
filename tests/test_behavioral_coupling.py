"""
tests/test_behavioral_coupling.py

Behavioral Reader-Self Coupling, Dose-Response, and Bootstrap CI Tests:
検証項目:
1. matched pair の差分計算: Delta_reader = reader_aff - reader_neu, Delta_self = self_aff - self_neu
2. Primary metric corr(Delta_reader, Delta_self) の算出と raw 相関との分離
3. pair_id 単位の Bootstrap 95% CI 算出
4. triplet 単位の repeated-measures linear slope の算出
5. 出力 CSV 形式の列名検証 (model, alignment, dimension, n_pairs, correlation_type, r, ci_low, ci_high, p_value)
"""

import numpy as np
import pandas as pd
import pytest

from behavioral.analysis.summarize_behavioral_aipsy import analyze_model_aipsy
from affective_empathy_eval.statistics import compute_correlation_bootstrap_ci, compute_bootstrap_ci


def test_compute_correlation_bootstrap_ci():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    y = 0.8 * x + 0.2 * rng.normal(0, 1, 100)

    r_est, ci_low, ci_high = compute_correlation_bootstrap_ci(x, y, method="pearson", n_boot=500)
    assert 0.6 < r_est < 0.99
    assert ci_low < r_est < ci_high

    assert not np.isnan(ci_low) and not np.isnan(ci_high)


def test_analyze_model_aipsy_coupling_and_dose_response(tmp_path):
    # モックの aipsy_4split CSV を作成
    n_pairs = 20
    rows = []
    rng = np.random.default_rng(123)

    for i in range(n_pairs):
        pair_id = f"p_{i:03d}"
        triplet_id = f"t_{i:03d}"

        # Neutral
        r_v_neu = 5.0 + rng.normal(0, 0.2)
        s_v_neu = 5.0 + rng.normal(0, 0.2)
        rows.append({
            "split": "neutral",
            "pair_id": pair_id,
            "triplet_id": triplet_id,
            "emotion": "ecstasy",
            "r_ev": r_v_neu,
            "s_ev": s_v_neu,
            "w_ev": 5.0,
            "r_ea": 5.0,
            "s_ea": 5.0,
            "w_ea": 5.0,
            "r_ed": 5.0,
            "s_ed": 5.0,
            "w_ed": 5.0,
        })

        # Moderate
        rows.append({
            "split": "moderate",
            "pair_id": pair_id,
            "triplet_id": triplet_id,
            "emotion": "ecstasy",
            "r_ev": r_v_neu + 1.0 + rng.normal(0, 0.2),
            "s_ev": s_v_neu + 0.8 + rng.normal(0, 0.2),
            "w_ev": 5.5,
            "r_ea": 5.5,
            "s_ea": 5.5,
            "w_ea": 5.5,
            "r_ed": 5.0,
            "s_ed": 5.0,
            "w_ed": 5.0,
        })

        # Clinical
        delta = rng.uniform(1.0, 3.0)
        r_v_clin = r_v_neu + delta + rng.normal(0, 0.1)
        s_v_clin = s_v_neu + 0.9 * delta + rng.normal(0, 0.1)
        rows.append({
            "split": "clinical",
            "pair_id": pair_id,
            "triplet_id": triplet_id,
            "emotion": "ecstasy",
            "r_ev": r_v_clin,
            "s_ev": s_v_clin,
            "w_ev": 6.0,
            "r_ea": 6.0,
            "s_ea": 6.0,
            "w_ea": 6.0,
            "r_ed": 5.0,
            "s_ed": 5.0,
            "w_ed": 5.0,
        })

    df = pd.DataFrame(rows)
    csv_file = tmp_path / "qwen2.5_1.5b_instruct_aipsy_4split.csv"
    df.to_csv(csv_file, index=False)

    df_rq1, df_rq2, df_rq3, df_rq4 = analyze_model_aipsy(str(csv_file))

    # 1. RQ4 (Coupling) の検証
    assert not df_rq4.empty
    expected_cols = ["model", "alignment", "dimension", "n_pairs", "correlation_type", "r", "ci_low", "ci_high", "p_value"]
    for col in expected_cols:
        assert col in df_rq4.columns, f"Missing column {col} in RQ4 output"

    # Primary metric: delta_coupling
    delta_rows = df_rq4[df_rq4["correlation_type"] == "delta_coupling"]
    assert len(delta_rows) >= 1
    v_delta = delta_rows[delta_rows["dimension"] == "V"].iloc[0]
    assert v_delta["r"] > 0.7, "Simulated positive delta coupling was not captured"
    assert v_delta["ci_low"] <= v_delta["r"] <= v_delta["ci_high"]
    assert v_delta["n_pairs"] == n_pairs

    # Secondary metric: raw_coupling
    raw_rows = df_rq4[df_rq4["correlation_type"] == "raw_coupling"]
    assert len(raw_rows) >= 1

    # 2. RQ2 (Dose-Response) の反復測定スロープ検証
    assert not df_rq2.empty
    assert "mean_slope" in df_rq2.columns
    assert "slope_ci_low" in df_rq2.columns
    v_dose = df_rq2[(df_rq2["task"] == "r") & (df_rq2["dimension"] == "V")].iloc[0]
    assert v_dose["mean_slope"] > 0.0, "Mean linear slope must be positive for monotonic response"
    assert v_dose["n_triplets"] == n_pairs
    assert v_dose["p_value"] < 0.05

    # 3. RQ1 (Sensitivity) の Bootstrap CI 検証
    assert not df_rq1.empty
    assert "mean_diff_ci_low" in df_rq1.columns
    assert "dz_ci_low" in df_rq1.columns
    v_rq1 = df_rq1[(df_rq1["task"] == "r") & (df_rq1["dimension"] == "V")].iloc[0]
    assert v_rq1["mean_diff_ci_low"] <= v_rq1["mean_diff"] <= v_rq1["mean_diff_ci_high"]
    assert v_rq1["dz_ci_low"] <= v_rq1["d_z"] <= v_rq1["dz_ci_high"]
