"""
tests/test_behavioral_specificity.py

AIPsy 4-split行動評価のRQ3特異性集計（Clinical vs Complex Neutral）が
cn_vals NameError を起こさず正常に完走することを検証する単体テスト。
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from behavioral.analysis.summarize_behavioral_aipsy import analyze_model_aipsy


def test_behavioral_rq3_specificity_runs_without_name_error():
    """
    clinical, complex_neutral, neutral を含むデータフレームで
    RQ3集計が cn_vals NameError なく正常に計算されることを検証。
    """
    rows = []
    # 5ペア分のダミーデータを作成
    for pair_id in range(1, 6):
        # neutral
        rows.append({
            "pair_id": pair_id,
            "split": "neutral",
            "target_emotion": "ecstasy",
            "w_ev": 5.0, "w_ea": 5.0, "w_ed": 5.0,
            "r_ev": 5.1, "r_ea": 4.9, "r_ed": 5.0,
            "s_ev": 5.0, "s_ea": 5.0, "s_ed": 5.0,
        })
        # clinical
        rows.append({
            "pair_id": pair_id,
            "split": "clinical",
            "target_emotion": "ecstasy",
            "w_ev": 7.5 + np.random.normal(0, 0.1),
            "w_ea": 7.0 + np.random.normal(0, 0.1),
            "w_ed": 6.5 + np.random.normal(0, 0.1),
            "r_ev": 7.2 + np.random.normal(0, 0.1),
            "r_ea": 6.8 + np.random.normal(0, 0.1),
            "r_ed": 6.3 + np.random.normal(0, 0.1),
            "s_ev": 6.9 + np.random.normal(0, 0.1),
            "s_ea": 6.5 + np.random.normal(0, 0.1),
            "s_ed": 6.1 + np.random.normal(0, 0.1),
        })
        # complex_neutral
        rows.append({
            "pair_id": pair_id,
            "split": "complex_neutral",
            "target_emotion": "ecstasy",
            "w_ev": 5.2 + np.random.normal(0, 0.1),
            "w_ea": 5.1 + np.random.normal(0, 0.1),
            "w_ed": 5.0 + np.random.normal(0, 0.1),
            "r_ev": 5.3 + np.random.normal(0, 0.1),
            "r_ea": 5.2 + np.random.normal(0, 0.1),
            "r_ed": 5.1 + np.random.normal(0, 0.1),
            "s_ev": 5.1 + np.random.normal(0, 0.1),
            "s_ea": 5.0 + np.random.normal(0, 0.1),
            "s_ed": 5.0 + np.random.normal(0, 0.1),
        })

    df = pd.DataFrame(rows)

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "test_model_aipsy.csv"
        df.to_csv(csv_path, index=False)

        # analyze_model_aipsy が NameError なしで完走すること
        rq1_df, rq2_df, rq3_df, rq4_df = analyze_model_aipsy(str(csv_path))

        assert rq3_df is not None, "rq3_df should not be None"
        assert len(rq3_df) > 0, "rq3_df should contain specificity records"
        # 必要なカラムが正しく計算されていること
        assert "mean_clinical_displacement" in rq3_df.columns
        assert "mean_complex_neutral_displacement" in rq3_df.columns
        assert "displacement_cohen_d" in rq3_df.columns
