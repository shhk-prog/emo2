# V2 RQ3 Net Causal & Dissociation 整合化 実装完了報告 (Walkthrough)

## 概要
本改修では、V2 RQ3 における Net Causal Leverage ($C_{\mathrm{net,rand}}$) の Primary 昇格、Causal & Decodability 双対 Positive-Peak ガードの導入、Consumer における防御的 Fallback の担保、および Confirmatory 解析（H3）との完全整合化を完了しました。

---

## 変更内容詳細

### 1. 幾何・解離計算モジュール (`src/affective_empathy_eval/geometry.py`)
- **`compute_decodability_peak()`**: 全層 $R^2 \le 0$ の場合、ピークを NaN とする（「最も悪くない失敗層」の誤採択防止）。
- **`compute_decodability_center_of_mass()`**: 正のデコード質量 $\sum \max(R^2, 0) \le 0$ の場合、重心を NaN とする。
- **`compute_causal_peak_from_net()`**: 全層 $C_{\mathrm{net}} \le 0$ の場合、ピークを NaN とする。
- **`compute_causal_center_of_mass_from_net()`**: 正の因果質量 $\sum \max(C_{\mathrm{net}}, 0) \le 0$ の場合、重心を NaN とする。
- **`compute_net_causal_dissociation_metrics()`**: 双対 positive-guard を適用し、有効なデコードピークと因果ピークが揃った場合のみ $\Delta d^*$ および $\Delta \bar{d}$ を算出。`no_positive_decodability_peak` と `no_positive_net_causal_peak` の診断フラグを返却。

### 2. RQ3 因果マッピング (`v2/primary/run_rq3_causal_map.py`)
- **全メトリクスの保存**: `c_v_raw`, `c_a_raw`, `c_v_rand`, `c_a_rand`, `c_v_perp`, `c_a_perp`, `c_v_net_rand`, `c_a_net_rand`, `c_v_net_perp`, `c_a_net_perp`, `c_v_zero`, `c_a_zero` を全て記録。`c_v` / `c_a` は raw の exact alias として保持。
- **Schema Version 2**: Layer チェックポイントおよび Condition チェックポイントに `schema_version: 2` を導入し、旧フォーマットのキャッシュを自動で無効化。
- **Axis 別 Primary Metric**: 解離量算出において Valence は `c_v_net_rand`、Arousal は `c_a_net_rand` を使用。
- **Legacy Fallback**: `_get_geometry_profile` ヘルパーにより、既存の `inst_r2_reader/self` からの読み込みを保証。
- **Combined CSV の再構築**: ファミリー完了ごとに保存される per-family CSV から `v2_causal_pair_level.csv` を再構築（再実行時の append 重複を防止）。
- **LMM Formula**: Primary 検定に `PRIMARY_CAUSAL_METRIC_V` (`c_v_net_rand`) および `PRIMARY_CAUSAL_METRIC_A` (`c_a_net_rand`) を使用。

### 3. RQ1/RQ2 Cross-Decoding (`v2/primary/run_rq1_rq2_cross_decoding.py`)
- **キー統一**: Canonical な `inst_native_*` キーを新設しつつ、既存の `inst_*` キーも exact alias として両面出力。

### 4. Confirmatory 解析 (`v2/primary/run_confirmatory_analysis.py`)
- **H1 Secondary Fallback**: `inst_native_r2_*` が存在しない場合、既存の `inst_r2_*` からフォールバック読み込み。
- **H3 Primary Metric**: `PRIMARY_CAUSAL_COLUMNS` (`c_v_net_rand` / `c_a_net_rand`) を使用して LMM を実行。

---

## 検証結果

### ユニットテスト (`tests/test_rq3_net_causal_effect.py`)
全14件のテストを実装し、すべて **PASS** を確認しました：

```text
tests/test_rq3_net_causal_effect.py::test_1_net_causal_subtraction PASSED [  7%]
tests/test_rq3_net_causal_effect.py::test_2_raw_peak_differs_from_net_peak PASSED [ 14%]
tests/test_rq3_net_causal_effect.py::test_3_causal_all_negative_nan_peak PASSED [ 21%]
tests/test_rq3_net_causal_effect.py::test_4_causal_all_negative_nan_com PASSED [ 28%]
tests/test_rq3_net_causal_effect.py::test_5_causal_all_negative_dissociation_metrics PASSED [ 35%]
tests/test_rq3_net_causal_effect.py::test_6_dry_run_mock_returns_all_net_and_control_keys PASSED [ 42%]
tests/test_rq3_net_causal_effect.py::test_7_layer_checkpoint_schema_v2_invalidation PASSED [ 50%]
tests/test_rq3_net_causal_effect.py::test_8_cond_checkpoint_schema_v2_validation PASSED [ 57%]
tests/test_rq3_net_causal_effect.py::test_9_aggregation_consistency_sample_mean_equals_layer_profile PASSED [ 64%]
tests/test_rq3_net_causal_effect.py::test_10_confirmatory_h3_uses_net_rand PASSED [ 71%]
tests/test_rq3_net_causal_effect.py::test_11_combined_csv_reconstruction_from_per_family_csvs PASSED [ 78%]
tests/test_rq3_net_causal_effect.py::test_12_intervention_magnitude_equivalence PASSED [ 85%]
tests/test_rq3_net_causal_effect.py::test_13_decodability_all_negative_nan_peak_and_com PASSED [ 92%]
tests/test_rq3_net_causal_effect.py::test_14_decodability_all_negative_dissociation_guards PASSED [100%]

=========================== 14 passed in 4.42s ============================
```

既存の関連テスト（`test_geometry.py`, `test_interventions.py`, `test_confirmatory_pipeline.py`）も全て 16 passed でリグレッションがないことを確認しました。

### Dry-run Smoke Test
- `v2/primary/run_rq3_causal_map.py --family olmo --dry-run --force`: 正常完了（Exit code 0）。
- 生成された CSV に `c_v_net_rand`, `c_a_net_rand`, `c_v_perp`, `c_a_perp` 等の全列が正しく格納されていることを確認。

---

## 本番実行手順（ターミナルで実行）

### 1. 旧成果物・チェックポイントのクリーンアップ
```bash
# RQ3 チェックポイント（両種類）
rm -f v2/results/raw/v2_causal_cond_*.json
rm -f v2/results/raw/v2_causal_layer_ckpt_*.json
# RQ3 manifest（キャッシュ判定を確実に無効化）
rm -f v2/results/raw/manifest_causal_map_*.json
# RQ3 成果物
rm -f v2/results/raw/v2_causal_map_*.json
rm -f v2/results/raw/v2_rq3_causal_relocation_*.json
rm -f v2/results/derived/v2_causal_pair_level.csv
rm -f v2/results/derived/pair_level/v2_causal_pair_level_*.csv
rm -f v2/results/derived/v2_causal_dissociation_summary.json
# Confirmatory
rm -f v2/results/derived/v2_lmm_confirmatory.json
# （※ RQ1/RQ2 および RQ4 の成果物は削除しない）
```

### 2. RQ3 本番実行（全4 family）
```bash
.venv/bin/python v2/primary/run_rq3_causal_map.py --family qwen  --device cuda:0 --force
.venv/bin/python v2/primary/run_rq3_causal_map.py --family llama --device cuda:0 --force
.venv/bin/python v2/primary/run_rq3_causal_map.py --family gemma --device cuda:0 --force
.venv/bin/python v2/primary/run_rq3_causal_map.py --family olmo  --device cuda:0 --force
```

### 3. pair-level CSV 統合確認
```bash
.venv/bin/python -c "
import pandas as pd
df = pd.read_csv('v2/results/derived/v2_causal_pair_level.csv')
print(df['family'].value_counts())
required = ['c_v_net_rand','c_a_net_rand','c_v_perp','c_a_perp','c_v_raw','c_a_raw']
missing = [c for c in required if c not in df.columns]
print('Missing columns:', missing)
assert not missing
"
```

### 4. Confirmatory 解析の再計算
```bash
.venv/bin/python v2/primary/run_confirmatory_analysis.py
```
