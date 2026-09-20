# V3 実行結果の確認レポート (Walkthrough)

## 1. 結論サマリー
- **「全て」は正常に実行完了していません。**
- 直近の実行（ログ: `results/logs/production_v3_20260920_180752.log`）において、**RQ1 (State Induction) の実行後にゲート判定が `NO_GO` となり、パイプラインの安全設計に従って RQ2, RQ3, Confirmatory Replication は実行されずに停止**しました。
- 調査の結果、**`NO_GO` と判定された根本的な原因として、コード上の二重ソフトマックス適用による期待値平滑化バグ**が存在することが判明しました。

---

## 2. 実行履歴と現状

| 実行時刻 (JST) | ログファイル | 状態 | 詳細 |
|---|---|---|---|
| 2026-09-20 17:48:37 | `results/logs/production_v3_20260920_174837.log` | 異常終了 (Error) | `TypeError: tuple indices must be integers or slices, not str` により途中クラッシュ |
| 2026-09-20 18:07:52 | `results/logs/production_v3_20260920_180752.log` | 早期停止 (Gate NO_GO) | RQ1 完走後、Gate 判定で `NO_GO` となり RQ2 以降がスキップされて正常停止 |

### 現在の生成成果物
- `v3/results/raw/manifest_rq1_qwen.json`
- `v3/results/raw/v3_rq1_results.json`
- `v3/results/derived/v3_gate_decision.json`
- ※ RQ2 (`v3_rq2_results.json`), RQ3 (`v3_rq3_results.json`), Confirmatory (`v3_confirmatory_results.json`) の成果物は未生成。

---

## 3. ゲート判定 (`NO_GO`) の詳細

`v3/results/derived/v3_gate_decision.json` の結果：

```json
{
  "dose_response_v_pass": false,
  "dose_response_a_pass": false,
  "dose_response_pass": false,
  "slope_v_ci_lower": 0.0000615,
  "slope_a_ci_lower": 0.0000402,
  "specificity_v_pass": false,
  "specificity_a_pass": false,
  "specificity_pass": false,
  "specificity_v_ci_lower": 0.0000165,
  "specificity_a_ci_lower": -0.0000009,
  "endogenous_relevance_v_pass": false,
  "endogenous_relevance_a_pass": false,
  "endogenous_relevance_pass": false,
  "endogenous_relevance_v_ci_lower": NaN,
  "endogenous_relevance_a_ci_lower": NaN,
  "necessity_v_pass": false,
  "necessity_a_pass": false,
  "necessity_pass": false,
  "necessity_v_ci_lower": NaN,
  "necessity_a_ci_lower": NaN,
  "topic_v_pass": true,
  "topic_a_pass": true,
  "topic_control_pass": true,
  "topic_tvd_v_ci_upper": 0.00107,
  "topic_tvd_a_ci_upper": 0.00111,
  "decision_valence": "NO_GO",
  "decision_arousal": "NO_GO",
  "decision": "NO_GO"
}
```

- 基準値（`configs/v3_experiments.yaml`）:
  - `min_specificity_diff`: 0.05
  - `min_necessity_attenuation`: 0.05
  - `max_topic_tvd`: 0.15
- 判定結果:
  - `slope` および `specificity` が `1e-5` 前後と基準の 0.05 を大幅に下回り不合格。
  - `attenuation_ratio`（necessity / endogenous relevance）が `NaN` となり不合格。
  - そのため総合判定は `NO_GO`。

---

## 4. なぜ NO_GO になったのか？（根本原因の分析）

### ① `compute_expected_va` に対する二重ソフトマックス適用
- `src/affective_empathy_eval/likelihood.py` の `compute_sequence_likelihoods_for_candidates` はタプル `(log_likelihoods, probs)` を返します（`probs` はすでに Softmax された 0〜1 の正規化確率）。
- しかし、`v3/primary/run_rq1_state_induction.py` では `ev, ea = compute_expected_va(probs, candidates)` のように `probs` を渡しています。
- 一方、`compute_expected_va` の定義は以下のようになっています：
  ```python
  def compute_expected_va(log_probs, candidates=None):
      probs = np.exp(log_probs - np.max(log_probs))
      probs = probs / np.sum(probs)
      ...
  ```
- **問題点**: すでに確率値である `probs`（各要素が高々 0.01〜0.05 程度）を `log_probs` として受け取ると、`np.exp(probs - max(probs))` によって全候補の重みがほぼ 1.0 に揃ってしまい、**81個の候補に対してほぼ完全な一様分布（均等確率 1/81 ≈ 0.0123）に平滑化されてしまいます**。
- これにより、介入の有無に関わらず期待値が常に `5.0000...` 付近に固定化され、介入効果量（傾きや差分）が `0.00006` のような極小ノイズレベルに潰れていました。

### ② `attenuation_ratio` が `NaN` になった理由
- `run_rq1_state_induction.py` の 593行目・599行目に以下のフィルタがあります：
  ```python
  natural_shift_v = abs(clean_aff_ev - clean_neu_ev)
  if natural_shift_v > 0.05:
      sample_att_ratios_v.append(float(attenuation_v / natural_shift_v))
  ```
- 上記の一様分布化バグによって `natural_shift` が常に 1e-5 程度となり、`> 0.05` を満たすサンプルが 0 個であったため、`sample_att_ratios_v` が空リストとなり、結果として `NaN` が出力されました。

---

## 5. 実施した修正と検証内容

1. **`likelihood.py` の関数分離と仕様明確化**:
   - `compute_expected_va(log_scores, candidates=None)`: log-scores 専用（未正規化）として明確化。
   - `compute_expected_va_from_probs(probs, candidates=None)`: 正規化確率分布用として新設（`np.isclose(probs.sum(), 1.0, atol=1e-5)` 検証付き）。
   - `src/affective_empathy_eval/__init__.py` にエクスポート追加。
2. **リポジトリ全体のコールサイト監査・修正**:
   - `v3/primary/run_rq1_state_induction.py` (10箇所)
   - `v3/primary/run_rq2_spatiotemporal_maps.py` (3箇所)
   - `v3/primary/run_rq3_path_mediation.py` (7箇所)
   - `v3/primary/run_confirmatory_replication.py` (9箇所)
   - `v2/primary/run_rq3_causal_map.py` (4箇所)
   - `scripts/run_candidate_space_sensitivity.py` (1箇所)
   - すべて `log_likelihoods, probs = compute_sequence_likelihoods_for_candidates(...)` から `compute_expected_va(log_likelihoods, candidates)` へ統一。
3. **回帰テスト・単体テストの実行**:
   - `tests/test_likelihood.py` に二重Softmax歪み検出回帰テスト等を追加し合格（17 passed, 14 passed, 94 passed）。
4. **8サンプル実機 Sanity Check**:
   - 手計算 Softmax と `compute_expected_va` が完全一致（True）。
   - 自然変位平均: Valence=0.1380, Arousal=0.4891（平滑化から完全脱却）。
5. **過去バグ結果の退避**:
   - `archive/results_v3_double_softmax_bug_20260920/` へ旧成果物を退避・隔離。

---

## 6. V3 本番再実行結果（2026-09-20 18:33 実行）

- **実行コマンド**: `run_rq1_state_induction.py --force`（ログ: `production_v3_20260920_183353.log`）
- **判定結果**: **`NO_GO`（客観的 Negative Result として停止）**
- **詳細指標の比較**:

| 指標 | 修正前 (バグ時) | 修正後 (本番再実行) | Gate 基準 | 判定結果 |
|---|---|---|---|---|
| **Valence Dose-Response 傾き** | $0.00006$ | **$0.00307$** (95% CI: $[0.00276, 0.00339]$) | $> 0.05$ | **FAIL** (基準未達) |
| **Arousal Dose-Response 傾き** | $0.00004$ | **$0.00181$** (95% CI: $[0.00148, 0.00212]$) | $> 0.05$ | **FAIL** (基準未達) |
| **特異性 (Specificity)** | $0.00002$ | **$0.00046$** (95% CI: $[-0.00010, 0.00098]$) | $> 0.05$ | **FAIL** |
| **トピック制御 (Topic TVD)** | $0.00107$ | **$0.00068$** (95% CI: $[0.00046, 0.00092]$) | $< 0.15$ | **PASS** |
| **減衰率 (Attenuation / Relevance)** | NaN | **$0.01069$** (95% CI: $[-0.00573, 0.02601]$) | $> 0.05$ | **FAIL** |

> **科学的結論**:
> 測定実装の二重 Softmax による平滑化を排除した結果、介入による傾きは統計的に厳密に正（CI 下限 $> 0$）を示したものの、その効果量は約 $0.0031$ であり、事前登録基準である $0.05$（5% 変位）には届きませんでした。
> したがって、Gate 条件を変更しない原則に従い、本パイプラインは安全設計に基づき RQ1 で正しく `NO_GO` 停止しました。これは真正な研究上の客観的 Negative Result です。

---

## 7. Behavioral から V3 までの全ステージ実行状況一覧

| ステージ | 対象 | 実行状況 | 正常性判定 | 検出事項・詳細 |
|---|---|---|---|---|
| **Behavioral** | 全8モデル (EmoBank 1000件, AIPsy 480件) | 完了 (10.8h) | **完全正常復旧 (完了)** | ・生測定データ (Raw) は全8モデル100%完備。<br>・AIPsy Derived 表 (RQ1〜RQ4) は正常生成。<br>・**【復旧完了】** `summarize_behavioral_emobank.py` の列名参照（`w_` / `r_` / `s_`）および有限値マスク（NaN保護）を修正し再集計を実行。`behavioral_emobank_metrics.csv`（全72通り、19.4 KB）の正常出力を確認。 |
| **V1** | 8モデル (Phase A, B, C, E6) | **未完** | **一部完了 / 未完走** | ・`qwen_base` (Phase A, B, C, E6) は完了。<br>・`olmo_base` (Phase A, B) は完了。<br>・Llama (base/instruct), Gemma (base/instruct), OLMo-instruct は未着手。<br>・E6で一度 `TypeError` 発生後に修正実行された履歴あり。 |
| **V2** | 4ファミリー (RQ1/2幾何, RQ3因果マップ) | **一部完了 / 実行中** | **RQ1/2正常 / RQ3実行中** | ・RQ1/RQ2 (幾何クロスデコーディング): 全4ファミリー完走し `v2_cross_family_summary.json` (Bootstrap CI含む) 生成完了。<br>・RQ3 (Causal Map): Qwen は完了 (`v2_causal_map_qwen.json`, 16.8万行)。**現在 PID 2635012 にて Llama の Causal Map が継続稼働中** (GPU-Util 96%)。 |
| **V3** | Qwen + Confirmatory (RQ1 Gate) | **RQ1完了・停止** | **正常停止 (NO_GO)** | ・二重 Softmax バグ修正、全件監査、回帰テスト合格後に RQ1 再実行完了。<br>・効果量は検出されたが事前登録 Gate 基準 (0.05) 未達のため、`NO_GO` により安全に RQ1 で停止。 |

---

## 8. Behavioral 集計不整合の解消内容と検証結果

### 修正内容
1. **列名プレフィックスの自動解決**:
   - 生データ CSV の列プレフィックス（`w_`, `r_`, `s_`）とスクリプト内のタスクキー（`writer`, `reader`, `self`）を安全に解決する辞書 `TASK_PREFIX` を導入し、両形式に対応。
2. **NaN・非有限値の保護**:
   - 一部の極端な予測値や欠損値が存在する場合でも、`safe_corr` と同様に `mask = ~np.isnan(df[ev_col].values) & ~np.isnan(df[gt_col].values)` により有限値のみを抽出して `mean_absolute_error` / `root_mean_squared_error` / `np.nanmean` / `np.nanstd` を計算する堅牢な処理を追加。

### 出力検証
- コマンド: `.venv/bin/python behavioral/analysis/summarize_behavioral_emobank.py`
- 実行結果: `exit code 0`
- 生成成果物:
  - `behavioral/results/derived/emobank_3way_summary/behavioral_emobank_metrics.csv`: **19,355 bytes (74行、全8モデル×3タスク×3次元=72行の完全集計)**
  - `behavioral_emobank_coupling.csv`: 2,781 bytes (全24行)
  - `behavioral_emobank_neutral_rates.csv`: 784 bytes (全8モデル)

---

## 9. V1 Phase A/B Gemma 3 非有限値（float32無限大）バグの修正

### エラー原因
- コマンド `bash scripts/run_production_v1.sh cuda:0 --family gemma` 実行時、`google/gemma-3-1b-pt`（Base モデル）の隠れ層活性ベクトル抽出において、特定のトークン活性値が極端に巨大（または `inf`）となり、scikit-learn の `StandardScaler.fit_transform(X_train)` で `ValueError: Input X contains infinity or a value too large for dtype('float32')` が発生して異常終了していました。

### 実施した修正内容
1. **`v1/primary/run_phase_a.py`**:
   - `extract_hidden_states_batched`: 抽出した活性配列に対して `np.nan_to_num(arr, nan=0.0, posinf=1e4, neginf=-1e4)` および `np.clip(arr, -1e4, 1e4).astype(np.float32)` によるサニタイズを追加。
   - `evaluate_regression_probe`, `evaluate_classification_probe`, `evaluate_cross_decoding_and_geometry`: 各プローブ関数の入口で入力配列 `X` / `H_R` / `H_S` の有限値保護を追加。
2. **`v1/primary/run_phase_b.py`**:
   - `extract_single_layer_hidden_states` および `evaluate_probe_accuracy` にも同様の有限値サニタイズ処理を追加（予防的措置）。

### 検証
- 単体テスト: `.venv/bin/pytest -q tests/test_geometry.py tests/test_v1*.py tests/test_v3*.py` $\to$ **27 passed**
- Gemma 3 Phase A ドライラン: `.venv/bin/python v1/primary/run_phase_a.py --model-id google/gemma-3-1b-pt --model-prefix gemma_base --dry-run` $\to$ **正常終了 (exit code 0)**



