# 変更内容の確認 (Walkthrough): V3 Reader-Grounded 情動方向への刷新および設定・因果サンプル数精緻化

## 1. 概要
本改修では、研究全体の連関と中心主張を飛躍的に強化するため、V3（Mechanistic State-to-Report）における主要な情動方向 $d_V, d_A$ を、自己報告（Self-report）からではなく、同じ刺激に対する**感情認知タスク（Reader Prediction）**から同定する「Reader-Grounded Affect Direction」へと刷新しました。

これにより、4つの実験ステージが以下の通り完全に連結されました：
$$\boxed{\text{Behavioral coupling}} \longrightarrow \boxed{\text{V1 shared representation}} \longrightarrow \boxed{\text{V2 reorganization}} \longrightarrow \boxed{\text{V3 Reader-grounded affect state} \longrightarrow \text{Self-report}}$$

あわせて、RQ2 Discovery における因果介入サンプル数を 5 件から 15 件へ増量し、設定ファイルの注記を整理しました。

---

## 2. 実施内容と検証結果

### 2.1 設定ファイルの整理 (`configs/v3_experiments.yaml`)
- `neutral_text_column: "neutral_text"`: 元CSVの列名ではなく、`load_v3_matched_pair_table()` が clinical 行に pair_id 対応の neutral テキストを自動結合して付与するローダー生成列名であることを注記明記。
- `spatiotemporal.n_causal_samples: 15`: 因果介入サンプル数を明示的に設定（推奨範囲 10〜20件の中央値）。

### 2.2 V3-RQ1 State Induction (`v3/primary/run_rq1_state_induction.py`)
- **Reader Prediction の抽出**:
  train split の各刺激に対して `TaskType.READER` のプロンプトを提示し、モデル自身の感情認知予測値 $y_{V,\text{train}}^R, y_{A,\text{train}}^R$ を算出。
- **Primary 情動方向の学習**:
  内部表現 $H_{\text{train}}$ から $y_{V,\text{train}}^R, y_{A,\text{train}}^R$ を予測するリッジ回帰により、Primary 情動方向 $d_V^R, d_A^R$ を同定。
- **Self-report への介入検証**:
  同定された $d^R$ を Self-report 処理に注入し、自己報告への十分性・必要性・特異性を検証。
- **Secondary 分析の保持**:
  自己報告自身から学習した $d_V^S, d_A^S$ も同時に推定し、両方向のコサイン類似度 $\cos(d^R, d^S)$ を算出して結果辞書に保持。

### 2.3 V3-RQ2 Spatiotemporal Maps (`v3/primary/run_rq2_spatiotemporal_maps.py`)
- **Reader-Grounded 4-Map**:
  各刺激に対する Reader Prediction を Primary デコード能 $D_V^R, D_A^R$ および局所介入方向 $d_V^R(l, s), d_A^R(l, s)$ のターゲットに設定。
- **因果介入サンプル数の拡張**:
  `n_causal_samples=15`（設定ファイル連動）とし、統計的安定性を向上。
- **Secondary 4-Map**:
  従来の自己報告をターゲットとするデコード能 $D_V^S, D_A^S$ も `secondary_maps` として計算・保持。

### 2.4 V3-RQ3 Path Mediation (`v3/primary/run_rq3_path_mediation.py`)
- **Discovery 候補層スクリーニング**:
  Discovery セットにおける情動ターゲットを Reader Prediction $y_V^R$ とし、Reader-grounded な情動状態 $d_{\text{stim}}^R$ を同定。
- **Confirmation 媒介推論**:
  $d_{\text{stim}}^R \rightarrow M \rightarrow Y$（Self-report）の自然間接効果（NIE）および媒介割合を検定。

### 2.5 V3 Confirmatory Replication (`v3/primary/run_confirmatory_replication.py`)
- **他 3 ファミリー（Llama, Gemma, OLMo）での統一検証**:
  各モデルの Reader Prediction を取得して各層固有の $d_V^{R,(l)}, d_A^{R,(l)}$ を推定し、Self-report への介入・層解離・時間的出現を統一的に検証。

---

## 3. テストと動作検証

### 3.1 単体・統合テスト
```bash
$ .venv/bin/pytest tests/test_production_entrypoints.py -v
tests/test_production_entrypoints.py::test_all_primary_entrypoints_exist PASSED [ 50%]
tests/test_production_entrypoints.py::test_production_dry_run_dispatch PASSED   [100%]
============================== 2 passed in 2.38s ==============================

$ .venv/bin/pytest -q
............................................................                            [100%]
60 passed in 8.06s
```

### 3.2 V3 各スクリプトのドライラン検証
- `python v3/primary/run_rq1_state_induction.py --dry-run`: **GATE DECISION = GO**
- `python v3/primary/run_rq2_spatiotemporal_maps.py --dry-run`: **Pass** (Delta Peak: V=0.185, A=0.222)
- `python v3/primary/run_rq3_path_mediation.py --dry-run`: **Pass** (Attenuation Ratio: 0.746)
- `python v3/primary/run_confirmatory_replication.py --dry-run`: **Pass** (Llama, Gemma, OLMo 全再現)

### 3.3 結果ディレクトリの初期化確認
- `bash scripts/clear_stage_results.sh v3` を実行し、全 results ディレクトリが `.gitkeep` のみのクリーンな状態に保たれていることを確認。

---

## 4. 本番全再実行コマンド

本番のフルデータ（4小型ファミリー: Qwen, Llama, Gemma, OLMo）に対する実行コマンドは以下の通りです：

```bash
# 1. Behavioral Stage（EmoBank & AIPsy）
bash scripts/run_production_behavioral.sh

# 2. V1 Stage（Phase A / B / C E3, E4, E6 / Summarize）
bash scripts/run_production_v1.sh

# 3. V2 Stage（RQ1/RQ2, RQ3 Causal Map, RQ4 Recovery Patching, Confirmatory）
bash scripts/run_production_v2.sh

# 4. V3 Stage（RQ1 State Induction, RQ2 Spatiotemporal, RQ3 Mediation, Confirmatory Replication）
bash scripts/run_production_v3.sh
```
