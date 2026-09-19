# V1 Phase A AIPsy 分類プローブ文字列ラベルエラー解消 修正確認 (Walkthrough)

## 概要
`bash scripts/run_production_v1.sh cuda:0` 実行時、`v1/primary/run_phase_a.py` の AIPsy 分類プローブ評価において発生した `ValueError: invalid literal for int() with base 10: 'rage'` を解消しました。

---

## 主な変更内容

### 1. `evaluate_classification_probe` のラベル正規化
- **対象ファイル**: [`v1/primary/run_phase_a.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_a.py)
- **変更前**:
  - `y_preds = np.zeros_like(y, dtype=int)` で `int` 配列を確保していたが、AIPsy の `y` は文字列（感情ラベル名 `'rage'`, `'sadness'` 等）。
  - `LogisticRegression.predict()` が返す文字列ラベルを `y_preds[val_idx]` に代入しようとして `ValueError: invalid literal for int() with base 10: 'rage'` が発生。
- **変更後**:
  - `sklearn.preprocessing.LabelEncoder` を適用し、入力ラベル `y`（文字列または数値）を一元的に `0, ..., C-1` の整数配列 `y_enc` にエンコード。
  - `StratifiedGroupKFold` / `GroupKFold` / `StratifiedKFold` 分割、ロジスティック回帰の学習・予測、各評価指標（balanced accuracy, macro F1, ROC-AUC）の算出をすべて `y_enc` を基準として実行。
  - 各 fold の学習済み分類器 `clf.classes_` のクラス順序・欠損を考慮し、多値分類時の確率行列 `y_probs` に正しくマッピング。

---

## 検証結果

### 単体テスト
[`tests/test_v1_token_and_probe_alignment.py`](file:///mnt/nas/home/hiromi/src/emo2/tests/test_v1_token_and_probe_alignment.py) に、文字列ラベル（多値感情名 4 クラス、二値ラベル 2 クラス）を用いた分類プローブテスト `test_evaluate_classification_probe_string_labels` を追加し、全 4 テストが成功することを確認しました。

```text
.venv/bin/pytest -v tests/test_v1_token_and_probe_alignment.py
================================== 4 passed in 30.12s ==================================
```

- 文字列ラベル（`'rage'`, `'sadness'`, `'joy'`, `'fear'`）での多値分類プローブが正常に動作することを確認。
- 二値文字列ラベル（`'clinical'`, `'neutral'`）でのプローブ評価および ROC-AUC / F1 の算出が正常に完了することを確認。
