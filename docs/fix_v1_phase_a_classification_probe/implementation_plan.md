# V1 Phase A AIPsy 分類プローブ文字列ラベルエラー解消 実装計画

## 概要
`bash scripts/run_production_v1.sh cuda:0` 実行時、`v1/primary/run_phase_a.py` の AIPsy 分類プローブ評価処理において、`ValueError: invalid literal for int() with base 10: 'rage'` が発生しました。
本計画では、分類プローブ関数 `evaluate_classification_probe` で入力ラベルを `LabelEncoder` により一元的に整数インデックス化し、文字列ラベルと数値ラベルの双方に対して型整合性とロバスト性を担保します。

## 原因分析
- `v1/primary/run_phase_a.py` の `evaluate_classification_probe`:
  ```python
  y_preds = np.zeros_like(y, dtype=int)
  ...
  clf.fit(X_train_scaled, y_train)
  y_preds[val_idx] = clf.predict(X_val_scaled)
  ```
- AIPsy データセット（`aipsy_4split_all.csv`）のターゲット変数（`emotion` など）は文字列（`'rage'`, `'sadness'` など）です。
- `y_train` が文字列配列の場合、`LogisticRegression.predict()` は予測されたクラス文字列を返します。
- これを `dtype=int` の `y_preds` のスライスに直接代入しようとしたため、Python が文字列を `int` に変換しようとして `ValueError: invalid literal for int() with base 10: 'rage'` が送出されました。

## 修正方針

### 1. `LabelEncoder` によるラベルの整数インデックス化
- 関数の先頭で `sklearn.preprocessing.LabelEncoder` を用いて、ラベル配列 `y` を `0, ..., C-1` の整数配列 `y_encoded` に変換します。
- `StratifiedGroupKFold` / `GroupKFold` / `StratifiedKFold` の分割生成、`LogisticRegression` の学習・予測、各種スコア（`balanced_accuracy_score`, `f1_score`, `roc_auc_score`）の算出をすべて `y_encoded` を基準に行います。

### 2. クラス確率 `predict_proba` の安全な格納
- 多値分類時、fold 内の学習サンプルに一部のクラスが含まれないケースを想定し、`clf.classes_` に応じた列へ確実に配置します：
  ```python
  probs = clf.predict_proba(X_val_scaled)
  for c_idx, c in enumerate(clf.classes_):
      y_probs[val_idx, c] = probs[:, c_idx]
  ```

## 変更対象ファイル
- **[MODIFY]** [v1/primary/run_phase_a.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_a.py)

## 検証計画
1. `tests/test_v1_token_and_probe_alignment.py` を実行し、既存の分類プローブテストが正常に通ることを確認。
2. 文字列ラベル（`['rage', 'joy', 'sadness', ...]`）を入力とする分類プローブのテストケースを追加し、正常にスコアが計算されることを確認。
