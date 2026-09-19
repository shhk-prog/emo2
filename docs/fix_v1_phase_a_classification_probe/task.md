# タスク: V1 Phase A AIPsy 分類プローブの文字列ラベルエラー解消

## 状況
- `bash scripts/run_production_v1.sh cuda:0` 実行時、`v1/primary/run_phase_a.py` の AIPsy 分類プローブ評価において以下のエラーが発生し停止した：
  `ValueError: invalid literal for int() with base 10: 'rage'`
- 原因: `evaluate_classification_probe` 内で `y_preds` が `dtype=int` の配列として初期化されているのに対し、AIPsy の目的変数 `y` は文字列（感情名 `'rage'`, `'sadness'` 等）であり、`clf.predict` が返した文字列を代入しようとして型変換エラーとなった。

## 目標
1. `v1/primary/run_phase_a.py` の `evaluate_classification_probe` に `LabelEncoder` を導入し、入力ラベル `y`（文字列または数値）を確実に 0 から C-1 の整数に正規化して学習・予測・評価を行う。
2. クラス確率 `y_probs` の代入時、学習 fold 内の `clf.classes_` に応じた適切な列配置を行い、不均衡やクラス欠損に対しても安全にする。
3. 単体テストを作成・実行して、文字列ラベルおよび数値ラベルの双方で分類プローブが正しく動作することを確認する。

## タスクリスト
- [x] 計画策定とタスクファイルの作成 (`docs/fix_v1_phase_a_classification_probe/`) <!-- id: 0 -->
- [x] `v1/primary/run_phase_a.py` の `evaluate_classification_probe` を `LabelEncoder` を用いて修正 <!-- id: 1 -->
- [x] 分類プローブの単体テストの実行・検証 <!-- id: 2 -->
- [x] 修正内容の確認 (Walkthrough) ドキュメントの作成 <!-- id: 3 -->
