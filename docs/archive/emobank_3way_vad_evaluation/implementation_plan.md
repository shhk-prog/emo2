# Implementation Plan: EmoBank 3軸（Writer / Reader / Self）× 3次元（VAD）集計システムの実装

EmoBank公式テストセット（$N \approx 1,006$）において、以下の3タスク × 3次元（V, A, D）のモデル予測・自己報告結果を集計し、モデルごとに9つの独立した表、およびモデル横断の比較表を出力する集計スクリプトを改修・整備する。

## 評価体系の定義
1. **① Writer-State Estimation ($W$)**:
   - プロンプト: `"Read the following text and estimate the affective state of the writer who wrote it."`
   - 評価基準: EmoBank **Writer VAD** との相関 ($r_V^W, r_A^W, r_D^W$)
2. **② Reader-Response Prediction ($R$)**:
   - プロンプト: `"Read the following text and estimate the affective response that this text is likely to evoke in an average human reader."`
   - 評価基準: EmoBank **Reader VAD** との相関 ($r_V^R, r_A^R, r_D^R$)
3. **③ Self-Report ($S$)**:
   - プロンプト: `"Read the following text and report your affective state."`
   - 外部参照: EmoBank **Reader VAD** との相関 ($r_V^S, r_A^S, r_D^S$)
   - 内部参照: 同一モデルの Reader Prediction との相関 ($r_V^{RS}, r_A^{RS}, r_D^{RS}$)

## 出力テーブルの構成仕様

### 1. モデル個別レポート（各モデルごとに出力）
- **【3×3 サマリーマトリクス】**:
  - 行：① Writer, ② Reader, ③ Self
  - 列：Valence, Arousal, Dominance
- **【9個の個別詳細表】**:
  - ① Writer: (1) Valence, (2) Arousal, (3) Dominance
  - ② Reader: (4) Valence, (5) Arousal, (6) Dominance
  - ③ Self-Report: (7) Valence, (8) Arousal, (9) Dominance
  - 各表に相関 ($r$, p値)、Spearman ($\rho$)、MAE、RMSE、モデル出力平均・標準偏差、人間正解平均・標準偏差を表示。

### 2. 全モデル横断レポート
- **【9大比較表（3タスク × 3次元）】**:
  - 各表で全モデル（Base / Instruct）の相関・性能をランキング比較。
- **【タスク別 統合表（3表）】**:
  - ① Writer総合表, ② Reader総合表, ③ Self総合表
- **【次元別 統合表（3表）】**:
  - Valence総合表, Arousal総合表, Dominance総合表

## 変更対象ファイル
- `[MODIFY]` [summarize_3way_vad.py](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/summarize_3way_vad.py)

## 検証方法
- 既に完了している `qwen2.5_1.5b_base` の結果（`qwen2.5_1.5b_base_3way_vad.csv`）に対して `summarize_3way_vad.py` を実行し、MarkdownおよびCSVが期待通り9個の表を含んで美しく出力されることを確認する。
