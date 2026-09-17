# 3-Way VAD集計結果の正当性検証タスク

## 目的
`v1/scripts/summarize_3way_vad.py` によって生成された集計結果（`v1/results/emobank_3way_vad_test1k/3way_vad_summary.csv` および `3way_vad_detailed_report.md`）が正しく計算・集計されているかを多角的に検証・精査する。

## 検証項目
1. **入力データ完全性の確認**:
   - 8モデル（4ファミリー × Base/Instruct）の各CSV（N=1000）の読み込み確認
   - 欠損値、カラム構造、サンプル数の確認
2. **計算ロジック・メトリクス定義の確認**:
   - スケーリング式：LLM出力（1〜9スケール、中立5）から人間EmoBank（1〜5スケール、中立3）への線形変換 `(y + 1) / 2`
   - 相関係数（Pearson $r$, Spearman $\rho$, Greedy $r$）の計算とスケール不変性
   - 較正誤差（MAE, RMSE）の算出
   - 退化・完全中立率（`exact_555_pct`, `pct_greedy_5`）の計算定義
   - 内部認知結合度（Internal Cognitive Coupling $\text{Corr}(R, S)$）の計算
3. **既存の個別サマリーJSONとの突合検証**:
   - 各モデルの `*_3way_vad_summary.json` と `3way_vad_summary.csv` の各値の照合
4. **エッジケース（分散0など）の処理確認**:
   - `llama3.2_1b_base` 等で見られる相関欠損（NaN）の理由精査
5. **最終報告の作成**:
   - `implementation_plan.md` および `walkthrough.md` の作成
   - ユーザーへの明確な報告
