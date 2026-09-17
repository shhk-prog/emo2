# 3-Way VAD集計結果（summarize_3way_vad.py）の正当性検証計画

## 概要
ユーザーが実行した `python v1/scripts/summarize_3way_vad.py --results-dir v1/results/emobank_3way_vad_test1k` の集計結果について、データ読み込み、前処理（スケーリング）、統計指標の算出、出力フォーマットの観点から正しく集計されているかを検証する。

## 検証項目と設計

### 1. 入力データセットおよび対象モデルの網羅性
- 対象フォルダ `v1/results/emobank_3way_vad_test1k` 内の全8モデル（4ファミリー：Qwen 2.5 1.5B, Mistral 7B v0.1, Llama 3.2 1B, Gemma 2 2B × Base/Instruct）が漏れなく集計されているか。
- 各CSV（N=1000）の行数・サンプル数が正しく認識されているか。

### 2. スケーリングおよび評価尺度の整合性
- LLMの出力期待値（1〜9スケール、中央値5.0）と EmoBank の人間アノテーション（1〜5スケール、中央値3.0）の変換式：
  $$y_{\text{scaled}} = \frac{y + 1.0}{2.0}$$
  - $1.0 \to 1.0$, $5.0 \to 3.0$, $9.0 \to 5.0$ の線形対応が正しいか。
  - 相関係数（Pearson $r$, Spearman $\rho$）が線形変換に対して不変であることの確認。
  - 較正誤差（MAE, RMSE）が人間スケール（1〜5）上で正しく算出されているか。

### 3. タスクとグラウンドトゥルースの対応
- ① Writer-State Estimation ($W$): 正解基準 `human_writer_v, a, d`
- ② Reader-Response Prediction ($R$): 正解基準 `human_reader_v, a, d`
- ③ Self-Report ($S$): 外部基準 `human_reader_v, a, d`
- ④ 内部結合度（Internal Cognitive Coupling）: 同一モデルの $R$ 予測と $S$ 報告の相関 $\text{Corr}(R, S)$

### 4. 退化率（Collapse）の定義と計算
- タスク全体の完全中立率: $(V=5) \land (A=5) \land (D=5)$ の比率（%）
- 単一次元の退化率: 当該次元の Greedy 出力が 5 となった比率（%）

### 5. 既存個別サマリーJSONとの突合検証
- 各モデルの `*_3way_vad_summary.json` の各数値（相関係数、平均値、中立率など）と新集計 `3way_vad_summary.csv` が一致しているか。

### 6. エッジケースのハンドリング
- 分散0（標準偏差0）の場合の安全な処理（`safe_corr`）およびレポートでの `N/A (std=0)` 表示の整合性。

## 検証方法
- ソースコードロジックの静的解析
- 既存データおよび生成CSV / Markdownレポートのデータ照合
