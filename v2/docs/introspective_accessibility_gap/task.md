# 論文改訂および追加実験の実装タスク

- [x] **フェーズ1: 複数モデルでのスケーリングと頑健性検証**
  - [x] `v2/scripts` 配下の既存スクリプト（推論・尤度評価）の確認
  - [x] 複数モデル（Qwen2.5, Llama-3.2, Gemma-2 の Base/Instruct）に対応する推論スクリプトの実装
  - [x] `statsmodels` を用いた混合効果モデル（Random Effect: モデル）による統計検定スクリプトの実装
- [x] **フェーズ2: SAEを用いたVA特徴抽出と介入**
  - [x] SAE利用環境のセットアップ（既存の公開SAEの調査、または自前学習の準備）
  - [x] 中間層におけるVA空間に対応するTop-k SAE特徴の単離スクリプト実装
  - [x] SAE特徴レベルのPatching / Ablation スクリプトの実装と評価
- [x] **フェーズ3: 内省精度テスト (Introspective Accessibility)**
  - [x] モデル自身に状態変化を推測させるプロンプトの設計
  - [x] 推測結果と実際の尤度変化の乖離を定量化するスクリプトの実装
- [x] **フェーズ4: 主結果の統合と論文の再構成**
  - [x] ベースライン比較（シャッフルラベル、文長、TF-IDF等）の結果整理
  - [x] `paper_draft.md` の全面改訂（Introduction〜Conclusionのリフレーミング）
  - [x] `walkthrough.md` の作成
