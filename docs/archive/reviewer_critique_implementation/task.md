# タスクリスト: 査読指摘への即応実装

- [x] 1. 論文ドラフト (`v3/docs/paper.md`) の防御的改訂 <!-- id: 0 -->
  - C1の格下げ（先行研究に基づく標準的測定プロトコルとして再配置）
  - C4（三人称ステアリング）の主本文からの除外（探索的補足知見への位置づけ変更）
  - Singh et al. (2026) への防御注記（内省・特権的自己アクセスの否定を明記）
  - Post-training因果帰属の厳密化（単一チェックポイントペアの限界と整合的表現）
- [x] 2. 【最優先】実験A（Within-model positive control）スクリプトの実装 <!-- id: 1 -->
  - 同一Instruct内での `Peak -> Neutral` パッチングスクリプト `v3/scripts/run_within_model_positive_control.py`
  - 介入部位の比較（MLP vs Residual stream、最終トークン vs 全プロンプトトークン、複数層）
- [x] 3. 実験B（Full-State Reconstruction）解析スクリプトの実装 <!-- id: 2 -->
  - `v3/scripts/analyze_alignment_fidelity_and_manifold.py`
  - $R^2_{\text{activation}}$ (全次元平均)、Linear CKA、Pair retrieval accuracy
- [x] 4. 実験C（Empirical Manifold Test）解析スクリプトの実装 <!-- id: 3 -->
  - Natural Instructの $D_M$ 経験的パーセンタイル評価
  - Two-sample classifier (Natural vs Aligned の線形判別AUC)
  - Cosine対照群（Matched vs Unmatched same/diff vs Natural-Natural）
- [x] 5. ドキュメント保存と今後の実行計画整理 <!-- id: 4 -->
