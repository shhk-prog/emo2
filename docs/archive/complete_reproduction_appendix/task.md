# Task: v1–v3 完全再現 Supplementary Methods (Appendix) の精緻化

## 目標
`v3/docs/paper2.md` の Appendix 全体を、正本データ・実在スクリプト・確認済みCLIに厳密に準拠した章立て（Appendix A〜G）へ再編し、v3論文の主主張と完全整合した完全再現仕様書（Supplementary Methods）として完成させる。

### 最新監査・是正項目（完了）
- [x] **E.10 Focused 39-Pair 中央値**: L18 Residual **49.31%**、L20 Residual **60.16%**、L24 Residual **61.57%**、L15 MLP **+0.50%** に完全一致。
- [x] **E.12 多層パッチングパイプライン分離**: Section 15.6 の独立パイプライン実測値（L18: **43.51%**, L20: **51.85%**, L24: **55.08%**, L20+24: **55.67%**, L18+20+24: **55.74%**, 7層: **55.35%**）を適用し、focused sweep との分離注記を明記。
- [x] **E.13 & E.14 介入位置（Token Position）の実装完全準拠**: 「Generation prefix last token」を排し、実装コード（`target_pos = inputs.input_ids.shape[1] - 1`）通り `Token Position: Prompt final token (target_pos = prompt_len - 1), not generation-prefix position.` に修正。時間的分離（Prompt-time accessibility / necessity vs Generation-time causal leverage）を厳格に保持。
- [x] **E.13 統計手法記述の厳密化**: 「Paired t-test on log-odds shift」を排し、実装通りの `2D Joint OT displacement`、等方的・直交ランダム帰無分布に対する `standardized Z-scores`、`pseudo-count empirical p-values`、`Benjamini–Hochberg FDR correction` の記述に刷新。
- [x] **E.14 N=100 特異性検定 p値範囲**: `p_value_perp` の範囲を **`0.2970 〜 1.000`**（L20 Resid: 0.2970, L15 MLP: 0.8614, L24 MLP: 0.9802, L24 Attn: 1.000）、全サイト FDR $q = 1.000$ に是正。
- [x] **E.17 結論表現の緩和**: 「モデルファミリーや事後学習手法に依存して異なり得ることが実証された」から「モデルファミリーや学習履歴に依存する可能性と整合する」へ修正。
- [x] **E.17 Llama Residual 範囲**: 全16層の負値回復範囲を **`-36.18% 〜 -4.23%`**（最大正回復は L4 Attention の **0.73%**）に是正。
- [x] **E.15 実在スクリプト名**: **`v3/scripts/run_dual_outcome_behavior.py`** に是正。
- [x] **E.16 Mood Congruency 実測値と位置づけ**: 保存先を `v3/results/derived/mood_congruency/...summary.csv` に指定し、本文確定値（L14 $\beta=-0.0323$, $p=1.18\times 10^{-70}$; L16 $\beta=-0.0169$, $p=6.20\times 10^{-40}$; L20 $\beta=+0.0244$, $p=3.23\times 10^{-61}$; L16 Random $\beta=+0.0444$）に基づき探索的分析として位置づけ。
- [x] **Appendix B / G 公開リモート追跡確認**: GitHub `main`（`origin/main`）上に `run_generation_multilayer_residual.py`、`plot_main_figure1.py`、`focused_causal_sweep_39pairs_pair_level.csv`、`generation_multilayer_residual_results.csv`、`figure1_four_panel_dissociation.png/pdf` が確実に push 済みであることを確認。
- [x] **Appendix D (v2) の監査**: 過度な断定表現を排し、実在するPhaseディレクトリ・スクリプトで確認できる客観的事実のみに抑制。
- [x] **全検証チェック 100% PASS**。
