# Task: V3 and Cross-Stage Final Refinements

## 1. V3 の本格修正 (最高優先度)
- [x] 1.1 `run_v3_state_induction.py`: Topic control の clean forward (`out_base_c`) を `with patch hook:` の外に移動して介入前後の純粋比較に修正
- [x] 1.2 `run_v3_state_induction.py`: Train/Test split を単純 index split から `pair_id` に基づく Group split (unique pair_id 分割) に修正
- [x] 1.3 `run_v3_state_induction.py`: Necessity 評価の自然変位基準を $|E[V] - 5|$ から matched-neutral baseline 差 $|E[V]_{\mathrm{aff}} - E[V]_{\mathrm{neutral}}|$ を Primary に変更
- [x] 1.4 `run_v3_spatiotemporal_maps.py`: $D(l,t)$ のプローブ fit/predict を厳格な held-out (K-Fold CV) に確認・保証
- [x] 1.5 `run_v3_spatiotemporal_maps.py`: $\beta(l,t)$ が internal score $\to$ report の偏回帰係数（重回帰の偏回帰係数）として正しく計算されていることを保証
- [x] 1.6 `run_v3_confirmatory_replication.py`: 人工関数による $C(l)$ 生成や固定値 (0.45) による necessity を完全排除し、H1〜H4 すべての実モデル解析パイプラインに刷新
- [x] 1.7 `plot_paper_figures.py`: ハードコードされた固定数値 (12.4, 98.6 等) や乱数生成を除去し、実測 CSV/JSON から読み込む設計に修正
- [x] 1.8 V3 の結果およびドキュメント: `attenuation_ratio` 命名の統一、旧「greedy collapse 98.6%」ストーリーを Primary 論文ドラフトから分離

## 2. V1 の修正
- [x] 2.1 `run_v1_phase_c_targeted_ablation.py` (E6): Auto-discovery のサンプル数を 10 pair から 50 pair（または Discovery split 全体）に拡大
- [x] 2.2 `run_v1_phase_c_targeted_ablation.py`: CLI 引数を `--split-eval` (default=True) から `--no-split-eval` に自然化
- [x] 2.3 E6 の論文表記を "Double Dissociation" から "Task-specific causal specialization" に統一
- [x] 2.4 E3 レポート: 平均ベクトル cosine と pairwise cosine の両方を明示・区別して出力
- [x] 2.5 Phase C: 本番実行時の `--limit` は 0 (all-pairs) を前提とするチェックを追加

## 3. V2 の修正
- [x] 3.1 `v2/scripts/run_annotation_proxy.py` および `run_rsa_and_controlled_coupling.py` (placeholder 方向含む) を `v2/scripts/legacy/` へ移動・隔離
- [x] 3.2 Cross-family summary: 4ファミリーの bootstrap CI を一般化の決定打とせず descriptive な要約として位置づけ、paired family の p値解釈を抑制
- [x] 3.3 RQ4: native と matched-plain の回復結果が最終 summary で明確に分離して報告されるよう整形

## 4. Behavioral & 全体テストの修正
- [x] 4.1 Behavioral: 旧独自尤度コードの残存確認と `src/affective_empathy_eval/likelihood.py` への完全一本化
- [x] 4.2 最終レポート構成: Human grounding / Affective sensitivity / Dose-response / Reader-Self coupling の4ブロックに固定
- [x] 4.3 `test_v1_refinements.py` / `metrics.py`: ブートストラップ相関テスト時の定数入力に対する `ConstantInputWarning` を NaN / zero 安全処理により解消
- [x] 4.4 全体テストの確認
