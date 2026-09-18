# タスク定義: v2 実験⑦ Unembedding / RMSNorm / lm_head Swap（8条件交差検証）の詳説

## タスク概要
- v2研究における「実験⑦ Unembedding / RMSNorm Swap Analysis: Locus of Self-Report Neutralization」（Phase 7 / Table 5）の目的、介入手法、定量的結果、および科学的結論を整理・解説する。
- 自己報告の中立化が「最終出力層（RMSNorm / lm_head）の重み更新」によるものか、「出力段に入る直前の残差ストリーム（Final Residual State）」によるものかを決定的に弁別した結果を詳述する。

## 参照ソース
- `v2/docs/post_training_readout_experiment/paper_draft.md` (Section 4.8, Table 5, Section 5)
- `v2/scripts/run_unembedding_norm_swap.py`
- `v2/results/derived/phase7_strict_path_patching/unembedding_norm_swap_results.csv`
- `v2/results/derived/phase7_strict_path_patching/unembedding_swap_results.csv`
