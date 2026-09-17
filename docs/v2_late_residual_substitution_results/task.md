# タスク定義: v2 実験⑥ Late-Residual Substitution（厳密な後期残差代入実験）の詳説

## タスク概要
- v2研究における「実験⑥ Late-Residual Substitution under Strict Identical Prompts」（Phase 8）の目的、設計、介入手法、定量的結果、および理論的帰結を整理・解説する。
- ユーザーからの要請に応じ、中間層パッチング（実験⑤）との比較、Direct Readout仮説の棄却、後続中間層への分散的依存性などの科学的意義を明確にする。

## 参照ソース
- `v2/docs/post_training_readout_experiment/paper_draft.md` (Section 4.7, Table 4)
- `v2/results/derived/phase8_causal_scrubbing/strict_causal_scrubbing_aggregated.csv`
- `v2/results/derived/phase8_causal_scrubbing/strict_causal_scrubbing_results.csv`
- `v2/scripts/run_strict_causal_scrubbing.py`
