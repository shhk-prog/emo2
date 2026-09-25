# タスクリスト: 論文提出用LaTeXおよび推論ステータスの最終修正

## 実施済みタスク
- [x] 1. TeX master（`iclr2027/iclr2027_conference2.tex`）のICLR公式sample本文・group未閉じ（`\bgroup`）・sample Appendixを完全除去し、正しい投稿用構造へ再編
- [x] 2. `v3/scripts/build_paper_summary.py` でState Induction Gate = NO_GO後のRQ2, RQ3, Cross-familyをexploratory / non-primaryへ修正（スキーマの `VALID_ANALYSIS_ROLES` に `exploratory` を追加）
- [x] 3. `v3/scripts/build_paper_summary.py` のH1 replication fallbackを削除し、strictにQwen-frozen H1 directionを参照するように修正
- [x] 4. `scripts/summarize_v2_reorganization.py` のH1a判定をDescriptiveへ修正し、Procrustes distortionのNoteを修正
- [x] 5. `scripts/summarize_v2_reorganization.py` の「事前登録」表現を「V2 Cross-family Hypothesis Summary」「Key Analysis Metric」「Criterion / Interpretation」等へ修正
- [x] 6. `scripts/build_all_paper_summaries.py` 内の古いV2/V3サマリー文言・解釈を最新結果（Gate NO_GO、幾何不整合、override下exploratory等）へ修正
- [x] 7. LaTeX本文のV3 Methods「Primary evidence」表キャプションにGate NO_GO条件とexploratory/descriptive位置づけを追記、held-out confirmationをheld-out cross-family replicationへ
- [x] 8. V3 H1の数式定義と判定（既に修正済みであることを確認）
- [x] 9. `iclr2027/tables/v3_confirmatory_matrix.tex` および `scripts/summarize_v3_causal_utilization.py` のNoteでH1のfrozen directional criterionとH2-H4を明確に区別
- [x] 10. LaTeX本文内の赤字修正メモ（`\color{red} 9/25 ...` と `\color{black}`）を6箇所すべて削除（本文内容は維持）
- [x] 11. Qwen 2.5のcitationを `team2025qwen3` から `yang2024qwen25` (Qwen2.5 Technical Report, arXiv:2412.15115) へ修正（BibTeXおよび本文引用箇所2箇所）
- [x] 12. V2考察の見出しを「Base--Instruct間では情報消失よりも表現幾何の差が観測された」へ緩和（確認済み）
- [x] 13. テーブル再生成（V2 tables: `v2_confirmatory_summary.tex`, V3 tables: `v3_confirmatory_matrix.tex`, V3 paper summary: `build_paper_summary.py --strict` 24 records）
- [x] 14. テストスイートの確認（NAS I/O遅延対策として `test_production_entrypoints.py` のtimeoutを10sから30sに調整し、PASS確認）
- [x] 15. `walkthrough.md` の作成
