# 実装計画: V2 Presentation および Confirmatory 整合性の修正

## 1. 目的
ICLR 2027論文成果物の最終監査において特定された、V2（事後学習に伴う内部表現再編）に関する10項目のpresentationおよびconfirmatory整合性の不備を解消し、論文用表・サマリーCSVを完全に同期・凍結可能（GO）な状態にする。

## 2. 変更対象ファイル
1. `v2/primary/run_confirmatory_analysis.py`:
   - H4 判定において `expected_families = {"qwen", "llama", "gemma", "olmo"}` に対する厳格な4-family completeness チェックを導入。
   - 観測ファミリーが不足している場合は `h4_status = "INCOMPLETE"`（Not evaluated）とし、確証的判定を行わない。
2. `v2/scripts/build_paper_summary.py`:
   - Table V2-1: `mean_procrustes_distortion` が center-of-mass だった問題を解消し、列名を `procrustes_distortion_center` に改名。必要に応じて真の layer-mean distortion も保持。
   - Table V2 Confirmatory:
     - H1a: Reader Procrustes Distortion, Self Procrustes Distortion に加え、RSA Reader, RSA Self も exact match 可能な形式で登録。
     - H1b: Reader / Self の両方（Valence Reader, Valence Self, Arousal Reader, Arousal Self 計4行）を出力。
     - H3: Valence と Arousal の両軸（各3 primary interaction terms + secondary main effect）を出力。
     - H4: 4-family 未完であることを明記し、`supported: "Incomplete / Not evaluated"`、CI は descriptive only として出力。
3. `scripts/summarize_v2_reorganization.py`:
   - `generate_h1_h2_table`: `hypothesis + metric` の exact match で `table_v2_confirmatory.csv` から取得。全 `---` 問題を解消。
   - `generate_causal_lmm_table`: Valence と Arousal の両軸（各3 primary terms + main effect）を表示。Note から古いハードコード（$\beta=0.181$ 等）を削除し、定性的な正確な記述に統一。
   - `generate_distribution_recovery_table`: Note の「GemmaおよびOLMoの実測値」を「現時点ではGemma familyのみ実測値が利用可能で、4-family H4 confirmatory integrationは未完」に修正。
   - `generate_confirmatory_summary_table`: Note を「H1a は支持、H1b/H2 は Not Supported、H3 は Not Supported、H4 は Incomplete」に修正。H1b Self 行や H3 Arousal 行も正しく描画。
   - `generate_markdown_summary`: ハードコード数値を全廃し、CSVから動的生成。
4. `tests/test_paper_summary_invariants.py`:
   - Table V2-1 の列名改名（`procrustes_distortion_center`）に対応するアサーションの整合確認。

## 3. 検証計画
1. `python v2/primary/run_confirmatory_analysis.py --strict` の実行確認（4-family 不足時の `INCOMPLETE` 動作）。
2. `python v2/scripts/build_paper_summary.py --strict` の実行確認。
3. `python scripts/build_all_paper_summaries.py --strict` で全ステージ一括再生成。
4. `python scripts/generate_paper_results_tables.py --repo-root . --out-dir iclr2027/tables` で全 LaTeX / Markdown 表再生成。
5. `pytest -q tests/test_paper_summary_invariants.py` の 8/8 pass 確認。
6. `iclr2027/tables/v2_*.tex` の差分確認（全 `---` 解消、両軸表示、Note 修正の確認）。
