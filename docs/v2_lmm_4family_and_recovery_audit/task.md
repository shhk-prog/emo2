# タスクリスト: V2 H3 LMM 4-Family カバレッジ検証および H4 Recovery 調査

## 背景
ICLR 2027論文の結果監査において、Behavioral、V1、V3、V2のPresentation / Confirmatory Note等はすべてfreeze可能と判定された。
残る確認・対応事項は以下の2点（+ 1点クリーンアップ）に絞られている。

## タスク項目

- [x] **1. `v2_causal_pair_level.csv` の family 内訳確認と RQ3 状況の特定**
  - 今朝 03:07 に Llama も含めて全4ファミリー（Qwen, Llama, Gemma, OLMo）の 516,000 件が揃い、Primary LMM (N=344,000) もフィッティング完了していることを確認。
- [x] **2. H4 Recovery のデータ実態調査**
  - Gemma, OLMo は完了済みキャッシュあり。Qwen は現在実行中（Reader完了、Self実行中）。Llama は Qwen 完了後に自動実行されることを確認。
- [x] **3. Behavioral RQ4 Note の科学的表現修正（凍結対応）**
  - output-level behavioral covariation である旨を明記し、内部ダイナミクスや事後学習への過大解釈を抑制する文面に改修。
- [x] **4. V3 Confirmatory Matrix Note の科学的表現修正（凍結対応）**
  - 「棄却」を「支持されなかった」に改め、事前 Gate NO_GO に基づき全体結論を正確に記載する文面に改修。
- [x] **5. V2 パイプライン完了およびテーブル最終再生成・全実験完了監査**
  - 今朝 06:23:34 に全 V2 ステージ（RQ1--RQ4、Confirmatory Analysis）が正常終了。
  - `build_all_paper_summaries.py --strict` により全 4 ステージの 25 テーブル、9 図表データ、Primary (341行) / Secondary (515行) を完全生成。
  - `generate_paper_results_tables.py` により全 TeX テーブルを最新生成。
  - `pytest tests/test_paper_summary_invariants.py` が 8/8 全件 PASSED。全実験・全結果の完全揃いを確認。
- [x] **6. V2 H3 Note および科学的解釈方針の反映**
  - `v2_confirmatory_summary.tex` / `v2_reorganization_summary.tex` / `v2_h3_causal_lmm.tex` の Note を修正：H3は全面的棄却ではなく、ValenceのAlignment $\times$ Task（$q = 0.044$）が部分的に支持され、層深度再配置は支持されなかった旨を明記。
  - `v2_causal_relocation.tex` / `v2_causal_controls.tex` の Note を修正：未評価（---）であり、H3の統計的結論は sample-level LMM を根拠とすることを明記。
  - Base/Instruct 差は randomized post-training intervention ではないため、「post-trainingのcausal effect」と呼ばず「post-training-associated reorganization」として扱う記述方針を統一。
- [x] **7. V2 H4 Distribution Recovery Note の介入方向修正**
  - `v2_distribution_recovery.tex` / `v2_reorganization_summary.tex` / `scripts/summarize_v2_reorganization.py` の Note を修正：逆になっていた介入方向を「Instructモデルの内部表現を同一familyのBaseモデル由来表現で置換またはalignmentした場合に、InstructのVA output distributionがBase distributionへどの程度接近するか」に正しく修正。Base/Instruct差をpost-trainingの因果効果とは呼ばない共通原則を維持。
