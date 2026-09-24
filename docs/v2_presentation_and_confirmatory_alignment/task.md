# タスクリスト: V2 Presentation および Confirmatory 整合性の修正

## 背景
ICLR 2027論文に向けた成果物監査において、Behavioral、V1、V3、paper-summary infrastructure（dry_run排除、manifest整合性、再生成パイプライン）は固定可能（GO）と判定された。
残る修正対象は、V2のpresentationおよびconfirmatory整合性に関する10項目に絞られている。

## タスク項目

- [x] **1. `v2_h1_h2_reorganization.tex` の全`---`問題の解消**
  - `scripts/summarize_v2_reorganization.py` 内の検索キーを `hypothesis + metric` の exact match に修正し、`table_v2_confirmatory.csv` の実値（distortion, shift, sharing）を正しく反映する。
- [x] **2. V2 H1b に Reader / Self の両行を追加**
  - Methodsの定義 $\Delta d_{D,T}^*$ ($T \in \{\text{Reader}, \text{Self}\}$, Valence/Arousal) に合わせ、`v2/scripts/build_paper_summary.py` の builder で Reader と Self の両行（計4行）を出力するようにする。
  - LaTeX生成側でも Reader / Self の両行を表示する。
- [x] **3. V2 confirmatory summary Note の結果整合**
  - `v2_confirmatory_summary.tex` の Note に「H1a, H1b, H2において有意差が確認された」とある誤記を修正。
  - H1a Supported、H1b/H2 Not Supported（4-family bootstrap CIがprespecified criterionを満たさず）、H3 primary interaction未達、H4未完という正確な結果に整合させる。
- [x] **4. `v2_h3_causal_lmm.tex` Note の古いハードコード削除**
  - Note に書かれている古い世代の数値（$\beta=0.181$, $p<0.001$ 等）を削除。
  - FDR補正後の有意水準に関する定性的な正確な記述に統一する。
- [x] **5. V2 H3 table の Valence / Arousal 両軸表示**
  - 現在 Valence しか表示していない `v2_h3_causal_lmm.tex` を、Methods に基づき Valence と Arousal の両軸（各3 primary interaction terms + secondary main effects）を表示するよう拡張。
- [x] **6. H4 を 4-family completeness 判定に厳格化**
  - `v2/primary/run_confirmatory_analysis.py` およびサマリー生成で、観測 family 数が事前定義された4 family（qwen, llama, gemma, olmo）に満たない場合は `h4_status = "INCOMPLETE"`（Not evaluated / descriptive only）とする。
- [x] **7. H4 Note の誤記修正**
  - 「GemmaおよびOLMoの実測値」という誤記を「現時点ではGemma familyのみ実測値が利用可能で、4-family H4 confirmatory integrationは未完」に修正。
- [x] **8. V2 geometry CSV の列名改名と曖昧性解消**
  - `table_v2_1_geometry.csv` の `mean_procrustes_distortion` は Center-of-Mass であるため `procrustes_distortion_center` に改名。
  - 真の mean distortion（layer-wise mean）も別途明示的に保存・分離する。
- [x] **9. V2 Markdown summary のハードコード全廃**
  - `iclr2027/tables/v2_reorganization_summary.md` に直接書かれている古いハードコード数値を全廃し、canonical CSVから動的に生成する。
- [x] **10. paper summary および tables の一括再生成と検証**
  - `results/derived/paper_summary/` を再生成（strict モード）。
  - `generate_paper_results_tables.py` で LaTeX/Markdown 表を再生成。
  - `pytest tests/test_paper_summary_invariants.py` および全関連テストが pass することを確認。
  - 再生成された表の数値・Noteが正確であることを検証。
