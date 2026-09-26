# Task: V2包括サマリーMarkdown (v2_reorganization_summary.md) の完全同期と全結果拡充

## 目的
`/mnt/nas/home/hiromi/src/emo2/iclr2027/tables/v2_reorganization_summary.md` に現在「Confirmatory Hypotheses Testing Summary」および「Causal Relocation LMM (H3)」しか掲載されておらず、V2 Stageの全分析結果（H1-H2幾何再編・共有度変化、H3a因果局在再配置、H3b因果特異性統制、H4出力分布回復）が欠落している問題を修正する。

## 現状の課題分析
1. `scripts/summarize_v2_reorganization.py` の `generate_markdown_summary(df_conf, df_lmm)` 関数が `df_conf` と `df_lmm` しか受け取っておらず、H1-H4の総合判定表とLMM表の2つしか出力していない。
2. 一方で対応するLaTeX版 `v2_reorganization_summary.tex` には以下の6つの表が全て統合されている：
   - 1. H1-H2 Representation Geometry and Sharing Reorganization Table (`v2_h1_h2_reorganization.tex`)
   - 2. H3a Causal Relocation Table (`v2_causal_relocation.tex`)
   - 3. H3b Causal Specificity Controls Table (`v2_causal_controls.tex`)
   - 4. H3c Linear Mixed-Effects Model LMM Table (`v2_h3_causal_lmm.tex`)
   - 5. H4 Output Distribution Recovery Table (`v2_distribution_recovery.tex`)
   - 6. H1-H4 Cross-family Hypothesis Confirmatory Summary Table (`v2_confirmatory_summary.tex`)
3. `v2_reorganization_summary.md` にも、LaTeX版と同様にV2 Stageの全6テーブル（および各セクション解説・ノート）を完全に反映し、スクリプト実行により再現可能に生成されるように改修する必要がある。

## タスク項目
- [x] 1. `scripts/summarize_v2_reorganization.py` の調査と改修設計
  - `generate_markdown_summary` の引数を拡張し、`df_conf, df_reloc, df_ctrl, df_lmm, df_recov` を渡すように変更
  - 全6セクション（H1-H2幾何・共有度、H3a因果局在、H3b因果特異性統制、H3c LMM、H4分布回復、およびH1-H4事前登録仮説総括）のMarkdownテーブル生成処理を実装
- [x] 2. `scripts/summarize_v2_reorganization.py` の実装
- [x] 3. 仮想環境下のPythonでスクリプトを実行し、`iclr2027/tables/v2_reorganization_summary.md` および関連ファイルを再生成
- [x] 4. 生成された `v2_reorganization_summary.md` の全テーブル内容、数値の整合性、TeX版との整合性を確認
- [x] 5. Section 2 (H3a) および Section 3 (H3b) が空欄（`---`）になっていた原因の調査
  - `v2/scripts/build_paper_summary.py` における `v2_causal_dissociation_summary.json` のキー参照不一致（`results` 階層の欠落）を特定
  - `build_paper_summary.py` を修正して全4ファミリーの層別因果効果・統制効果データを正常ロード
- [x] 6. `build_all_paper_summaries.py` および `summarize_v2_reorganization.py` を再実行し、全テーブルに実測値を完全反映
- [x] 7. pytest による全テストパスの確認（156 passed）
- [x] 8. `walkthrough.md` の作成と報告
