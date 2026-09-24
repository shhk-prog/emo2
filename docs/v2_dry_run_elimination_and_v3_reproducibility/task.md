# タスクリスト: V2 dry_run 混入根絶・全一括再生成および V3 再現性修正

## 1. 概要
ICLR 2027 論文結果フォルダの完全凍結（打ち止め）に向けて、以下の残存問題を修正・検証する：
1. V2 paper summary からの dry_run 混入の完全排除
2. paper_summary 一式の世代不整合解消・クリーン環境からの全一括再生成
3. V2 Causal Relocation 表における Base Peak 列の欠損バグ修正
4. V2 H1/H2 確証的指標の実測プロダクションデータからの再算出
5. V2 H3/H4 表生成スクリプトにおけるハードコードの完全排除
6. V3 確証的表の外部 JSON 依存撤廃および canonical CSV からの完全再生成化
7. V3 H3 表における内因性減衰量 $M$ および正味減衰量 $M_{\text{net}}$ の両行併記
8. V3 Gate Note の表記修正および Markdown サマリーの符号統一

---

## 2. チェックリストと完了状況

- [x] **1. V2 dry_run 混入の根絶**
  - `v2/scripts/build_paper_summary.py` の `find_v2_artifact` を修正し、`allow_dry_run=False` を強制。
  - `v2/results/derived/` 直下で `v2/primary/run_confirmatory_analysis.py` を実測 168,000 件データから実行し、本番の `v2_lmm_confirmatory.json` を生成。
  - `grep -R "dry_run" results/derived/paper_summary` → **0件（ZERO MATCHES FOUND）** を確認。
  - `tests/test_paper_summary_invariants.py` に `test_paper_summary_contains_no_dry_run_sources` テストを追加し、不変条件テスト 8/8 passed。

- [x] **2. paper_summary 一式の世代不整合解消・クリーン全一括再生成**
  - `rm -rf results/derived/paper_summary` を実行後、`python scripts/build_all_paper_summaries.py --strict` により全ステージを一世代で再生成。
  - `qc_summary.json` の `updated_at_utc` および各テーブル行数が実測CSVと完全に同期。

- [x] **3. V2 Causal Relocation 表の Base Peak 欠損バグ修正**
  - `scripts/summarize_v2_reorganization.py` における条件フィルタを `primary_conditions = {"base_plain_reader", "base_plain_self", "inst_matched_plain_reader", "inst_matched_plain_self"}` に修正。
  - Base Peak および Instruct Peak の実測値、$\Delta d_C^*$, $\Delta d_{\text{center}}$ がすべて表に正しく出力されることを確認（Llama は実測未集約のため `---` を保持）。

- [x] **4. V2 H1/H2 の本番実測値反映**
  - dry_run mock 値（0.352等）を排除し、本番の Procrustes distortion（Reader: 1.455, Self: 2.872）、Peak Shift、Sharing Reorganization の実数値を反映。

- [x] **5. V2 H3/H4 表生成スクリプトのハードコード排除**
  - `table_v2_3c_lmm.csv` に FDR $q$ 値列を追加。
  - `table_v2_confirmatory.csv` に H1--H4 の全 prespecified 行（推定値、CI、p、q、supported）を格納し、`summarize_v2_reorganization.py` はこの CSV から直接読み出して LaTeX 表を出力するよう改修。

- [x] **6. V3 再現性の修復（外部 JSON 依存の撤廃）**
  - `scripts/summarize_v3_causal_utilization.py` から `v3_cross_model_replication_summary.json` への依存を完全削除。
  - canonical な `table_v3_4_confirmatory.csv` および `table_v3_confirmatory_matrix.csv` のみから `v3_confirmatory_details.tex` と `v3_confirmatory_matrix.tex` を 100% 確実に再生成可能にした。

- [x] **7. V3 H3 表における判定根拠の可視化 ($M$ & $M_{\text{net}}$)**
  - `v3/scripts/build_paper_summary.py` で `mediated_M` と `net_vs_random` の2行を CSV に格納。
  - `v3_confirmatory_details.tex` で両行を表示し、Llama Arousal 等で $M$ は PASS でも $M_{\text{net}}$ が FAIL であるため総合判定が FAIL である理由を読者が一目で理解できるようにした。
  - Note にも `$\text{CI}_{\text{low}}(M) > 0$ かつ $\text{CI}_{\text{low}}(M_{\text{net}}) > 0$` の双方を要求する旨を明記。

- [x] **8. 細部表記ミスの修正**
  - V3 Gate Note: 「全軸でSufficiency / Specificity / Endogenousの事前定義thresholdを満たさなかったため...」に修正。
  - V3 Markdown Summary: `H1 (Dissociation Δd > 0)` に統一。
