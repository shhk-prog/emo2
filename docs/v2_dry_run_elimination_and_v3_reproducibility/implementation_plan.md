# 実装計画: V2 dry_run 混入根絶・全一括再生成および V3 再現性修正

## 1. 背景と課題
前回の検証により、`tests/test_paper_summary_invariants.py` の 7 つの不変条件は通過したものの、内部監査において以下の重大な課題が発見された：
1. `v2/results/derived/dry_run/` の仮生成データが `find_v2_artifact` のフォールバックにより `results/derived/paper_summary/` に混入していた（Primary 30行、Secondary 12行）。
2. `qc_summary.json` と実際の CSV 行数が異なる世代で混在していた。
3. `v2_causal_relocation.tex` の Base Peak がフィルタの不一致により `---` になっていた。
4. `summarize_v2_reorganization.py` に LMM $q$ 値や H4 CI のハードコードが残存していた。
5. `summarize_v3_causal_utilization.py` が canonical CSV ではなく外部 JSON に依存していたため、zip 展開環境で表が `---` に化けていた。
6. V3 H3 において $M$ と $M_{\text{net}}$ の判定根拠が表に分かれていなかった。

---

## 2. 改修方針と手順

### 2.1 V2 本番アーティファクトの生成と dry_run の完全遮断
- `v2/primary/run_confirmatory_analysis.py --raw-dir v2/results/raw --derived-dir v2/results/derived` を実行し、実測 168,000 件のデータから本番の `v2_lmm_confirmatory.json` を生成。
- `v2/scripts/build_paper_summary.py` の `find_v2_artifact` で `allow_dry_run=False` をデフォルト化し、本番集約での dry_run 探索を完全遮断。
- `tests/test_paper_summary_invariants.py` に `test_paper_summary_contains_no_dry_run_sources` を追加。

### 2.2 paper_summary のクリーン全一括再生成
- `rm -rf results/derived/paper_summary`
- `python scripts/build_all_paper_summaries.py --strict`
- 一世代のタイムスタンプと行数で全 CSV, manifest, QC を整合。

### 2.3 V2 表生成スクリプトの改修
- `summarize_v2_reorganization.py`:
  - `primary_conditions = {"base_plain_reader", "base_plain_self", "inst_matched_plain_reader", "inst_matched_plain_self"}` により Base と Instruct を正確にマッチング。
  - `table_v2_confirmatory.csv` から全数値を直接読み出して `v2_confirmatory_summary.tex` を構築し、ハードコードを完全排除。

### 2.4 V3 再現性の担保と判定根拠の明確化
- `v3/scripts/build_paper_summary.py`:
  - `table_v3_4_confirmatory.csv` に `metric` 列を追加し、H1 (peak, center) および H3 ($M$, $M_{\text{net}}$) を独立行として出力。
- `summarize_v3_causal_utilization.py`:
  - 外部 JSON (`v3_cross_model_replication_summary.json`) への依存を完全撤廃。
  - `df_conf` と `df_conf_matrix` のみから `v3_confirmatory_details.tex` と `v3_confirmatory_matrix.tex` を生成。
  - Gate Note および Markdown サマリーの表記ミスを修正。

---

## 3. 検証項目
1. `grep -R "dry_run" results/derived/paper_summary` → 0件確認
2. `pytest -q tests/test_paper_summary_invariants.py` → 8 passed 確認
3. `pytest -q` → 154 passed 確認
4. `generate_paper_results_tables.py` による全テーブル再生成と TeX ファイル差分検証
