# 改修内容の確認 (Walkthrough): V2 dry_run 混入根絶・全一括再生成および V3 再現性修正

## 1. 実施内容と修正結果

ユーザーからの指摘事項8点について、すべての修正・検証を完了しました。

| # | 指摘項目 | 修正内容 | 検証結果 |
|---|---|---|---|
| 1 | **V2 dry_run 混入根絶** | `find_v2_artifact` で `allow_dry_run=False` を強制。実測16.8万件データから本番 `v2_lmm_confirmatory.json` を生成し、summary を再構築。不変条件テストに `test_paper_summary_contains_no_dry_run_sources` を追加。 | `grep -R "dry_run" results/derived/paper_summary` → **0件（ZERO MATCHES FOUND）**。テスト 8/8 pass。 |
| 2 | **paper_summary 世代整合化** | `rm -rf results/derived/paper_summary` を実行後、`build_all_paper_summaries.py --strict` により全ステージを一括再生成。 | `qc_summary.json` のタイムスタンプ・行数が全CSVと完全に一致。 |
| 3 | **V2 Causal Relocation Base Peak 欠損** | `summarize_v2_reorganization.py` のフィルタを `primary_conditions = {"base_plain_reader", "base_plain_self", "inst_matched_plain_reader", "inst_matched_plain_self"}` に修正。 | [v2_causal_relocation.tex](file:///mnt/nas/home/hiromi/src/emo2/iclr2027/tables/v2_causal_relocation.tex) の Base Peak に実測値が正常出力。 |
| 4 | **V2 H1/H2 本番実測値反映** | dry_run mock 値（0.352等）を排除し、本番データから Distortion（Reader: 1.455, Self: 2.872）、Peak Shift、Sharing Reorganization を反映。 | 実測値に基づく堅牢な数値に更新。 |
| 5 | **V2 表生成のハードコード排除** | `table_v2_3c_lmm.csv` に FDR $q$ 値列を追加し、`table_v2_confirmatory.csv` に全仮説を網羅。表生成スクリプトは CSV から直接読み出して LaTeX 化。 | スクリプト内のハードコードを完全撤廃。 |
| 6 | **V3 再現性の担保** | `summarize_v3_causal_utilization.py` から外部 JSON への依存を完全削除し、canonical な `table_v3_4_confirmatory.csv` と `table_v3_confirmatory_matrix.csv` のみから完全再生成可能にした。 | 外部 JSON がない環境でも 100% 正確に再生成可能。 |
| 7 | **V3 H3 判定根拠の可視化 ($M$ & $M_{\text{net}}$)** | `v3_confirmatory_details.tex` で `H3: Mediation (M)` と `H3: Net vs. Random (M_net)` の両方を表示。 | Llama Arousal 等で $M$ は PASS でも $M_{\text{net}}$ が FAIL である理由が読者に一目で分かるように整理。Note にも二重基準を明記。 |
| 8 | **細部表記ミスの修正** | V3 Gate Note の「$\text{CI}_{\text{low}} \le 0$」を「事前定義thresholdを満たさなかったため」に修正。Markdown サマリーの `H1 (Dissociation Δd < 0)` を `> 0` に統一。 | 不整合解消完了。 |

---

## 2. 動作確認結果

### 2.1 dry_run ゼロ件検証
```bash
$ grep -R "dry_run" results/derived/paper_summary || echo "ZERO MATCHES FOUND"
ZERO MATCHES FOUND
```

### 2.2 不変条件テスト (`test_paper_summary_invariants.py`)
```bash
$ .venv/bin/pytest -q tests/test_paper_summary_invariants.py
........                                                               [100%]
8 passed in 2.17s
```

### 2.3 全テストスイート
```bash
$ .venv/bin/pytest -q
154 passed, 2 deselected, 5 warnings in 25.39s
```

### 2.4 テーブル一括自動生成スクリプト
```bash
$ .venv/bin/python scripts/generate_paper_results_tables.py --repo-root . --out-dir iclr2027/tables
======================================================================
All tables successfully generated!
Location: /mnt/nas/home/hiromi/src/emo2/iclr2027/tables
======================================================================
```

---

## 3. 結論
これにより、結果フォルダは真の意味で**「完全実測・完全自律再現可能」**な状態として確定（Freeze）されました。
V2 の数値も dry_run のない正真正銘の本番実測値となり、考察の執筆に安心して使用していただけます。
