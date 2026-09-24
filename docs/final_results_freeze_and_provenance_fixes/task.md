# タスクリスト: 結果フォルダの最終凍結・確証的判定およびプロベナンス修正

## 1. 概要
ICLR 2027論文 (`iclr2027/iclr2027_conference2.tex`) における結果章の完全固定（打ち止め）に向けて、残存していたV2（3点）、V3（1点）、プロベナンス/Provenance（1点）、および細部フォーマット（3点）の計8項目を厳密に修正・検証する。

---

## 2. チェックリストと対応状況

- [x] **1. V2 H3確証的判定の修正 (Prespecified Primary Terms)**
  - 主効果 (Alignment main effect) を H3 の Primary 判定から除外し、事前登録された Primary interaction 3項 (`Alignment × Depth`, `Alignment × Task`, `Alignment × Task × Depth`) で判定。
  - いずれも $q > 0.05$ のため、`Not Supported` に修正。
  - `Alignment main effect` ($\beta=0.181, p<0.001$) は Secondary / descriptive 行として別行に配置。
  - `v2/scripts/build_paper_summary.py`, `scripts/summarize_v2_reorganization.py`, `iclr2027/tables/v2_confirmatory_summary.tex` に反映。

- [x] **2. V2 H3における FDR $q$ 値の表示**
  - raw $p$ ではなく、事前登録通り多重比較補正後の `primary_fdr_adjusted_p_values` ($q=0.878, 0.878, 0.961$) を表に出力。

- [x] **3. V2 H4確証的要約のMethods整合化**
  - Primary 指標を `Self--Reader Matched-Plain AUC Difference` とし、ブートストラップ95%信頼区間 `[-0.015, 0.078]` を明記。
  - `Matched-Plain AUC recovery` の平均値は Descriptive 行へ移行。

- [x] **4. V2 H4のファミリー網羅性（4-Family Coverage）および判定保留の明記**
  - 実データ上、GemmaとOLMoのみ実測値が存在し、LlamaとQwenは未実施（2/4 family）であることを明記。
  - 4-family cohort基準未達のため、H4確証的判定を `Supported` とせず、`Not Supported (Incomp.)`（未完）と判定。
  - `v2_distribution_recovery.tex` のNoteにも明記。

- [x] **5. V2 RQ3/RQ4のファミリー網羅性の表示 (4-Family Coverage)**
  - `v2_causal_relocation.tex` および `v2_causal_controls.tex` を OLMo 単体から 4-family（Gemma, Llama, OLMo, Qwen）表示へ拡張。
  - 実測値（Gemma, OLMo, Qwen）を掲載し、未集約の Llama は `---`（欠損）として正確に明示。

- [x] **6. V3 H3判定ロジックの厳密化 ($M_{\text{net}} > 0$)**
  - `v3/scripts/build_paper_summary.py` および `scripts/summarize_v3_causal_utilization.py` において、単なる $M > 0$ だけでなく、ランダム統制を差し引いた $M_{\text{net}} = M - M_{\text{rand}}$ の 95% CI 下限が 0 を上回るか（$\text{CI}_{\text{low}}(M_{\text{net}}) > 0$）を両軸（Valence, Arousal）で要求するよう判定コードを厳密化。

- [x] **7. V2 H2 $\Delta\text{Sharing}$ の点推定値の掲載**
  - `v2_h1_h2_reorganization.tex` および `v2_confirmatory_summary.tex` において、点推定値（Valence: $-0.082$, Arousal: $-0.055$）を追記。

- [x] **8. `results/derived/paper_summary/` の追跡・同梱（Provenance保証）**
  - `.gitignore` 内の `**/results/derived/**` 除外指定により、`results/derived/paper_summary/` が git/zip から漏れていた問題を修正。
  - `.gitignore` に `!results/derived/paper_summary/` を正しく設定し、全CSV（`table_b*.csv`, `table_v1_*.csv`, `table_v2_*.csv`, `table_v3_*.csv`）およびマニフェスト、サマリーを git/zip 追跡対象に包含。
  - `pytest -q tests/test_paper_summary_invariants.py` が 7 passed (100%) となることを検証完了。
