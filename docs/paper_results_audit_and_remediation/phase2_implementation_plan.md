# Phase 2: 実装計画 (Implementation Plan)

## 概要
ユーザーから指摘された15項目の生成ロジック/表示上の不整合、およびLaTeXコンパイルエラーを段階的に解消する。

## 修正対象コンポーネント

1. **V2 関連スクリプト**:
   - `v2/scripts/build_paper_summary.py`:
     - `v2_distribution_recovery_summary.json` から全4ファミリーの reader / self 各タスクの実データを抽出して `table_v2_4_recovery.csv` を出力。
     - causal center-of-mass 計算の検証。
   - `scripts/summarize_v2_reorganization.py`:
     - `generate_h1_h2_table()` 内のハードコードされたデフォルト配列を削除。欠損時は例外または `---` 表示。
     - `v2_confirmatory_summary.tex` に H3, H4 行を追加。ハードコード `< 0.001` を削除し実q値または `---` に。
     - `v2_causal_controls.tex` に `Task` 列を追加し、Reader / Self の二重出力を解消。
     - `C_{net,rand} > 0` に関するNote文言を修正（点推定の方向性であり、有意性はCI/検定で判断）。
     - LMM 表のキャプションを `C_{net,rand}` を目的変数とする記述に修正。

2. **V3 関連スクリプト**:
   - `v3/scripts/build_paper_summary.py` & `scripts/summarize_v3_causal_utilization.py`:
     - H1 判定: $\Delta d^* = d_C^* - d_D^* > 0$ かつ $\Delta \bar{d} > 0$（正方向の解離）。peak_ci_low > 0 かつ center_ci_low > 0 を合格基準とする。
     - H1 の詳細表示: Peak と Center の両方を表示。
     - H3 判定: $CI_{\mathrm{low}}(M) > 0$ かつ $CI_{\mathrm{low}}(M_{\mathrm{net}}) > 0$（0.05 ではなく 0.0）。
     - H4 判定: $CI_{\mathrm{low}}(\Delta C^{\mathrm{temporal}}) > 0$（0.05 ではなく 0.0）。
     - `table_v3_3_mediated_attenuation.csv`: $T, R, M, M_{\mathrm{rand}}, M_{\mathrm{net}}$ の実数値を格納し、表へ直接出力。逆算・ハードコードを全廃。
     - V3 Note の修正: H4 Arousal 単軸合格と二軸joint confirmation 不達成の整合性。
     - `v3/scripts/build_paper_summary.py` の判定ロジックを最新の二軸・厳密仕様へ同期。

3. **テーブル再生成およびパイプライン実行**:
   - 各スクリプトを実行し、`results/derived/paper_summary/tables/*.csv` および `iclr2027/tables/*.tex` を再生成。
   - `scripts/generate_paper_results_tables.py` で `iclr2027/tables/` への配置を最新化。

4. **LaTeXコンパイル監査**:
   - `iclr2027/iclr2027_conference2.tex` を `pdflatex` / `xelatex` でコンパイルし、`Missing } inserted` の原因箇所を特定・修正。
   - エラーフリーでのビルドを確認。

5. **Walkthrough / 報告**:
   - `phase2_walkthrough.md` に修正内容・検証結果を記載。
