# 実装計画: V2包括サマリーMarkdown (v2_reorganization_summary.md) の完全同期と全結果拡充

## 1. 背景と改修目的
`iclr2027/tables/v2_reorganization_summary.md` は、ICLR 2027 論文における V2 Stage（事後学習に伴う表現・因果構造の再編）の包括的サマリー Markdown である。
現状、同ディレクトリの LaTeX 包括サマリー `v2_reorganization_summary.tex` には 6 つの主要テーブルがすべて統合されているのに対し、Markdown 版 `v2_reorganization_summary.md` には第1節（Confirmatory Hypotheses Testing Summary）と第2節（Causal Relocation LMM）の2つしか出力されておらず、結果が全部入っていない状態になっている。

本改修では、`scripts/summarize_v2_reorganization.py` の `generate_markdown_summary` を拡張し、LaTeX 版と完全に 1:1 で対応する全 6 つの結果テーブルおよび解説 Note を含む完全な包括サマリー Markdown を出力できるようにする。

## 2. 拡充対象テーブルとセクション構成
Markdown サマリーを以下のセクション構成に再編・拡充する：

1. **## 1. Representation Geometry and Sharing Reorganization (H1--H2)**
   - 表: Geometric Distortion (H1a), Decodability Peak Shift ($\Delta d^*$, H1b), Sharing Reorganization ($\Delta\text{Sharing}$, H2)
   - 列: Hypothesis, Metric, Estimate, 95% CI, Condition
   - Note: $\Delta d^*$ および $\Delta\text{Sharing}$ の解釈基準

2. **## 2. Causal Peak Relocation (H3a)**
   - 表: Family, Task, Axis, Base Peak ($d^*$), Instruct Peak ($d^*$), $\Delta d_C^*$, $\Delta d_{\text{center}}$
   - Note: 未評価（---）についての事前登録注記および sample-level LMM への誘導

3. **## 3. Causal Specificity Controls (H3b)**
   - 表: Family, Task, Condition, Axis, $C_{\text{raw}}$, $C_{\text{rand}}$, $C_{\text{perp}}$, $C_{\text{net,rand}}$ (Primary), Zero-Ablation
   - Note: $C_{\mathrm{net,rand}}$ の意味および事前登録注記

4. **## 4. Causal Relocation Mixed-Effects Model (H3c LMM)**
   - 表: Axis, Predictor / Parameter, Estimate ($\beta$), 95% CI, p-value, FDR $q$
   - Note: Prespecified interaction terms および Valence の Alignment $\times$ Task ($q = 0.044$) の解釈

5. **## 5. Output Distribution Recovery (H4)**
   - 表: Family, Task, Matched AUC, $\Delta\text{EMD AUC}$, Max Recovery, Best Depth ($d^*$), Native AUC, Aligned AUC
   - Note: Base 表現介入による Instruct 出力分布の接近度合いと 4 ファミリー評価

6. **## 6. Pre-registered Hypotheses Testing Summary (H1--H4 Confirmatory)**
   - 表: Hypothesis, Key Analysis Metric, Estimate, 95% CI, $p$ / FDR $q$, Criterion / Interpretation
   - Note: 各仮説の採択／棄却および記述的サマリーの総括

## 3. 実装手順
1. `scripts/summarize_v2_reorganization.py` の `generate_markdown_summary` 関数のシグネチャを `generate_markdown_summary(df_conf, df_reloc, df_ctrl, df_lmm, df_recov)` に変更。
2. 上記 6 セクションのテーブル生成ロジックを実装。数値フォーマットは `format_num` を共通利用し、LaTeX 数式（`$\Delta d^*$`, `$\beta$`, `q = 0.044` など）を Markdown 準拠で美しく表示。
3. `main()` 関数内での呼び出し箇所を更新。
4. スクリプトを実行し、`iclr2027/tables/v2_reorganization_summary.md` を更新。
5. 更新後の `v2_reorganization_summary.md` を検証。
6. `docs/fix_v2_reorganization_summary_md/walkthrough.md` を作成。

## 4. 期待される成果
- `iclr2027/tables/v2_reorganization_summary.md` に V2 の全分析結果（H1, H2, H3a, H3b, H3c, H4, Confirmatory）が網羅される。
- LaTeX 版 `v2_reorganization_summary.tex` と完全に同一の Canonical CSV データを基に動的生成され、100% 同期が保たれる。
