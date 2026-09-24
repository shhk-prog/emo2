# 実装計画: 論文結果表・生成ロジックの監査と修正 (Paper Results Audit & Remediation)

## 1. 概要
ICLR 2027論文 (`iclr2027/iclr2027_conference2.tex`) の結果章（§5 Behavioral, §9 V1, §13 V2, §17 V3）に関して、実データと乖離・矛盾している箇所、placeholder / fallback 値、強すぎるまたは誤った Note の解釈、不整合を監査フィードバックに基づき完全に修正する。

## 2. 修正対象ファイルと変更内容

### 2.1 LaTeX 本体 (`iclr2027/iclr2027_conference2.tex`)
- preamble に `\usepackage{booktabs,tabularx,array,multirow}` を指定し、`\multirow` の Undefined control sequence エラーを解消。
- 本文中の数式外 `\text{}`（Line 7456, Line 706）を `\textit{}` に修正。

### 2.2 V2 集計スクリプト (`v2/scripts/build_paper_summary.py` & `scripts/summarize_v2_reorganization.py`)
- `dec_peak = 0.5` を全廃し `np.nan` 化。
- Causal Relocation の数式矛盾（$\Delta d_C^* = \text{Instruct} - \text{Base}$）を解消。
- `table_v2_3c_lmm.csv` の読み込みを `TERM_MAP` による exact mapping に変更し、推定量と CI のズレを修正。
- `v2_confirmatory_summary.tex` の hard-coded 配列を撤廃し、`table_v2_confirmatory.csv` から動的生成。
- recovery の placeholder / fallback 値を排除。

### 2.3 V1 集計スクリプト (`scripts/summarize_v1_internal_sharing.py`)
- E1: 相対深度0.2以上の乖離を明記。
- E2: RSA正・direct負・Procrustes改善の実態を正確に記述。
- E3: cosine near 0 を反映し、同一回路との過大主張を回避。
- E4: Specificity が FDR 補正後に全条件で非有意（$q > 0.3$）である実態に合わせ、「因果交換可能」から「robust evidence は得られなかった」へ反転。
- E6: 全ファミリーではなく一部モデルのみ有意であることを明記し、FDR $q$ 列を追加。
- Phase B: shortcut 否定を支持しつつ、Paraphrase/Reversal はサンプル数が小さいため補助的エビデンスと位置づけ。

### 2.4 Behavioral 集計スクリプト (`scripts/summarize_behavioral_emobank.py` & `scripts/summarize_behavioral_aipsy.py`)
- EmoBank: 事後学習の因果効果とは解釈しない旨を明記。
- RQ1: OLMo Valence ($d_z = 0.32\sim0.36$) の実数を反映。
- RQ3: 特異性のモデル依存性を明記。

### 2.5 V3 集計スクリプト (`scripts/summarize_v3_causal_utilization.py`)
- Gate 表に Endogenous Relevance 列（`attenuation_ci_low`, `attenuation_pass`）を追加し 4 条件化。
- Confirmatory Details 表に Valence と Arousal の両軸を出力（計24行）。
- Confirmatory Details 表で各ファミリー固有の 95% CI を表示。
- Pass/Fail 判定を事前登録された CI criterion による厳密判定へ変更。
- Mediated Attenuation 表に $M_{\text{rand}}$ と $M_{\text{net}} = M - M_{\text{rand}}$ 列を追加。

### 2.6 テーブルファイル構成の整理 (`iclr2027/tables/`)
- canonical な 19 ファイルのみを配置し、古い duplicate / alias を `iclr2027/tables/archive/` に退避。

## 3. 検証方針
- `python scripts/generate_paper_results_tables.py --repo-root . --out-dir iclr2027/tables` を実行し、全テーブルが警告なく生成されることを確認。
- 各テーブルの数値と Note が実データと完全に整合していることを検証。
