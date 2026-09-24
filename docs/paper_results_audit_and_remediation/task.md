# タスクリスト: 論文結果表・生成ロジックの監査と修正 (Paper Results Audit & Remediation)

## 状況・目的
ICLR 2027論文 (`iclr2027/iclr2027_conference2.tex`) の結果章（§5 Behavioral, §9 V1, §13 V2, §17 V3）に関わる20項目の監査指摘事項に基づき、placeholder/fallback値の完全排除、数式矛盾の解消、Noteの学術的厳密化、V3 Gate/Confirmatoryの4条件化・両軸化、LaTeXコンパイルエラーの修正を実施する。

## タスク進捗

- [x] **1. LaTeX 環境・コンパイル環境の修正**
  - [x] `iclr2027/iclr2027_conference2.tex` の preamble に `\usepackage{multirow}` を追加
  - [x] 数式外の `\text{}`（Line 7456, Line 706 等）を `\textit{}` に修正

- [x] **2. V2 の placeholder / fallback 値の完全排除**
  - [x] `v2/scripts/build_paper_summary.py` 内の `dec_peak = 0.5` を `np.nan` に修正
  - [x] recovery のデフォルト値（0.72, 0.15 等）を `np.nan` 化

- [x] **3. V2 数式矛盾の解消（Causal Relocation）**
  - [x] `\Delta d_C^* = \text{Instruct} - \text{Base}` ($1.0 - 1.0 = 0.0$) として正しく計算
  - [x] decodability との乖離は `causal_decodability_gap` として分離

- [x] **4. V2 LMM の exact mapping 化**
  - [x] `scripts/summarize_v2_reorganization.py` に `TERM_MAP` を導入し、部分一致による不整合（Intercept = 0.625, CI=[-0.472, -0.416]）を解消（Intercept = -0.444, CI=[-0.472, -0.416] に修正）

- [x] **5. V2 Confirmatory Summary の実データ動的生成**
  - [x] `conf_items = [...]` の hard-coded 配列を完全撤廃
  - [x] `table_v2_confirmatory.csv` から動的に行をパース・生成

- [x] **6. V1 表 Note の実数値準拠・学術的厳密化**
  - [x] E1: 0.2以上の乖離が存在することを明記
  - [x] E2: RSA正・direct負・Procrustes改善の実態を正確に記述（「同一表現幾何」の断定を回避）
  - [x] E3: cosine near 0 の条件を反映し、同一回路との過大主張を回避
  - [x] E4: Specificity が FDR 補正後に全条件で非有意（$q > 0.3$, CI crosses zero）である実態に合わせ、「因果交換可能」から「robust evidence は得られなかった」へ反転
  - [x] E6: 一部モデルのみ有意であることを明記し、表に FDR $q$ 列を追加
  - [x] Phase B: shortcut 否定を支持しつつ、Paraphrase/Reversal はサンプル数が小さいため補助的エビデンスと位置づけ

- [x] **7. Behavioral 表 Note の修正**
  - [x] EmoBank: Base/Instruct間でprompt formatが異なるため事後学習の因果効果とは解釈しない旨を明記
  - [x] AIPsy RQ1: OLMo Valence ($d_z = 0.32\sim0.36$) の実数を反映
  - [x] AIPsy RQ3: 特異性のモデル依存性を明記

- [x] **8. V3 Gate 表・Confirmatory・Mediated Attenuation の修正**
  - [x] Gate 表に Endogenous Relevance 列（`attenuation_ci_low`, `attenuation_pass`）を追加し 4 条件化
  - [x] Gate 表の Note に事前登録閾値（0.10, 0.05, 0.05, 0.15）を明記
  - [x] Confirmatory Details 表に Valence と Arousal の両軸を出力（計24行）
  - [x] Confirmatory Details 表で各ファミリー固有の 95% CI を表示
  - [x] Pass/Fail 判定を point estimate から事前登録された CI criterion による厳密判定へ変更
  - [x] Mediated Attenuation 表に $M_{\text{rand}}$ と $M_{\text{net}} = M - M_{\text{rand}}$ 列を追加

- [x] **9. 重複・旧テーブルのアーカイブ退避**
  - [x] `iclr2027/tables/archive/` を作成し、旧 alias や不要ファイルを移動
  - [x] canonical な 19 ファイルのみを配置

- [x] **10. テーブル再生成と総合検証**
  - [x] `python scripts/generate_paper_results_tables.py` の一括実行確認
  - [x] 各表の出力内容と Note の整合性を確認
