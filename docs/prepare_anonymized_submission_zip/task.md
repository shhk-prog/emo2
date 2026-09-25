# タスク計画: 投稿前最終修正 & 完全匿名化提出パッケージ作成

## 目的
ICLR 2027二重盲検要件・学術的整合性・再現性基準を満たすため、論文LaTeX原稿、BibTeX、V2統計分析、コード匿名化、および49件の上流artifactを完全に整備し、提出可能な状態に仕上げる。

## タスクリスト
- [x] 1. **CRITICAL: コードZIPの匿名性 & パッケージャーの自己検出バグ修正**
  - `scripts/package_submission_code.py` 自身をZIPから除外し、自己検出バグを解消
  - `iclr2027/`（実名・所属・メール等）を確実にZIPから除外
  - 禁止文字列（`hiromi`, `/mnt/nas`, `/mnt/data`, `iag-02`, 実名, 所属等）が1件も含まれないことを解凍先で自動完全保証
- [x] 2. **CRITICAL: 49件の上流 derived artifacts の同梱 & paper-summary provenance**
  - `paper_summary_manifest.json` の要求する49個のupstream artifactを確実にZIPに同梱
  - 解凍環境で `python scripts/build_all_paper_summaries.py --strict` が100%成功することを自己検証
- [x] 3. **MAJOR: V2 LMM $q=0.044$ の収束性・頑健性確認 & 感度分析**
  - `v2/results/derived/v2_lmm_confirmatory.json` の詳細監査
  - random-effects variance, convergence warnings の確認
  - cluster-robust SE / GEE / bootstrap 等による感度分析スクリプトを作成・実行し、結果を文書化
- [x] 4. **MAJOR: 統計記述の修正（Appendix N.4 の $q=0.027$）**
  - 「SecondaryのPost-training主効果も $q=0.027$ で有意」を「Secondary analysisではPost-training主効果も観測された（uncorrected $p=0.027$）。」に修正
- [x] 5. **MAJOR: 参考文献（BibTeX）の修正 & クリーンアップ**
  - `Martorell`: Nicolas Martorell 単著、タイトル `Quantitative Introspection in Language Models: Tracking Internal States Across Conversation` (arXiv:2603.18893)
  - `Reichman / Avsian / Heck`: COLM 2025 会議論文に修正、著者順整理
  - `OLMo 2`: 重複エントリの統合
  - `.bib` 内の非BibTeX生テキストの除去、欠落キー（`qwen2025qwen25technicalreport` 等）の整合
- [x] 6. **MAJOR: LaTeX source のクリーンアップ & 本文表現の精緻化**
  - 赤い内部メモタグ（`\color{red} ... \color{black}`）の完全除去
  - AbstractのV2表現精緻化（「representation geometryに差がみられ、Valenceではtask-dependentなcausal-profile differenceが限定的に支持された」）
  - 中心式を $\neq$ から $\not\Rightarrow$ に変更
  - likelihood 記述を $\tilde{P}(c \mid p)$ （length-normalized score-induced candidate distribution）として明確化
- [x] 7. **LaTeX ビルド確認 & 最終提出ディレクトリ（PDF + 匿名ZIP）の分離と最終検証**
  - LaTeX のエラーのないビルド確認
  - `results/submission/` に最終PDFと匿名コードZIPを配置
  - 最終ディレクトリに対する個人情報・絶対パス全文スキャン
