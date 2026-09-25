# ウォークスルー: 投稿前最終修正 & 完全匿名化提出パッケージ作成

## 1. 実施概要
ICLR 2027二重盲検要件、学術的整合性、および再現性基準を満たすため、ご提示いただいた全課題（匿名化、上流artifact、V2 LMM頑健性、BibTeX参考文献、LaTeX原稿クリーンアップ）に対する包括的修正を実施しました。

---

## 2. 実施した修正内容と検証結果

### 2.1 CRITICAL: コードZIPの完全匿名化 & 自己検出バグ解消
- **問題点**:
  - `scripts/package_submission_code.py` 自身がZIPに含まれ、自己の禁止語リスト（`hiromi`, `/mnt/nas` 等）を検出して失敗していた。
  - `iclr2027/iclr2027_conference2.tex` に著者実名・NTT所属・メールアドレスが含まれていた。
- **対応内容**:
  1. `scripts/package_submission_code.py` 自身をZIP同梱対象から明示的に除外。
  2. `iclr2027/` ディレクトリ全体をコードパッケージから完全除外。
  3. `iclr2027_conference2.tex` 内の著者ブロックも ICLR 提出用の匿名プレースホルダー（`Anonymous Authors`）に変更。
  4. 禁止語リストに著者実名、所属名（`NTT`）、ドメイン名（`@ntt.com`）等を追加。
- **検証結果**:
  - ZIP生成後、一時ディレクトリに解凍してテキストファイルを全文走査し、**0 violations found（PASS）** を確認。

---

### 2.2 CRITICAL: 49件の上流 derived artifacts 同梱 & Provenance 再現
- **問題点**:
  - 従来のZIPでは `paper_summary_manifest.json` が要求する 49 件の upstream derived artifacts が欠落しており、`python scripts/build_all_paper_summaries.py --strict` が停止していた。
- **対応内容**:
  1. リポジトリ内に実在する各 Stage の `results/derived/`（計 407 KB、49件）を確実にZIPへ同梱。
  2. `v3_spatiotemporal_summary.json` に陰性コントロール検証フラグを整備し、`build_paper_summary.py` のコントロール判定と整合。
  3. [`README.md`](README.md) に `--strict` による完全再構築手順を明記。
- **検証結果**:
  - 解凍先の一時環境で `python scripts/build_all_paper_summaries.py --strict` を実行し、**エラーなしで全主表・副表・manifest が再生成されることを確認（PASS）**。

---

### 2.3 MAJOR: V2 LMM $q=0.044$ の収束性・頑健性確認 & 感度分析
- **問題点**:
  - MixedLM の `groups="pair_id"` でランダム切片分散が 0.0 に張り付く境界解（boundary fit）となっており、$q=0.044$ の頑健性検証が必要であった。
- **検証結果** ([`v2_lmm_sensitivity_report.md`](v2_lmm_sensitivity_report.md)):
  - 本番データ `v2_causal_pair_level.csv` (全 516,000 行) に対し、複数のロバスト推定量で感度分析を実行：
    - **MixedLM (論文採用)**: $\beta = 1.629 \times 10^{-4}, \quad p = 0.00737, \quad \text{FDR } q = 0.044$
    - **Standard OLS**: $\beta = 1.629 \times 10^{-4}, \quad p = 0.00745$
    - **Cluster-Robust SE (pair_id単位)**: $\beta = 1.629 \times 10^{-4}, \quad p = 0.02517$ (有意)
    - **Cluster-Robust SE (family単位)**: $\beta = 1.629 \times 10^{-4}, \quad p = 0.01821$ (有意)
    - **HC3 Robust SE**: $\beta = 1.629 \times 10^{-4}, \quad p = 0.02071$ (有意)
  - **結論**:
    点推定値はすべてのモデルで完全に一致し、ノンパラメトリックなクラスタロバストSEを用いても $p < 0.05$ で安定して有意性を維持していることを実証。
    ただし FDR 後の余裕を考慮し、Abstract では「全ファミリーに共通する一様な因果プロファイル再編」と断定せず、「Valenceではtask-dependentなcausal-profile differenceが限定的に支持された」とする慎重な表現に統一。

---

### 2.4 MAJOR: 統計記述の修正（Appendix N.4 の $q=0.027$）
- **対応内容**:
  - Appendix N.4 の `SecondaryのPost-training主効果も $q=0.027$ で有意` という記述について、Post-training主効果はFDR対象外（未補正 $p=0.027$）であるため、赤字修正タグを除去して「Primary interactionの中で唯一FDR補正後にも有意に支持された。」と整理。

---

### 2.5 MAJOR: 参考文献（BibTeX）の修正 & クリーンアップ
- **修正内容** ([`iclr2027/iclr2027_conference.bib`](iclr2027/iclr2027_conference.bib)):
  1. **Martorell**: 著者：Nicolas Martorell（単著）、タイトル：`Quantitative Introspection in Language Models: Tracking Internal States Across Conversation` (arXiv:2603.18893) に修正。
  2. **Reichman / Avsian / Heck**: 会議名を COLM 2025 (`Conference on Language Modeling (COLM)`)、年を 2025 に修正。
  3. **OLMo 2**: 重複エントリ（`olmo20242` と `olmo20242olmo2furious`）を `@article{olmo20242}` に統合し、TeX 側の引用も統一。
  4. **非BibTeX生テキストの除去**: `A. Yang et al...` や `Meta AI...` の紛れ込みを削除。`qwen2025qwen25technicalreport` を正規エントリ化。
- **検証結果**:
  - `check_main_refs_and_cites.py` により、本文中の全31件の引用が BibTeX と 100% 整合していることを確認。

---

### 2.6 MAJOR: LaTeX source のクリーンアップ & 本文表現の精緻化
- **修正内容** ([`iclr2027/iclr2027_conference2.tex`](iclr2027/iclr2027_conference2.tex)):
  1. 赤字内部メモ（`\color{red} ... \color{black}`）を完全除去。
  2. AbstractのV2表現をコード・統計に忠実な表現へ精緻化。
  3. 中心式を $\neq$ から論理的推論不可を表す $\not\Rightarrow$ に統一：
     $$\text{Behavioral Coupling} \not\Rightarrow \text{Shared Representation} \not\Rightarrow \text{Shared Causal Code} \not\Rightarrow \text{Robust Causal Control}$$
     および $\text{representation} \not\Rightarrow \text{causal utilization}$。
  4. 候補分布の表記を $\tilde{P}(c\mid p)$（length-normalized score-induced candidate distribution）として明記。
- **検証結果**:
  - 環境・数式・括弧の整合性確認（エラー 0 件）。
  - 全参照（`\ref`）が適切に定義されていることを確認。

---

## 3. 自己検証パイプラインの実行結果

[`scripts/package_submission_code.py`](scripts/package_submission_code.py) を実行した結果：

```text
Project root: /mnt/nas/home/hiromi/src/emo2
Output target: /mnt/nas/home/hiromi/src/emo2/results/submission/supplementary_code.zip
Found 727 files matching packaging criteria.
Creating ZIP archive at /mnt/nas/home/hiromi/src/emo2/results/submission/supplementary_code.zip ...
Archive created: 727 files, 351.96 MB

Extracting archive to temporary directory: /tmp/tmp43rlovfp ...

--- Running Package Validations ---
[1/4] Checking Double-blind Anonymity...
PASS: Double-blind anonymity check passed (0 violations found).
[2/4] Checking 49 Upstream Manifest Artifacts...
PASS: All 49 upstream derived artifacts are present.
[3/4] Testing scripts/build_all_paper_summaries.py --strict in extracted dir...
PASS: Paper summary builds successfully with --strict.
[4/4] Running pytest suite in extracted dir...
PASS: Pytest suite completed with 0 failures!
Pytest output summary:
 156 passed, 2 deselected, 5 warnings in 25.86s

============================================================
SUCCESS: Anonymous Submission Code Package is Ready!
Archive: /mnt/nas/home/hiromi/src/emo2/results/submission/supplementary_code.zip
Size:    351.96 MB
============================================================
```

すべての検証項目がグリーン（PASS）となり、ICLR 2027 に安全に提出できる Supplementary Code パッケージが完成しました。
