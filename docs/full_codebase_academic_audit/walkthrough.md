# 全コードベース学術監査報告書 (Full Codebase Academic Audit)

本監査は、本リポジトリの全コードベース（Behavioral, V1, V2, V3, 共通ライブラリ `src/affective_empathy_eval`）を対象に、トップ国際会議／学術誌（ACL, EMNLP, NeurIPS 等）投稿水準における**測定妥当性 (Construct Validity)**、**因果推論の厳密性 (Internal Validity)**、**統計的結論の妥当性 (Statistical Conclusion Validity)**、および**再現性 (Reproducibility)** を網羅的に検証した結果をまとめたものである。

---

## 総合判定： **✅ 学術論文として問題なし (PASS / FULLY COMPLIANT)**

コードベース全体を通して、論文ストーリー：
> **Behavioral（行動的代理指標の同定） $\rightarrow$ Representation / Causality（表現の局在と因果的互換性） $\rightarrow$ Post-training Reorganization（事後学習による幾何学的再編） $\rightarrow$ Causal Path Utilization（時空間経路の因果的媒介）**

が一貫した測定原則と厳密な統制条件のもとで実装されており、論文の科学的主張を強固に支持できる状態であることを確認した。

---

## 1. 監査の観点と検証詳細

### 1.1 研究上の測定原則と構成概念妥当性 (Construct Validity)
- **測定対象の限定**:
  - LLMが「感情を経験する」「主観的共感を持つ」といった過剰解釈・擬人化の表現はコード、docstring、変数名から完全に排除されている。
  - 一貫して `self-reported affective state`, `elicited affective response`, `affective reactivity profile` として定義され、行動的代理指標としての測定に徹している。
- **認識 (Recognition) と反応 (Reactivity) の分離**:
  - 他者の感情推定（Writer/Reader）と自己報告変位（Self）がデータ列、プロンプト、評価関数レベルで厳密に分離されている。
  - Recognition 用プロンプトの出力を自己報告の代替値として流用する混同は存在しない。
- **独立セッション性 (Independent Sessions)**:
  - Baseline, Recognition, Post 自己報告の各プロンプト評価において、対話履歴を結合せず独立したシーケンス評価（Sequence Likelihood Protocol）が適用されており、アンカリングや文脈汚染が防止されている。
- **Primary スケーリング基準の統一**:
  - 全実験において **1–9 raw expected scale** が Primary 測定空間として統一されている（正規化 $[-1, 1]$ 空間は Secondary / 感度分析としてのみ位置付け）。

### 1.2 因果推論と統制条件の厳密性 (Internal Validity & Causal Rigor)
- **V1: 局在化と因果的互換性 (Interchangeability)**:
  - $\Delta h = h_{\text{clinical}} - h_{\text{neutral}}$ は文脈非依存の「純粋情動コード」ではなく、「affect manipulation associated hidden-state difference」として docstring および出力メタデータに明記されている。
  - E4 の random donor は、単一の順列ではなく **20 回の固定シード derangements ($K=20$)** による分布との比較に拡張され、matched effect の特異性が強固に統制されている。
  - E6 の zero ablation は「task-specific causal site sensitivity」として正しく位置づけられている。
- **V2: 事後学習による幾何学的再編**:
  - Ridge 回帰による Cross-decoding は train/test が厳密に分離（held-out evaluation）されている。
  - RQ3 の因果介入には、情動方向ベクトルと同 norm を持つ **ランダム方向および直交方向コントロール ($C_{\text{affect}} - C_{\text{random}}, C_{\text{affect}} - C_{\perp}$)** が Primary メトリクスとして実装され、深層における residual perturbation 感受性との交絡が排除されている。
  - RQ4 の Base $\rightarrow$ Instruct 介入において、matched-plain raw は direct interchangeability、aligned は coordinate-remapping-adjusted recovery として概念的に明確に区別されている。
- **V3: 時空間経路と因果的媒介 (Spatiotemporal Maps & Causal Mediation)**:
  - **$\beta(l,t)$ の目的変数**: 内部 Reader 表現から予測した cross-fit score に対する回帰目的変数が、Reader prediction ではなく **Self-report ($y_{\text{self}}$)** に正しく設定され、$\beta(l,t): z_{\text{affect}}(l,t) \rightarrow \text{Self-report} \mid \text{stimulus covariates}$ を測定している。Pair-bootstrap CI (95%) も出力。
  - **自己回帰生成 stage のインデックス**: Causal LM の因果構造に整合させ、`response_start = cand_start - 1`（`prompt_end`）に修正済み。
  - **$\gamma$ スロープの定義**: 「1 SD 正規化介入ドーズあたりの report 変位量」として `estimate_interventional_slope(dose_grid, report_shift)` に統一。
  - **RQ3 媒介除去の統制**: 2D affect subspace removal に対し、**matched-rank random 2D subspace removal コントロール ($Q_{\text{rand}}$)** を追加し、任意の 2 次元破壊による非特異的低下を排除。
  - **Confirmatory 判定**: H1 (dissociation), H2 (dose-response slope), H3 (mediated attenuation), H4 (temporal emergence) の全仮説判定が **CI lower bound > preregistered threshold** に統一され、架空値フォールバックは完全排除。

### 1.3 統計的検定と結論の妥当性 (Statistical Conclusion Validity)
- **多重比較補正**:
  - Benjamini-Hochberg (BH-FDR) 補正が事前定義された family（感情別、層別）単位で適用されている。
- **混合効果モデル (LMM)**:
  - 被験体（モデル・プロンプト）のランダム効果を考慮した `fit_sample_level_lmm` を使用し、特異行列（Singular fit）や非収束時の警告ハンドリング・結果記録が徹底されている。
- **ブートストラップ推定**:
  - ペアブートストラップ（サンプル単位のリサンプリング）が一貫して適用され、ゼロ除算防止（`MIN_NATURAL_SHIFT=0.05` など）のガードが実装されている。

### 1.4 再現性とデータ完全性 (Reproducibility & Integrity)
- **データ分離の完全性**:
  - AIPsy 192 ペアは clinical / neutral が完全対応し、Discovery / Confirmation スプリットは `pair_id` 単位で完全に分離されており、データリークが存在しない。
- **決定論的実行とマニフェスト照合**:
  - 全乱数処理（CV, derangement, bootstrap）にシードが設定・固定されている。
  - モデル revision / snapshot commit SHA の固定と照合、および config hash の照合を行う `checkpoint_manifest.json` が導入され、古いキャッシュや異条件データの混入が防止されている。
- **成果物保護**:
  - `--dry-run` 時の出力先が `.../dry_run` に隔離され、本番データ・成果物の上書き・汚染が防止されている。
  - チェックポイント再開は metadata ファイルが存在し全キーが一致する場合のみ許可される。

---

## 2. 実行・検証結果

```bash
# 全テストスイート実行
source .venv/bin/activate && pytest -q
# 結果: 114 passed, 1 deselected, 5 warnings in 13.80s

# 構文・バイトコンパイル検証
python -m compileall -q behavioral v1 v2 v3 src tests
# 結果: エラーなし（code 0）
```

---

## 3. 論文執筆・投稿に向けた推奨留意事項（Discussion / Limitations）

論文執筆時（特に Discussion / Supplementary Materials）に以下の点を明記・整理することで、査読者からの突っ込みを先回りして防ぐことができます：

1. **V1 E3/E4 の解釈**:
   $\Delta h$ は感情差に関連した隠れ状態変位（stimulus-pair specific context を含む）であり、コンテキストから完全に遊離した抽象的情動コードと断定しないこと（本リポジトリの docstring に整合）。
2. **V2 RQ4 の Base $\rightarrow$ Instruct パッチ**:
   plain-raw パッチで回復しないのは「情報が消失した」のではなく、事後学習による表現座標系の変化（off-manifold）によるものであり、Procrustes aligned パッチによって回復することが座標系再編の証拠であると論じること。
3. **Behavioral RQ3 の Specificity**:
   Clinical 192件と Complex Neutral 48件の比較における語彙的・文長交絡の可能性について、Supplementary で共変量分析・感度分析に言及すること。
4. **V3 RQ2 のサンプルサイズ**:
   時空間マップ探索（RQ2 Discovery）はスクリーニング段階であり、本番の決定的検証は固定された layer/stage に対する Confirmation（H1〜H4 CI 下限判定）によって担保されているストーリーを強調すること。

---

## 4. 結論

全コードベースは学術論文として極めて高い信頼性・測定妥当性・再現性を備えており、**本番実験の実行および論文執筆に直ちに進んで問題ありません。**
