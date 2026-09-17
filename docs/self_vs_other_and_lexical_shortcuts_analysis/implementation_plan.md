# 実装計画: V1 (Self vs. Other) — 内部表現・因果経路の共有検証と意味的妥当性の厳密化

## 概要と全体研究ロードマップ

EmoBank（人間アノテーション付きVAD）および AIPsy-Affect 4-Split（キーワードフリー臨床ヴィネット最小対）の行動実験（Behavioral Stage）により、以下の事実が実証されました：
1. **強い行動的連動 (Behavioral Coupling)**: Reader Prediction（他者認識）と Self-Report（自己報告）の変化量には $r(\Delta V_R, \Delta V_S) \approx 0.80 \sim 0.97$ という極めて高い相関が存在する。
2. **モデルファミリー依存の事後学習変容**: 事後学習（Base $\rightarrow$ Instruct）による変容はモデルファミリー依存であり、特にMistralやGemmaではArousal応答の増幅や感情空間の劇的な再編が顕著に確認された。

これを受け、研究を「何が起きているか（行動観察）」から**「出力が似ているからといって、内部でも同じ表現・同じ因果経路を使っているのか？」**という内部機構の解明へと進めます。

---

### 全体研究ロードマップ（EmoBank / AIPsy $\rightarrow$ V1 $\rightarrow$ V2 $\rightarrow$ V3）における役割分担

先行実験と将来フェーズの間で実験・問いが重複することを完全に排し、各フェーズの中心比較軸と科学的問いを以下のように一貫して固定します：

| Stage | 中心比較軸 | 科学的問い (Core Question) | 主な手法 |
|:---|:---|:---|:---|
| **Behavioral Stage**<br>(EmoBank / AIPsy) | **Behavior (行動)** | **何が起きているのか？**<br>- 人間VADとの一致度<br>- 刺激特異性・用量反応・行動Coupling | 期待値尤度測定 (Continuous Expected VAD)<br>3大感情クラスタ分析 |
| **V1 (本計画)** | **Reader vs. Self**<br>*(Who shares what?)* | **高い行動連動は、共有された表現・意味・因果実装によって支えられているのか？**<br>- 行動的Couplingは内部表現の共有を意味するか？<br>- その表現は語彙ショートカットではなく文脈意味か？<br>- 共有情報は同じ因果回路で利用されるか？ | **7ステップ（Phase A $\rightarrow$ B $\rightarrow$ C）**<br>- E1: Layer-wise Probe (Regression / Classification)<br>- E2: **Reader↔Self Cross-Decoding + Alignment**<br>- E5: **Lexical Audit + 4大Controls (Judge検証付)**<br>- E3: **Causal Map (Magnitude + Direction Vector)**<br>- E4: **Matched Difference Patching (αスイープ付)**<br>- E6: **Mixed-effects Interaction Model** |
| **V2 (次期フェーズ)** | **Base vs. Instruct**<br>*(What does post-training change?)* | **Post-trainingで何が変わるのか？**<br>- 感情表現そのものが消去されたのか (Erasure)？<br>- 幾何変換か (Transformation)？<br>- 一様抑制か (Suppression)？<br>- 分散再配置か (Distributed Remapping)？ | - **Base↔Instruct Cross-Decoding**<br>- Procrustes / Ridge alignment<br>- Component Patching<br>- Late Residual injection<br>- Output-head / Norm swap |
| **V3 (最終フェーズ)** | **Decodability vs. Causality**<br>*(Does readable mean causally used?)* | **読める場所＝使う場所なのか？**<br>- 情報をdecodeできる場所と、出力を因果的に動かす場所は一致するか？ | - Site-wise probe vs. causal map<br>- Probe-direction ablation<br>- Temporal / layer localization |

> [!IMPORTANT]
> **本研究計画が検証する核心的仮説（中立的問い）**:
> $$\boxed{\text{Does Similar Behavior imply Shared Representation and Shared Causal Implementation?}}$$
> （行動的類似性は、内部表現の同一性や因果回路の共有を必然的に意味するのか？）

---

## ユーザー確認事項 (User Review Required)

> [!NOTE]
> - **Phase A（E1: Decodability & E2: Geometry）および Phase B（E5のLexical Confound Audit）** は、すでに抽出・保存済みの隠れ状態テンソルおよびテキストコーパスを活用して実行可能なため、**GPUの新規大量計算を行わずに即時実行可能**です。
> - **Phase C（E3: Causal Map, E4: Cross-Task Patching, E6: Targeted Ablation）** については、介入推論を伴うため、AGENTS.md規約に基づきモデル規模（1B〜7B）やバッチ数を絞り、**事前に計算規模とGPU使用の承認**を得た上で進めます。

---

## 1. V1 の論理展開：7ステップの階段構造

本研究（V1）の主RQは以下のように定式化されます：

$$\boxed{RQ:\ \text{Reader PredictionとSelf-Reportの高い行動的Couplingは、共有されたaffective representation、共有された文脈意味表現、そして共有された因果実装によって支えられているのか？}}$$

この問いに対し、査読者の懸念（「単なる相関では？」「語彙共起では？」「プローブが拾っただけでは？」）を一段ずつ反証しながら進む以下の7ステップ（Phase A $\rightarrow$ B $\rightarrow$ C）を厳密に構築します：

```text
Step 1: 行動的結合の確認 (Behavioral Coupling)
  ● AIPsy-Affect において Reader Prediction と Self-Report が強く連動 (r = .80 - .97)
                                  │
                                  ▼
【Phase A: Behavior → Representation (存在と形式)】
Step 2: E1 Shared Decodability (存在)
  ● 共通の外部感情情報が、Reader/Self の双方から同じ領域でデコード可能か？
Step 3: E2 Shared Geometry (形式)
  ● その情報は同じ座標系・同じ幾何形式（readout / alignment）で保持されているか？
                                  │
                                  ▼
【Phase B: Representation の意味的妥当性 (Semantic Validity)】
Step 4: E5 Semantic vs. Lexical (意味)
  ● 観測された表現は、単純な語彙共起だけでは説明しにくく、文脈意味に追従するか？
  ● Lexical Confound Audit ＋ 独立Judge検証付き4段階統制 (Minimal pair, Outcome reversal, Paraphrase, Shuffle)
                                  │
                                  ▼
【Phase C: Causality (因果回路の検証)】
Step 5: E3 Shared Causal Map (場所)
  ● 同じsiteが、同じ強さ・同じVA方向に出力を動かしているか？ (Magnitude + 2D Direction Vector)
Step 6: E4 Causal Interchangeability (交換可能性)
  ● Reader/Self 間でペア単位の感情差分ベクトルを移植できるか？ (Matched Difference Patching & αスイープ)
Step 7: E6 Double Dissociation (特異化)
  ● task-specific な因果的特異化 (Causal Specialization) があるか？ (Mixed-effects Interaction Model)
```

---

## 2. 詳細な実験設計 (E1〜E6)

### Phase A: 表現の存在と幾何構造

#### E1: Shared Decodability（層別復元能の比較）
- **問い**: ReaderとSelfは、モデル内部の同じ層・同じコンポーネントから共通の感情情報をデコードできるか？
- **目的変数（Target Labels）の二層固定とモデル選択**:
  - **Primary Target（共通外部感情指標 $Y_{\text{stimulus}}$）**:
    - **EmoBank (連続値回帰)**: 人間評価値（$V_{\text{human}}, A_{\text{human}}$） $\rightarrow$ **Ridge Regression ($R^2$)**
    - **AIPsy-Affect (カテゴリ分類)**: 3大感情クラスタ（Negative/Positive/Alert）、8感情カテゴリ、Clinical vs. Neutral $\rightarrow$ **Multinomial / Binary Logistic Probe (ROC-AUC, Balanced Accuracy)**
    - **AIPsy-Affect (刺激強度)**: None $\rightarrow$ Moderate $\rightarrow$ Clinical $\rightarrow$ **Ordinal Logistic Regression / Spearman $\rho$**
  - **Secondary Target（各タスク自身の行動出力）**: モデル自身の生成期待値（$E[V_R], E[A_R]$ および $E[V_S], E[A_S]$）
- **手法**:
  - 全層（Layer $0 \dots L$）の隠れ状態から Primary Target を予測するプローブを学習し、層別デコーダビリティ曲線を算出。
  - Attention / MLP / Residual stream それぞれの寄与度を分解。
- **解釈の厳密化**:
  - 「同じ層で高いデコード能 $\rightarrow$ 同じ表現」と飛躍せず、**「Reader / Self の双方で affective information が同じ領域から decodable である」** という記述的同定に留める（$\text{Decodability} \neq \text{Causal Implementation}$ の徹底）。

#### E2: Shared Geometry（幾何構造の共有と整列可能性：3段階検証）
- **問い**: ピーク層が同じだとして、両者は**同じ線形読み出しが可能か、あるいは異なる座標系だが変換可能（Alignable）なのか**？
- **比較対象の厳密な限定**: 本実験は **Reader $\leftrightarrow$ Self 間のみ** を対象とする（V2の Base $\leftrightarrow$ Instruct と比較軸を完全に分離）。
- **手法（3段階アプローチ）**:
  1. **Direct Cross-Decoding (直接線形転移能)**:
     - Readerプローブ $W_R$ をSelfの隠れ状態に適用: $W_R(H_S)$
     - Selfプローブ $W_S$ をReaderの隠れ状態に適用: $W_S(H_R)$
  2. **RSA / CKA (幾何空間のトポロジー相関)**:
     - 刺激間非類似度行列（RDM）をReaderとSelfで計算し、層ごとの相関を評価。
  3. **Procrustes / Ridge Alignment (座標整列可能性)**:
     - 直交Procrustes変換 $Q$ または正則化アライメント行列 $A$ を学習し、座標回転・伸縮によって一致するかを検証。
- **3つの幾何パターンの判定**:
  | パターン | 指標 | 科学的解釈 |
  |:---|:---|:---|
  | **Shared Geometry** | Direct Transfer High | **共通の座標系・線形読み出しを利用** |
  | **Alignable Geometry** | Direct Low, Alignment High | **異なる座標系だが線形変換可能な幾何形式** |
  | **Directly Non-transferable** | Both Low | **直接の線形転移・整列が困難な表現形式**（非線形写像の余地を残す） |

---

### Phase B: 意味的妥当性の検証（語彙ショートカットの排除）

#### E5: Semantic vs. Lexical（語彙交絡監査と独立Judge検証付き4段階統制）
- **問い**: 観測された発火場所・表現は、単純な語彙共起（Bag-of-Words / Lexical Shortcuts）だけで説明できるか？
- **手法**:
  1. **Lexical Confound Audit (事前の語彙交絡監査)**:
     - 「AIPsy-Affect は minimal pair である」とアプリオリに断定せず、各ペアについて定量評価を実施：
       - Token Jaccard similarity, Levenshtein edit distance
       - Sentence embedding similarity (Sentence-BERT)
       - Word count difference, Sentiment lexicon difference (VADER / SentiWordNet)
       - Emotion-word count, Perplexity difference
     - 語彙交絡因子を共変量として統制した偏回帰・線形混合効果モデルを適用。
  2. **独立Judge / 品質検証プロトコルを伴う4段階統制**:
     - **(a) Lexically matched minimal pairs**: 語彙監査を通過した統制対での基本効果。
     - **(b) Compositional / Outcome Reversal (文脈・結末反転)**:
       - 単純な "not" だけでなく、同一主要語彙を残しながらイベント結果を反転させた文対（例: *"He expected to lose and lost."* vs. *"He expected to lose, but unexpectedly won."*）。
       - **品質検証基準**: Lexical overlap $\ge 0.80$, Intended polarity reversal, Grammaticality を独立チェッカーで検証。
     - **(c) Paraphrase Invariance (意味保持・語彙変更)**:
       - 異なる単語を用いて同一の情動事態を記述。
       - **品質検証基準**: Semantic equivalence, Affective equivalence, Fluency を独立チェッカーで検証。
     - **(d) Word Shuffle (語彙保持・意味破壊のストレス解消テスト)**:
       - 語彙集合を100%保持したまま語順をランダム化し意味を破壊。プローブ精度・発火が崩壊することを確認。
- **解釈の厳密化（査読耐性）**:
  - 「人間と同じ抽象的感情概念の立証」といった過剰な表現を排し、**「観測されたaffective representationが、単純な語彙共起だけでは説明しにくく、文脈意味に追従することを検証する（representation is not readily explained by simple lexical shortcuts）」** と堅固に定式化。

---

### Phase C: 因果回路の検証

#### E3: Shared Causal Map（強さと2次元方向ベクトルを統合した因果部位オーバーラップ）
- **問い**: 感情情報が存在する場所ではなく、**実際に出力を動かす因果的サイト（Causal Sites）** は同じか？
- **強さと2次元方向ベクトルの同時測定**:
  1. **Magnitude Map (強さの分布変化)**:
     - $C^{\text{mag}}(l, c) = W_1(P^{\text{patch}}, P^{\text{control}})$（Wasserstein距離）または JSD
  2. **Directional Vector Map (2次元VA方向の変化)**:
     - $C^{\text{dir}}_{VA}(l, c) = (\Delta E[V], \Delta E[A])$
     - ReaderとSelfの間で2次元方向ベクトルのコサイン類似度 $\cos(C^{\text{dir}}_{R}, C^{\text{dir}}_{S})$ を算出。
- **3大オーバーラップ指標**:
  1. **Spatial Overlap**: 上位 $k$ 個の因果サイトの重複率（Jaccard 係数 $J(S_R, S_S)$）
  2. **Rank Similarity**: 全因果サイトにおける Spearman 順位相関 $\rho_{\text{Spearman}}(C_R, C_S)$
  3. **Peak Displacement**: Reader の因果ピーク層と Self の因果ピーク層の距離差（$|l^*_R - l^*_S|$）
- **評価の核心**:
  $$\boxed{\text{Where? (場所)} \quad + \quad \text{How strongly? (強度)} \quad + \quad \text{In which direction? (方向ベクトル)}}$$
  「同じ場所が強い」だけでなく、「同じ場所をpatchするとReader/Selfが同じVA方向に出力を動かすか」まで評価。

#### E4: Causal Interchangeability（ペア単位差分パッチングと用量反応スイープ）
- **問い**: Reader実行時の活性化は、Selfの出力生成においてそのまま機能的に利用可能か？
- **手法**:
  1. **Full-state Cross-Task Patching**: $h_S^{(l)} \leftarrow h_R^{(l)}$
  2. **Matched Difference-vector Patching (ペア単位の感情差分ベクトル注入)**:
     - 刺激固有の語彙・状況差を最小化するため、**同一刺激ペア単位** で差分を計算：
       $$\Delta h_{R, i} = h_{R, i}^{\text{aff}} - h_{R, i}^{\text{matched-neutral}}$$
       $$h_{S, i}^{\text{neutral}} \leftarrow h_{S, i}^{\text{neutral}} + \alpha \Delta h_{R, i}$$
     - 注入係数 $\alpha$ をスイープ（$\alpha \in \{-1.0, 0.0, 0.5, 1.0, 1.5\}$）し、**用量依存的（Dose-dependent）な出力シフト** を検証。
- **必須の4大コントロールと特異性指標**:
  - ① Matched same-stimulus、② Random-stimulus、③ Same-task Reader、④ Same-task Self
  - 特異性: $\text{Specificity} = \text{Effect}_{\text{matched}} - \text{Effect}_{\text{random}}$
- **解釈の安全性**:
  - パッチ不成立時も「完全に別系統」と断定せず、**Task-context compatibility（プロンプト文脈の違いによる off-manifold 化）の可能性**に配慮し、「直接的な cross-task interchangeability は確認されない」と慎重に定式化。

#### E6: Double Dissociation（線形混合効果モデルによる統計的交互作用検定）
- **問い**: ReaderとSelfに特化した因果的特異化（Causal Specialization）が存在するか？
- **手法**:
  - E3/E4で同定された Reader 特異的サイト $S_R$ および Self 特異的サイト $S_S$ を標的として消去（Targeted Ablation）。
  - **線形混合効果モデル（Linear Mixed-Effects Model: LMM）を Primary 検定とする**:
    $$\text{Outcome} \sim \text{Task} \times \text{SiteType} + (1 \mid \text{pair})$$
    - 固定効果: $\text{Task (Reader vs Self)} \times \text{SiteType (Reader-site vs Self-site)}$
    - ランダム効果: 刺激ペアごとの切片 $(1 \mid \text{pair})$
  - 補助検定として 2-way ANOVA を併記。
- **判定基準**:
  - LMMにおける交互作用項が有意（$p < .01$）であることを示し、「完全独立回路」ではなく**因果的特異化／部分解離（Causal Specialization / Partial Dissociation）**を主張する。

---

## 3. 実装スクリプト構成

1. **`v1/scripts/run_v1_phase_a_probe.py`** [NEW]:
   - E1 (Ridge for EmoBank V/A, Logistic for AIPsy categories, Ordinal for intensity) & E2 (Direct Cross-Decoding, RSA/CKA, Procrustes Alignment for Reader↔Self) を実行。
2. **`v1/scripts/run_v1_phase_b_semantic_audit.py`** [NEW]:
   - E5 (Lexical Confound Audit: Jaccard, Edit distance, S-BERT, VADER, PPL) および独立Judge検証付きの文脈反転・言い換え対の生成・検証。
3. **`v1/scripts/run_v1_phase_c_causal_patching.py`** [NEW]:
   - E3 (Magnitude & 2D Directional Vector Causal Map) & E4 (Matched Difference-vector Patching with $\alpha$-sweep & 4 Controls) を実行。
4. **`v1/scripts/run_v1_phase_c_targeted_ablation.py`** [NEW]:
   - E6 (Targeted Ablation & Linear Mixed-Effects Model Interaction 検定) を実行。
