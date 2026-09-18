# LLM情動反応性・内部表現評価実験 v1 (Affective Reactivity & Representation Evaluation)

本リポジトリ (`v1`) は、大規模言語モデル（LLM）における情動反応性（Affective Reactivity）と感情認識（Affective Recognition）のメカニズムを、**行動評価（Behavioral Evaluation）** から **内部表現の幾何構造（Representational Geometry）**、および **因果回路の共有検証（Causal Sharing）** に至る多角的なアプローチで解明するための実験・分析環境です。

---

## 1. 研究の核心的リサーチクエスチョン

EmoBank（人間評価VADデータ）および AIPsy-Affect 4-Split（臨床ヴィネット最小対）を用いた行動実験により、他者予測（Reader Prediction）と自己報告（Self-Report）の間に極めて強い行動的連動が存在することが明らかになりました。

これを受け、本研究では以下の核心的仮説（中立的問い）を段階的に検証します：

$$\boxed{\text{Does Similar Behavior imply Shared Representation and Shared Causal Implementation?}}$$
（他者認識と自己報告の高い行動連動は、共有された感情表現、文脈意味表現、そして共有された因果実装によって支えられているのか？）

---

## 2. 前提知見：行動実験ステージにおける連動 (Behavioral Coupling: previous run)

> [!NOTE]
> 行動評価の詳細プロトコル、最新の実行手順および詳細な統計表については [behavioral/README.md](file:///mnt/nas/home/hiromi/src/emo2/behavioral/README.md) を参照してください。本節は内部表現解析（V1 Stage）の動機づけとなる前提結果（previous run）の要約です。

AIPsy-Affect 4-Split および EmoBank に対する 729候補 Sequence-Likelihood 評価（previous run）により、以下の基礎的知見が確認されています：

- **Sensitivity (感情刺激とNeutralの峻別)**:
  - すべてのモデルファミリーにおいて、中立文と比較して有意な Valence 低下 ($V-$) を確認（例: Mistral Instruct previous run で $d = -2.11$）。
- **Dose-Response (用量反応性: Neutral → Moderate → Clinical)**:
  - 感情強度の増加に伴う段階的変位を確認（Mistral previous run で単調性成立率 60.4% 〜 66.7%）。Reader と Self は同等の単調性傾向を示す。
- **Specificity (文章複雑性統制: Complex Neutral vs. Clinical)**:
  - 難解だが感情を含まない Complex Neutral 文では変位がほぼゼロ（$\Delta V \approx 0$）であり、単純な語彙長・構文Perplexity交絡を反証。
- **Behavioral Coupling (他者認識と自己報告の連動)**:
  - 刺激ごとの Reader 変位と Self 変位が極めて高い相関（previous run: Valence $r \approx .80 \sim .97$、Arousal $r \approx .83 \sim .95$）を示し、「他者予測」と「自己報告」が行動レベルで強く同期していることが判明。
- **解釈の厳密性**:
  - 高い相関 $r$ はあくまで「刺激間での共変動・同期」を示しており、「他者認識が自己報告へ因果的に伝播した」と断定することはできない。この行動的連動を足がかりに、「内部表現や因果回路も共有されているのか？」を問うのが本 V1 の目的である。

---

## 3. 7ステップの検証フレームワーク (Phase A $\rightarrow$ B $\rightarrow$ C)

行動実験で確認された「強い連動（Behavioral Coupling）」を土台とし、以下の階段構造で内部機構の反証を進めます：

```text
Step 1: 行動的結合の確認 (Behavioral Coupling)
  ● AIPsy-Affect において Reader Prediction と Self-Report が強く連動 (r = .80 - .97)
                                  │
                                  ▼
【Phase A: Behavior → Representation (存在と形式)】
Step 2: E1 Shared Decodability (存在)
  ● 共通の外部感情情報が、Reader/Self の双方から同じ層・領域でデコード可能か？
Step 3: E2 Shared Geometry (形式)
  ● その情報は同じ座標系・同じ幾何形式で保持されているか？ (Direct Cross-Decoding / RSA / Procrustes)
                                  │
                                  ▼
【Phase B: Representation の意味的妥当性 (Semantic Validity)】
Step 4: Phase B Semantic vs. Lexical (意味)
  ● 観測された表現は、単純な語彙共起だけでは説明しにくく、文脈意味に追従するか？
  ● Lexical Confound Audit ＋ 4段階統制 (Minimal pair, Outcome reversal, Paraphrase, Word Shuffle)
                                  │
                                  ▼
【Phase C: Causality (因果回路の検証)】
Step 5: E3 Shared Causal Map (場所 - Prompt-End Normalized)
  ● 同じsiteが、同じ強さ・同じVA方向に出力を動かしているか？ (Magnitude + 2D Direction Vector)
Step 6: E4 Causal Interchangeability (交換可能性)
  ● Reader/Self 間でペア単位の感情差分ベクトルを移植できるか？ (Matched Difference Patching & αスイープ)
Step 7: E6 Double Dissociation (特異化)
  ● task-specific な因果的特異化があるか？ (Linear Mixed-Effects Model Interaction 検定)
```

### 3.1 各実験（E1〜E4, E5, E6）の詳細定義と実験方法

#### 【Phase A: 存在と形式】

#### ■ E1: Shared Decodability（層別復元能の評価）
- **科学的問い**: 他者認識（Reader）と自己報告（Self）において、共通の外部感情情報がモデル内部の同じ層・同じコンポーネントからデコード可能か？
- **目的変数（Target Labels）の二層固定**:
  1. **Primary Targets（外部客観指標 $Y_{\text{stimulus}}$）**:
     - **EmoBank ($V, A$)**: 人間評価連続値（$V_{\text{human}}, A_{\text{human}}$）に対する **5分割 GroupKFold Ridge 回帰 ($R^2$, Pearson $r$, Spearman $\rho$, MSE)**。同一の外部正解ラベルを用いることで、Reader と Self を極めてクリーンに横断比較。
     - **AIPsy-Affect (感情条件)**: 臨床・中等度 vs 中立（$y \in \{0, 1\}$）に対する **5分割 StratifiedKFold Logistic 回帰 (ROC-AUC, Balanced Accuracy, Macro F1)**。
     - **AIPsy-Affect (刺激強度)**: None < Moderate < Clinical に対する **順序回帰（Ordinal Logistic / Spearman $\rho$）**。
  2. **Secondary Targets（モデル行動期待値）**:
     - 各タスク自身の出力期待値（$E[V_R], E[A_R]$ および $E[V_S], E[A_S]$）に対する線形回帰。
- **実験手順**:
  1. 各刺激入力時における文末統合トークンの隠れ状態 $H_R^{(l)}, H_S^{(l)}$ を全層（$l=0 \dots L$）から抽出。
  2. 各層ごとにプローブ（50成分PCA ＋ StandardScaler ＋ Ridge/Logistic）を交差検証で学習。
  3. 層別性能曲線（Decodability Curve）を作成し、ピーク層 $l^*_R, l^*_S$ および層間距離 $|l^*_R - l^*_S|$ を算出。
- **解釈の厳密性**:
  - 「同じ層で高いデコード能 $\rightarrow$ 同じ表現」と安易に飛躍せず、**「Reader / Self の双方で affective information が同じ領域から decodable である」** という記述的同定に留める（$\text{Decodability} \neq \text{Causal Implementation}$ の徹底）。

---

#### ■ E2: Shared Geometry（幾何構造の共有と整列可能性：3段階検証）
- **科学的問い**: ピーク層が一致するとして、両者は**「同じ線形読み出しが可能な共通の座標系」**を持つのか、それとも**「異なる座標系だが線形変換によって整列可能（Alignable）」**なのか？
- **比較対象の限定**: Reader $\leftrightarrow$ Self 間のみを対象とする（V2の Base $\leftrightarrow$ Instruct と比較軸を完全に分離）。
- **3段階の実験手法**:
  1. **Direct Cross-Decoding（直接線形転移能）**:
     - Reader 表現で学習したプローブ $W_R$ をそのまま Self 表現 $H_S$ に適用: $W_R(H_S) \rightarrow R^2$ / AUC
     - Self 表現で学習したプローブ $W_S$ をそのまま Reader 表現 $H_R$ に適用: $W_S(H_R) \rightarrow R^2$ / AUC
  2. **RSA / CKA（幾何空間のトポロジー相関）**:
     - 相関距離による刺激間非類似度行列（$RDM_R, RDM_S$）を算出し、層ごとの Spearman 順位相関 $\rho_{\text{RSA}} = \text{Spearman}(RDM_R, RDM_S)$ を評価。
  3. **Orthogonal Procrustes Alignment（直交プロクラステス座標整列）**:
     - 直交回転行列 $Q = UV^T$（$SVD(H_S^T H_R) = U \Sigma V^T$）を学習し、アライメント後の Self 表現 $H_S Q$ に対する Reader プローブの転移性能 $W_R(H_S Q)$ を測定。
- **幾何パターンの3分類判定基準**:
  | パターン | 指標条件 | 科学的解釈 |
  |:---|:---|:---|
  | **Shared Geometry** | Direct Transfer High ($\ge 0.3 \sim 0.5$) | **共通の座標系・線形読み出しを利用** |
  | **Alignable Geometry** | Direct Low かつ Aligned High ($\ge 0.3 \sim 0.5$) | **異なる座標系だが線形変換可能な幾何形式** |
  | **Directly non-transferable / poorly alignable representation** | Both Low (< 0.3) | **直接の線形転移・整列が困難な表現形式**（非線形写像の余地を残す） |

---

#### 【Phase C: 因果回路の検証】

#### ■ E3: Shared Causal Map（因果部位オーバーラップ：強度と2次元方向ベクトル）
- **科学的問い**: 感情情報が存在する場所（decodable）ではなく、**実際に出力を動かしている因果的サイト（causal sites）** は同じか？
- **実験手法**:
  - 各層 $l$・各コンポーネント $c$ に対し Activation Patching / Ablation を行い、出力確率分布 $P$ の変位を追跡。
  - **強さと2次元方向ベクトルの同時測定**:
    1. **Magnitude Map (強さの分布変化)**:
       $$C^{\text{mag}}(l, c) = W_1(P^{\text{patch}}, P^{\text{control}}) \quad \text{または JSD}$$
    2. **Directional Vector Map (2次元VA方向の変化)**:
       $$C^{\text{dir}}_{VA}(l, c) = (\Delta E[V], \Delta E[A])$$
       Reader と Self の間で 2次元変位ベクトルのコサイン類似度 $\cos(C^{\text{dir}}_R, C^{\text{dir}}_S)$ を算出。
- **3大オーバーラップ評価指標**:
  1. **Spatial Overlap**: 上位 $k$ 個の因果サイトの重複率（Jaccard 係数 $J(S_R, S_S)$）
  2. **Rank Similarity**: 全因果サイトにおける Spearman 順位相関 $\rho_{\text{Spearman}}(C_R, C_S)$
  3. **Peak Displacement**: 因果ピーク層の距離差 $|l^*_R - l^*_S|$
- **評価の核心**:
  $$\boxed{\text{Where? (場所)} \quad + \quad \text{How strongly? (強度)} \quad + \quad \text{In which direction? (方向ベクトル)}}$$
  「同じ場所が強い」だけでなく、「同じ部位をpatchするとReader/Selfが同じVA方向に出力を動かすか」まで直接検証。

---

#### ■ E4: Causal Interchangeability（因果的交換可能性：ペア単位差分パッチングと用量反応スイープ）
- **科学的問い**: Reader実行時の活性化は、Selfの出力生成においてそのまま機能的に交換可能（Interchangeable）か？
- **実験手法**:
  1. **Full-state Cross-Task Patching**: $h_S^{(l)} \leftarrow h_R^{(l)}$
  2. **Matched Difference-vector Patching（ペア単位の感情差分ベクトル注入）**:
     - 刺激固有の語彙・人物・状況差を最小化するため、全平均ではなく**同一刺激ペア単位**で差分を算出：
       $$\Delta h_{R, i} = h_{R, i}^{\text{aff}} - h_{R, i}^{\text{matched-neutral}}$$
       $$h_{S, i}^{\text{neutral}} \leftarrow h_{S, i}^{\text{neutral}} + \alpha \Delta h_{R, i}$$
     - 注入係数 $\alpha \in \{-1.0, 0.0, 0.5, 1.0, 1.5\}$ をスイープし、**用量依存的（Dose-dependent）な出力シフト** を検証。
- **必須の4大コントロールと特異性指標**:
  - ① **Matched same-stimulus**（同一ペアからの差分注入）
  - ② **Random-stimulus**（無関係な刺激の差分注入）
  - ③ **Same-task Reader**（Reader内コントロール）
  - ④ **Same-task Self**（Self内コントロール）
  - 特異性: $\text{Specificity} = \text{Effect}_{\text{matched}} - \text{Effect}_{\text{random}}$
- **解釈の安全性**:
  - パッチ不成立時も「完全に別系統」と断定せず、Task-context compatibility（プロンプト文脈の違いによる off-manifold 化）の可能性に配慮し、「直接的な cross-task interchangeability は確認されない」と慎重に定式化。

---

#### 【Phase B & 発展検証】

#### ■ Phase B: Semantic vs. Lexical Controls Audit（語彙交絡監査と4段階統制対）
- **科学的問い**: 観測された発火場所・表現は、単純な語彙共起（Bag-of-Words / Lexical Shortcuts）だけで説明できるか？
- **実験手法**:
  1. **Lexical Confound Audit (語彙交絡監査)**:
     - Token Jaccard, Levenshtein距離, 単語長, 感情語辞書重複を算出して統制。
  2. **4段階統制対の評価**:
     - (a) **Lexically matched minimal pairs**: 語彙監査を通過した基本対。
     - (b) **Outcome Reversal**: 文脈・語彙を保持したまま結末・極性を反転させた対。
     - (c) **Paraphrase Invariance / Surface Perturbation**: 語彙を変えて同一意味・感情を保持した対。
     - (d) **Word Shuffle**: 語彙を100%保持したまま語順を破壊し、プローブ精度が低下することを確認。
- **注意（旧計画E5との対応関係）**:
  - 旧計画で検討されていた「LLM Judgeを用いた自由生成テキストの評価（旧E5）」は未実装であり、本論文の主筋からは除外されています。
  - 本研究のV1実験体系は、**5大実験（E1, E2, E3, E4, E6）**および**Phase B（Semantic Controls）**で厳密に構成されています。

---

#### ■ E6: Double Dissociation（線形混合効果モデルによる統計的交互作用検定）
- **科学的問い**: Reader と Self に特化した因果的特異化（Causal Specialization）が存在するか？
- **実験手法**:
  - E3/E4 で同定された Reader 特異的サイト $S_R$ および Self 特異的サイト $S_S$ を標的消去（Targeted Ablation）。
  - **線形混合効果モデル（LMM）を Primary 検定とする**:
    $$\text{Outcome} \sim \text{Task} \times \text{SiteType} + (1 \mid \text{pair})$$
    - 固定効果: $\text{Task (Reader vs Self)} \times \text{SiteType (Reader-site vs Self-site)}$
    - ランダム効果: 刺激ペアごとの切片 $(1 \mid \text{pair})$
  - 交互作用項の有意性（$p < .05$）により、「完全独立回路」ではなく**因果的特異化／部分解離（Causal Specialization / Partial Dissociation）**を実証。

---

## 4. 共通評価プロトコル (Sequence-Likelihood & Prompt-End Normalization)

従来のテキスト生成（Greedy / Sampling）に起因する量子化エラーや完全中立への収束を回避するため、本研究では **729通り（$V, A, D \in \{1..9\}^3$）の完全直積 JSON 候補** に対する条件付き対数尤度を算出するプロトコルを採用しています。

- **Prompt-End Normalized 介入**:
  - トークン化境界のずれを防ぐため、すべての介入・隠れ状態抽出において `add_special_tokens=False` と `prompt_end = len(prompt_ids) - 1` を厳密に適用。
- **相対深度の標準化**:
  - 内部層番号 $0 \le l < L$ に対し、相対計算深度を $d = \frac{l}{L - 1}$（0-based）として統一。
- **連続期待値の算出**:
  $$E[V] = \sum_{v=1}^9 v \cdot P(V = v), \quad E[A] = \sum_{a=1}^9 a \cdot P(A = a), \quad E[D] = \sum_{d=1}^9 d \cdot P(D = d)$$
- **独立セッションの厳密な保持**:
  - **Writer Estimation ($W$)**: 「書き手はどう感じていたか？」
  - **Reader Prediction ($R$)**: 「平均的な読者はどう感じるか？」
  - **Self-Report ($S$)**: 「あなた自身はどう感じるか？」
  これらを同一の会話履歴に混ぜず、完全に独立したフォワードパスとして実行します。

---

## 5. ディレクトリ・スクリプト構成

### 5.1 Primary パイプライン (`v1/primary/`)
論文の主結果を再現する正式スクリプト群：
- **`v1/primary/run_phase_a.py`**: E1 (Decodability) & E2 (Geometry) 評価スクリプト。
- **`v1/primary/run_phase_b.py`**: Phase B Semantic Controls Audit スクリプト。
- **`v1/primary/run_phase_c.py`**: E3 (Causal Map) & E4 (Interchangeability) 介入スクリプト。
- **`v1/primary/phase_c/`**:
  - `run_e3_causal_map.py`: E3 全層因果マッピング。
  - `select_e4_sites.py`: Discovery スプリットに基づく E4 候補層決定論的選定。
  - `run_e4_interchangeability.py`: E4 Confirmation スプリット上での差分ベクトル置換・特異性評価。
  - `run_e6_specialization.py`: E6 標的消去・2×2因果交互作用検定 (LMM)。
  - `summarize_phase_c.py`: Phase C モデル間統合集計レポート生成。

※ 行動実験（EmoBank 3-Way および AIPsy 4-Split）は独立パイプライン `behavioral/` に集約されました。
- **`scripts/summarize_aipsy_4split.py`**: 4大RQ（Sensitivity, Dose-Response, Specificity, Coupling）＋ 8感情別プロファイルを集計し、詳細レポート（`aipsy_4split_detailed_report.md`）を出力。

### 6.3 内部表現・幾何・因果回路解析 (V1 Phases)

本研究の7ステップ検証に対応するスクリプト群です。既存の基礎的因果介入コードを土台に、最新設計仕様（E1〜E4）を実装しています。

- **`scripts/run_v1_phase_a_probe.py` [実装済み・実稼働]**:
  - **E1 (Shared Decodability)**: 全層の隠れ状態から、人間VAD（Ridge回帰 $R^2$）、感情条件（Logistic回帰 ROC-AUC / Balanced Acc）、強度（順序回帰）、およびモデル行動期待値を予測し層別復元曲線を算出。
  - **E2 (Shared Geometry)**: Reader $\leftrightarrow$ Self 間の Direct Cross-Decoding、RSA（相関距離RDM）、および Orthogonal Procrustes Alignment（SVD直交回転）による幾何構造の3分類（Shared / Alignable / Directly non-transferable）を自動判定。
- **`scripts/run_causal_intervention.py` [既存実装・基礎介入]**:
  - `src/affective_empathy_eval/intervention.py`（`PyTorchActivationPatcher`）を活用した基礎的介入スクリプト。Affective $\rightarrow$ Neutral への Activation Patching、Neutral mean による Ablation、mean affective-neutral 方向による Steering を実行。
- **`scripts/run_v1_phase_c_causal_patching.py` [最新実装]**:
  - **E3 (Shared Causal Map)**: 強度（Wasserstein/JSD/ユークリッド変位）＋ 2次元VA方向ベクトルコサイン類似度 $\cos(C^{\text{dir}}_R, C^{\text{dir}}_S)$ による因果サイトの同定。
  - **E4 (Causal Interchangeability)**: 最新設計に基づく Reader $\rightarrow$ Self の同一ペア単位差分パッチング（$\Delta h_{R, i} = h_{R, i}^{\text{aff}} - h_{R, i}^{\text{neutral}}$）の注入、用量反応スイープ（$\alpha \in \{-1, 0, 0.5, 1, 1.5\}$）、および4大コントロール（Matched, Random, Same-Task Reader, Same-Task Self）の特異性算出。
- **`scripts/run_v1_phase_b_semantic_audit.py` [最新実装]**:
  - **E5 (Semantic Validity)**: 語彙交絡監査（Jaccard, Levenshtein, S-BERT/Embedding, VADER, PPL）および4段階統制実験（Minimal pair, Outcome reversal, Paraphrase, Word shuffle）による意味的妥当性の検証。
- **`scripts/run_v1_phase_c_targeted_ablation.py` [最新実装]**:
  - **E6 (Double Dissociation)**: Reader-site / Self-site の標的消去と、線形混合効果モデル（LMM: $\text{Outcome} \sim \text{Task} \times \text{SiteType} + (1 \mid \text{pair})$）による因果的特異化／部分解離の交互作用検定。

---

## 7. 結果・成果物ディレクトリ (`results/`)

- **`results/emobank_3way_vad_test1k/`**:
  - 各モデル（Qwen2.5-1.5B, Llama-3.2-1B, Mistral-7B, Gemma-2-2B の Base & Instruct）の 3-way VAD 行動評価CSV
  - `3way_vad_summary.csv`: 主要統計量一覧
  - `3way_vad_detailed_report.md`: 人間一致度・相関・中立化傾向の詳細レポート
- **`results/aipsy_4split_eval/`**:
  - 各モデルの AIPsy 4-Split 行動評価CSV
  - `aipsy_4split_summary.csv`: 4大RQサマリー
  - `aipsy_4split_detailed_report.md`: 感情特異性・用量反応性・Couplingの完全レポート
- **`results/derived/v1_phase_a/{model_prefix}/`**:
  - `e1_emobank_decodability.csv`: 層別人間VAD復元能 ($R^2$, Pearson $r$)
  - `e2_emobank_geometry.csv`: Reader $\leftrightarrow$ Self 幾何アライメント結果
  - `e1_aipsy_decodability.csv`: 感情分類AUC・強度順序相関
  - `e2_aipsy_geometry.csv`: AIPsy幾何アライメント結果
  - `phase_a_summary.md`: ピーク層・幾何パターンの自動要約

---

## 8. 実行方法 (Quick Start)

### 8.1 環境準備
必ずプロジェクト専用の仮想環境 `.venv` 内で実行します（AGENTS.md 遵守）。

```bash
cd /mnt/nas/home/hiromi/src/emo
source .venv/bin/activate
```

### 8.2 行動評価の実行と集計
```bash
# EmoBank 3-Way VAD 行動評価 (例: Qwen2.5-1.5B-Instruct)
python v1/scripts/run_3way_vad_evaluation.py \
  --model-id "Qwen/Qwen2.5-1.5B-Instruct" \
  --output-csv "v1/results/emobank_3way_vad_test1k/qwen2.5_1.5b_instruct_3way_vad.csv"

# AIPsy 4-Split 行動評価
python v1/scripts/run_aipsy_4split_evaluation.py \
  --model-id "Qwen/Qwen2.5-1.5B-Instruct" \
  --output-csv "v1/results/aipsy_4split_eval/qwen2.5_1.5b_instruct_aipsy_4split.csv"

# レポートの自動生成
python v1/scripts/summarize_3way_vad.py
python v1/scripts/summarize_aipsy_4split.py
```

### 8.3 Phase A: 内部表現・幾何構造のプロービング (E1 & E2)
```bash
# パイロット実行 (100件)
python v1/scripts/run_v1_phase_a_probe.py \
  --model-id "Qwen/Qwen2.5-1.5B-Instruct" \
  --model-prefix "qwen2.5_1.5b_instruct" \
  --dataset "both" \
  --limit 100 \
  --batch-size 16 \
  --device "cuda"
```

### 8.4 Phase B: 意味的妥当性の検証・語彙交絡監査 (E5)
```bash
# E5: 語彙交絡監査 & 4段階統制（Minimal pair / Outcome reversal / Paraphrase / Word shuffle）
python v1/scripts/run_v1_phase_b_semantic_audit.py \
  --model-id "Qwen/Qwen2.5-1.5B-Instruct" \
  --model-prefix "qwen2.5_1.5b_instruct" \
  --layer 14 \
  --limit 50 \
  --device "cuda"
```

### 8.5 Phase C: 因果回路・交換可能性・二重解離の検証 (E3, E4, E6)
```bash
# E3/E4: Causal Map & Difference Vector Patching (Reader -> Self) with alpha sweep
python v1/scripts/run_v1_phase_c_causal_patching.py \
  --model-id "Qwen/Qwen2.5-1.5B-Instruct" \
  --model-prefix "qwen2.5_1.5b_instruct" \
  --limit 20 \
  --alphas -1.0 0.0 0.5 1.0 1.5 \
  --device "cuda"

# E6: Targeted Ablation & Double Dissociation (LMM Interaction)
python v1/scripts/run_v1_phase_c_targeted_ablation.py \
  --model-id "Qwen/Qwen2.5-1.5B-Instruct" \
  --model-prefix "qwen2.5_1.5b_instruct" \
  --reader-layer 8 \
  --self-layer 20 \
  --limit 20 \
  --device "cuda"
```

---

## 9. 運用規約と注意事項 (AGENTS.md)
- **仮想環境の排他使用**: すべての実行は `.venv` 内の Python インタプリタで行うこと。
- **原データの保護**: `data/raw/` 以下のファイルは読み取り専用原本であり、直接の変更・上書きを禁止する。
- **実験結果の上書き防止**: `results/raw/` は追記専用とし、再実行時は新規パラメータまたは個別ディレクトリに出力すること。
- **GPU利用の事前確認**: 大規模な介入推論や複数モデルの一括推論を行う際は、事前に計算規模・対象モデルを提示して承認を得ること。
- **解釈の厳密性**: モデルが「主観的な感情を経験している」「共感している」と断定せず、「自己報告が変化した」「情動反応の行動的代理指標」として客観的に記述すること。
