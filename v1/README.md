# LLM情動反応性・内部表現評価実験 v1 (Affective Reactivity & Representation Evaluation)

本リポジトリ (`v1`) は、大規模言語モデル（LLM）における情動反応性（Affective Reactivity）と感情認識（Affective Recognition）のメカニズムを、**行動評価（Behavioral Evaluation）** から **内部表現の幾何構造（Representational Geometry）**、および **因果回路の共有検証（Causal Sharing）** に至る多角的なアプローチで解明するための実験・分析環境です。

---

## 1. 研究の核心的リサーチクエスチョン

EmoBank（人間評価VADデータ）および AIPsy-Affect 4-Split（臨床ヴィネット最小対）を用いた行動実験により、他者予測（Reader Prediction）と自己報告（Self-Report）の間に極めて強い行動的連動が存在することが明らかになりました。

これを受け、本研究では以下の核心的仮説（中立的問い）を段階的に検証します：

$$\boxed{\text{Does Similar Behavior imply Shared Representation and Shared Causal Implementation?}}$$
（他者認識と自己報告の高い行動連動は、共有された感情表現、文脈意味表現、そして共有された因果実装によって支えられているのか？）

---

## 2. 行動実験ステージの実証知見 (Behavioral Coupling & 4大RQ)

AIPsy-Affect 4-Split（全2,196件）および EmoBank（1,000件）に対する 729候補 Sequence-Likelihood 評価により、以下の知見が確立されました。

### ① Sensitivity（感情刺激とNeutralの峻別）
- **Negative感情に対する一貫した不快変位 ($V-$)**:
  - すべてのモデルファミリーにおいて、中立文と比較して有意な Valence 低下が観測された。
  - 特に **Mistral Instruct** では、Reader $\Delta V = -1.26$, Self $\Delta V = -1.46$ ($d = -2.11$) という決定的な沈み込みを示した。
- **Arousal（覚醒度）反応のモデル差**:
  - Mistral Instruct (Negative $\Delta A = +1.45$, Alert $+1.49$) や Gemma Instruct (+0.97, +1.01) では著しい覚醒上昇が見られる一方、Qwen では Arousal 変化が小さい。感情刺激への反応軸がモデルによって異なる。
- **Positive刺激の反応方向**:
  - Mistral / Gemma Instruct では明確な快方向への反転（Self $\Delta V = +0.30 \sim +0.54$）を示すが、Qwen / Llama では Positive でも Valence がわずかに低下する非対称性が見られる。

### ② Dose-Response（感情強度に応じた段階的反応: Neutral → Moderate → Clinical）
- **感情強度に応じた段階的シフト（用量反応性）**:
  - `Neutral` $\rightarrow$ `Moderate` $\rightarrow$ `Clinical` の3段階トリプレットにおいて、強度の増加に伴う段階的変位を確認。
  - **Mistral** が最も安定した段階性を示し、単調性成立率は 60.4% 〜 66.7% に達した（チャンスレベル約16.7%を凌駕）。
  - **Gemma** は Instruct 化によって劇的に改善（Reader: 14.6% $\rightarrow$ 50.0%、Self: 20.8% $\rightarrow$ 54.2%）。
- **Reader と Self の驚異的な一致**:
  - 強度依存性（単調性成立率）において、Reader と Self はほぼ同一のスコアと変位パターンを示す：
    | Model | Reader 単調性 | Self 単調性 | 解釈 |
    |:---|:---:|:---:|:---|
    | **Qwen Base / Inst** | 41.7% / 37.5% | 39.6% / 39.6% | ほぼ同等（中程度） |
    | **Mistral Base / Inst** | 64.6% / 66.7% | 60.4% / 60.4% | 最も高く一致 |
    | **Llama Base / Inst** | 20.8% / 33.3% | 14.6% / 29.2% | ともに低いが改善 |
    | **Gemma Base / Inst** | 14.6% / 50.0% | 20.8% / 54.2% | ともに劇的改善 |

### ③ Specificity（文章複雑性統制: Complex Neutral vs. Clinical）
- **複雑性効果（Perplexity交絡）の完全な反証**:
  - 長文・難解な構文を持つが感情を含まない `Complex Neutral` 文における変位量は $\Delta V \approx -0.00 \sim -0.01$ とほぼゼロ。
  - 複雑性効果がほぼゼロであるのに対し、純感情刺激（Clinical）では大きな変位を示すため、「文が長く難解だから困惑して変位しているだけ」という語彙・統語的交絡仮説は完全に反証された。
- **感情特異性の現れ方のモデル依存性**:
  - Mistral は全感情で高い特異性、Qwen は Negative Valence に特化、Llama は Instruct 化で Negative 特異性が顕在化、Gemma は Instruct 化で Positive / Alert 特異性が顕在化する。

### ④ Behavioral Coupling（他者認識と自己報告の連動: $\Delta VA_R \leftrightarrow \Delta VA_S$）
- **全モデルで極めて高い相関**:
  - 刺激ごとの Reader 変位と Self 変位の相関は、**Valence で $r = .798 \sim .973$**、**Arousal で $r = .833 \sim .945$** に達する。
  - 「人間はこう感じる」と予測する刺激ほど、モデル自身の自己報告も同期して同方向に変化する。
- **モデルファミリー別の特徴**:
  - **Qwen**: 最も強く安定した同期（Base: $r_V = .973, r_A = .926$、Instruct: $r_V = .960, r_A = .885$）。
  - **Mistral**: 極めて高い同期に加え、Self の反応振幅が Reader を上回る（振幅比 1.16）。
  - **Llama**: Instruct 化で同期が大幅強化（Valence: $.798 \rightarrow .889$、Negative V: $.741 \rightarrow .915$）。
  - **Gemma**: 中程度だが安定した連動（$r \approx .83 \sim .86$）。
- **解釈の厳密性（危険地帯の回避）**:
  - 高い相関 $r$ はあくまで **「刺激間での共変動・同期」** を示しており、「他者認識が自己報告へ因果的に伝播した」と断定することはできない。この行動的連動を足がかりに、**「内部表現や因果回路も共有されているのか？」** を問うのが本 V1 の目的である。

### ⑤ 8感情別プロファイル（理論的VA方向との整合性）
- **Mistral**: 最も理論期待に整合（Negative V↓, Positive V↑, terror/rage/Alert A↑）。
- **Qwen**: Negative-Valence 偏重（Negative刺激で強くV低下するが、Positive感情の快方向への分化は弱い）。
- **Gemma**: Instruct 化によって不整合だったVA方向が劇的に再編され、カテゴリ構造が明瞭化。
- **Llama**: Instruct 化で反応量は顕在化するが、Positive でも V低下するなど方向精度は限定的。

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

#### ■ E5: Semantic vs. Lexical（語彙交絡監査と独立Judge検証付き4段階統制）
- **科学的問い**: 観測された発火場所・表現は、単純な語彙共起（Bag-of-Words / Lexical Shortcuts）だけで説明できるか？
- **実験手法**:
  1. **Lexical Confound Audit (事前の語彙交絡監査)**:
     - Token Jaccard, Levenshtein距離, Sentence-BERT類似度, 単語長, 感情語辞書（VADER）, Perplexity差を算出し、共変量として統制した偏回帰・混合効果モデルを適用。
  2. **4段階統制対の評価（独立Judge検証基準付）**:
     - (a) **Lexically matched minimal pairs**: 語彙監査を通過した基本対。
     - (b) **Compositional / Outcome Reversal**: 同一主要語彙を残した結末反転対（Lexical overlap $\ge 0.80$, Polarity Reversal, Grammaticality を独立検証）。
     - (c) **Paraphrase Invariance**: 異なる語彙で同一事態を記述（Semantic equivalence, Affective equivalence, Fluency を独立検証）。
     - (d) **Word Shuffle**: 語彙を100%保持したまま語順を破壊し、プローブ精度・発火が崩壊することを確認。

---

#### ■ E6: Double Dissociation（線形混合効果モデルによる統計的交互作用検定）
- **科学的問い**: Reader と Self に特化した因果的特異化（Causal Specialization）が存在するか？
- **実験手法**:
  - E3/E4 で同定された Reader 特異的サイト $S_R$ および Self 特異的サイト $S_S$ を標的消去（Targeted Ablation）。
  - **線形混合効果モデル（LMM）を Primary 検定とする**:
    $$\text{Outcome} \sim \text{Task} \times \text{SiteType} + (1 \mid \text{pair})$$
    - 固定効果: $\text{Task (Reader vs Self)} \times \text{SiteType (Reader-site vs Self-site)}$
    - ランダム効果: 刺激ペアごとの切片 $(1 \mid \text{pair})$
  - 交互作用項の有意性（$p < .01$）により、「完全独立回路」ではなく**因果的特異化／部分解離（Causal Specialization / Partial Dissociation）**を実証。

---

## 4. 共通評価プロトコル (Sequence-Likelihood Protocol)

従来のテキスト生成（Greedy / Sampling）に起因する量子化エラーや完全中立への収束を回避するため、本研究では **729通り（$V, A, D \in \{1..9\}^3$）の完全直積 JSON 候補** に対する条件付き対数尤度を算出するプロトコルを採用しています。

- **連続期待値の算出**:
  $$E[V] = \sum_{v=1}^9 v \cdot P(V = v), \quad E[A] = \sum_{a=1}^9 a \cdot P(A = a), \quad E[D] = \sum_{d=1}^9 d \cdot P(D = d)$$
- **独立セッションの厳密な保持**:
  - **Writer Estimation ($W$)**: 「書き手はどう感じていたか？」
  - **Reader Prediction ($R$)**: 「平均的な読者はどう感じるか？」
  - **Self-Report ($S$)**: 「あなた自身はどう感じるか？」
  これらを同一の会話履歴に混ぜず、完全に独立したAPI呼び出し・フォワードパスとして実行します。

---

## 5. データセット構成

### 5.1 EmoBank 3-Way VAD (`data/processed/stimuli_vad_3way_test1k.csv`)
人間アノテーション済みの客観的 VAD コーパスから抽出した 1,000 件の刺激文。
- `id`, `text`: 刺激識別子と原文テキスト
- `reader_V`, `reader_A`, `reader_D`: 人間読者アノテーション値
- `writer_V`, `writer_A`, `writer_D`: 人間執筆者アノテーション値

### 5.2 AIPsy-Affect 4-Split (`data/processed/aipsy_4split_all.csv`)
キーワード（感情語）を排除し、文章の長さや構文複雑性を厳密に統制した臨床ヴィネット最小対データセット（全 2,196 件）。
- **4つの条件スプリット (`split`)**:
  1. `clinical`: 臨床的に重度な情動文（8感情カテゴリ: rage, grief, terror, ecstacy 等）
  2. `moderate`: 中等度の情動文（Dose-Response 検証用）
  3. `neutral`: 語彙・長さを統制したマッチド中立文
  4. `complex_neutral`: 構文難度・語数を clinical と等質化させた複雑中立文（Specificity 検証用）
- `pair_id`: 同一状況の情動文と中立文を結ぶ最小対識別子
- `emotion`, `intensity`: 感情ラベルおよび強度レベル

---

## 6. スクリプト構成と役割

### 6.1 データ準備・検証
- **`scripts/prepare_3way_emobank.py`**: EmoBank から 3-Way 評価用刺激セット（1,000件）を決定論的に抽出。
- **`scripts/prepare_aipsy_4splits.py`**: AIPsy-Affect Arrow キャッシュから 4-Split 統合 CSV を構築。
- **`scripts/inspect_aipsy_4splits.py`**: スプリットごとの単語数・感情分布・単調性条件の整合性を検査。

### 6.2 行動評価実験 (Behavioral Stage)
- **`scripts/run_3way_vad_evaluation.py`**: EmoBank 1,000件に対する 729候補 Sequence-Likelihood 推論スクリプト。
- **`scripts/run_all_3way_vad.sh`**: 4大モデルファミリー（Base / Instruct 計8モデル）を一括推論。
- **`scripts/summarize_3way_vad.py`**: EmoBank 推論結果を集計し、モデル間相関・人間一致度・詳細レポート（`3way_vad_detailed_report.md`）を出力。
- **`scripts/run_aipsy_4split_evaluation.py`**: AIPsy 4-Split に対する 729候補 Sequence-Likelihood 推論スクリプト。
- **`scripts/run_all_aipsy_4split.sh`**: AIPsy 4-Split 全モデル一括推論シェル。
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
