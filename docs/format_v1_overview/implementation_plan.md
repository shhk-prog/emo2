# paper_v1.md 「V1の概要」整形・コード整合化 実装計画

## 概要
`/mnt/nas/home/hiromi/src/emo/v3/docs/paper_v1.md` の **「# v1の概要」（Line 3217〜末尾 4684）** について、学術論文の章としてふさわしい端正なレイアウト・体系的見出し・学術的文体（である調）に整流化するとともに、**現行の実装コード（`v1/scripts/run_v1_phase_*.py` 等）に即した完全・正確な説明**へと改訂します。

---

## ユーザーレビュー必須事項
> [!IMPORTANT]
> **現行コードとの照合による重要改訂点**:
> 1. **未実装・非存在という記述の完全是正**:
>    草稿内に残っていた「ZIPに run_v1_phase_b_semantic_audit.py や run_v1_phase_c_targeted_ablation.py が存在しない」という記述は過去の外部ドラフトの残骸であり、**現在は E1〜E6 すべてのスクリプトが実稼働コードとして完全実装されています**。これを現行コードベースに即した「実装済み・実稼働の検証系」として正しく記述します。
> 2. **講義・雑談調の学術的昇格**:
>    「全体像はこうです」「悲しい結論になります」「人類は相関を見つけるとすぐ因果と言いたくなるので」等の口語表現を、査読に耐えうる論理的・学術的な論述スタイル（「である調」）へ洗練します。
> 3. **コード実装仕様の完全反映**:
>    `extract_hidden_states_batched`、`PyTorchActivationPatcher`、GroupKFold / StratifiedKFold、PCA 50次元 + Ridge / Logistic、RSA 相関距離、Procrustes SVD回転、$\alpha$-sweep、LMM（線形混合効果モデル）など、実際のコードで採用されている数理・実装アルゴリズムを正確に解説します。

---

## 提案する章立てと構成案

### `# V1 実験計画概要：内部表現・幾何構造・因果回路の多段階解明`

1. **`## 1. 核心的リサーチクエスチョンと多段階検証フレームワーク`**
   - 行動実験（EmoBank / AIPsy-Affect）で確認された超高連動（Behavioral Coupling: $r \approx 0.80 \sim 0.97$）を出発点とする問題設定。
   - 核心的問い：$$\boxed{\text{Does Similar Behavior imply Shared Representation and Shared Causal Implementation?}}$$
   - 存在（E1） $\to$ 形式（E2） $\to$ 意味（E5） $\to$ 因果部位（E3） $\to$ 交換可能性（E4） $\to$ 機能的特異化（E6）の論理的階段構造。

2. **`## 2. 評価タスクの前提：Reader と Self のプロンプト設計と独立フォワードパス`**
   - `format_prompt()` に基づく厳密なタスク設計（Reader: 読者反応推定 vs. Self: 自己状態報告）。
   - 単一会話コンテキストの汚染を排した、完全独立のフォワードパス（$H_R(x), H_S(x)$）による内部表現抽出。

3. **`## 3. Phase A：存在と幾何形式の解明（E1 & E2）`**
   - **`### 3.1 E1: Shared Decodability（層別復元能の評価）`**
     - スクリプト: `v1/scripts/run_v1_phase_a_probe.py`
     - 全層プロンプト末尾トークンベクトルの抽出。
     - EmoBank: 人間VAD連続値に対する 5分割 GroupKFold Ridge回帰（PCA 50成分）。
     - AIPsy-Affect: 臨床・中等度 vs 中立の 5分割 StratifiedKFold Logistic回帰（ROC-AUC / Balanced Acc）、および感情強度順序回帰。
     - モデル行動期待値（$E[V], E[A]$）の予測。
     - 記述的同定に留める科学的境界づけ（$\text{Decodability} \neq \text{Causal Circuit}$）。
   - **`### 3.2 E2: Shared Geometry（幾何構造の共有と整列可能性：3段階検証）`**
     - スクリプト: `v1/scripts/run_v1_phase_a_probe.py`
     - Direct Cross-Decoding（プローブの直接相互転移能）。
     - RSA（相関距離RDMに基づくトポロジー順位相関 $\rho_{\text{RSA}}$）。
     - Orthogonal Procrustes Alignment（SVD直交回転 $Q = UV^T$ による整列能）。
     - 幾何パターンの3分類自動判定（Shared / Alignable / Directly non-transferable）。

4. **`## 4. Phase B：意味的妥当性の検証と語彙交絡の完全排除（E5）`**
   - **`### 4.1 E5: Semantic vs. Lexical Validity（語彙交絡監査と4段階統制実験）`**
     - スクリプト: `v1/scripts/run_v1_phase_b_semantic_audit.py`
     - Lexical Confound Audit: Jaccard類似度、Levenshtein距離、感情語辞書（EMOTION_LEXICON）、Perplexity (PPL) の事前監査。
     - 4段階統制対の実験:
       1. Minimal Pair Contrast（文脈・構文を固定した臨床 vs 中立対）
       2. Outcome Reversal（主要語彙を残し結末の感情極性のみを反転）
       3. Paraphrase Invariance（同一事態を異なる語彙で表現）
       4. Word Shuffle（単語を100%保持し語順破壊で文脈意味を解体）

5. **`## 5. Phase C：因果回路と機能的交換可能性の検証（E3, E4, E6）`**
   - **`### 5.1 E3: Shared Causal Map（因果部位オーバーラップ：強度と2次元方向ベクトル）`**
     - スクリプト: `v1/scripts/run_v1_phase_c_causal_patching.py`
     - `PyTorchActivationPatcher` による同一タスク差分パッチング（Neutral + $\Delta h_S$, Neutral + $\Delta h_R$）。
     - 729候補 Sequence-Likelihood による変位量（Magnitude）と 2次元VA変位ベクトル（$C^{\text{dir}} = (\Delta V, \Delta A)$）の測定。
     - 2次元方向ベクトルのコサイン類似度 $\cos(C_R^{\text{dir}}, C_S^{\text{dir}})$ およびピーク変位 $|l^*_R - l^*_S|$ による因果サイト同定。
   - **`### 5.2 E4: Causal Interchangeability（因果的交換可能性：差分パッチングと用量反応）`**
     - スクリプト: `v1/scripts/run_v1_phase_c_causal_patching.py`
     - Reader $\to$ Self のペア単位差分パッチング（$\tilde{h}_S = h_{S,\text{neu}} + \alpha \Delta h_R$）。
     - $\alpha \in \{-1.0, 0.0, 0.5, 1.0, 1.5\}$ による用量反応性（Dose-response）検証。
     - 4大コントロール（Matched, Random, Same-Task Reader, Same-Task Self）と特異性指標（$\text{Specificity} = \text{Effect}_{\text{matched}} - \text{Effect}_{\text{random}}$）。
   - **`### 5.3 E6: Double Dissociation（線形混合効果モデルによる統計的交互作用検定）`**
     - スクリプト: `v1/scripts/run_v1_phase_c_targeted_ablation.py`
     - Reader-site / Self-site の標的消去（Targeted Ablation）と 2×2 因果介入マトリクス。
     - 線形混合効果モデル（LMM: `Outcome ~ Task * SiteType + (1 | pair)`）による統計的交互作用検定。
     - 共通表現からのタスク特異的読み出し回路の分岐（Causal Specialization / Partial Dissociation）の実証。

6. **`## 6. E1〜E6の統合的論理関係と反証の階段構造`**
   - 6つの実験が果たす役割のマトリクス一覧。
   - 査読者の代替仮説を先回りして反証する堅牢な論理体系。

7. **`## 7. 実装スクリプト構成と実行仕様`**
   - `v1/scripts/` 内の実稼働スクリプト一覧と実行コマンド（クイックスタート）。

---

## 変更対象ファイル
- [paper_v1.md](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper_v1.md)
  - 変更行: Line 3217 〜 Line 4684（末尾まで）

---

## 検証計画
1. **現行コードとの完全一致検証**: 各実験（E1〜E6）の説明が、`v1/scripts/` の実際のPythonコードと完全に合致しているか確認。
2. **Markdownレンダリング検証**: 数式ブロック（`$$ \boxed{...} $$`）、表、リストが崩れなく表示されることを確認。
