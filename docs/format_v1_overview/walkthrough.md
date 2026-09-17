# paper_v1.md 「V1の概要」整形・コード整合化 完了レポート（Walkthrough）

`/mnt/nas/home/hiromi/src/emo/v3/docs/paper_v1.md` 内の **「# V1 実験計画概要：内部表現・幾何構造・因果回路の多段階解明」（Line 3217〜3560）** について、現行の実稼働コード（`v1/scripts/run_v1_phase_*.py` 等）に即した完全・正確な説明への改訂、および学術論文スタイルの体裁整形を実施しました。

---

## 1. 実施した主要な改訂・整形内容

### ① 実稼働コードベース（E1〜E6）との完全整合
- **過去の事実誤認の完全排除**:
  旧草稿に残っていた「ZIPに run_v1_phase_b_semantic_audit.py や run_v1_phase_c_targeted_ablation.py が存在しない」という記述を完全に削除し、**現行コードベースに E1〜E6 の全スクリプトが実稼働コードとして完全実装されている実態**に即した正確な記述に改めました。
- **実装アルゴリズム・関数の完全反映**:
  - **E1 (Shared Decodability)**: `extract_hidden_states_batched` による全層プロンプト末尾統合トークン抽出、EmoBank に対する 5分割 GroupKFold Ridge回帰（StandardScaler + PCA 50成分）、AIPsy-Affect に対する 5分割 StratifiedKFold Logistic回帰（ROC-AUC, Balanced Acc）および感情強度順序回帰。
  - **E2 (Shared Geometry)**: `evaluate_cross_decoding_and_geometry` による Direct Cross-Decoding、RSA（相関距離RDMに基づくSpearman相関）、Orthogonal Procrustes Alignment（SVD直交回転 $Q=UV^T$）と、幾何パターンの3分類自動判定（Shared / Alignable / Directly non-transferable）。
  - **E5 (Semantic Validity)**: `run_v1_phase_b_semantic_audit.py` による語彙交絡監査（Jaccard, Levenshtein, EMOTION_LEXICON, PPL）および4段階統制対（Minimal Pair, Outcome Reversal, Paraphrase Invariance, Word Shuffle）。
  - **E3 (Shared Causal Map)**: `run_v1_phase_c_causal_patching.py` による同一タスク差分パッチング、729候補 Sequence-Likelihood による変位強度（Magnitude）および 2次元方向ベクトルコサイン類似度 $\cos(C_R^{\text{dir}}, C_S^{\text{dir}})$。
  - **E4 (Causal Interchangeability)**: Reader $\to$ Self のペア単位差分パッチング（$\tilde{h}_S = h_{S,\text{neu}} + \alpha \Delta h_R$）、$\alpha \in \{-1.0, 0.0, 0.5, 1.0, 1.5\}$ による用量反応性（Dose-response）検証、Matched / Random コントロールと特異性指標。
  - **E6 (Double Dissociation)**: `run_v1_phase_c_targeted_ablation.py` による Reader-site / Self-site の標的消去、2×2 因果介入マトリクス、線形混合効果モデル（LMM: `Outcome ~ Task * SiteType + (1 | pair)`）による統計的交互作用検定。

### ② 学術論文としての文体・レイアウト整流化
- 口語調・講義調（「悲しい結論になります」「人類は相関を見つけるとすぐ因果と言いたくなるので」等）を完全に排除し、学術論文にふさわしい論理的で格調高い「である調」に統一しました。
- 体系的な見出し階層（`## 1.`〜`## 7.` および `###`）と `$$ \boxed{...} $$` 数式ブロックを適用し、可読性を大幅に向上させました。

---

## 2. 実験構成とコード対応マトリクス

| 実験 | フェーズ | 科学的問い | 実装スクリプト | 主要な手法・統計モデル |
|:---|:---:|:---|:---|:---|
| **E1** | Phase A | **感情情報の存在確認** | `run_v1_phase_a_probe.py` | 5-fold GroupKFold Ridge / StratifiedKFold Logistic / 順序回帰 |
| **E2** | Phase A | **表現幾何形式の共有** | `run_v1_phase_a_probe.py` | Cross-Decoding / RSA / Orthogonal Procrustes SVD回転 |
| **E5** | Phase B | **意味的妥当性の検証** | `run_v1_phase_b_semantic_audit.py` | 語彙交絡監査（Jaccard/Levenshtein/PPL）＋ 4段階統制対 |
| **E3** | Phase C | **因果部位の重なり** | `run_v1_phase_c_causal_patching.py` | Activation Patching ＋ 2D方向ベクトルコサイン類似度 |
| **E4** | Phase C | **機能的交換可能性** | `run_v1_phase_c_causal_patching.py` | ペア単位差分パッチング ＋ $\alpha$-Sweep ＋ Matched/Random対照 |
| **E6** | Phase C | **機能的特異化の実証** | `run_v1_phase_c_targeted_ablation.py` | Targeted Ablation ＋ LMM 統計的交互作用検定 |

---

## 3. 変更対象ファイル
- [paper_v1.md](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper_v1.md)（Line 3217 〜 Line 3560）
