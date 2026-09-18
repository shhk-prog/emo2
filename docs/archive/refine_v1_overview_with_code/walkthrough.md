# Walkthrough: V1の概要リファイン（実コード準拠および学術的整形）

## 実施内容

`/mnt/nas/home/hiromi/src/emo/v3/docs/paper_v1.md` 内の「# V1の概要」セクション（旧 Line 3220〜4688）について、実際のリポジトリ内コードおよび実験プロトコルに基づき、以下の刷新・整形を実施しました。

### 1. 実装コードとの完全一致
- **Phase A (E1, E2)**:
  - [`v1/scripts/run_v1_phase_a_probe.py`](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/run_v1_phase_a_probe.py) の実装に基づき、全層文末トークン隠れ状態の抽出、StandardScaler + 50成分PCA、EmoBank（5-fold GroupKFold Ridge回帰）および AIPsy-Affect（5-fold StratifiedKFold Logistic回帰・強度順序回帰）の二層ターゲットを明記。
  - E2 の Direct Cross-Decoding、RSA（相関距離RDMのSpearman順位相関）、Orthogonal Procrustes Alignment（SVDによる回転転移）および 3大幾何判定基準（Shared, Alignable, Non-transferable）を正確に定式化。
- **Phase B (E5)**:
  - [`v1/scripts/run_v1_phase_b_semantic_audit.py`](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/run_v1_phase_b_semantic_audit.py) の実装に基づき、Lexical Confound Audit（Jaccard, 編集距離, 感情辞書差, PPL比）と 4大直交統制対（Original, Paraphrase, Word Shuffle, Outcome Reversal）の反証論理を体系化。
- **Phase C (E3, E4, E6)**:
  - [`v1/scripts/run_v1_phase_c_causal_patching.py`](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/run_v1_phase_c_causal_patching.py) の実装に基づき、PyTorch forward hook による Activation Patching（効果量Magnitudeと2次元VA変位ベクトルのコサイン類似度）、Matched Difference Patching と $\alpha \in \{-1.0, 0.0, 0.5, 1.0, 1.5\}$ スイープによる用量反応性、4大コントロール（Matched, Random, Same-R, Same-S）を明記。
  - [`v1/scripts/run_v1_phase_c_targeted_ablation.py`](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/run_v1_phase_c_targeted_ablation.py) の実装に基づき、Targeted Ablation と 線形混合効果モデル（LMM）$\text{Outcome} \sim \text{Task} \times \text{SiteType} + (1 \mid \text{pair})$ による Double Dissociation 検定を明記。

### 2. 古い暫定メモの完全排除
- 過去のZIP調査時の残骸記述（「ZIP内にスクリプトが存在しない」「E5/E6のスクリプトがない」など現実に反するメモ）をすべて排除し、現在すでに全8モデルで完走している本番実装に基づく確固たる研究プロトコルとして記述。

### 3. LaTeX数式構文およびMarkdown階層の美化
- 崩れていた角括弧 `[` `]` を `$$ ... $$` やインライン `$ ... $` へ修正。
- 大見出し `# V1の概要：実験フレームワークと実装プロトコル` の下に `## 1. 全体像...` 〜 `## 6. V1全体が構成する論理フレームワークの総括` を体系的に配置。
- 概念図（ASCIIダイアグラム）、数式ブロック、定量的基準表を美しくレイアウト。
