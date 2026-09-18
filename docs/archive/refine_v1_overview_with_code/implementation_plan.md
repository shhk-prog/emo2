# 実装計画: V1の概要のリファイン（実装コード準拠および学術的整形）

## 目的
`/mnt/nas/home/hiromi/src/emo/v3/docs/paper_v1.md` の「# V1の概要」セクション（Line 3220〜4693）には、過去のZIP受領時の暫定メモ（「ZIP内にスクリプトが存在しない」等）や、崩れたLaTeX数式、細切れなテキストが残存している。
本作業では、実際のリポジトリ内コード（`v1/scripts/` 配下の Phase A/B/C スクリプト）および `v1/README.md` の仕様に完全に準拠させ、学術論文の骨子として正確かつ美しく読める体系的な概要へと刷新する。

## 主な更新方針

1. **実コードに基づく正確なプロトコル記述**:
   - **Phase A (E1, E2)**:
     - 実コード `v1/scripts/run_v1_phase_a_probe.py` に準拠。
     - E1: EmoBank（人間VADへの5-fold Ridge回帰）、AIPsy-Affect（5-fold Logistic回帰によるAffective vs Neutral二値分類、および刺激強度の順序回帰）。文末トークン隠れ状態 $H_R^{(l)}, H_S^{(l)}$ の全層抽出。
     - E2: Direct Cross-Decoding ($W_R(H_S)$, $W_S(H_R)$)、RSA（相関距離RDMのSpearman順位相関）、Orthogonal Procrustes ($Q = UV^T$ による回転転移能)。3大幾何判定（Shared, Alignable, Non-transferable）。
   - **Phase B (E5)**:
     - 実コード `v1/scripts/run_v1_phase_b_semantic_audit.py` に準拠。
     - E5-1: Lexical Confound Audit（Jaccard, Edit distance, Sentiment cue, Perplexity比）。
     - 4段階統制対（192最小対）：Original、Paraphrase（表層変更・意味維持）、Word Shuffle（語彙維持・構造破壊）、Outcome Reversal（文脈近似・結末意味反転）。
   - **Phase C (E3, E4, E6)**:
     - 実コード `v1/scripts/run_v1_phase_c_causal_patching.py` および `v1/scripts/run_v1_phase_c_targeted_ablation.py` に準拠。
     - E3: PyTorch forward hook による Activation Patching。Magnitude変化（JSD/Wasserstein）と2次元VA方向ベクトル（$\Delta E[V], \Delta E[A]$）のコサイン類似度。
     - E4: ペア単位 Matched Difference Patching（$\tilde h_{S, i} = h_{S, i}^{\text{neutral}} + \alpha \Delta h_{R, i}$）。$\alpha \in \{-1.0, 0.0, 0.5, 1.0, 1.5\}$ スイープによる用量反応性検証、4大コントロール（Matched, Random, Same-R, Same-S）。
     - E6: Targeted Ablation による Reader-selective / Self-selective サイトの同定と、Linear Mixed-Effects Model (LMM) による交互作用検定 $\text{Outcome} \sim \text{Task} \times \text{SiteType} + (1 | \text{pair})$（Double Dissociation）。
2. **古い暫定メモの完全払拭**:
   - 「ZIPにファイルがない」等の過去の古いコメントを削除し、実装済みパイプラインとしての客観的記述に修正。
3. **LaTeX数式およびMarkdown階層の学術的整形**:
   - 生の角括弧 `[` `]` を `$$ ... $$` やインライン `$ ... $` へ修正。
   - 見出しを `# V1の概要`、`## 1. ...`、`### ...` として論理的に整理。
   - テーブル記法を整頓し、数式と図解を美しく配置。
