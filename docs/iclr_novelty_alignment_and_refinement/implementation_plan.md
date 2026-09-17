# ICLR向け新規性位置づけの厳密化と論文改訂の実装計画 (ICLR Novelty Alignment & Refinement)

本計画は、提供された査読者・新規性分析に基づき、本研究の独自性を最大化し、ICLR本会議での採択可能性を大幅に引き上げるための論文原稿改訂（[`v3/docs/paper.md`](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper.md)）および実験ロードマップの策定方針を示すものである。

---

## 1. 方針と論点整理

### 1.1 主張の優先順位と新規性ポジショニング
査読フィードバックで指摘された通り、「感情表現の存在」や「ステアリングによる出力変化」は既に先行研究で広く扱われており、これを新規性として主張することは大きなリジェクトリスクを伴います。
本研究の真の独自性は以下の命題に集約されます：

> **Core Proposition:**  
> *"A representation can be linearly decodable and cross-model prediction-restorable, yet still fail to be causally substitutable for a downstream, policy-constrained first-person reporting distribution."*  
> （表現が線形デコード可能かつクロスモデルで予測復元可能であっても、ポリシ制約された下流の一人称報告分布に対して因果的に代替可能であるとは限らない）

| 項目 | 査読での評価 | 論文における位置づけと変更方針 |
|---|---|---|
| **N2: Decodability $\neq$ Causal Substitutability** | **最重要・最高新規性** | **論文の絶対的コアに昇格**。<br>①Direct transfer失敗 $\rightarrow$ ②Ridge alignmentによるheld-out予測復元 $\rightarrow$ ③それでもAligned Patchingによる自己報告分布回復は0%、の3段階を明瞭に提示。 |
| **N1: Greedy中立化と尤度感度の解離** | **中〜高（測定論的貢献）** | 「LLMの隠された感情の発見」ではなく、「制約付き一人称報告候補列に対する潜在選好の測定において、Greedy decodeが貧弱な測定手法となり得ること（**Sequence-likelihood measurement protocol**）」という評価論・測定論の貢献として提示。 |
| **N4: Valence-Congruent Recognition Shift** | **中〜高（因果交互作用）** | 「世界初のmood congruency」という心理学的過剰主張を完全排除し、**$\text{Steering strength} \times \text{valence direction} \times \text{stimulus ambiguity}$** という曖昧性特異的・方向特異的なタスク横断因果効果に純化。 |
| **N3: 分散的再マッピング** | **中程度（注意が必要）** | 「分散回路の新規メカニズム同定」としてではなく、「**単一コンポーネントや出力層単独説（lm_head swap）の反証（Negative result）および分散的変化との整合性**」として慎重に記述を修正。 |

### 1.2 採択時に使用する公式な4大貢献文 (Official Contributions)
論文の導入および結論部において、以下の4点に完全に統一します：

1. **Likelihood-based measurement (N1)**:  
   *We introduce a constrained sequence-likelihood protocol for measuring first-person affect self-report distributions beyond a single greedy completion, and show that greedy neutralization can coexist with stimulus-sensitive likelihood variation.*
2. **Post-training dissociation (N1/前提)**:  
   *Across paired base/instruct language models, we show that post-training can preserve linearly decodable affect-relevant information while altering its relationship to constrained first-person reporting behavior.*
3. **Predictive–causal dissociation (N2 - Core)**:  
   *We demonstrate that a held-out linear alignment that restores cross-model affect prediction is insufficient to restore the corresponding first-person report distribution under pre-specified cross-model activation patching, despite positive within-model controls.*
4. **Cross-task causal influence (N4)**:  
   *We show that steering a valence-associated direction shifts third-person affect recognition judgments most strongly for ambiguous stimuli, with random and arousal-direction controls, indicating a task-crossing computational influence rather than a generic output offset.*

### 1.3 必須引用・先行研究6系統との直接的対比
先行研究との重複批判を先回りで回避するため、以下の6系統の最新文献を `Section 5 (Related Work)` に明記し、本研究との明確な境界線を引きます：
1. **Anthropic (2024/2026)** (*Emotion concepts and their function in a large language model*): 感情概念の因果的関与と事後学習による活性化分布変化を扱う。 $\rightarrow$ 本研究は「表現の消去ではなく、下流の制約付き自己報告プロトコルとの結合変化（decoupling）」を解明する点で差別化。
2. **Emotion Inference MI (2025/2026)** (*Mechanistic Interpretability of Emotion Inference in LLMs*): 感情推論の活性化パッチングと局所化。 $\rightarrow$ 本研究は一人称報告におけるクロスモデルパッチングの失敗（解離）を対照的に提示。
3. **Unified View on Emotion Representation (ACL 2026)**: 感情表現へのパッチングの体系化。 $\rightarrow$ 「パッチが効く」報告に対し、事後学習ペア間での「線形復元後のパッチ不全」という重要な境界条件（negative result）を提示。
4. **Asymmetric Valence Processing (2026)**: Valence方向の同定とステアリング。 $\rightarrow$ Valence方向の発見自体ではなく、事後学習後のreadout couplingおよび三人称曖昧性タスクへの因果波及に特化。
5. **Latent Structure of Affective Representations (2026)**: VA幾何とステアリング。 $\rightarrow$ Sanity checkとして位置づけ、主貢献から切り離す。
6. **Emotions, Where Art Thou (ICLR 2026)**: 潜在空間射影と隠れ状態操作。 $\rightarrow$ Sequence likelihood評価とクロスモデル因果アライメントの差分として説明。

---

## 2. 論文原稿 ([`v3/docs/paper.md`](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper.md)) の具体的改訂計画

### [MODIFY] [`v3/docs/paper.md`](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper.md)

1. **Title & Abstract (概要)**:
   - コア命題（Predictive vs Causal dissociation）を前面に押し出し、4つの貢献を簡潔に凝縮。
   - N3の表現を「分散的再マッピングの同定」から「局所的説明の反証と分散的変化との整合」へ修正。
2. **Section 1 (Introduction)**:
   - 貢献の箇条書き部分を、上記の4大貢献文に置き換え。
   - 擬人観（「モデルが感情を持つか」）の完全排除ステートメントを冒頭で再強調。
3. **Section 2 (Problem Formulation & Hypotheses)**:
   - 表1の H4 (Distributed coupling change) の記述を精緻化し、「単一部位・出力層単独説の棄却」を主旨とする。
4. **Section 4 (Results)**:
   - **4.1 (N1)**: 「尤度空間での感度の逆転」を測定論的・評価論的観点（Greedy decodeの限界）から記述。
   - **4.4 (N2)**: 本文の最大の山場として再構成。Positive control（Within-model）とOOD診断を強調し、「事前指定されたアライメントパッチングでの不全」を克明に描写。
   - **4.6 (N3)**: 「We find no evidence that a single tested component or the output head alone constitutes a sufficient bottleneck...」の公式定式化を採用。
   - **4.7 (N4)**: $\text{Steering strength} \times \text{valence direction} \times \text{stimulus ambiguity}$ の三要因交互作用モデルを明示。
5. **Section 5 (Related Work)**:
   - 先行研究6系統を直接組み込み、表6（対比表）に具体的な著者・論文名・先行知見・本研究の差別化ポイントを網羅。
6. **Section 6 (Discussion & Limitations)**:
   - サンプルサイズ（$n=10$ pair）の制限を率直に開示し、Case studyとしての位置づけと、採択に向けた今後の拡張ロードマップ（$n=50\sim 100$ への拡大、Multi-layer simultaneous patching、Mahalanobis OOD診断）を明記。

---

## 3. 実験・データセット拡張ロードマップ（今後の採択強化施策）

査読者が指摘する「$n=10$ pairではICLR本会議でborderline〜weak rejectになり得る」というリスクを解消するための具体的な実験計画：

1. **Strict Minimal-Pair Dataset の拡張 ($n=10 \rightarrow n=50\sim 100$)**:
   - AIPsy-Affect の厳密統制トリプレットを50〜100ペアに拡充。
   - 人手評価または厳密な語彙・構文フィルタによる交絡因子の排除。
2. **Multi-layer Simultaneous Patching の実装**:
   - 単一層だけでなく、連続するブロック（例: Layer 12-16同時パッチ）による回復率の変化を検証し、分散性仮説を積極検証。
3. **Cross-model patch の多変量 OOD 診断**:
   - コサイン類似度・ノルム比に加え、ターゲット活性化分布に対する Mahalanobis 距離および次層出力への伝播影響を定量化。

---

## 4. 検証計画

- **整合性チェック**: 論文全体の用語（N1〜N4、S1〜S5、H1〜H4）が完全に統一され、矛盾や過剰主張が残っていないかを確認。
- **プロジェクトルール遵守**:
  - `docs/iclr_novelty_alignment_and_refinement/task.md` の更新
  - `docs/iclr_novelty_alignment_and_refinement/implementation_plan.md` の保存
  - 完了後の `walkthrough.md` の作成
