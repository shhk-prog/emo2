# ICLR向け新規性位置づけの厳密化と論文改訂の確認 (Walkthrough)

本ドキュメントは、提供された査読・新規性分析に基づき実施した論文原稿（[`v3/docs/paper.md`](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper.md)）のポジショニング刷新および学術的厳密化の完了報告である。

---

## 1. 改訂の全体概要

| 領域 | 従来の記述リスク | 改訂後のポジショニング（ICLR採択基準） |
|---|---|---|
| **コア命題** | 「事後学習が表現と自己報告の結合を変える」という一般的な記述 | **"A representation can be linearly decodable and cross-model prediction-restorable, yet still fail to be causally substitutable for a downstream, policy-constrained first-person reporting distribution."** を論文の最高命題として確立。 |
| **N2 (核心)** | 単なる「パッチングが効かなかった」という結果報告 | **論文の絶対的コア**へ昇格。①Direct transfer失敗 $\rightarrow$ ②Ridge alignmentによるheld-out線形予測復元 $\rightarrow$ ③事前指定パッチングでの自己報告回復率0.0% の3段階論法を確立。Within-model陽性対照（24.5%回復）との対比を明確化。 |
| **N1 (測定論)** | 「LLMの隠された感情の発見」と読まれるリスク | **「制約付き一人称報告の潜在選好測定において、Greedy decodeが貧弱な測定手法となり得ること」** という評価論・測定論上の貢献（Sequence-likelihood protocol）に純化。 |
| **N3 (機構)** | 「Distributed remappingの同定」という過剰主張リスク | **「単一コンポーネントや出力ヘッド（`lm_head`）単独説を反証（Negative result）し、分散的結合変化と整合する」** という慎重かつ堅固な定式化に変更。 |
| **N4 (課題間波及)** | 心理学的「Mood congruency」との等価視リスク | 心理学的過剰主張を排除し、**$\text{Steering strength} \times \text{valence direction} \times \text{stimulus ambiguity}$ の三要因因果交互作用**（曖昧刺激への選択的波及）に純化。 |
| **先行研究** | 既存のemotion steeringやpatching研究との重複懸念 | **最前線6系統の研究（Anthropic, Emotion Inference MI, Unified View, Asymmetric Valence, Latent Structure, Emotions Where Art Thou）** を明記し、先行知見と本研究の差分・境界条件を対比表（表6）として体系化。 |
| **標本規模** | $n=10$ pairの因果分析が過小評価されるリスク | **「Controlled Mechanistic Case Study」としての厳密統制の意義**を明記しつつ、Limitationにおいて $n=50\sim 100$ pairへの拡張計画とMulti-layer patching等のロードマップを率直に提示。 |

---

## 2. 論文原稿 ([`v3/docs/paper.md`](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper.md)) の主な変更箇所

### 2.1 タイトルとアブストラクトの刷新
- **タイトル**:  
  `Decodability Without Causal Substitutability: How Post-Training Decouples Affect-Relevant Representations from Constrained First-Person Reports in Language Models`
- **Abstract**:  
  論文のCore Propositionを明示し、4つの貢献（N1〜N4）を凝縮。

### 2.2 導入（Introduction）における公式4大貢献文の配置
査読推薦文をそのまま採用し、本文と完全に一致させました：
1. **Likelihood-based measurement (N1)**:
   > *We introduce a constrained sequence-likelihood protocol for measuring first-person affect self-report distributions beyond a single greedy completion, and show that greedy neutralization can coexist with stimulus-sensitive likelihood variation.*
2. **Post-training dissociation (N1/前提)**:
   > *Across paired base/instruct language models, we show that post-training can preserve linearly decodable affect-relevant information while altering its relationship to constrained first-person reporting behavior.*
3. **Predictive–causal dissociation (N2 - 主たる貢献)**:
   > *We demonstrate that a held-out linear alignment that restores cross-model affect prediction is insufficient to restore the corresponding first-person report distribution under pre-specified cross-model activation patching, despite positive within-model controls.*
4. **Cross-task causal influence (N4)**:
   > *We show that steering a valence-associated direction shifts third-person affect recognition judgments most strongly for ambiguous stimuli, with random and arousal-direction controls, indicating a task-crossing computational influence rather than a generic output offset.*

### 2.3 先行研究との正面対比表（表6）の拡充
Section 5（Related Work）において、以下の6系統を網羅した詳細対比表を導入しました：
- **Anthropic (2024/2026)** (*Emotion concepts and their function in an LLM*)
- **Emotion Inference MI (2025/2026)** (*MI of Emotion Inference in LLMs*)
- **Unified View (ACL 2026)** (*A Unified View on Emotion Representation in LLMs*)
- **Asymmetric Valence Processing (2026)**
- **Latent Structure (2026)** (*Latent Structure of Affective Representations in LLMs*)
- **Emotions, Where Art Thou (ICLR 2026)**

### 2.4 標本制約の率直な開示と採択強化ロードマップ
Section 6（Discussion & Limitations）において、現在の $n=10$ pairの制約を「Controlled Mechanistic Case Study」として位置づけつつ、ICLRでの評価を決定づける以下の3大施策を公式ロードマップとして明記しました：
1. **Strict Minimal-Pair Dataset の 50〜100 pair への拡張**
2. **Multi-layer Simultaneous Patching による分散性仮説の直接検証**
3. **共分散に基づく Mahalanobis 距離を用いた多変量 OOD 診断**

---

## 3. 保存・管理状況

- **論文正本**: [`v3/docs/paper.md`](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper.md)
- **タスクリスト**: [`docs/iclr_novelty_alignment_and_refinement/task.md`](file:///mnt/nas/home/hiromi/src/emo/docs/iclr_novelty_alignment_and_refinement/task.md)
- **実装計画書**: [`docs/iclr_novelty_alignment_and_refinement/implementation_plan.md`](file:///mnt/nas/home/hiromi/src/emo/docs/iclr_novelty_alignment_and_refinement/implementation_plan.md)
- **完了報告書**: [`docs/iclr_novelty_alignment_and_refinement/walkthrough.md`](file:///mnt/nas/home/hiromi/src/emo/docs/iclr_novelty_alignment_and_refinement/walkthrough.md)
