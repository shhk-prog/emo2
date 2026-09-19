# 現行設計に合わせた論文構成（正本は README）

`iclr2027/iclr2027_conference.tex` はテンプレート、`iclr2027_conference2.tex` は旧稿である。旧 4 family（Mistral を Primary 扱い）、旧結果、「Arousal の劇的増幅」、Behavioral を V1 と呼ぶ構造は使わない。
再実行前に結果を本文へ書き込まない。構成と用語だけを現行コードに揃える。

## 提案タイトル

大規模言語モデルにおける感情表現から自己報告への因果的利用過程
(From Internal Affect Representations to Self-Report: Causal Utilization in Large Language Models)

## 論文の中心主張 (Central Thesis)

> **LLM self-reports cannot be characterized by output-level covariation alone, nor as a uniform direct readout of all decodable affect-relevant information.**
> **LLM self-reports are systematically related to affect-relevant internal representations, but this relationship is partial and task-dependent, is reorganized in association with post-training, and becomes causally consequential only at particular stages of computation.**

証拠の流れ:
$$\text{Covariation} \rightarrow \text{Representation / Causal Overlap} \rightarrow \text{Post-training-Associated Reorganization} \rightarrow \text{Localized Causal Leverage}$$

## 本文（ICLR 9ページ想定）

1. **Introduction**
2. **Related Work**
3. **Unified Experimental Framework**
   - 操作定義: Reader = recognition task、Self = self-report / reactivity task
   - 共感概念との安易な一対一対応は行わない（Discussion に留める）
   - 基本仮説モデル（結論を先取りせず、並列する情報処理経路と部分共有度を検証する構造）:

```text
Stimulus
  ↓
Affect-relevant internal states
  ├→ Reader computation
  └→ Self-report computation
```
   - 候補空間の規約: Behavioral/V1 = 729 VAD、V2/V3 = 81 VA。
     - **重要**: Stage 間で絶対的な $E[V], E[A]$ を直接比較してはならない。
     - 補足として、同一刺激サブセットにおいて 729 候補と 81 候補で主要傾向（相関・変位の方向性）が保たれる感度分析（Sensitivity Analysis）を明記。
   - Primary コホート: Qwen 2.5 1.5B / Llama 3.2 1B / Gemma 3 1B / OLMo 2 1B (Base & Instruct)
   - Supplementary: Mistral 7B v0.3
4. **Behavioral Characterization of Reader–Self Covariation**
   - 問: *Do Reader and Self covary in their responses to controlled affective changes?*
   - Primary: AIPsy matched manipulation における $\Delta\text{Reader}$ vs $\Delta\text{Self}$ の変位共変（$\Delta$ coupling）
   - Secondary: EmoBank における出力レベルの相関（外部参照対応）
5. **Internal Representation and Causal Sharing**
   - 問: *To what extent do Reader and Self share representational and causal structure internally?*
   - （※旧称 "in Base Models" は削除。Base/Instruct 双方を扱い、事後学習による差の本格検証は §6 に渡す）
   - Decodability, Representational Overlap, Causal Mediation / Intervention / Specialization (E6)
6. **Post-training-Associated Reorganization**
   - 問: *How does post-training associate with the reorganization of affect-relevant computations?*
   - Direct cross-decoding, Procrustes/Ridge alignment, Component patching, Output-head vs Residual 分離
   - 単なる「Instruct で出力が変わった」ではなく、representation-use relationship の再編として実証
7. **From Representation to Causal Utilization**
   - 問: *Where and when does affect-relevant information exert measurable causal leverage over self-report?*
   - 「デコード可能（decodable）であること」と「自己報告生成に因果的に使われること」を明確に分離
   - AIPsy matched-neutral に基づく厳格な因果介入（RQ1 十分性・用量反応性・直交空間除去）
   - 時空間 4-Map 解析（RQ2）: `pre_V` を candidate-independent な Primary ステージとし、`pre_A` 以降を canonical neutral teacher-forced trajectory として位置づけ
   - 統合 Confirmatory Replication: Discovery RQ2 と完全に統一された local direction 推定・介入プロトコルによる独立再現
8. **Discussion and Limitations**
   - 4つの証拠レベルの統合的解釈
   - 認知的共感／情動的共感との概念的接点と限界
   - 測定上の制約

## Appendix

モデル別詳細表、全層曲線、Phase B の rule-based perturbation 詳細、E6 専門化分析、Scale validation、729 VAD / 81 VA 感度分析。

## 用語規約

- 厳禁: 「モデルが感情を経験する」「モデルが共感を持つ」「認知的共感 = Reader」「情動的共感 = Self」
- 推奨: 「自己報告情動状態 (self-reported affective state)」「刺激誘発性情動反応 (elicited affective response)」「部分的に共有された情動関連内部表現 (partially shared affect-relevant representation)」
