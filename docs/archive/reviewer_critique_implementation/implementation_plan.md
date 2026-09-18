# 実装計画: 査読指摘への即応実装

## 目的
査読者（Weak Reject / Borderline: 4-5/10）の指摘を受け、論文草稿の論理的弱点（C1の過大評価、C4のrandom control問題、内省の懸念、単一ペアの因果帰属）を即座に修正するとともに、最重要課題である追加実験（Exp A: Within-model positive control、Exp B: Full-state reconstruction、Exp C: Empirical manifold test）を実行可能なPythonスクリプトとして実装します。

## 実施内容

### 1. 論文ドラフト (`v3/docs/paper.md`) の防御的改訂
- **C1の位置づけ変更**: 新規手法としての過大主張を抑え、Martorell (2024) 等の先行研究に準拠した「表層のgreedy collapseを回避してモデルの制約付き報告分布を精密に測定するプロトコル」としてMethodに再配置。主要貢献リストからは除外またはMeasurement Setupとして明記。
- **C4の扱い**: Layer 16でrandom controlが大きく振れる（generic disruption）問題を直視し、主結果から除外してAppendixへ退避。論文の中心命題（Core Proposition）である「Decodability Without Causal Substitutability」に焦点を完全に絞る。
- **Singh et al. (2026) への準拠注記**: 「first-person report」は出力の文法およびプロンプト形式を指すものであり、モデルの内省（introspection）や主観的体験の証拠ではないことをConstruct SeparationおよびLimitationsに明記。
- **Post-training因果帰属の限定**: 単一ペア（Qwen2.5-1.5B）の比較であることから、「post-trainingが結合を変化させた」という強い主張を「Base/Instruct比較において観察される結合差異（consistent with post-training-associated changes）」へと厳密化。

### 2. 実験A（Within-model positive control）スクリプトの実装
- ファイル: `v3/scripts/run_within_model_positive_control.py`
- 目的: 同一Instructモデル内で、Peak刺激の活性化をNeutral刺激へパッチした場合に自己報告がPeak側へシフトするか（Positive Control）を検証。
- 介入部位の比較:
  - MLP output (layer 15, target_pos)
  - Residual stream (layer 15, target_pos)
  - All prompt tokens (layer 15, residual stream)
  - Multi-layer residual streams (layers 13-16)

### 3. 実験B & C（Full-State Reconstruction & Empirical Manifold Test）スクリプトの実装
- ファイル: `v3/scripts/analyze_alignment_fidelity_and_manifold.py`
- 指標:
  - $R^2_{\text{activation}}$ (全1536次元平均)
  - CKA (Centered Kernel Alignment)
  - Pair retrieval accuracy (Nearest Neighbor)
  - Natural Instruct活性化の $D_M$ 経験的分布（5%, median, 95%）とAligned Baseのパーセンタイル
  - Two-sample Linear Classifier (Natural vs Aligned) による判別AUC
  - Cosine類似度対照群（Matched vs Unmatched same-valence vs Unmatched diff-valence vs Natural-Natural）

## 成果物と保存先
- ドキュメント: `docs/reviewer_critique_implementation/`
  - `task.md`
  - `implementation_plan.md`
  - `walkthrough.md`
- スクリプト:
  - `v3/scripts/run_within_model_positive_control.py`
  - `v3/scripts/analyze_alignment_fidelity_and_manifold.py`
- 論文ドラフト:
  - `v3/docs/paper.md`
