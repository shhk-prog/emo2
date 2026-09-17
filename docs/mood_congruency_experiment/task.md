# Task: LLMにおける「気分一致バイアス（Mood Congruency Bias）」の因果実証実験の設計と実装

## 背景と目的
先行研究では「LLMの内部表現から感情を抽出できるか（プロービング）」および「ステアリングにより自由生成テキストのトーンを操作できるか」が検証されてきた。しかし、「内部感情表現の操作（Induced Mood）が、モデルによる他者の客観的感情認識（Recognition）に因果的バイアスを及ぼすか？」という心理学的な**気分一致効果（Mood Congruency Effect）** の検証は世界でまだ行われていない。

本タスクでは、本研究基盤（`v2`, `v3`）のActivation SteeringおよびSequence Likelihood Protocolを拡張し、内部感情ベクトルの介入が他者感情推定に与える影響を厳密に測定する因果実験パイプラインを設計・実装する。

## 主な作業項目
1. **実験設計（Protocol & Hypotheses）**
   - 刺激セット（曖昧刺激・中立刺激・明確刺激）の選定
   - 客観認識プロンプト（Recognition）の標準化
   - Steering強度（$\alpha \in \{-3, -1.5, 0, 1.5, 3\}$）とコントロール条件（Random / Arousal）の定義
2. **スクリプト実装（Pipeline Implementation）**
   - `v3/scripts/run_mood_congruency_experiment.py` の新規作成
   - 尤度空間（$E[V_{\text{rec}}], E[A_{\text{rec}}]$）およびGreedy出力の同時記録
3. **分析・統計モデリング**
   - 混合効果モデルによる気分一致係数（$\beta_{\text{mood}}$）の推定
   - 曖昧刺激における効果量（感度）の検証
4. **論文（`v3/docs/paper.md`）への統合計画**
   - 新しい実験セクションまたは発展的検証としての位置づけ整理
