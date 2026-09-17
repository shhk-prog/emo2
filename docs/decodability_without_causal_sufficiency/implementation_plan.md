# 実装計画: 論文の中心命題転換と実測データの再解釈

## 概要
ユーザーの指摘に基づき、本論文の中心命題を無理に「post-trainingによるdecoupling」へ押し込むことを止め、実測データに完全に忠実な **「Decodability Without Causal Sufficiency: Predictive Representations Need Not Control Downstream Reports in Language Models」** へとパラダイムシフトを行います。
この修正により、査読者からの「within-modelでも効かないのにpost-trainingのせいとは言えない」「$D_M=4.98$ は自然多様体ではなく中心凝縮である」という致命的な批判を完全に先回りし、解釈可能性研究に対する強固な警鐘論文へと昇華させます。

## 改訂方針

### 1. タイトルと核となる主張の変更
- **タイトル**: `Decodability Without Causal Sufficiency: A Case Study of Affect-Relevant Representations Across Base and Post-Trained Language Models`
- **Core Proposition**:
  > A representation may be linearly decodable with high precision, achieve high cross-model predictive alignment ($R^2_{\mathrm{activation}} \approx 0.50$, $\mathrm{CKA} \approx 0.82$), and substantially reduce raw distribution shift, yet still fail to exert causal control over a downstream policy-constrained report. Crucially, within-model substitution controls reveal that the very site from which the feature is decodable lacks causal sufficiency for the downstream report (recovery $\approx 0.00\%$), demonstrating that probe-accessible locations do not necessarily identify the causal bottlenecks of downstream behavior.

### 2. Within-model実験の再定義
- 「Positive control（正の対照）」という表現を完全に排除。
- **「Within-model causal sufficiency test」** または **「Within-model substitution control」** と改称。
- 「Instruct内部ですら、Peak刺激の自然活性化をパッチしてもNeutral報告が動かない（Recovery 0.00%）」という結果を、プロービング部位の因果的不十分性を暴く決定的な証拠として位置づける。

### 3. 多様体診断（$D_M$ / Cosine / AUC）の解釈の厳密化
- 高次元ガウス分布におけるデータ集中（半径 $\sqrt{d}$ の薄殻 / Annulus theorem）を踏まえ、自然なInstruct（$D_M \approx 39.66 \approx \sqrt{1536}$）に対し、Aligned Base（$D_M \approx 4.98$）は「自然多様体に入った」のではなく、**「Ridge正則化による平均 $\boldsymbol{\mu}$ への中心凝縮（Center Collapse / Over-regularization）」**であることを理論的・数学的に明記。
- Two-sample AUC = 0.6386 は「ほぼ識別不能」ではなく、「極端なシフトは解消されたが、依然として統計的に非典型的（atypical）」として正確に記述。
- Cosine（0.9950）も、入力ごとの微細な差異よりも巨大な共有平均成分が支配的に再構築されている可能性として考察。

### 4. ドキュメント保存先
- `docs/decodability_without_causal_sufficiency/`
  - `task.md`
  - `implementation_plan.md`
  - `walkthrough.md`
