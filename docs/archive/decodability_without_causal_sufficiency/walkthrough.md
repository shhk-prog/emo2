# ウォークスルー: 論文の中心命題転換と実測データの再解釈

## 改訂の総括
ユーザーから提示された鋭い査読者視点（「Within-modelが0%である以上、post-trainingによるdecouplingとは言えない」「$D_M=4.98$ は自然多様体ではなく中心凝縮である」）に基づき、論文のタイトル、中心命題、および実測データの解釈を根本から再構築しました。

当初の「post-trainingによるdecoupling」という無理筋なストーリーを退け、実測データが直接示している**「Decodability Without Causal Sufficiency（線形デコード可能性は局所的な因果的十分性を意味しない）」**を真の中心命題に据えることで、査読者の批判の余地を完全に封じ込め、機械論的解釈可能性（Mechanistic Interpretability）分野において極めて説得力のある論文原稿へと昇華させました。

---

## 主な改訂ポイント

### 1. タイトルと中心命題の刷新
- **新タイトル**:  
  `Decodability Without Causal Sufficiency: A Case Study of Affect-Relevant Representations Across Base and Post-Trained Language Models`
- **中心命題（Core Proposition）**:
  > 線形プローブで感情価が高精度にデコードでき、モデル間で高次元活性化全体の分散の約50%（$R^2_{\mathrm{act}} = 0.4966$）、Linear CKA 0.8236でアライメント可能な中間表現であっても、同一モデル内における置換実験（Within-model substitution control）において下流報告を全く動かせない（回復率 0.00%）。  
  > したがって、「プローブで情報が読める場所」は「下流計算を司る因果的ボトルネック」を意味しない。

### 2. Within-model実験の再定義（Positive Control呼称の排除）
- 「Positive control」という表現を完全に排除し、**「Within-model causal sufficiency test」** または **「Within-model substitution control」** と改称。
- Instruct内部ですら、Peak刺激の自然な活性化をNeutral刺激へパッチしても報告が全く動かない（Recovery 0.00%）という結果を、プロービング部位の因果的不十分性を暴く決定的な証拠として正面から論理の主役に据えました。

### 3. 多様体診断（$D_M$ / Cosine / AUC）の数学的・幾何学的適正化
- **高次元ガウス薄殻定理（Annulus theorem）に基づく解釈**:
  - $d=1536$ 次元空間では、データは平均近傍ではなく半径 $\sqrt{d}$ の薄殻に集中する。
  - 自然なInstruct活性化は理論値 $\sqrt{1536} \approx 39.19$ と完全に一致する中央値 **39.66**（5th: 34.20, 95th: 51.43）に位置する。
  - Aligned Base活性化の $D_M = 4.98$ は「自然多様体に入った」のではなく、**Ridge正則化によって平均ベクトル $\boldsymbol{\mu}$ の近傍へ異常に中心凝縮（Center Collapse / Over-regularization）した非典型的な状態**であることを明記。
- **Two-sample AUC = 0.6386**:
  - 「ほぼ識別不能」という表現を退け、「Raw Baseの完全OOD（AUC 1.000）から大幅に改善したものの、自然な分布とは依然として統計的に非典型的（atypical）」と厳密に記述。
- **Cosine類似度（0.9950）**:
  - 共有平均成分の支配的な復元を反映している可能性を考察。

---

## 成果物
- 論文草稿: [`v3/docs/paper.md`](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper.md)
- ドキュメント記録:
  - [`docs/decodability_without_causal_sufficiency/task.md`](file:///mnt/nas/home/hiromi/src/emo/docs/decodability_without_causal_sufficiency/task.md)
  - [`docs/decodability_without_causal_sufficiency/implementation_plan.md`](file:///mnt/nas/home/hiromi/src/emo/docs/decodability_without_causal_sufficiency/implementation_plan.md)
  - [`docs/decodability_without_causal_sufficiency/walkthrough.md`](file:///mnt/nas/home/hiromi/src/emo/docs/decodability_without_causal_sufficiency/walkthrough.md)
