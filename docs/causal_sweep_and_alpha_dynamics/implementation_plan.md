# 実装計画: 全28層因果スイープとRidge正則化動態の完全統合

## 概要
ユーザーによって実行された2つの決定的な実験データ：
1. **全28層 Causal Localization Sweep**（Layer 0〜27 のプローブ精度 $D_\ell$ vs 因果回復率 $C_\ell$）
2. **Ridge $\alpha$ Regularization Sweep**（$\alpha \in [10^{-5}, 10^4]$ における幾何・多様体動態）
を論文ドラフト（`v3/docs/paper.md`）に統合します。

## 主な追加・改訂内容

### 1. 全28層因果スイープの追加（結果 4.6 & 表6）
- **驚異的な発見**:
  - 線形デコード精度は Layer 0 ($R^2=0.306$) から Layer 15 ($R^2=0.548$) に向けて綺麗なベル型カーブを描き、中盤層で極大化する。
  - しかし、同一モデル内因果置換（Within-model substitution）による報告回復率は、**全28層のMLPおよび残差ストリームのすべてにおいて完全に 0.00%**。
  - これは「Layer 15のパッチングがたまたま効かなかった」という局所的な反論を完全に粉砕し、**「最終トークンの局所活性化スライスという介入族全体が、全ネットワーク深度にわたって下流報告に対して因果的十分性を欠いている（Causally Insufficient across all depths）」**という体系的・普遍的エビデンスを提供する。

### 2. Ridge $\alpha$ スイープの追加（結果 4.3 & 表3）
- **数理的動態の可視化**:
  - $\alpha=10^{-5}$ から $10^4$ への変化に伴い、$D_M$ は $9.78 \rightarrow 0.12$ へと平均 $\boldsymbol{\mu}$ に向かって単調に収縮（Center Collapse の証明）。
  - 最適領域 $\alpha \le 0.1$ において、**Pair Retrieval Top-1 精度は 86.6%**、CKAは $0.829$、Two-sample AUCは $0.616$ に達する。
  - $N_{\mathrm{dev}}=83 \ll d=1536$ の小標本制約下では、正則化を極限まで弱めても出力テンソルの分散が自然な薄殻（$D_M \approx 39.63$）に達しないという高次元収縮現象（dimensional shrinkage）を手法論的診断として確定。

### 3. 考察（Discussion）の深化
- 「プローブのピーク層（L15）と因果的効果（全層ゼロ）の完全な乖離」
- 「高次元正則化回帰における Center Collapse と表現類似度指標（CKA/R²）の限界」

## ドキュメント保存先
- `docs/causal_sweep_and_alpha_dynamics/`
  - `task.md`
  - `implementation_plan.md`
  - `walkthrough.md`
