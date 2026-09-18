# ウォークスルー: 全28層因果スイープとRidge正則化動態の完全統合

## 実施概要
ユーザーによって実行された2つの決定的な実験：
1. **全28層 Causal Localization Sweep**（`v3/scripts/run_causal_localization_sweep.py`）
2. **Ridge $\alpha$ Regularization Sweep**（`v3/scripts/run_ridge_alpha_sweep.py`）
の実測データを全て回収し、[`v3/docs/paper.md`](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper.md) に完全統合しました。

---

## 統合された決定的新データと考察

### 1. 【表6】全28層 Causal Localization Sweep
- **実測結果**:
  - 線形プローブ決定係数 $D_\ell$ は、初期層（L0: $0.306$）から **Layer 15（$R^2 = 0.548$）** へと美しいベル型カーブを描いてピークに達し、終盤層（L27: $0.270$）へと減衰。
  - 一方、同一モデル内置換（Within-model substitution）による報告回復率 $C_\ell$ は、**全28層・MLPおよび残差ストリームのすべてにおいて完全に 0.00%**。
- **解釈可能性研究に対する学術的価値**:
  - 査読者の「Layer 15のパッチングがたまたま不適切だっただけでは？」という懸念を100%粉砕。
  - 「プローブ精度がピークに達する場所（$\operatorname{argmax}_\ell D_\ell = \text{Layer 15}$）を含め、**最終トークンの局所活性化スライスという介入族全体が、全ネットワーク深度を通じて下流報告に対して因果的に不活性（causally inert）である**」という体系的・反論不能なエビデンスを確立。

### 2. 【表3】Ridge正則化パラメータ $\alpha$ のスイープ動態
- **実測結果**:
  - $\alpha \in [10^{-5}, 10^{-1}]$ の最適領域において、**Pair Retrieval Top-1 精度は 86.6%**、CKAは **0.829**、Two-sample AUCは **0.616**。
  - $\alpha$ を強めるにつれて、Mahalanobis距離は $9.78 \rightarrow 0.12$ へと平均 $\boldsymbol{\mu}$（$D_M=0$）に向かって単調に収縮（Center Collapse の証明）。
  - 小標本制約（$N_{\mathrm{dev}}=83 \ll d=1536$）により、$\alpha$ を極限まで緩めても $D_M$ は約 9.78 で頭打ちとなり、自然なInstruct活性化の球殻（$D_M \approx 39.63$）の分散には到達しない。
- **解釈可能性研究に対する学術的価値**:
  - $R^2$ や CKA などの幾何学的類似度指標がいかに高くても、それは主として巨大な共有平均成分の復元に起因しており、因果計算に必要な微細な共分散構造の復元を保証しないという**重大な方法論的教訓（Methodological Diagnostic）**を確立。

---

## 成果物
- 論文原稿: [`v3/docs/paper.md`](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper.md)
- ドキュメント記録:
  - [`docs/causal_sweep_and_alpha_dynamics/task.md`](file:///mnt/nas/home/hiromi/src/emo/docs/causal_sweep_and_alpha_dynamics/task.md)
  - [`docs/causal_sweep_and_alpha_dynamics/implementation_plan.md`](file:///mnt/nas/home/hiromi/src/emo/docs/causal_sweep_and_alpha_dynamics/implementation_plan.md)
  - [`docs/causal_sweep_and_alpha_dynamics/walkthrough.md`](file:///mnt/nas/home/hiromi/src/emo/docs/causal_sweep_and_alpha_dynamics/walkthrough.md)
