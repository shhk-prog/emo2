# 実装計画: 実験⑦ Unembedding / RMSNorm / lm_head Swap 詳細解説

## 目的
v2研究の実験⑦（Unembedding / RMSNorm Swap Analysis; Phase 7 / Table 5）について、以下の体系的構成で学術的・定量的に解説する。

## 検討項目
1. **リサーチクエスチョン（Locus of Neutralization）**:
   - 事後学習（post-training）に伴う自己報告の中立化は、語彙出力段（RMSNorm および Unembedding / lm_head）の重み更新によって生じているのか、それとも出力段直前の残差ストリーム（Final Residual State）の時点で既に決まっているのか。
2. **実験手法（8条件の完全交差スワップ設計）**:
   - 3要素（Final Residual, RMSNorm, Unembedding）× 2ソース（Base, Instruct）= 8条件（BBB, BBI, BIB, BII, III, IIB, IBB, IBI）。
   - 81通りの感情自己報告JSON候補（Valence 1〜9 × Arousal 1〜9）に対する厳密な同時尤度評価。
   - 評価指標：期待値 $E[V]$, $E[A]$, Wasserstein距離 $WD_V$, Jensen-Shannon Divergence (JSD), 2D Earth Mover's Distance (2D EMD)。
3. **定量的結果（Table 5 の網羅的提示）**:
   - Base残差群（B-条件）と Instruct残差群（I-条件）での分布指標の明確な二極化。
   - 残差ソース固定下のRMSNorm/Head置換の無視できる微小効果（$\Delta WD_V \approx 0.01$, $\Delta \text{2D EMD} \approx 0.02$, $\text{JSD} \le 2\times 10^{-5}$）。
   - 残差ソース切り替え時の劇的な分布シフト（$\Delta WD_V \approx 0.20$, $\Delta \text{2D EMD} \approx 0.26$, $\text{JSD} \approx 3.3\times 10^{-3}$）。
4. **実験⑥（Late-Residual Substitution）との統合的理論解釈**:
   - 出力層単独説の棄却と、Deep Distributed Routing（深層分散ルーティング）仮説の確立。
