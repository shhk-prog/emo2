# タスクリスト: 全28層因果スイープとRidge正則化動態の完全統合

- [x] 1. 全28層 Causal Localization Sweep 実測データの整理 <!-- id: 0 -->
  - 全28層のプローブ精度（L15でピーク $R^2=0.548$）と因果回復率（全層一貫して 0.00%）
  - $\operatorname{argmax}_\ell D_\ell = \text{L15}$ と $\forall \ell, C_\ell = 0.00\%$ の完全な解離
- [x] 2. Ridge $\alpha$ スイープ実測データの整理 <!-- id: 1 -->
  - $\alpha \in [10^{-5}, 10^4]$ における $R^2$, CKA, Top-1 Retrieval (86.6%), $D_M$ (9.78 -> 0.12), AUC (0.616 -> 0.796) の数理的推移
  - 小標本高次元写像における分散縮小（Shrinkage / Center Collapse）の実証
- [x] 3. `v3/docs/paper.md` の全面更新（新表3・新表6の追加と全28層データの網羅） <!-- id: 2 -->
- [x] 4. ドキュメント保存（`task.md`, `implementation_plan.md`, `walkthrough.md`） <!-- id: 3 -->
