# タスクリスト: 査読耐性を極大化する精密リフレーミングと全層因果スイープの設計

- [x] 1. 論文ドラフト (`v3/docs/paper.md`) の厳密なリフレーミング <!-- id: 0 -->
  - タイトル修正: `Decodability Without Causal Sufficiency: A Case Study of Affect-Relevant Representations in a Paired Base/Instruct Language Model` (単一ペアの誠実な明記)
  - 先行研究（Amnesic Probing, Elazar et al., 2021等）の明示的引用と差別化（「probe $\neq$ behavioral use」の再発明批判を先回り封殺）
  - Base/Instruct比較の位置づけ変更: 因果帰属ではなく「表現アライメントのストレステスト (stress test of representational transfer)」へ
  - 過度な一般化の抑制: 「representation」ではなく「local activation slice / tested site」の causal insufficiency へ限定
  - Center Collapse の位置づけ変更: 主要新規貢献から「R²/CKAだけに頼る評価への手法論的診断 (Methodological diagnostic)」へ
  - N4（三人称ステアリング）を主本文から完全に除外し、Appendixへ退避
  - 自然-自然コサインの定義注記および mean-centered cosine の必要性の記載
  - 過剰表現の微修正（「直接的に実証する」→「provides direct evidence that」、「保持」→「remains strongly linearly decodable」、「完全に整合」→「consistent with tested interventions」）
- [x] 2. Sランク実験（全層Causal Localization Sweep & Decodability相関）のスクリプト実装 <!-- id: 1 -->
  - 全層（Layer 0〜27） $\times$ コンポーネント（mlp, resid）の自動因果スイープ `v3/scripts/run_causal_localization_sweep.py`
  - プローブ精度 $D_\ell$ と因果回復率 $C_\ell$ の層別相関 $\rho(D_\ell, C_\ell)$
- [x] 3. Ridge $\alpha$ sweep スクリプトの実装 <!-- id: 2 -->
  - 正則化強度と高次元再構築度・中心凝縮（$D_M$）・判別AUCの推移解析 `v3/scripts/run_ridge_alpha_sweep.py`
- [x] 4. ドキュメント保存（`task.md`, `implementation_plan.md`, `walkthrough.md`） <!-- id: 3 -->
