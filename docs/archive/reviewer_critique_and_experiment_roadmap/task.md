# タスクリスト: 査読フィードバックへの対応と実験ロードマップの策定

- [x] 1. 査読クリティークの構造化と致命的脆弱性の特定 <!-- id: 0 -->
  - 問題1: Within-model positive controlの欠落（MLPパッチング自体の因果的十分性の未証明）
  - 問題2: Ridge alignmentのfull-state再構築度（$d=1536, N=124$ のunderdetermined問題と低ランクプローブ復元への偏り）
  - 問題3: Mahalanobis距離・Cosineの解釈（$\chi^2_{1536}$ と empirical reference distribution / anisotropy control）
  - 問題4: 1モデルペア依存（Qwen2.5-1.5Bのみ）
  - 問題5: C1の独自性過大評価（Martorell既報）および C4（L16 random control逆転）の脆弱性
  - 問題6: Singh et al. (2026) に対応した「first-person report」の非内省的定義
- [x] 2. 優先度S〜Aの追加実験設計（最小追加セット: Exp A, B, C, D）の具体化 <!-- id: 1 -->
  - Exp A: Within-model positive causal control (Instruct Peak -> Instruct Neutral, MLP vs Residual stream vs All-token)
  - Exp B: Full-representation alignment fidelity ($R^2_{\text{activation}}$, CKA, PWCCA, pair retrieval)
  - Exp C: Empirical manifold test (Natural Instruct vs Aligned vs Raw の $D_M$ percentile / two-sample classifier AUC)
  - Exp D: Second model family (Llama-3-8B or Gemma-2-2B でのC2再現)
- [ ] 3. 現状の草稿 (`v3/docs/paper.md`) の即時修正計画（防御的書き換え・クレーム格下げ） <!-- id: 2 -->
  - C1を主貢献からMeasurement setupへ格下げ
  - C4を主結果から除外またはAppendixへ落とす準備
  - Singh et al. (2026) への防御的注釈（内省能力を主張しない旨の明記）
  - EMD Recoveryの分母問題への言及と代替指標の併記計画
- [x] 4. プロジェクト `docs/reviewer_critique_and_experiment_roadmap/` への保存 <!-- id: 3 -->
