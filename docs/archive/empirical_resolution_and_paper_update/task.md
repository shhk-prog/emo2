# タスクリスト: 全実験の詳細実測値を網羅した論文草稿の改訂

- [x] 1. リポジトリ内の全実験結果データの収集と整理 <!-- id: 0 -->
  - 表1: 表層的Greedy崩壊と候補列尤度感応性の共存 (Instruct 98.6% vs Base, $r=0.629$ vs $0.365$)
  - 表2: 内部表現の線形デコード可能性と交絡要因統制 ($R^2$: Valence 0.58, Arousal 0.42, Token 0.05, Surface VAD 0.12, Narrative 0.08)
  - 表3: クロスモデル表現予測の回復度比較 (L12, L16, L20 の Direct vs Procrustes vs Ridge)
  - 表4: Base -> Instruct Ridgeアライメントの高次元再構築度詳細 ($R^2_{\mathrm{act}} = 0.4966$, CKA = 0.8236, Retrieval Top-1 = 75.61%)
  - 表5: 全層における幾何シフトおよびOOD診断詳細 (L11-18 の $D_M$, Cosine, Norm Ratio)
  - 表6: 自然なInstruct多様体の経験的参照分布とAligned Baseの比較 ($D_M$ 5th-95th %tile, AUC = 0.6386, Cosine controls)
  - 表7: 単一層および多層同時活性化パッチングにおける回復率詳細 (L15: 0.11%, L14-15: 0.07%, L13-16: 0.35%, L11-18: -0.33%)
  - 表8: 同一Instructモデル内におけるPeak -> Neutral パッチング詳細 (Within-model positive controls, 5プロトコルすべて 0.00%)
  - 表9: 三人称感情認識における活性化ステアリング効果 (L14, L16, L20 の $\beta$, SE, $t$, $p$, $\Delta V$)
- [x] 2. `v3/docs/paper.md` の全面改訂（全表・統計量・対照群の網羅） <!-- id: 1 -->
- [x] 3. ドキュメント保存とユーザーへの報告 <!-- id: 2 -->
