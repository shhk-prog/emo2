# 実装計画: paper3.md 総合改訂（不整合解消と厳密化）

## 概要
ユーザーからの査読・整合性レビューに基づき、11項目の重要不整合を完全に解消します。
「第II部 統合再編」を削除し、そこに存在した有益な統合表（Table 1, Table 2, Table 3）を用語・数値を厳密に修正した上で正式本文側へ配置し、全体の再現性と学術的一貫性を担保します。

## 修正計画

### 1. Related Work の修正 (Points 10, 11)
- `Hase et al. (2024)` → `Hase et al. (2023)` (NeurIPS 2023 "Does Localization Inform Editing?")
- "A Unified View on Emotion Representation in Large Language Models" を `Maheswaran and Desarkar (2026)` (EACL 2026) として正式引用
- 既存感情研究に対する全称命題を緩和：「Several recent studies primarily characterize emotion representations through probing or representation similarity; systematic matched-substitution localization across layer, component, and generation time remains comparatively underexplored.」

### 2. 用語・測定定義の厳密化 (Points 2, 7, 8, 9)
- **距離尺度**: 「Wasserstein-2」を完全に排除し、「2D Joint Optimal Transport cost with Manhattan ground cost」に統一
- **実験位置づけ**: 「確証的」「バイアスのない」を排除し、「Exploratory full-layer screen」「Focused full-cohort effect-size evaluation」へ統一
- **因果レバーの表現**: 「因果レバー皆無」等の誇張を避け、「tested matched-substitution intervention under the evaluated token position showed little local causal recovery」「今回検証した局所介入では強い局所因果レバレッジを示さなかった」に統一
- **プローブ必要性の表現**: 「FDRで棄却された」等の誤用を避け、「No evidence for probe-aligned local necessity was found.」に統一

### 3. 数値・実験条件の統一 (Points 3, 5, 6)
- **プローブ必要性のランダム方向数**:
  - 全層スクリーニング (全84サイト): 20 probe-orthogonal random directions ($N=20$)
  - 代表層検証 (代表5層・15サイト): 100 random directions ($N=100$)
- **多層Residualパッチング**:
  - 正本 `generation_multilayer_residual_results.csv` の条件・数値に完全統一：
    L18 (43.51%), L20 (51.85%), L24 (55.08%), L20+L24 (55.67%), L18+L20+L24 (55.74%), L18–24 (55.35%)
  - 不整合な別条件（L22+24等）は完全削除
- **Llama-3.2-1B-Instruct 追試**:
  - 「最大1.82%」を排除し、「Attention最大 L4 +0.73%」「MLP最大 L15 -0.36%」「Residual全層負（L15 -4.23%）」に統一
  - 結論を「Qwenで観察されたlate-residual localizationは、今回の初期Llama partial replicationでは再現されなかった」に留める

### 4. 正式本文への表の統合 (Points 1, 4)
- **Table 1 (実験設計・データセット諸元)**: 「4. モデルとデータ」に挿入（用語修正適用）
- **Table 2 (代表サイト結果サマリー)**: 「8.6 Direct Peak-Site Contrast」に挿入（数値・CI・用語修正適用）
- **Table 3 (主張とエビデンス総括表)**: 「10. Discussion」または「12. Conclusion」に挿入
- **Generation-time 表の分離 (Point 4)**:
  - Table A2a: Exploratory full-layer generation-time sweep (全28層, Stage 1, 15 evaluated / 13 valid)
  - Table A2b: Focused full-cohort generation-time evaluation (代表6層, Stage 2, 39 complete pairs)
- **「第II部 統合再編」の削除**: 1950行以降の重複・不整合部分を完全削除

## 検証手順
- `grep_search` で「Wasserstein-2」「50本」「因果レバー皆無」「確証的」「L22+L24」「1.82%」等の不整合文字列がゼロであることを確認。
- 各テーブルの数値が正本CSVと完全に合致していることを検証。
