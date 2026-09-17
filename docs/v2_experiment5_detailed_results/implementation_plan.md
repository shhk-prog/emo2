# 実装計画: v2における実験⑤詳細結果の体系的提示

## 1. 概要
v2において「実験5」に対応する2つの主要な分析（①Phase 5: 全層コンポーネント・パッチング・スクリーニング、②研究計画書上の実験5: Output-Gating仮説の検証）について、その目的、実験設定、実測数値、および結論を網羅的にまとめる。

## 2. 検討項目と構成

### 2.1 主軸：Phase 5「全層コンポーネント・パッチング・スクリーニング」
- **目的**: Instructモデルに対して、Baseモデルの各層・各モジュール（residual, attention, mlp）の活性化をパッチ（上書き）し、どのコンポーネントが自己報告をBase型へ回復させる因果的影響を持つかを網羅的（Layer 10〜27, 計54コンポーネント）にスクリーニングする。
- **データソース**: `v2/results/derived/phase5_screening/patching_screening_summary.csv`
- **主要な指標**:
  - `mean_delta_v` ($\Delta V$): パッチによるValence期待値の変化量
  - `mean_delta_a` ($\Delta A$): パッチによるArousal期待値の変化量
  - `mean_entropy`: パッチ後の自己報告分布のエントロピー
  - `mean_p55`: 中立値 $(5, 5)$ の生起確率
- **上位コンポーネントのランキングと層別傾向**:
  - 最大効果コンポーネント: Layer 10 `res` ($\Delta V = -0.108$), Layer 15 `res` ($\Delta V = -0.094$), Layer 16 `res` ($\Delta V = -0.094$), Layer 10 `mlp` ($\Delta V = -0.091$)
  - 逆方向コンポーネント: Layer 11 `mlp` ($\Delta V = +0.026$), Layer 14 `attn` ($\Delta V = +0.021$)
  - 後期層（Layer 19〜27）: ほぼ $\Delta V \approx 0.000$ となり、単一パッチングの効果が消失。
- **科学的結論**:
  - 単一のコンポーネントだけで劇的に Base 状態へ戻る（$\Delta V = -1.0$ のような完全回復）ものは存在しない。
  - これは「単一の感情抑制ボトルネック（H3: Single Gating）」を明確に否定し、「複数の中間層コンポーネントが分散して関与している（H4: Distributed Remapping）」ことを証明した。

### 2.2 補足：研究案.md上の実験5「Output-Gating仮説の検証」
- **目的**: 自己報告の中立化が「JSONフォーマットの学習バイアス」によるものかを検証。
- **結果**: プロンプトを「JSON」から「自然言語 (`My valence is X and arousal is Y`)」に変更しても、Instructモデルの分布は中立（5付近）のまま維持された。単なるフォーマット依存ではない深いマッピング変化であることを確認。

## 3. ドキュメント保存
- `docs/v2_experiment5_detailed_results/` 配下に保存。
