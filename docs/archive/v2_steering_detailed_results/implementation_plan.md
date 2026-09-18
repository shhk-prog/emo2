# 実装計画: v2におけるSteering（活性化ステアリング）詳細結果の体系的提示

## 1. 概要
v2で実施された「実験③ Steering SlopeによるReadout感度の因果検証」について、実験プロトコル、対象層（Layer 20, 24, 27）、介入強度（$\alpha \in [-3.0, +3.0]$）、得られたプロット・数値結果、および最終的な科学的解釈（初期の仮説と、厳密統制下でのNegative Resultの重要性）を整理する。

## 2. 検討項目と構成

### 2.1 実験設計と介入プロトコル
- **目的**: 内部の感情方向（Valence Contrastive Direction $d_V$）に直接介入した際、自己報告の期待値 $E[V]$ がどれだけ感度よく応答するか（Steering Slope: $\frac{\partial E[V]}{\partial \alpha}$）をBaseとInstructで比較し、Readout Suppression (H3) を因果的に検証する。
- **対象モデル**: `Qwen2.5-1.5B` (Base) vs `Qwen2.5-1.5B-Instruct`
- **介入層**: Layer 20, 24, 27（後期残差ブロック）
- **介入式**: $h_{l} \leftarrow h_{l} + \alpha \cdot \sigma_l \cdot d_V$ （$\alpha \in \{-3.0, -1.5, 0.0, +1.5, +3.0\}$）
- **対象トークン**: プロンプト最終トークン以降の全生成ポジションにフックを継続適用。

### 2.2 実測結果とプロット分析（Layer 20, 24, 27）
- **Layer 20**:
  - Base: $\alpha$ に対する傾きはほぼフラット（$E_V \approx 5.22 \to 5.19$）。
  - Instruct: $E_V \approx 5.52 \to 5.54$ とほぼフラットだが、分散（95% CI: 4.65〜6.40）が極めて大きい。
- **Layer 24**:
  - Base: 傾きは極小（$E_V \approx 5.20 \to 5.23$）。
  - Instruct: 傾きは微小な負の傾斜（$E_V \approx 5.57 \to 5.51$）だが、CIが大きく有意な単調増加なし。
- **Layer 27**:
  - Base: 微小な増加傾向（$E_V \approx 5.14 \to 5.28$）。
  - Instruct: $\alpha \le -1.5$ で $E_V \approx 5.51$ だが、$\alpha = 0$ で急激に $6.28$ へ跳ね上がり、正方向（$\alpha = 1.5, 3.0$）では出力の崩壊・フォーマット破綻が発生。

### 2.3 科学的結論とNegative Resultの意味（包括的レポートからの知見）
- **単調な因果的制御の不成立**:
  - 初期の単純な期待（線形に自己報告が回復する）に反し、品質維持領域（Quality-preserving range）で評価すると、単一の線形感情ベクトルへのステアリングではロバストかつ単調な自己報告の方向特異的制御は達成できなかった。
- **重要な洞察（Decodability $\neq$ Steerability）**:
  - 内部表現から感情情報が線形プローブで高精度に読める（Decodable）からといって、その線形方向を直接揺さぶれば出力が制御できる（Steerable）とは限らない。
  - これは、感情の読み出し機構が単純な1次元線形パスではなく、多層・多モジュールに分散した非線形なネットワーク（Distributed Remapping, H4）であることを補強する決定的な証拠となった。
