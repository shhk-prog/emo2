# 実験⑥ Late-Residual Substitution 詳細解説 Walkthrough

## 1. 実験の概要とリサーチクエスチョン
- **実験名**: Late-Residual Substitution under Strict Identical Prompts（厳密な同一プロンプト下での後期残差代入実験、Phase 8 / Table 4）
- **リサーチクエスチョン**:
  実験⑤（Activation Patching）で判明した中間層の主要コンポーネント（特に `mlp_10` や `attn_14`）の感情信号は、**「残差ストリーム（Residual Stream）を通じて直接最終層・Unembeddingに読み出されている（Direct Readout）」** のか？
  それとも、**「後続の中間層ネットワーク（Downstream intermediate layers）を経由して複雑に変形・統合されて初めて自己報告に影響している（Distributed Downstream Processing）」** のか？

## 2. 介入設計とプロトコル
- **プロンプト**: Value Forcing プロトコル（`{\n  "valence": ` でプロンプトを終え、直後の1トークンが数字 1〜9 となるよう統一）。
- **数式設計**:
  ソースコンポーネント $c$（$l \in \{10, 14, 15\}$）の Instruct 活性化 $c_l^{(\mathrm{Inst})}$ を Base 活性化 $c_l^{(\mathrm{Base})}$ で置換した差分を、最終層直前（Layer 27 入力 $h_{26}$）に直接注入：
  $$h_{26}^{(\mathrm{patched})} = h_{26}^{(\mathrm{clean})} - c_l^{(\mathrm{Inst})} + c_l^{(\mathrm{Base})}$$
- **比較対照（コントロール）**:
  - **Matched Base**: 当該テキストと同一サンプルの Base コンポーネント寄与。
  - **Random Source**: 別サンプルの Base コンポーネント寄与をランダムに代入。
  - **特異的効果**: $\Delta_{\mathrm{specific}} = \Delta WD_{\mathrm{matched}} - \Delta WD_{\mathrm{random}}$（ペア単位 10,000回クラスタ・ブートストラップによる 95% CI）。

## 3. 定量結果まとめ（Table 4）
| Source Component | $\Delta E[V]$ [95% CI] | $\Delta WD_V$ [95% CI] | $\Delta_{\mathrm{specific}}$ vs random [95% CI] |
|:---|---:|---:|---:|
| `attn_14` matched Base | +0.0051 [0.0031, 0.0073] | +0.0013 [-0.0007, 0.0032] | -0.0014 [-0.0029, 0.0003] |
| `attn_14` random source | +0.0064 [0.0042, 0.0089] | +0.0009 [-0.0017, 0.0031] | — |
| `mlp_10` matched Base | -0.0089 [-0.0123, -0.0056] | -0.0023 [-0.0054, 0.0011] | +0.0010 [-0.0017, 0.0038] |
| `mlp_10` random source | -0.0099 [-0.0139, -0.0061] | -0.0006 [-0.0042, 0.0032] | — |
| `mlp_15` matched Base | +0.0131 [0.0080, 0.0180] | +0.0018 [-0.0038, 0.0068] | -0.0012 [-0.0041, 0.0017] |
| `mlp_15` random source | +0.0143 [0.0084, 0.0202] | +0.0037 [-0.0021, 0.0087] | — |

## 4. 科学的結論
1. **Direct Readout Shortcut 仮説の完全な棄却**:
   中間層コンポーネントの寄与を後期残差ストリーム（$h_{26}$）に直接注入しても、Base型自己報告分布への回復は生じない（$\Delta WD_V \approx 0$）。また、Matched と Random Source の間に有意差は認められない（$\Delta_{\mathrm{specific}} \approx 0$）。
2. **分散的な後続中間層依存性（Distributed Downstream Processing）の確定**:
   実験⑤で見られた劇的な Base 型分布回復（例: `mlp_10` で $\Delta WD_V \approx -0.187$）は、中間層から最終層への直接的な線形加算によるものではなく、**Layer 10 以降の後続ブロック群（Layer 11〜26）を経由した複雑なカスケード処理** を通じて初めて自己報告に波及している。
