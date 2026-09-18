# 実装計画: 実験⑥ Late-Residual Substitution 詳細解説

## 目的
v2研究の実験⑥（Late-Residual Substitution under Strict Identical Prompts; Phase 8 / Table 4）について、以下の観点を網羅した詳細な学術的報告を作成する。

## 検討項目
1. **問題設定と仮説（Why Late-Residual Substitution?）**:
   - 実験⑤（Activation Patching）で見出された中間層コンポーネント（`mlp_10` 等）の因果効果が、「最終層への直接読み出し（Direct Readout Shortcut）」なのか、「後続の中間層を経由した分散処理（Distributed Downstream Processing）」なのかを弁別する。
2. **実験手法と厳密なプロトコル（Protocol & Formulation）**:
   - 厳密な同一プロンプト（Strict Identical Prompts / Value Forcing）プロトコル。
   - 介入の数式：$h_{26}^{(\mathrm{patched})} = h_{26}^{(\mathrm{clean})} - c_l^{(\mathrm{Inst})} + c_l^{(\mathrm{Base})}$
   - Layer 27 直前入力（$h_{26}$）代入と Unembedding 直前代入の比較。
   - Random Source Control と 特異的効果（$\Delta_{\mathrm{specific}}$）の検証（10,000回ブートストラップ 95% CI）。
3. **定量的結果（Empirical Results: Table 4）**:
   - `attn_14`, `mlp_10`, `mlp_15` の $\Delta E[V]$, $\Delta WD_V$, $\Delta_{\mathrm{specific}}$ の完全な数値。
   - 中間層パッチング（$\Delta WD_V \approx -0.187$）との劇的な対比（$\Delta WD_V \approx 0$）。
4. **科学的インプリケーション（Scientific Implications）**:
   - Direct Readout 仮説の棄却。
   - 後続中間層ネットワーク依存性の確立。
   - Instructモデルにおける中立化メカニズムの局在に対する洞察。
