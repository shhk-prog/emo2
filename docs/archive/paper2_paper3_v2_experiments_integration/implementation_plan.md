# 実装計画: paper2 および paper3 への v2 実験②・③・⑤詳細結果の統合

## 1. 概要
`v3/docs/paper2.md` および `v3/docs/paper3.md` の Appendix D（v2 実験群のアーカイブ）および本文において、これまで要約にとどまっていた v2 実験②（RSA & 統制回帰）、実験③（Steering Slope & Negative Result）、実験⑤（Phase 5 全層パッチングスクリーニング & Output-Gating）の詳細な実験プロトコル、実測データ、テーブル、科学的示唆を全面的に拡充・統合する。

## 2. 反映内容の具体設計

### 2.1 実験②（三空間RSA & 統制回帰分析）の反映箇所：Appendix D.1
- **三空間幾何距離行列**:
  - Human Ground Truth ($D^H$), Model Internal ($D^I$), Model Self-Report ($D^S$)
- **全層RSA指標の実測値**:
  - $\mathrm{RSA}_{H,S} = 0.124$（自己報告が中立 (5,5) 付近に強く拘束されていることの証拠）
  - $\mathrm{RSA}_{I,S}$: 中間層（Layer 10 で $-0.176$、Layer 18 で $-0.197$）において顕著な**負の相関（幾何構造のねじれ・解離）**を示すこと。
- **統制重回帰分析（Controlled Regression）**:
  - $E_V \sim z_V + V_H + \text{word\_count}$
  - 内部表現 $z_V$ の寄与は非有意（$\beta = -0.0113, p = 0.676$）
  - 表層感情辞書スコア $V_H$ のみ有意（$\beta = +0.2933, p = 0.003$）
  - 内部感情表現と自己報告の因果的切断（**System Dissociation**）の決定的証拠として明記。

### 2.2 実験③（活性化ステアリング & Steering Slope）の反映箇所：Appendix D.3
- **介入プロトコル**:
  - Layer 20, Layer 24, Layer 27 の残差ストリームにおける Valence 対立方向 $d_V$ への介入（$\alpha \in \{-3.0, -1.5, 0.0, +1.5, +3.0\}$）
- **実測挙動（Steering Slope）**:
  - Layer 20/24: 傾きはほぼ平坦（$E_V \approx 5.2 \to 5.5$）で単調な自己報告制御は不可。Instructでは分散（95% CI: 4.65〜6.40）のみが増大。
  - Layer 27: $\alpha = 0$ で $E_V \approx 6.28$ へ局所的に跳ね上がり、正の過大介入で出力崩壊・フォーマット破綻が発生。
- **学術的示唆**:
  - **Decodability $\neq$ Steerability**: 内部で線形プローブが高精度であっても、単一線形軸のステアリングで出力を単調制御できるとは限らない。
  - 単一の蛇口（Readout Gain）が絞られているだけではないことの証明。

### 2.3 実験⑤（Phase 5 パッチングスクリーニング & Output-Gating）の反映箇所：Appendix D.5 & D.7
- **全層54コンポーネント網羅スクリーニング（Phase 5）**:
  - Layer 10〜27 の全18層 × 3モジュール（Res, Attn, MLP）のパッチング結果テーブルを記載。
  - 最大効果コンポーネント: Layer 10 `res` ($\Delta V = -0.108$), Layer 10 `mlp` ($\Delta V = -0.091, \Delta A = -0.081$), Layer 15/16 `res` ($\Delta V = -0.094$)。
  - 後期層（Layer 19〜27）では単一パッチングの効果が急速に消失（Layer 27 では $\Delta V = 0.000$）。
  - **単一ボトルネック（Single Gating）の明確な否定**: 単一モジュールでは Base 型への劇的回復は起きず、複数の中間層コンポーネントが加算的に分散して中立出力を形成（H4: Distributed Remapping）していることを実証。
- **Output-Gating検証（実験5）**:
  - 自然言語形式（`My valence is X...`）でも自己報告期待値は中立（$E_V \approx 5.0$）に留まり、JSONフォーマット特異的バイアスではないことを明記。

## 3. 編集対象ファイル
- `v3/docs/paper3.md`
- `v3/docs/paper2.md`
