# 実装計画: paper2およびpaper3へのSynergy Patching実測データの統合

## 1. 概要
`v3/docs/paper3.md` および `v3/docs/paper2.md` の Appendix D.5（Component-Wise Patching Screening and Synergy）において、Phase 6 Synergy Patchingの実測データ表（Table D.2）を新設し、線形加算性（Linear Additivity）および3コンポーネント同時介入（`10_mlp + 14_attn + 16_res`）の累積効果を明記する。

## 2. 追記する具体的内容

### Table D.2: Synergy Patching (Joint Component Intervention) Results
| 介入条件 (Intervention) | 実測変化量 ($\Delta V$) | 予測値 (Expected Sum) | シナジー差分 (Obs - Exp) | 挙動と理論的解釈 |
|:---|---:|---:|---:|:---|
| **`10_mlp` (単独)** | -0.0911 | — | — | 独立した負の因果効果 |
| **`14_attn` (単独)** | +0.0207 | — | — | 独立した正の因果効果 |
| **`10_mlp` + `14_attn` (同時実測)** | **-0.0561** | **-0.0704** | **+0.0143** | **線形加算性が概ね成立（わずかな減衰、非線形相殺なし）** |
| **`10_mlp` + `14_attn` + `16_res`** | **-0.1420** | **-0.1501** | **+0.0081** | **3コンポーネントの累積的線形加算を確認** |

### 理論的結論の補強
- 非線形相乗による劇的跳躍（単一ボトルネックの解除）は起きず、各コンポーネントが分離可能な線形重みを持って加算的に機能していること。
- これにより、事後学習に伴う自己報告の中立化が「分散的再写像（Distributed Remapping, H4）」によって支配されているという結論を決定的に補強する。

## 3. 対象ファイル
- `v3/docs/paper3.md` (Appendix D.5)
- `v3/docs/paper2.md` (Appendix D.5)
