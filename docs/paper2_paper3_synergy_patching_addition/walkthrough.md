# 修正内容・解説の確認 (Walkthrough): paper2 & paper3 への Synergy Patching 詳細結果の追記

## 1. 実施概要
ユーザーからの「paper2,paper3に追記して」という指示に基づき、`v3/docs/paper2.md` および `v3/docs/paper3.md` の Appendix D.5（Component-Wise Patching Screening and Synergy）に、**Phase 6 Synergy Patching の専用実測テーブル（Table D.2）および分析考察** を追記・拡充しました。

---

## 2. 追記された具体的内容

### Table D.2: Synergy Patching (Joint Multi-Component Intervention) Results
`paper2.md` および `paper3.md` の Appendix D.5 に、以下の対比テーブルを新設しました：

| 介入条件 (Intervention) | 実測変化量 ($\Delta V$) | 線形予測値 (Expected Sum) | シナジー差分 (Obs - Exp) | 挙動と理論的解釈 |
|:---|---:|---:|---:|:---|
| **`10_mlp` (単独)** | **-0.0911** | — | — | 独立した負の因果効果（Valence低下） |
| **`14_attn` (単独)** | **+0.0208** | — | — | 独立した正の因果効果（Valence上昇） |
| **`10_mlp` + `14_attn` (2箇所同時)** | **-0.0561** | **-0.0703** | **+0.0142** | **線形加算性が成立（わずかな減衰、相殺や破綻なし）** |
| **`10_mlp` + `14_attn` + `16_res` (3箇所同時)** | **-0.1420** | **-0.1643** | **+0.0223** | **3コンポーネントの累積的線形加算を確認** |

### 理論的結論の記述
1. **線形加算性の成立**:
   - 実測された2コンポーネント複合効果（$-0.0561$）は、個別の効果の単純な線形和（$-0.0703$）とおおむね一致し、非線形な相殺や分布の虚脱（Collapse）は生じないことを明記。
2. **3コンポーネント同時パッチの累積**:
   - Layer 16 `res`（単独効果 $-0.0940$）を加えた3モジュール同時介入においても、各モジュールの独立した寄与が累積的に重畳（$\Delta V \approx -0.142$）することを明記。
3. **分散的再写像（Distributed Remapping, H4）の決定打**:
   - 自己報告の中立化が単一の特効的ボトルネックによるものではなく、中間層の複数の独立したコンポーネント（Layer 10 MLP, Layer 14 Attention, Layer 16 Residual等）がそれぞれ分離可能な因果的重み（Separable Causal Contributions）を持ち寄り、ネットワーク全体で中立ペルソナを出力するように再配線されていることを決定づけた。

---

## 3. 更新完了ファイル
- `v3/docs/paper3.md` (Appendix D.5)
- `v3/docs/paper2.md` (Appendix D.5)
