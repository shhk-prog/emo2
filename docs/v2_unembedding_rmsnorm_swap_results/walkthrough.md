# 実験⑦ Unembedding / RMSNorm / lm_head Swap 詳細解説 Walkthrough

## 1. 実験の目的と背景
- **実験名**: Unembedding / RMSNorm Swap Analysis: Locus of Self-Report Neutralization（Phase 7 / Table 5）
- **核心の問い**:
  Instructモデルにおける感情自己報告の中立化（平坦化）は、
  - **仮説A（Output Head Locus）**: 最終出力段のパラメータ（RMSNorm や $W_U$ / lm_head）が「感情語彙へのロジット投影を抑制するフィルター」として更新されたためか？
  - **仮説B（Residual Stream Locus）**: 最終層出力時点の残差ストリーム（Final Residual State）において、既に感情信号が出力層へ届かない形式に変形・抑圧されているためか？

## 2. 8条件完全交差スワップ設計
モデルの最終ロジット生成式：
$$\text{logits} = W_U \cdot \text{RMSNorm}(h_{\mathrm{final}})$$
に対し、$h_{\mathrm{final}}$（残差）、$\text{RMSNorm}$（重み $\gamma$）、$W_U$（lm_head）の3箇所を Base (B) と Instruct (I) で独立に入れ替えた $2^3 = 8$ 条件を検証：

| Condition | Residual ($h_{\mathrm{final}}$) | RMSNorm ($\gamma$) | Unembedding ($W_U$) | 説明 |
|:---|:---:|:---:|:---:|:---|
| **BBB** | Base | Base | Base | 完全な Base モデル（基準） |
| **BBI** | Base | Base | Instruct | Base 残差に Instruct の Head のみ適用 |
| **BIB** | Base | Instruct | Base | Base 残差に Instruct の Norm のみ適用 |
| **BII** | Base | Instruct | Instruct | Base 残差に Instruct の Norm と Head を適用 |
| **III** | Instruct | Instruct | Instruct | 完全な Instruct モデル（基準） |
| **IIB** | Instruct | Instruct | Base | Instruct 残差に Base の Head のみ適用 |
| **IBB** | Instruct | Base | Base | Instruct 残差に Base の Norm と Head を適用 |
| **IBI** | Instruct | Base | Instruct | Instruct 残差に Base の Norm のみ適用 |

評価は 81通りの Valence 1〜9 × Arousal 1〜9 の JSON 候補尤度から厳密な結合分布を算出して実施。

## 3. 定量的結果（Table 5）
基準条件 BBB（Base純正）に対する距離・発散の比較：

| Condition | Res | Norm | Head | Expected Valence ($E[V]$) | $WD_V$ to BBB | JSD to BBB | 2D EMD |
|:---|:---:|:---:|:---:|---:|---:|---:|---:|
| **BBB** | Base | Base | Base | 5.101 | 0.000 | 0.00000 | 0.000 |
| **BBI** | Base | Base | Instruct | 5.089 | 0.013 | 0.00002 | 0.019 |
| **BIB** | Base | Instruct | Base | 5.101 | 0.000 | 0.00000 | 0.000 |
| **BII** | Base | Instruct | Instruct | 5.089 | 0.013 | 0.00002 | 0.019 |
| **III** | Instruct | Instruct | Instruct | 5.113 | 0.198 | 0.00333 | 0.266 |
| **IIB** | Instruct | Instruct | Base | 5.136 | 0.201 | 0.00344 | 0.263 |
| **IBB** | Instruct | Base | Base | 5.136 | 0.201 | 0.00344 | 0.263 |
| **IBI** | Instruct | Base | Instruct | 5.113 | 0.198 | 0.00333 | 0.266 |

## 4. 科学的発見と理論的結論
1. **出力層（RMSNorm / Unembedding）の寄与は極小**:
   - 残差が Base（B群）の場合、Norm や Head を Instruct のものに差し替えても（BBI, BIB, BII）、Base分布からのズレは $WD_V \le 0.013$, $\text{JSD} \le 0.00002$, $\text{2D EMD} \le 0.019$ と極めて軽微。
   - 特に BIB（Normだけ変更）は BBB と $WD_V=0$, $\text{JSD}=0$ で完全に一致。
2. **残差ストリーム（Final Residual State）が分布形状を完全に支配**:
   - 残差が Instruct（I群: III, IIB, IBB, IBI）に切り替わった瞬間、$WD_V \approx 0.20$, $\text{2D EMD} \approx 0.26$, $\text{JSD} \approx 0.0034$ へと約14〜150倍跳ね上がる。
   - Instruct 残差に対し、出力段を Base の Norm と Head（IBB）に差し戻しても、Base型分布への復元は生じず、$WD_V = 0.201$, $\text{2D EMD} = 0.263$ のまま維持される。
3. **中立化の真の所在（Locus）**:
   - 事後学習に伴う自己報告の中立化は、最終出力層の重み更新（線形投影フィルター）によるものではなく、**出力層に入る直前の残差ストリーム（Final Residual State）の時点で既に完成している**ことが決定的に証明された。
4. **実験⑥との総合結論（Deep Distributed Routing）**:
   - 実験⑥（単一コンポーネントの後期直接代入の失敗）と実験⑦（最終出力層重み説の棄却）を統合すると、自己報告の中立化は「出力層のゲート」でも「浅い直通回路」でもなく、**「中間層から最終層に至るディープな中間ブロック群（MLP/Attention）が織りなす分散ネットワーク（Circuit）」によって内部で形成されている**ことが確定した。
