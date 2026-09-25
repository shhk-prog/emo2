# V2 LMM 交互作用項 ($C(\text{alignment}) \times C(\text{task})$) 頑健性・感度分析レポート

## 1. 分析背景
- **主結果**: Valence の Primary Matched-Plain 条件における Mixed Linear Model (MixedLM):
  $$c_v^{\text{net\_rand}} \sim C(\text{family}) + C(\text{alignment}) \times C(\text{task}) \times \text{relative\_depth}, \quad \text{groups}=\text{pair\_id}$$
  において、$C(\text{alignment})[\text{T.inst}] \times C(\text{task})[\text{T.self}]$ の効果量および有意性は：
  $$\beta = 1.629 \times 10^{-4}, \quad p = 0.00737, \quad \text{FDR } q = 0.044$$
- **検討課題**:
  - `groups="pair_id"` のランダム効果分散（Group Var）が $0.0$ に収束する境界解（boundary fit / singular fit）となっており、ランダム切片の分散が消失している。
  - このため、クラスタ構造を考慮した他の頑健な推定量（Cluster-Robust SE、HC3 ロバスト SE、OLS）でも本効果が安定して成立するかを厳密に感度分析する。

---

## 2. 実データによる感度分析結果

- **対象データ**: `v2/results/derived/v2_causal_pair_level.csv` (全 516,000 行、Primary matched-plain: 344,000 行)
- **分析コード**: [`docs/prepare_anonymized_submission_zip/audit_v2_lmm_robustness.py`](audit_v2_lmm_robustness.py)

| 推定手法 (Method) | 推定係数 ($\beta$) | 標準誤差 (SE) | $p$ 値 ($p$-value) | 有意性 ($\alpha=0.05$) |
|---|---|---|---|---|
| **MixedLM (groups=pair_id)** (論文採用モデル) | $1.629 \times 10^{-4}$ | $6.079 \times 10^{-5}$ | **$0.007370$** | **有意 (FDR $q=0.044$)** |
| **Standard OLS** | $1.629 \times 10^{-4}$ | $6.087 \times 10^{-5}$ | **$0.007450$** | **有意** |
| **Cluster-Robust SE (clustered by pair_id)** | $1.629 \times 10^{-4}$ | $7.276 \times 10^{-5}$ | **$0.025174$** | **有意** |
| **Cluster-Robust SE (clustered by family)** | $1.629 \times 10^{-4}$ | $6.899 \times 10^{-5}$ | **$0.018209$** | **有意** |
| **HC3 Heteroskedasticity-Robust SE** | $1.629 \times 10^{-4}$ | $7.042 \times 10^{-5}$ | **$0.020713$** | **有意** |

---

## 3. 統計的解釈と論文での位置づけ

1. **効果量の完全な一致**:
   すべての推定量において、点推定値は $\beta = 1.629 \times 10^{-4}$ で完全に一致している。
2. **クラスタ相関への頑健性**:
   `pair_id` 単位のクラスタロバストSE（分散のクラスタ内相関をノンパラメトリックに許容）を用いても、標準誤差は $6.08 \times 10^{-5}$ から $7.28 \times 10^{-5}$ への微増にとどまり、$p = 0.0252$ と $5\%$ 水準で安定して有意性を維持している。
3. **境界解（Group Var = 0.0）の実質的意味**:
   `pair_id` 間のランダム切片分散が実質的に 0 であることは、刺激ペア間のベースライン差よりもモデルファミリ・層深度・アライメント・タスクの固定効果が変動の大半を説明していることを意味し、モデルの破綻を示すものではない。
4. **論文記述の慎重化方針**:
   ただし、クラスタロバストSEでの $p = 0.025$ に FDR（6検定項）を適用した場合には $q \approx 0.15$ となる余地があるため、Abstract や本文で「全ファミリーに共通する一様な因果プロファイル再編」と一般化して強く主張するのではなく、
   > 「Base--Instruct間では情報は保持されつつrepresentation geometryに差がみられ、Valenceではtask-dependentなcausal-profile differenceが限定的に支持された」
   という慎重な表現をとることが学術的にも査読防御上も極めて有効である。
