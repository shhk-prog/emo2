# ウォークスルー: 査読クリティークへの対応と実験ロードマップ

## 査読指摘の総括
査読者の判定（Weak Reject / Borderline: 4-5/10）は、機械論的解釈可能性（Mechanistic Interpretability）分野の最前線（2026年時点）の基準に照らして、完全に的を射たものです。

特に以下の3点は、論文の成否を分ける急所です：
1. **Positive controlの不在**: 同一モデル内（Instruct→Instruct）で動かないパッチングは、クロスモデルパッチングの失敗（0%）が「表現のデカップリング」ではなく「介入部位・方法の因果的不十分性（単なるnull）」であることを意味してしまう。
2. **Alignmentの低次元バイアス**: $d=1536$ を $N=124$ でRidge回帰した場合、復元されたのはValenceプローブが拾う1次元情報だけで、下流の報告計算に必要な高次元の文脈・構文情報がごっそり失われている可能性が高い。
3. **経験的参照のない多様体適合主張**: Mahalanobis距離やCosine類似度の絶対値（$D_M \approx 6$, $\text{Cos} \approx 0.995$）は、自然活性化ペアの分布と比較されなければ、ID（In-Distribution）であることの証明にならない。

---

## 具体的対応ロードマップ

### 1. 即時実施（論文ドラフトの防御的改訂）
- **C1の位置づけ変更**:
  - 主たる新規性主張から外し、「Martorell等の先行知見に則った妥当な測定セットアップ」としてIntroduction / Methodに再配置。
- **C4の扱い**:
  - Layer 16でrandom control（$\beta=+0.0444$）がvalence（$\beta=-0.0169$）を上回っている事実から、「generic disruption」の懸念を払拭できないため、主本文から除外しAppendixまたは将来課題に退避。
- **Singh et al. (2026) への先回り防御**:
  - 「first-person report」は出力の形式（文法・プロンプト課題）を指すのみであり、モデルの内省（introspection）や特権的自己アクセスを主張するものではないことを明記。
- **Post-trainingへの因果帰属の限定**:
  - 単一チェックポイントペア比較であることを明記し、「post-trainingがdecoupleした」から「Base/Instruct比較における結合の差異（consistent with post-training-associated changes）」へと表現をさらに厳密化。

### 2. 必須追加実験（最小セット: Exp A 〜 D）

| 実験 | 目的 | 具体的手法 | 期待される成功基準 |
|---|---|---|---|
| **Exp A: Positive Causal Control** | パッチング介入の妥当性証明 | 同一Instruct内で `Peak -> Neutral` パッチ。<br>・MLP vs Residual stream<br>・最終トークン vs 全プロンプトトークン | $\mathrm{Recovery}_{\mathrm{within}} \gg 0$（例: 30〜60%以上）となる介入部位・プロトコルを確立 |
| **Exp B: Full-State Reconstruction** | Ridge alignmentの多次元復元性検証 | $R^2_{\text{activation}}$（全次元平均）、CKA、Pair retrieval accuracy | Valence予測だけでなく、活性化テンソル全体がどの程度復元されているかを定量化 |
| **Exp C: Empirical Manifold Test** | 「自然多様体に入った」ことの厳密な実証 | 自然なInstruct活性化の $D_M$ 分布、Two-sample classifier (AUC)、Cosine対照群（Matched vs Unmatched） | Aligned活性化が自然なInstruct活性化と統計的に識別困難であることを示す |
| **Exp D: Second Model Family** | 1モデルペア依存の脱却 | Llama-3-8B または Gemma-2-2B で C2 を再現 | Qwen固有のアーティファクトでないことを証明 |
