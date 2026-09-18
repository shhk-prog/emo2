# V3 Documentation: Alignment with Primary Experimental Pipeline

本ディレクトリには、V3（Representational-Causal Dissociation および生成ステージ介入）に関する文書が格納されています。

## 論文導線の1本化に関する重要方針

旧ドラフト群（旧 `paper_restructured.md`, `paper.md` 等）に記述されていた **「98.6% greedy collapse」** や **「neutralization」** を主軸とするストーリーは、初期の greedy 生成時におけるアーティファクトに基づくものであり、**論文の Primary 導線からは除外され、`legacy/`（または補助的解析・感度分析）として位置づけられます。**

現在の統一された研究導線（Behavioral → V1 → V2 → V3）における正式プロトコルは以下の4本柱に基づきます：

1. **Human Grounding**:
   - EmoBank / AIPsy の人間アノテーションとの正準アライメント（Pearson / Spearman 相関）。
2. **Distributional Sensitivity**:
   - 729 状態の同時確率分布に対する完全条件付き対数尤度に基づく期待値 $E[V], E[A]$ の高感度な変位測定。
   - `exact_neutral_argmax` は主指標ではなく、生成時の縮約傾向を説明するための**補助指標**として扱う。
3. **Dose-Response & Specificity**:
   - 情動介入強度 $\alpha$ の掃引に対する単調な反応および、Matched vs. Random / Orthogonal コントロールに対する特異度（Specificity）。
4. **Reader–Self Coupling / Dissociation**:
   - 人間感情推定（Reader）と自己報告（Self）の表現共有・因果的交差互換性（E4）および課題特異的因果分離（E6, V3）。

## 用語の固定（現行導線）

- V3-RQ3 の主指標名は **mediated attenuation**（および residual shift / attenuation ratio）である。Pearl 流の NDE/NIE は用いない。
- V3-RQ1 の Topic control は Self VA shift と同尺度の効果量ではない。Topic 課題への非特異的摂動が小さいことを確認する統制として扱う。

