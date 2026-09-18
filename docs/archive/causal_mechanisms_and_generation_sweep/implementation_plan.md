# 実装計画書: 査読耐性を極限まで高めるための微細表現精緻化（第11版 投稿完全版）

本計画は、査読者からの指摘を先回りで完全に封殺するため、論文原稿（`v3/docs/paper2.md` および `v3/docs/paper.md`）における微細な断定表現、非有意からの不存在への論理飛躍、および集計指標の定義の曖昧さを徹底的に排除するものです。

---

## 1. 改訂の要点と方針

### (1) Section 6 の効果量表現修正
- 旧: 「いずれの解析手法によっても、定性的な層別プロファイルおよび効果の大きさは完全に同一であり、MLP解読能のピークが一貫してLayer 15に現れることが確認された」
- 新: 「いずれの解析でも層別プロファイルは定性的に一致し、MLP解読能のピークはLayer 15に位置した。効果量も近い範囲にあった。」

### (2) Abstract の “no evidence for Probe-Aligned Local Necessity” への変更
- 統計的非有意から「必要性が低い」と直接同一視する断定を回避：
  - 旧: “High Decodability coexists with Low Local Sufficiency and Low Probe-Aligned Necessity”
  - 新: **“High Decodability coexists with Low Local Sufficiency and no evidence for Probe-Aligned Local Necessity”**

### (3) Section 16.4 の不存在断定の回避
- 非有意結果から不存在を断定しない正確な統計的記述へ：
  - 旧: 「特異的な局所的必要性（local necessity）を持たない」
  - 新: **「特異的な局所的必要性を支持する証拠は得られなかった（no evidence for probe-aligned local necessity）」**

### (4) Generation-time pairwise median の定義明示（誤読の完全排除）
- “median across layers” との誤読を防ぐため、Abstract および Methods (Section 15) にて定義を一文で明記：
  > **“For each layer-component condition, recovery was computed across valid matched pairs; the median across those pairs was 0.00%.”**

### (5) Section 19.2 の局所支配表現の緩和
- 最大 5.02% の微小変位が存在することを踏まえ、完全な不存在断定を回避：
  - 旧: 「単一層の局所介入は生成時においても出力を支配しない」
  - 新: **「強い局所的支配を示す証拠は得られなかった（no evidence for strong local causal dominance）」**

### (6) Discussion 19.1 のタイトルの精緻化
- 論文全体の検証範囲（操作的定義）とぴったり一致させる：
  - 旧: `19.1 Decodability Is Neither Sufficiency Nor Necessity`
  - 新: **`19.1 Decodability Does Not Identify Local Sufficiency or Probe-Aligned Local Necessity`**

---

## 2. 変更対象ファイル

1. [MODIFY] [paper2.md](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper2.md)
2. [MODIFY] [paper.md](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper.md)
3. [MODIFY] [task.md](file:///mnt/nas/home/hiromi/src/emo/docs/causal_mechanisms_and_generation_sweep/task.md)
4. [MODIFY] [walkthrough.md](file:///mnt/nas/home/hiromi/src/emo/docs/causal_mechanisms_and_generation_sweep/walkthrough.md)
