# 変更内容の確認 (Walkthrough): paper3.md 総合改訂・不整合解消

本ドキュメントは、`/mnt/nas/home/hiromi/src/emo/v3/docs/paper3.md` に対する11項目の重要不整合修正および学術的厳密化の作業結果を記録したものである。

---

## 1. 実施した修正項目の概要

| # | 指摘・改善項目 | 修正前の状態 | 修正後の状態・対応 |
|---|---|---|---|
| **1** | 「第II部 統合再編」の重複・矛盾数値の混入 | 1952行以降に別稿が重複挿入され、Ridge $R^2 \approx 0.88$, CKA=0.94 等の矛盾値が混入 | **第II部を完全削除**。必要な図表（Table 1, Table 2, Table 3）を正本値・慎重な表現で正式本文側へ統合。 |
| **2** | Joint OT を「Wasserstein-2」と呼ぶ誤謬 | 地球移動距離を「Wasserstein-2」と呼称していた箇所が存在 | **2D Joint Optimal Transport cost with Manhattan ground cost**（または Joint OT on the discrete VA grid）へ完全統一。 |
| **3** | Probe necessity の方向数誤認 | 「50本のランダム直交方向」と誤記 | **全層探索: 20方向**（raw $p \ge 1/21=0.0476$, BH-FDR $q \ge 0.857$）、**代表5層検証: 100方向**（$p \ge 0.297, q=1.000$）に統一。 |
| **4** | Generation-time 全層表のサンプルサイズ混在 | Stage 1（13有効ペア）と Stage 2（39完全ペア）が同一表に混在 | **Table A2a**（Exploratory full-layer, 13 valid pairs, 全28層）と **Table A2b**（Focused full-cohort, 39 complete pairs, 6代表層）に明確に分離。 |
| **5** | Multi-layer Residual の実験条件・値の混在 | L22+L24, L20+L22+L24 等の架空条件・推定値が存在 | `generation_multilayer_residual_results.csv` 正本（L18, L20, L24, L20+24, L18+20+24, L18–24）に統一し、架空条件を完全排除。 |
| **6** | Llama 追試結果の記述矛盾と過度な一般化 | 「最大 1.82%」「モデル固有の特性であることが示された」と強硬に断定 | 正本値（MLP最大 L15 -0.36%, Attention最大 L4 +0.73%, Residual全層負）を明記し、「初期追試では再現されなかった」と慎重に限定。 |
| **7** | 「確証的」「バイアスのない」の不適切呼称 | 「確証的集中評価」「バイアスのない全層プロファイル探索」と呼称 | **Exploratory full-layer screen** および **Focused full-cohort effect-size evaluation** へ統一。 |
| **8** | 「因果レバー皆無」等の強すぎる全称的表現 | 「因果レバー皆無」「特異的な因果的必要性を一切持たない」 | 「tested local matched-substitution intervention 下で強い局所因果レバレッジを示さなかった」と操作定義に沿って厳密化。 |
| **9** | 「FDRで棄却された」の表現修正 | 帰無仮説を証明したかのような「特異的中和は全層で棄却された」 | **No evidence for probe-aligned local necessity was found.** に統一。 |
| **10**| Related Work の文献年誤認 | Hase et al. (2024) と誤記 | **Hase et al. (2023; NeurIPS 2023)** に修正。 |
| **11**| Emotion 先行研究の引用と一般化の抑制 | 最新研究の年号曖昧化、「因果介入が一切ない」とする過度な全称命題 | **Maheswaran and Desarkar (2026; EACL 2026)** として正式引用し、先行研究の動向を「主としてprobing等で特徴づけており、体系的局所化介入は未開拓」と安全な記述に緩和。 |

---

## 2. 論文全体の構造（改訂後）

```text
paper3.md (総行数: 2,010行)
├── タイトル・著者・要旨 (Abstract)
├── 1. Introduction (問題設定、3つの分離された問い)
├── 2. Related Work (線形プロービング、因果介入、感情表現, Hase 2023, Maheswaran & Desarkar 2026)
├── 3. Problem Formulation & Methodological Framework (Accessibility, Causal Leverage, Directional Necessity)
├── 4. Dataset & Experimental Setup (Table 1: 仕様総括表を追加)
├── 5. Baseline Decodability vs. Surface Output (Greedy崩壊と尤度感度の共存)
├── 6. Linear Decodability Peak at Mid-Layers (L15 MLP, L18 Attn, L14 Resid)
├── 7. Prompt-Time Local Causal Recovery is Near-Zero Across Layers
├── 8. Generation-Time Causal Shift and Peak Dissociation
│   ├── 8.1〜8.5 実験結果と検定
│   ├── 8.6 Table 2: 代表サイト結果総括表 (正確なブートストラップCIと慎重な因果記述)
│   └── 8.7 多層Residual同時介入の飽和特性
├── 9. Robustness, Controls, and Boundaries
├── 10. Discussion (Table 3: 主張とエビデンス対照表)
├── 11. Conclusion
└── Appendices (付録アーカイブ)
    ├── Appendix A: Prompt-Time Joint OT 全28層完全数値表
    ├── Appendix B: Generation-Time スイープ数値表
    │   ├── Table A2a: 探索的全層スイープ (Stage 1, 13 valid pairs)
    │   └── Table A2b: 集中全数コホート評価 (Stage 2, 39 complete pairs)
    ├── Appendix C: v1 開発時実験記録
    ├── Appendix D: v2 事後学習メカニズム実験記録
    ├── Appendix E: v3 因果局所化実験記録 (E.1〜E.17, 正本スクリプト・データ対応)
    ├── Appendix F: 探索的・補助的解析の整理
    └── Appendix G: 完全な成果物・再現性マニフェスト
```

---

## 3. 検証結果

1. **矛盾・重複の完全排除**:
   - `Wasserstein-2`: 0件検出
   - `50本`: 0件検出
   - `L22+L24` 等の架空条件: 0件検出
   - `因果レバー皆無`: 0件検出
   - `確証的`: 0件検出（v2のスクリプト名等の歴史的記録を除く）
   - `バイアスのない`: 0件検出
   - `FDRで棄却された`: 0件検出
2. **Appendix 体系の連番一意性**:
   - Appendix A から G まで重複なく綺麗に整理され、論文本文および付録相互参照が正しく維持されていることを確認。
3. **データ完全性**:
   - `data/raw/` および `v3/results/` の生データ・CSV正本は一切変更せず、論文原稿上の数値記述を完全にCSV原本と合致させた。
