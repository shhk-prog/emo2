# 論文結果章（Results）統合サマリーレポート

- **生成日時 (UTC)**: 2026-09-25T04:13:34.743812+00:00
- **総 Primary レコード数**: 341
- **総 Secondary/Control レコード数**: 515

---

## 論文ストーリーラインの全体構造
```text
§4 Behavioral: ReaderとSelfは制御された情動変化に対して共変動するか（出力レベルの連動）
       ↓ [しかし同じ内部表現を利用しているかは不明]
§5 V1: 共通の内部表現幾何・因果経路が存在するか（同一モデル内の共有と特殊化）
       ↓ [しかしBaseとInstructでなぜ関係が異なるかは不明]
§6 V2: 事後学習（Post-training）によって幾何・共有・因果配置はどう再編されるか
       ↓ [しかしInstruct生成過程のどこ・いつSelf-reportへ因果的に利用されるか不明]
§7 V3: 自律的情動情報は計算過程のどの時空間ステージで因果的に利用されるか
```

---

## §4 Behavioral Characterization (行動評価サマリー)
- **中心問い**: ReaderとSelfは、制御された情動刺激の変化に対して共変動するか。
- **主要成果物**:
  - [Table B1: EmoBank Human Correspondence](file:///results/derived/paper_summary/tables/table_b1_emobank_correspondence.csv)
  - [Table B2: AIPsy Sensitivity](file:///results/derived/paper_summary/tables/table_b2_sensitivity.csv)
  - [Table B3: AIPsy Dose-Response](file:///results/derived/paper_summary/tables/table_b3_dose_response.csv)
  - [Table B4: AIPsy Specificity](file:///results/derived/paper_summary/tables/table_b4_specificity.csv)
  - [Table B5: Reader-Self Change Coupling](file:///results/derived/paper_summary/tables/table_b5_coupling.csv)
  - [Table B Summary: Model Representatives](file:///results/derived/paper_summary/tables/table_b_summary.csv)
- **最重要結果 (Table B5)**: Clinical–Neutral ペアにおける $\Delta R$ と $\Delta S$ の相関 $r_\Delta = \text{Corr}(\Delta R, \Delta S)$ が極めて有意に確認され、出力変化の共変動が立証された。

---

## §5 Internal Representation and Causal Sharing (V1 内部表現・因果共有サマリー)
- **中心問い**: 出力連動の背後に、共有された内部表現と因果構造が存在するか。
- **主要成果物**:
  - [Table V1-1: Shared Decodability](file:///results/derived/paper_summary/tables/table_v1_1_peak_decodability.csv)
  - [Table V1-2: Shared Geometry](file:///results/derived/paper_summary/tables/table_v1_2_shared_geometry.csv)
  - [Table V1-3: Semantic Controls](file:///results/derived/paper_summary/tables/table_v1_3_semantic_controls.csv)
  - [Table V1-4: Shared Causal Map](file:///results/derived/paper_summary/tables/table_v1_4_causal_map.csv)
  - [Table V1-5: Causal Interchangeability](file:///results/derived/paper_summary/tables/table_v1_5_interchangeability.csv)
  - [Table V1-6: Task-Specific Specialization](file:///results/derived/paper_summary/tables/table_v1_6_specialization.csv)
- **核心的証拠順序**: Shared Decodability $\to$ Alignable Geometry $\to$ Shared Causal Maps $\to$ Partial Task-Specific Specialization。

---

## §6 Post-training-Associated Reorganization (V2 事後学習再編サマリー)
- **中心問い**: Base–Instruct 間で表現–報告関係はどのように再編されるか。
- **主要成果物**:
  - [Table V2-1: Representation Geometry](file:///results/derived/paper_summary/tables/table_v2_1_geometry.csv)
  - [Table V2-2: Reader-Self Sharing](file:///results/derived/paper_summary/tables/table_v2_2_sharing.csv)
  - [Table V2-3A: Causal Relocation](file:///results/derived/paper_summary/tables/table_v2_3a_causal_relocation.csv)
  - [Table V2-3B: Control Comparison](file:///results/derived/paper_summary/tables/table_v2_3b_causal_controls.csv)
  - [Table V2-3C: LMM Results](file:///results/derived/paper_summary/tables/table_v2_3c_lmm.csv)
  - [Table V2-4: Distribution Recovery](file:///results/derived/paper_summary/tables/table_v2_4_recovery.csv)
  - [Table V2 Confirmatory Summary](file:///results/derived/paper_summary/tables/table_v2_confirmatory.csv)
- **核心的知見**: 単なる表現の消去（Erasure）ではなく、幾何変換・共有度再編・因果配置の深層への移行（Relocation）および介入による回復可能性が確認された。

---

## §7 From Representation to Causal Utilization (V3 時空間因果利用サマリー)
- **中心問い**: Instructモデル内部の自己報告生成において、情動情報はいつ・どこで因果的に利用されるか。
- **主要成果物**:
  - [Table V3-1: State-Induction Gate](file:///results/derived/paper_summary/tables/table_v3_1_gate.csv)
  - [Table V3-2: Spatiotemporal 4-Maps (Discovery)](file:///results/derived/paper_summary/tables/table_v3_2_spatiotemporal_summary.csv)
  - [Table V3-3: Mediated Attenuation (Confirmatory)](file:///results/derived/paper_summary/tables/table_v3_3_mediated_attenuation.csv)
  - [Table V3-4: Confirmatory Replication](file:///results/derived/paper_summary/tables/table_v3_4_confirmatory.csv)
  - [Table V3 Confirmatory Matrix](file:///results/derived/paper_summary/tables/table_v3_confirmatory_matrix.csv)
- **厳格な科学的境界**: RQ2 full 4-map は `discovery` として報告し、独立 Confirmation set による attenuation および cross-model replication は `confirmatory` として厳密に直交分離。
