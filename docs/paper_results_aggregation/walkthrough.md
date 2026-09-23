# 結果章用集約パイプライン（Paper Results Aggregation）実装検証レポート (Walkthrough)

## 1. 概要と基本方針

本実装は、論文の構成（**§4 Behavioral $\to$ §5 V1 $\to$ §6 V2 $\to$ §7 V3**）に完全準拠し、各実験ステージの確定済み結果から本文・付録・図表に必要なデータを抽出・集約する**純粋なプレゼンテーション層（read-only presentation layer）**を構築したものです。

### 遵守された厳格な設計原則
1. **統計量の再計算禁止（No Re-computation）**:
   - Primary の $p$ 値, $q$ 値, 信頼区間 (CI) は、元実験スクリプトが出力した検証済み数値をそのまま引き継ぎ（direct copy）、集計コード側での再検定や新たな統計モデリングは一切行っていません。
2. **原データ・元結果の不変性（Immutability）**:
   - `data/raw/` および各ステージの `results/raw/`, `results/derived/` は読み取り専用として扱い、新規出力はすべて `results/derived/paper_summary/` 配下に完全分離しました。
3. **Primary と Analysis Role の直交化（Orthogonality）**:
   - `is_primary`（論文本文の主結果か否か）と `analysis_role`（`discovery`, `confirmatory`, `primary`, `secondary`, `control`, `diagnostic`）を直交概念として定義。これにより、V3 RQ2 の 4-Map（`is_primary=True, analysis_role="discovery"`）を本文の主結果として維持しつつ、科学的探索（Discovery）と検証（Confirmatory）の境界を厳密に分離しました。
4. **V2 NaN 保持 & Canonical Metric 直接転送**:
   - 正の因果ピークが存在しない場合は 0 埋めを行わず `np.nan` を厳格に保持。`c_net_rand` 等の指標も元 derived artifact から直接転送。
5. **V3 Gate 文字列保持 & パイプライン継続判定**:
   - Gate 判定結果は 4 値の文字列（`GO`, `GO (Valence-only)`, `GO (Arousal-only)`, `NO_GO`）のまま保持し、`pipeline_continues = (overall_decision == "GO")` に基づいて下流の存在妥当性を判定。
6. **完全な追跡可能性（Provenance & Derivation）**:
   - 生成されたすべてのレコードは、`paper_summary_manifest.json` において元成果物パス（`sources`）および導出種別（`derivation`）を記録。

---

## 2. 実装されたコンポーネント

| コンポーネント | ファイルパス | 役割・特徴 |
|---|---|---|
| **共通スキーマ** | [schema.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/paper_summary/schema.py) | 共通19列スキーマ、`filter_primary_results`, `filter_secondary_results`, UNIQUE_KEY 一意性検証 |
| **共通ユーティリティ** | [common.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/paper_summary/common.py) | 複数 `sources` 配列対応 `PaperSummaryManifest`, `QCSummaryBuilder`, safe I/O, Gate ロジック |
| **§4 Behavioral 集約** | [behavioral/analysis/build_paper_summary.py](file:///mnt/nas/home/hiromi/src/emo2/behavioral/analysis/build_paper_summary.py) | Table B1〜B5, Table B Summary (総合点・ランクなしの横並び), Figure B1〜B4 データ |
| **§5 V1 集約** | [v1/scripts/build_paper_summary.py](file:///mnt/nas/home/hiromi/src/emo2/v1/scripts/build_paper_summary.py) | Table V1-1〜V1-6, Figure V1-1〜V1-4 データ, Table V1-3 nonfallback N 併記 |
| **§6 V2 集約** | [v2/scripts/build_paper_summary.py](file:///mnt/nas/home/hiromi/src/emo2/v2/scripts/build_paper_summary.py) | Table V2-1〜V2-4, Confirmatory Summary, Figure V2-1〜V2-4, NaN 保持, LMM term 一意化 |
| **§7 V3 集約** | [v3/scripts/build_paper_summary.py](file:///mnt/nas/home/hiromi/src/emo2/v3/scripts/build_paper_summary.py) | Table V3-1〜V3-4, Confirmatory Matrix, Gate 文字列保持, response_end 追跡 |
| **マスターオーケストレーター** | [scripts/build_all_paper_summaries.py](file:///mnt/nas/home/hiromi/src/emo2/scripts/build_all_paper_summaries.py) | `importlib.util` による動的ロード・名前衝突回避、全ステージ実行、Markdown Summary 生成 |
| **不変条件テスト** | [tests/test_paper_summary_invariants.py](file:///mnt/nas/home/hiromi/src/emo2/tests/test_paper_summary_invariants.py) | 7つの学術的不変条件の自動検証（pytest） |

---

## 3. 生成された論文用成果物一覧

集計スクリプト（`scripts/build_all_paper_summaries.py --strict`）により、以下のファイルが `results/derived/paper_summary/` 配下に完全生成されました。

### 3.1 主結果・副次結果・統括文書
- [primary_results.csv](file:///mnt/nas/home/hiromi/src/emo2/results/derived/paper_summary/primary_results.csv) (329 行)
- [secondary_results.csv](file:///mnt/nas/home/hiromi/src/emo2/results/derived/paper_summary/secondary_results.csv) (492 行)
- [paper_summary_manifest.json](file:///mnt/nas/home/hiromi/src/emo2/results/derived/paper_summary/paper_summary_manifest.json) (全レコードの provenance & derivation マップ)
- [qc_summary.json](file:///mnt/nas/home/hiromi/src/emo2/results/derived/paper_summary/qc_summary.json) (ステージ別行数、Gate判定、QC情報)
- [results_summary.md](file:///mnt/nas/home/hiromi/src/emo2/results/derived/paper_summary/results_summary.md) (論文本文執筆用 Markdown サマリー)

### 3.2 論文掲載用テーブル (Tables: 25 ファイル)
- **Behavioral (6ファイル)**:
  - `table_b1_baseline_stability.csv`
  - `table_b2_recognition_human_concordance.csv`
  - `table_b3_controlled_reactivity_emobank.csv`
  - `table_b4_reactivity_neutral_divergence.csv`
  - `table_b5_coupling.csv` (Clinical–Neutral ペア連動)
  - `table_b_summary.csv` (モデル別代表値一覧)
- **V1 (6ファイル)**:
  - `table_v1_1_peak_decodability.csv`
  - `table_v1_2_shared_geometry.csv`
  - `table_v1_3_semantic_controls.csv` (nonfallback 有効N明記)
  - `table_v1_4_causal_map.csv`
  - `table_v1_5_interchangeability.csv`
  - `table_v1_6_specialization.csv`
- **V2 (7ファイル)**:
  - `table_v2_1_geometry.csv`
  - `table_v2_2_sharing.csv`
  - `table_v2_3a_causal_relocation.csv` (NaN保持)
  - `table_v2_3b_causal_controls.csv`
  - `table_v2_3c_lmm.csv` (LMM 係数)
  - `table_v2_4_recovery.csv`
  - `table_v2_confirmatory.csv` (H1〜H4 仮説検証結果)
- **V3 (6ファイル)**:
  - `table_v3_1_gate.csv` (Gate 文字列保持)
  - `table_v3_2_spatiotemporal_summary.csv` (Discovery 4-Maps, negative control response_end)
  - `table_v3_3_mediated_attenuation.csv` (Confirmatory)
  - `table_v3_4_confirmatory.csv` (Cross-model replication)
  - `table_v3_confirmatory_matrix.csv`
  - `table_v3_gate.csv`

### 3.3 図プロット用データ (Figure Data: 9 ファイル)
- `fig_b1_baseline_scatter.csv`, `fig_b2_controlled_reactivity_scatter.csv`, `fig_b3_neutral_divergence_kde.csv`, `fig_b4_coupling_paired_shift.csv`
- `fig_v1_1_decodability_depth.csv`, `fig_v1_2_geometry_depth.csv`, `fig_v1_3_causal_depth.csv`, `fig_v1_4_interchangeability_scatter.csv`
- `fig_v2_3_causal_relocation_depth.csv`

---

## 4. 検証結果

### 4.1 不変条件テスト（7/7 PASSED）
`pytest tests/test_paper_summary_invariants.py -v` を実行し、全 7 件のテストをパスしました。

```text
tests/test_paper_summary_invariants.py::test_invariant_1_primary_orthogonality PASSED [ 14%]
tests/test_paper_summary_invariants.py::test_invariant_2_v2_nan_preservation PASSED [ 28%]
tests/test_paper_summary_invariants.py::test_invariant_3_v3_negative_control_gate_dependent PASSED [ 42%]
tests/test_paper_summary_invariants.py::test_invariant_4_v1_phase_b_nonfallback_n PASSED [ 57%]
tests/test_paper_summary_invariants.py::test_invariant_5_v3_gate_decision_logic PASSED [ 71%]
tests/test_paper_summary_invariants.py::test_invariant_6_strata_coverage_and_no_duplicate_keys PASSED [ 85%]
tests/test_paper_summary_invariants.py::test_invariant_7_manifest_provenance_completeness PASSED [100%]
=========================== 7 passed in 2.10s ===========================
```

### 4.2 静的コード解析（Ruff Lint & Format 100% Clean）
- `ruff check`: All checks passed!
- `ruff format --check`: All formatted!

---

## 5. 結論
これにより、論文の各セクション（§4, §5, §6, §7）執筆に必要な表・図データおよび要約ファイルが、学術的不変性と追跡可能性を完全に保証した状態で整備完了しました。
