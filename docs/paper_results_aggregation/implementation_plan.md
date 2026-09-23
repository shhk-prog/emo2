# 論文結果章に対応する各Stage結果集計コードの実装計画（確定最終版・完全合意版）

## 概要と目的

本計画は、論文構成仕様（§4 Behavioral → §5 V1 → §6 V2 → §7 V3）に基づき、各実験ステージの結果を集約・整理し、論文の本文・表・図に直結するクリーンな集計データを生成するコード群を体系的に実装することを目的とします。
本コード群の責務は、**「統計の再解析」ではなく、「元実験スクリプトが確定した結果を論文構造へ写像する純粋なプレゼンテーション層（read-only presentation layer）」**です。

---

## 基本設計原則（確定版）

1. **Experiment scripts = inference/statistics の正本、Paper summary scripts = read-only presentation layer**:
   - 集計コードでは統計量（$p$, $q$, CI, bootstrap値）を一切再計算せず、元実験スクリプトが出力した値をそのまま引き継ぐ（copy）。
   - **canonical derived metric が実験成果物に存在する（例: V2 の `c_v_net_rand`, `c_a_net_rand` 等）場合は必ず direct copy** とする。summary 側での差分計算は元成果物に存在しない場合にのみ厳格に限定。
2. **`is_primary` と `analysis_role` の直交化**:
   - `is_primary`: 論文本文の Primary result か否か（True / False）。
   - `analysis_role`: その証拠の性質（`primary`, `confirmatory`, `discovery`, `secondary`, `control`, `diagnostic`）。
   - Primary結果抽出フィルター:
     ```python
     primary_df = df[
         df["is_primary"].eq(True)
         & ~df["analysis_role"].isin(["secondary", "control", "diagnostic"])
     ]
     ```
     これにより、`is_primary=True` かつ `analysis_role="discovery"` である V3 RQ2 の 4-map 成果も正しく `primary_results.csv` に残る。
3. **文字列・カテゴリ値を保持する共通19列スキーマ**:
   - 数値推定値 `estimate` に加え、テキスト/カテゴリ値 `value_text` を導入。
   - Gate決定（`GO`, `GO (Valence-only)`, `GO (Arousal-only)`, `NO_GO`）や、状態ラベル（`No distinct sites identified`）をそのまま保存。
4. **完全な追跡可能性（複数 artifact / 複数 key に対応した Provenance & Derivation）**:
   - CSV の 19 列において、複数 artifact / 複数 key に由来する場合は JSON 文字列（例: `source_artifact = '["a.json", "b.json"]'`, `source_key = '["x", "y"]'`）として格納可能。
   - `paper_summary_manifest.json` では、単一 source のみならず複数 source オブジェクトのリスト `sources: [{"artifact": "...", "key": "..."}, ...]` と計算式・導出ロジックを記録する `derivation`（例: `"direct_copy"`, `"argmax finite layer-wise r2, then layer/(L-1)"`）を完全保持。
5. **Unique-Key 衝突防止規約（1 scientific estimate = 1 unique record）**:
   - LMM の各項などの細分化された推定値は、`metric = "lmm_beta::<term>"`（例: `lmm_beta::alignment_x_depth`, `lmm_beta::alignment_x_task`）のように metric 名へ term を含める命名規則を厳格に適用し、`["stage", "rq", "family", "alignment", "task", "axis", "condition", "metric"]` での一意性を保証。
6. **Stage横断総合スコア・ランキングの排除**:
   - `model_summary.csv` は各Stageの代表値を並べるのみとし、測定空間（729 VAD vs 81 VA）や単位が異なる指標の合算・平均・順位付けは行わない。
7. **V2 Derived 正本の直接参照**:
   - `v2_causal_dissociation_summary.json`、`v2_lmm_confirmatory.json`、`v2_cross_family_summary.json`、`v2_recovery_*` 等の最新 derived artifact を正本として読み込む。
8. **NaN と Negative Control の保持**:
   - V2 peak 未検出時の NaN は科学的意味（peak なし）を持つため `fillna(0)` や除外を行わない。
   - V3 `response_end` も negative temporal control として表・図に必ず残す（RQ2 実行時）。
9. **`pipeline_continues` に基づく State-aware な `--strict` バリデーション**:
   - Gate判定は 4 状態（`GO`, `GO (Valence-only)`, `GO (Arousal-only)`, `NO_GO`）を取り、パイプライン継続条件は `pipeline_continues = (overall_decision == "GO")` である。
   - strict validator は以下のように判定する：
     ```python
     if pipeline_continues:
         require(rq2_artifact)
         require(rq3_artifact)
     else:
         # NO_GO, GO (Valence-only), GO (Arousal-only) では下流の非存在を正常受理
         record_qc(status="not_run_due_to_gate")
     ```
   - もし `--force-after-no-go` で強制実行されたデータが存在する場合は `execution_status="forced_after_no_go"` として記録。
10. **条件付き不変条件テスト（Gate依存の整合性）**:
    - RQ2 が実行された場合（`completed` または `forced_after_no_go`）にのみ `response_end` の存在を必須とし、Gate により未実行（`not_run_due_to_gate`）の場合は非存在を正常と判定する。

---

## 共通スキーマ設計

### 出力ディレクトリ構成
```text
results/derived/paper_summary/
├── primary_results.csv              # 全Stage統合Primaryテーブル（is_primary=True かつ non-secondary/control）
├── secondary_results.csv            # 全Stage統合Secondary・Controlテーブル
├── model_summary.csv                # モデル別代表値一覧（総合点・ランク付けなし）
├── statistical_tests.csv            # 各実験が出力した検定統計量（t, p, q, FDR等）一覧
├── qc_summary.json                  # QC・欠損・非fallback N・ゲート合否
├── paper_summary_manifest.json      # 出力値 -> 元ファイル・キー配列・導出ルールの完全対応
├── results_summary.md               # 論文執筆用の全体Markdownサマリー
├── figure_data/                     # 図作成専用データ（コントロールを削除しない）
│   ├── figure_b1_emobank_corr.csv
│   ├── figure_b2_sensitivity.csv
│   ├── figure_b3_dose_response.csv
│   ├── figure_b4_coupling_scatter.csv
│   ├── figure_v1_1_decodability.csv
│   ├── figure_v1_2_geometry.csv
│   ├── figure_v1_3_causal_map.csv
│   ├── figure_v1_4_interchangeability.csv
│   ├── figure_v2_1_geometry_decodability.csv
│   ├── figure_v2_2_sharing.csv
│   ├── figure_v2_3_causal_relocation.csv
│   ├── figure_v2_4_recovery.csv
│   ├── figure_v3_1_gate_dose_response.csv
│   ├── figure_v3_2_v3_3_spatiotemporal_maps.csv
│   └── figure_v3_4_mediated_attenuation.csv
├── tables/                          # 論文本文・補足用の各表CSV
│   ├── table_b1_emobank_correspondence.csv
│   ├── table_b2_sensitivity.csv
│   ├── table_b3_dose_response.csv
│   ├── table_b4_specificity.csv
│   ├── table_b5_coupling.csv
│   ├── table_b_summary.csv
│   ├── table_v1_1_peak_decodability.csv
│   ├── table_v1_1_aipsy_probes.csv
│   ├── table_v1_2_shared_geometry.csv
│   ├── table_v1_3_semantic_controls.csv
│   ├── table_v1_4_causal_map.csv
│   ├── table_v1_5_interchangeability.csv
│   ├── table_v1_6_specialization.csv
│   ├── table_v2_1_geometry.csv
│   ├── table_v2_2_sharing.csv
│   ├── table_v2_3a_causal_relocation.csv
│   ├── table_v2_3b_causal_controls.csv
│   ├── table_v2_3c_lmm.csv
│   ├── table_v2_4_recovery.csv
│   ├── table_v2_confirmatory.csv
│   ├── table_v3_1_gate.csv
│   ├── table_v3_2_spatiotemporal_summary.csv
│   ├── table_v3_3_mediated_attenuation.csv
│   ├── table_v3_4_confirmatory.csv
│   └── table_v3_confirmatory_matrix.csv
└── stage_summaries/                 # 各Stage固有の出力CSV
    ├── behavioral/
    ├── v1/
    ├── v2/
    └── v3/
```

### 共通19列レコードスキーマ
すべての集計行は以下の19列で統一します：
1. `stage`: `behavioral`, `v1`, `v2`, `v3`
2. `rq`: 研究課題（例: `rq1_correspondence`, `rq4_coupling`, `rq3_causal_relocation` 等）
3. `family`: モデルファミリー（例: `llama`, `gemma`, `olmo`, `qwen`）
4. `alignment`: `base`, `instruct`
5. `task`: `reader`, `self`, `writer` (該当する場合)
6. `axis`: `valence`, `arousal`, `dominance`
7. `condition`: 実験条件（例: `matched_plain`, `native_chat`, `clinical_vs_neutral`, `alpha_1.0` 等）
8. `metric`: 評価指標名（例: `pearson_r`, `mean_aligned_delta`, `r2_peak`, `overall_gate_decision`, `lmm_beta::alignment_x_depth` 等）
9. `estimate`: 数値推定値（該当しない/未検出時は NaN）
10. `value_text`: 文字列/カテゴリ値（例: `"GO (Valence-only)"`, `"No distinct sites identified"`, 数値の場合は空文字列または文字列表現）
11. `ci_low`: 95%信頼区間下限（元実験出力からコピー）
12. `ci_high`: 95%信頼区間上限（元実験出力からコピー）
13. `p`: p値（元実験出力からコピー）
14. `q`: FDR補正後q値（元実験出力からコピー）
15. `n`: 有効サンプル数 / ペア数
16. `is_primary`: `True` / `False`
17. `analysis_role`: `primary`, `confirmatory`, `discovery`, `secondary`, `control`, `diagnostic`
18. `source_artifact`: 元ファイルパス（複数ファイルの場合は JSON 文字列 `'["path/a.json", "path/b.json"]'`）
19. `source_key`: 元ファイル内のキー/カラム名（複数キーの場合は JSON 文字列 `'["key_a", "key_b"]'`）

### Manifest構造 (`paper_summary_manifest.json`)
```json
{
  "summary_schema_version": "1.0.0",
  "generated_at_utc": "YYYY-MM-DDTHH:MM:SSZ",
  "provenance_map": {
    "behavioral.table_b1.reader_valence_pearson_r": {
      "sources": [
        {
          "artifact": "behavioral/results/derived/emobank_3way_summary/behavioral_emobank_metrics.csv",
          "key": "r_continuous"
        }
      ],
      "derivation": "direct_copy"
    },
    "v1.table_v1_1.peak_relative_depth": {
      "sources": [
        {
          "artifact": "v1/results/derived/v1_phase_a/{model}/e1_emobank_decodability.csv",
          "key": "r2"
        }
      ],
      "derivation": "argmax finite layer-wise r2, then layer/(L-1)"
    },
    "v2.table_v2_3a.c_v_net_rand": {
      "sources": [
        {
          "artifact": "v2/results/derived/v2_causal_dissociation_summary.json",
          "key": "c_v_net_rand"
        }
      ],
      "derivation": "direct_copy"
    },
    "v3.table_v3_1.overall_gate_decision": {
      "sources": [
        {
          "artifact": "v3/results/derived/v3_gate_decision.json",
          "key": "overall_decision"
        }
      ],
      "derivation": "direct_copy"
    }
  }
}
```

---

## 各Stageの集計仕様

### §4 Behavioral
- **Table B1: EmoBank Human Correspondence**
  - Reader V/A, Self V/A, Writer V/A の `pearson_r`, `pearson_p`, `spearman_rho`, `spearman_p`, `n`。Primaryは Reader/Self の $r, \rho$。MAE/RMSEは Secondary。
- **Table B2: Sensitivity**
  - `mean_aligned_delta`, 95% CI, `cohen_dz`, $t$, $p$, $q$, `mean_raw_delta`。
- **Table B3: Dose-Response**
  - $\Delta_1^*$, $\Delta_2^*$, $p_1, p_2, p_{IUT}, q_{IUT}$, `monotonicity_rate`, `secondary_slope`。
- **Table B4: Specificity**
  - $D_C, D_{CN}$, 差分, Cohen's $d$, Welch $t$, $p$, $q$。
- **Table B5: Reader–Self Coupling**
  - Primary: $\Delta R$ と $\Delta S$ の相関 $r_\Delta$, CI, $p, q, \rho_\Delta$。Secondary: raw相関。
- **Summary**: 各モデルの代表値を並べる（総合点・ランキングなし）。

### §5 V1
- **Table V1-1: Shared Decodability**
  - EmoBank layer-wise $R^2(l)$ の peak layer, peak depth, peak $R^2$, $|d_R^* - d_S^*|$、AIPsy probes。
- **Table V1-2: Shared Geometry**
  - Direct cross-decoding, RSA, Procrustes alignment gain。
- **Table V1-3: Semantic / Contextual Controls**
  - Original vs Paraphrase Bal Acc, **V1 Phase B nonfallback $N$ (必須併記)**, Shuffle drop, Outcome reversal $\Delta P$。
- **Table V1-4: Shared Causal Map**
  - Prompt-end layer causal profile $C_R(l), C_S(l)$ の peak layer/depth/magnitude, rank corr, cosine。
- **Table V1-5: Causal Interchangeability**
  - Matched donor vs random derangements: Specificity, $t$, permutation $p$, $q$, transfer ratio（Primary: $l=l_R^*, \alpha=1$）。
- **Table V1-6: Task-Specific Specialization**
  - Distinct site selection（ない場合は `No distinct sites identified`）、Site ablation effect、Task $\times$ SiteType interaction。

### §6 V2
- **Table V2-1: Representation Geometry**
  - Matched-plain: Base/Instruct peak depth, $\Delta$ peak depth, Mean RSA, Procrustes distortion, distortion center。Native/Format は Secondary。
- **Table V2-2: Reader–Self Sharing**
  - Base vs Instruct plain の全layer平均 sharing, $\Delta S_{\text{matched}}$, Native chat sharing, Format effect。
- **Table V2-3A & V2-3B: Causal Relocation & Controls**
  - Primary: $C_{\text{net,rand}}$（元出力 `c_v_net_rand`, `c_a_net_rand` の direct copy）。Secondary: $C_{\text{net,perp}}$。
  - 正のピーク層/深さ、重心、$\Delta d_{\text{peak}}$, $\Delta d_{\text{center}}$。**ピーク未検出時は NaN を維持（`fillna(0)` や除外禁止）**。
  - Control 表: raw, rand, perp, net_rand, net_perp, zero。
  - LMM結果表: 固定効果（元 LMM 出力からコピー。`metric="lmm_beta::<term>"` で一意化）。
- **Table V2-4: Distribution Recovery**
  - Matched-plain AUC recovery, $\Delta \text{EMD AUC}$, Max recovery, Best depth, Native AUC, Aligned AUC, Self minus Reader 差分。
- **Confirmatory Summary Table**: H1a, H1b, H2, H3, H4 の実数値・CI・$p/q$。

### §7 V3
- **Table V3-1: State-Induction Gate**
  - Sufficiency slope, Specificity, Attenuation ratio, Topic TVD（推定値・CI・合否）。
  - **4値 overall_decision**: `GO`, `GO (Valence-only)`, `GO (Arousal-only)`, `NO_GO`。
  - パイプライン継続判定: `pipeline_continues = (overall_decision == "GO")`。
  - `valence_pass = True/False`, `arousal_pass = True/False` をそれぞれ明記。
- **Table V3-2: Spatiotemporal 4-Maps Summary**
  - **`analysis_role = discovery`** として明記（Confirmatory と混在させない）。
  - $D_V, \beta_V, \gamma_V, C_V$ および Arousal の global peak layer/depth/stage。
  - A priori dissociation 表: $pre_V, pre_A$ での $\Delta d_{\text{peak}}, \Delta d_{\text{center}}$。
  - **必須QC**: 陰性コントロール $C(\text{response\_end})$ を必ず残す（RQ2 実行時）。
- **Table V3-3: Mediated Attenuation**
  - **独立 Confirmation set による集計（`analysis_role = confirmatory`）**。
  - Total shift, Residual shift, Absolute attenuation, Attenuation ratio, Net attenuation vs random。
- **Table V3-4: Confirmatory Replication & Matrix**
  - Llama, Gemma, OLMo における H1〜H4 の推定値、95% CI、閾値、合否。

---

## 修正・新規作成ファイル一覧

| 操作 | ファイルパス | 役割 |
|---|---|---|
| [NEW] | `src/affective_empathy_eval/paper_summary/schema.py` | 19列共通スキーマ定義、複数artifact/key対応、Provenance/Derivation定義、バリデータ |
| [NEW] | `src/affective_empathy_eval/paper_summary/common.py` | 共通ファイルI/O、Manifestビルダー、条件付き不変条件チェッカー |
| [NEW] | `behavioral/analysis/build_paper_summary.py` | §4 Behavioral 結果集計CLI（Read-only presentation） |
| [NEW] | `v1/scripts/build_paper_summary.py` | §5 V1 結果集計CLI（Read-only presentation） |
| [NEW] | `v2/scripts/build_paper_summary.py` | §6 V2 結果集計CLI（Derived正本直接ロード、NaN保持） |
| [NEW] | `v3/scripts/build_paper_summary.py` | §7 V3 結果集計CLI（Gate文字列保持、Discovery/Confirmatory直交分離、response_end保持） |
| [NEW] | `scripts/build_all_paper_summaries.py` | 全ステージ統括オーケストレーターCLI |
| [NEW] | `tests/test_paper_summary_invariants.py` | 論文上の条件付き不変条件テスト |

---

## 論文上の条件付き不変条件テスト

`tests/test_paper_summary_invariants.py` にて以下を検証します：
1. **Primaryファイルの純粋性と直交性**:
   - `primary_results.csv` は `is_primary == True` かつ `analysis_role not in ("secondary", "control", "diagnostic")` の行のみ。
   - `is_primary=True` かつ `analysis_role="discovery"` である V3 RQ2 が正しく含まれること。
2. **V2 NaN 保持**:
   - V2 で peak が未検出の場合、`0.0` 等で埋められず `np.nan` のまま保持されていること。
3. **V3 陰性コントロールの存在（Gate依存）**:
   - RQ2 が実行された場合（`completed` または `forced_after_no_go`）にのみ図表データに $C(\text{response\_end})$ が必ず存在すること。Gate により未実行（`not_run_due_to_gate`）の場合は非存在を正常受理すること。
4. **V1 Phase B nonfallback N の非欠落**:
   - V1 Phase B のセマンティックコントロール表において、`n_nonfallback_paraphrase_pairs`, `n_nonfallback_reversal_pairs` が必ず明記されていること。
5. **V3 Gate 文字列とパイプライン継続判定**:
   - `overall_decision` が `"GO"`, `"GO (Valence-only)"`, `"GO (Arousal-only)"`, `"NO_GO"` のいずれかであること。
   - `overall_decision == "GO"` の場合のみ `pipeline_continues == True` となること。
6. **必須 strata と Unique-Key 重複なし**:
   - 主要モデル・軸・条件が存在し、LMM term を含む `metric` 等によって `["stage", "rq", "family", "alignment", "task", "axis", "condition", "metric"]` の重複が一切ないこと。
7. **`pipeline_continues` に基づく State-aware な `--strict` モード**:
   - `pipeline_continues == False`（`NO_GO`, `GO (Valence-only)`, `GO (Arousal-only)`）の場合は下流成果物の非存在をエラーとせず `not_run_due_to_gate` を記録し、`pipeline_continues == True` の場合にのみ後段成果物の存在を必須とすること。
