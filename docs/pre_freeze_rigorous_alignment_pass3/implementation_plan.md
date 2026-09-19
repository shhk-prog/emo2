# コードフリーズ前の最終厳格化 実装計画（Pass 3）

本計画は、Behavioral $\rightarrow$ V1 $\rightarrow$ V2 $\rightarrow$ V3 の一貫した論文ストーリー（Covariation $\rightarrow$ Representation / Causal Overlap $\rightarrow$ Post-training-Associated Reorganization $\rightarrow$ Localized Causal Leverage）を維持しつつ、本番実験の最終再実行およびコードフリーズに向けた残存の曖昧性・バグ・命名の不整合（全15項目）を厳密に解消するための実装計画である。

---

## ユーザー確認事項（User Review Required）

- **論文本体ストーリーおよび実験フレームワークの変更はありません**。
- **新規の実験・新規RQは一切追加せず**、既存の実装・出力形式・命名を事前登録仕様および論文主張と完全に一致させます。
- **後方互換性**: 既存の集計スクリプトや互換性のために、旧キー（`necessity_*`, `inst_reader` 等）は deprecated alias として安全に保持または明示的に指定します。

---

## 変更内容の概要（全15項目）

### 1. 【P0】V3 RQ2 `is_manifest_matching` import & キャッシュ検証テスト
- **対象**: `v3/primary/run_rq2_spatiotemporal_maps.py`, `tests/test_refinement_suite.py`
- **内容**:
  - `from affective_empathy_eval.manifests import create_run_manifest, is_manifest_matching` に修正。
  - キャッシュヒット時に計算ブランチに入らず正しく既存結果が再利用される単体テスト `test_rq2_cache_hit_does_not_recompute` を追加。

### 2. 【P0】V2 RQ4 `task_comparison` の matched-plain Primary 化
- **対象**: `v2/primary/run_rq4_recovery_patching.py`
- **内容**:
  - family 単位の返り値 `task_comparison` を明示的に分離：
    ```python
    "task_comparison": {
        "primary_matched_plain": {
            "diff_max_recovery_self_vs_reader": diff_max_ratio_matched,
            "diff_auc_recovery_self_vs_reader": diff_auc_matched,
        },
        "secondary_native_chat": {
            "diff_max_recovery_self_vs_reader": diff_max_ratio_native,
            "diff_auc_recovery_self_vs_reader": diff_auc_native,
        },
    }
    ```
  - トップレベルの generic key `diff_auc_recovery_self_vs_reader` も matched-plain の値を指すよう保証。

### 3. 【P1】V2 RQ3 の native-chat alias 統一
- **対象**: `v2/primary/run_rq3_causal_map.py`
- **内容**:
  - 内部表現 `fam_causal["inst_reader"]` / `fam_causal["inst_self"]` を `fam_causal["inst_native_reader"]` / `fam_causal["inst_native_self"]` に統一。
  - `inst_matched_reader` / `inst_matched_self` との混同や、Instruct=native-chat と誤認されるリスクを完全に排除。

### 4. 【P1】V2 RQ3 `post_training_comparison` generic alias の整理
- **対象**: `v2/primary/run_rq3_causal_map.py`
- **内容**:
  - `post_training_comparison` の generic alias を整理し、`post_training_comparison_matched` と `post_training_comparison_native` を明示。後方互換性用には `deprecated_alias_of: "post_training_comparison_matched"` を付与。

### 5. 【P1】V3 Confirmatory の層指定由来を config & manifest に明記
- **対象**: `configs/v3_experiments.yaml`, `v3/primary/run_confirmatory_replication.py`
- **内容**:
  - config に `selection_source: "qwen_discovery_frozen"` を明記。
  - manifest のメタデータにも `"confirmatory_site_selection_source": "qwen_discovery_frozen"` を永続化。

### 6. 【P1】V3 Confirmatory H3 QC 判定を Primary absolute attenuation CI lower に準拠
- **対象**: `v3/primary/run_confirmatory_replication.py`, `configs/v3_experiments.yaml`
- **内容**:
  - config に `confirmatory.qc.min_mediated_attenuation_ci_lower: 0.0` を定義。
  - QC 判定を `h3_pass_v = bool(atten_v_low > min_mediated_attenuation)` に修正（ratio による判定は Secondary 記録のみに限定）。

### 7. 【P1】`all_confirmed` / `CONFIRMED` の論文主結果化防止
- **対象**: `v3/primary/run_confirmatory_replication.py`
- **内容**:
  - 出力キーを `auxiliary_qc_all_pass: bool(...)` に変更。
  - status を `"status": "EFFECT_ESTIMATES_AVAILABLE"` に変更し、1bit判定ゲームに見えないよう整理。

### 8. 【P1】V3 RQ2 cache manifest の厳格化 (config_hash, dataset_hash, code_version)
- **対象**: `v3/primary/run_rq2_spatiotemporal_maps.py`
- **内容**:
  - `manifest_config` 辞書（analysis_role, family, model_id, dataset_path, semantic_stages, alpha_sweep, causal_reference_alpha, n_causal_samples, seed, subsample）を作成。
  - `expected_config_hash`, `expected_dataset_hash`, `expected_code_version` を `is_manifest_matching` に渡してキャッシュ判定。

### 9. 【P1】V3 RQ3 / Confirmatory の cache manifest 厳格化
- **対象**: `v3/primary/run_rq3_path_mediation.py`, `v3/primary/run_confirmatory_replication.py`
- **内容**:
  - RQ3 および Confirmatory でも上記と同様に config 全体ハッシュ、データセットハッシュ、コードバージョンを用いた厳格なキャッシュ判定を実装。

### 10. 【P1】V2 RQ3 / RQ4 cache manifest の full config hash 化
- **対象**: `v2/primary/run_rq3_causal_map.py`, `v2/primary/run_rq4_recovery_patching.py`
- **内容**:
  - 部分辞書ではなく実験設定全体（`v2_config` を含む辞書）をハッシュ化して manifest 生成・検証に使用。

### 11. 【P1】V3 README の介入手法記述の正確化
- **対象**: `v3/primary/README.md`
- **内容**:
  - Direction injection ($h' = h + \alpha \sigma_h \hat{d}$) と Subspace removal ($h' = h - QQ^T (h - \mu_{\text{neu}})$) の2種類を峻別して正確に記述。

### 12. 【P1】V3 Primary 用語を `Endogenous relevance` に統一
- **対象**: `v3/primary/run_rq1_state_induction.py` 等
- **内容**:
  - `endogenous_relevance_v_pass` を Primary 出力キーにし、`necessity_v_pass` は後方互換 alias として維持。

### 13. 【P1】Root README の generation stages を teacher-forced と明記
- **対象**: `README.md`
- **内容**:
  - "causal leverage is concentrated at particular layers and stages along the teacher-forced candidate sequence" と記述を整合。

### 14. 【P2】V3 RQ3 dry-run simulation の ratio 定義を real path に整合
- **対象**: `v3/primary/run_rq3_path_mediation.py`
- **内容**:
  - `simulate_path_mediation_confirmation` 内でも `MIN_NATURAL_SHIFT = 0.05` を適用し、実経路のロジックと一致させる。

### 15. 【P2】V3 Confirmatory の hard-coded threshold を config `confirmatory.qc` に集約
- **対象**: `configs/v3_experiments.yaml`, `v3/primary/run_confirmatory_replication.py`
- **内容**:
  - `min_sufficiency_slope: 0.1`, `min_mediated_attenuation_ci_lower: 0.0`, `min_temporal_contrast: 0.0` を config に集約。

---

## 検証計画（Verification Plan）

1. **構文・コンパイルチェック**:
   - `python -m py_compile` を対象スクリプト全件に対して実行。
2. **自動単体・結合テスト**:
   - `pytest tests/test_confirmatory_pipeline.py tests/test_production_entrypoints.py tests/test_refinement_suite.py tests/test_manifest_provenance.py -v`
   - 新設の `test_rq2_cache_hit_does_not_recompute` を含む全テストの通過を確認。
3. **Dry-run 統合実行**:
   - `python tests/run_all_v3_dryruns.py`
   - V2 / V3 entrypoint の dry-run 実行を確認し、結果 JSON のキー構造（`primary_matched_plain`, `auxiliary_qc_all_pass`, `endogenous_relevance_*` 等）を直接検証。
4. **ドキュメント成果物**:
   - `docs/pre_freeze_rigorous_alignment_pass3/walkthrough.md` に実施内容・テスト結果・差分を詳細に記録。
