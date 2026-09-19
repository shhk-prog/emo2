# タスクリスト: コードフリーズ前の最終厳格化（Pass 3）

## 1. 【P0】V3 RQ2 `is_manifest_matching` 未インポート修正 & キャッシュ検証テスト
- [x] `v3/primary/run_rq2_spatiotemporal_maps.py` に `is_manifest_matching` のインポートを追加 <!-- id: p0-v3-rq2-import -->
- [x] キャッシュヒット時に再計算がスキップされるテスト `test_rq2_cache_hit_does_not_recompute` を作成 <!-- id: p0-v3-rq2-cache-test -->

## 2. 【P0】V2 RQ4 `task_comparison` の matched-plain Primary 化
- [x] `v2/primary/run_rq4_recovery_patching.py` で `task_comparison` を `primary_matched_plain` と `secondary_native_chat` に構造化 <!-- id: p0-v2-rq4-task-comp -->
- [x] generic key が matched-plain を指すよう修正 <!-- id: p0-v2-rq4-generic-key -->

## 3. 【P1】V2 RQ3 の native-chat alias 統一
- [x] `v2/primary/run_rq3_causal_map.py` 内の `inst_reader` / `inst_self` を `inst_native_reader` / `inst_native_self` に置換・統一 <!-- id: p1-v2-rq3-aliases -->

## 4. 【P1】V2 RQ3 `post_training_comparison` generic alias の整理
- [x] `post_training_comparison` の曖昧性を排除し、`deprecated_alias_of` または明示キーに限定 <!-- id: p1-v2-rq3-post-comp -->

## 5. 【P1】V3 Confirmatory の層指定由来を config & manifest に明記
- [x] `configs/v3_experiments.yaml` の `confirmatory` セクションに `selection_source: "qwen_discovery_frozen"` を追加 <!-- id: p1-v3-conf-source-cfg -->
- [x] `v3/primary/run_confirmatory_replication.py` の manifest に `confirmatory_site_selection_source` を記録 <!-- id: p1-v3-conf-source-manifest -->

## 6. 【P1】V3 Confirmatory H3 QC 判定を Primary absolute attenuation CI lower に変更
- [x] `v3/primary/run_confirmatory_replication.py` の H3 QC を `atten_v_low > min_mediated_attenuation` (0.0) に変更 <!-- id: p1-v3-conf-h3-qc -->
- [x] ratio による判定は Secondary 記録のみに限定 <!-- id: p1-v3-conf-ratio-sec -->

## 7. 【P1】`all_confirmed` / `CONFIRMED` の論文主結果化防止
- [x] `auxiliary_qc_all_pass: bool(...)` に変更 <!-- id: p1-v3-conf-qc-name -->
- [x] status を `"EFFECT_ESTIMATES_AVAILABLE"` に整理 <!-- id: p1-v3-conf-status -->

## 8. 【P1】V3 RQ2 cache manifest の厳格化 (config_hash, dataset_hash, code_version)
- [x] `manifest_config` 辞書を構築し、`expected_config_hash`, `expected_dataset_hash`, `expected_code_version` を検証 <!-- id: p1-v3-rq2-manifest-hash -->

## 9. 【P1】V3 RQ3 / Confirmatory の cache manifest 厳格化
- [x] `v3/primary/run_rq3_path_mediation.py` の cache 検証を厳格化 <!-- id: p1-v3-rq3-manifest-hash -->
- [x] `v3/primary/run_confirmatory_replication.py` の cache 検証を厳格化 <!-- id: p1-v3-conf-manifest-hash -->

## 10. 【P1】V2 RQ3 / RQ4 cache manifest の full config hash 化
- [x] `v2/primary/run_rq3_causal_map.py` の manifest_config を full config 化 <!-- id: p1-v2-rq3-manifest-full -->
- [x] `v2/primary/run_rq4_recovery_patching.py` の manifest_config を full config 化 <!-- id: p1-v2-rq4-manifest-full -->

## 11. 【P1】V3 README の介入手法記述の正確化
- [x] `v3/primary/README.md` で direction injection (additive) と subspace removal を峻別して明記 <!-- id: p1-v3-readme-operators -->

## 12. 【P1】V3 の Primary 用語を `Endogenous relevance` に統一
- [x] `v3/primary/run_rq1_state_induction.py` 等で `endogenous_relevance_v_pass` を Primary 化、`necessity_*` は互換 alias に整理 <!-- id: p1-v3-endogenous-relevance -->

## 13. 【P1】Root README の generation stages を teacher-forced と明記
- [x] `README.md` の記述を "teacher-forced candidate sequence" に揃える <!-- id: p1-root-readme-teacher-forced -->

## 14. 【P2】V3 RQ3 dry-run simulation の ratio 定義を real path に整合
- [x] `simulate_path_mediation_confirmation` 内で `MIN_NATURAL_SHIFT = 0.05` を適用 <!-- id: p2-v3-rq3-dryrun-ratio -->

## 15. 【P2】V3 Confirmatory の hard-coded threshold を config `confirmatory.qc` に集約
- [x] `configs/v3_experiments.yaml` に `confirmatory.qc` 設定を追加し、スクリプトから参照 <!-- id: p2-v3-conf-qc-cfg -->

## 16. 全体検証 & ドキュメント更新
- [x] `py_compile` による文法・構文チェック <!-- id: p0-verify-compile -->
- [x] `pytest` による単体・結合テスト実行 <!-- id: p0-verify-pytest -->
- [x] dry-run パイプライン実行の確認 <!-- id: p0-verify-dryrun -->
- [x] `docs/pre_freeze_rigorous_alignment_pass3/walkthrough.md` の作成・保存 <!-- id: p0-verify-walkthrough -->
