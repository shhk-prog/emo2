# タスクリスト: 実験結果のアーカイブ退避、出力モジュール化・逐次保存・再開機能の実装

## 1. 既存結果のアーカイブ退避
- [x] 1.1 `behavioral/results/`, `v1/results/`, `v2/results/`, `v3/results/` の既存全結果を `old_results/archive_20260921_040722_pre_modular/` へ移動 <!-- id: archive_results -->
- [x] 1.2 各ステージの `results/`（raw, derived, checkpoints, cache）をクリーン初期化（.gitkeep配置） <!-- id: clean_results_dirs -->

## 2. 実験結果ファイル名の命名規則統一と段階・実験の明記
- [x] 2.1 **Behavioral**: EmoBank（`behavioral_emobank_*`）と AIPsy（`behavioral_aipsy_*`）の raw/derived 出力名統一 <!-- id: naming_behavioral -->
- [x] 2.2 **V1 E1 & E2**: `run_phase_a.py` の出力を E1（`v1_e1_decodability_*`）と E2（`v1_e2_cross_decoding_*`, `v1_e2_rsa_*`, `v1_e2_alignment_*`）に完全分離 <!-- id: naming_v1_e1_e2 -->
- [x] 2.3 **V1 E5**: `run_phase_b.py` の出力を `v1_e5_semantic_controls_*` に明記 <!-- id: naming_v1_e5 -->
- [x] 2.4 **V1 E3 & E4**: `run_phase_c.py` の出力を E3（`v1_e3_causal_map_*`）と E4（`v1_e4_interchangeability_*`）に明確分離 <!-- id: naming_v1_e3_e4 -->
- [x] 2.5 **V1 E6**: `run_e6_specialization.py` の出力を `v1_e6_double_dissociation_*` に明記 <!-- id: naming_v1_e6 -->
- [x] 2.6 **V2 RQ1 & RQ2**: `run_rq1_rq2_cross_decoding.py` の出力を RQ1（`v2_rq1_decodability_preservation_*`）と RQ2（`v2_rq2_geometry_transformation_*`）に分離 <!-- id: naming_v2_rq1_rq2 -->
- [x] 2.7 **V2 RQ3 & RQ4**: RQ3（`v2_rq3_causal_relocation_*`）と RQ4（`v2_rq4_recovery_patching_*`）に明記 <!-- id: naming_v2_rq3_rq4 -->
- [x] 2.8 **V3 RQ1 - RQ3 & Confirmatory**:
  - RQ1: `v3_rq1_gate_*`
  - RQ2: `v3_rq2_spatiotemporal_maps_*`
  - RQ3: `v3_rq3_path_mediation_*`
  - Confirmatory: `v3_confirmatory_replication_*` <!-- id: naming_v3 -->

## 3. 逐次保存・成否フラグ（execution_success: true/false）・途中再開（resume）機能
- [x] 3.1 各出力 JSON に標準メタデータブロック（`execution_status: "success"` / `"failed"`, `execution_success: true/false`, `stage`, `experiment_id`, `completed_at` 等）を追加 <!-- id: success_flag -->
- [x] 3.2 ループ内処理完了ごとの即時逐次保存の実装（一括保存の完全撤廃） <!-- id: incremental_save -->
- [x] 3.3 途中再開（resume）ロジックの導入: `execution_success == true` かつ manifest 一致の場合のみスキップ、未完了・失敗ファイルは再実行 <!-- id: resume_support -->

## 4. 統合ランナー・集計スクリプト・テストの整合性更新
- [x] 4.1 `src/affective_empathy_eval/run.py` および各ステージ集計スクリプトの出力パス・ファイル名参照を更新 <!-- id: update_runners -->
- [x] 4.2 `scripts/run_production_reruns.sh`, `scripts/reaggregate_*.py` の更新 <!-- id: update_scripts -->
- [x] 4.3 構文コンパイル (`compileall`) と pytest スイート実行による検証 <!-- id: verify_tests -->
- [x] 4.4 stage all dry-run スモークテストによる途中再開・逐次保存の動作検証 <!-- id: verify_dryrun -->
- [x] 4.5 `walkthrough.md` の作成と報告 <!-- id: complete_walkthrough -->
