# タスクリスト: 静的監査指摘28項目の修正および再実行対象の退避

## 1. 準備および結果退避
- [x] 1.1 `docs/audit_fixes_and_archive_for_rerun/` の初期化 <!-- id: 1 -->
- [x] 1.2 指定された結果（V2 RQ4, V3 全体, V1 Phase A, V1 Phase B, V1 Phase C E6）を安全に退避するスクリプト（`scripts/archive_targets_for_rerun.sh`）を配備 <!-- id: 2 -->

## 2. A. 必ず修正する箇所 (P0/P1 バグ修正)
- [x] 2.1 【P0】`v3/primary/run_confirmatory_replication.py`: Arousal の `d_profile_a` が未生成のバグ修正、raw $R^2$ への変更、`assert len(...) == num_layers` 追加 <!-- id: 3 -->
- [x] 2.2 【P1】`v3/primary/run_rq2_spatiotemporal_maps.py`: `rq2_sites` を軸別（`temporal_relative_depth_v/a`, `temporal_stage_v/a`, `a_priori_test_stage_v/a`）に分離 <!-- id: 4 -->
- [x] 2.3 `v3/primary/run_rq3_path_mediation.py`: `frozen_sites` を V/A 軸別に分離保存 <!-- id: 5 -->
- [x] 2.4 `v3/primary/run_confirmatory_replication.py`: V/A 別 temporal layer (`temporal_layer_v`, `temporal_layer_a`) での検証と intervention 分離 <!-- id: 6 -->
- [x] 2.5 【P1】V3 の $R^2$ クリップ撤廃: `run_rq2_spatiotemporal_maps.py`, `run_rq3_path_mediation.py`, `run_confirmatory_replication.py` で raw $R^2$ を使用 <!-- id: 7 -->
- [x] 2.6 `v3/primary/run_rq2_spatiotemporal_maps.py`: `pair_id` グループ不足時に通常 KFold へフォールバックせずエラー終了（dry-run 除く） <!-- id: 8 -->
- [x] 2.7 【P1】`v2/primary/run_rq4_recovery_patching.py`: `best_l` を 3 条件（Native, Matched-Plain, Aligned）に分離、`auc_recovery_matched_plain` と `delta_emd` の Primary 化 <!-- id: 9 -->
- [x] 2.8 【P1】`src/affective_empathy_eval/likelihood.py`: delimiter 挿入による厳密 prefix 保証と `require_strict_prefix=True` 化、およびテスト検証 <!-- id: 10 -->

## 3. B. キャッシュ・再現性の改善
- [x] 3.1 `src/affective_empathy_eval/manifests.py`: `dataset_hash` をファイル内容 sha256 化、`is_manifest_matching()` に `git_commit` 等の厳格照合を追加 <!-- id: 11 -->
- [x] 3.2 `v1/primary/run_phase_a.py`, `run_phase_b.py`: `is_manifest_matching()` による完全一致スキップ判定の導入 <!-- id: 12 -->
- [x] 3.3 `scripts/clear_stage_results.sh` & `scripts/archive_and_clean_results.py`: 削除型から timestamp 付き archive 型へ安全化 <!-- id: 14 -->
- [x] 3.4 Prompt hash の保存と `is_manifest_matching` の厳格化 <!-- id: 16 -->

## 4. C & D & E. 実験定義・細部の改善
- [x] 4.1 V3 RQ1: `resolved_relative_depth` の保存と RQ3 での 0.50 fallback 廃止 <!-- id: 17 -->
- [x] 4.2 V3: unmatched pair の暗黙 drop を防止し `exclusions_v3.csv` 記録 <!-- id: 19 -->
- [x] 4.3 V3: 「いつ」の表記・キー名を `a_priori_primary_stage_v/a` に整理 <!-- id: 20 -->
- [x] 4.4 V1 Phase B: カラム名を `n_nonfallback_*` および `n_primary_test_*` に変更 <!-- id: 21 -->
- [x] 4.5 V2 RQ4: `delta_emd` の全レイヤー保存 <!-- id: 22 -->

## 5. F & G & H. スクリプト・感度分析・数値条件の修正
- [x] 5.1 `scripts/run_candidate_space_sensitivity.py`: $\Delta V, \Delta A$ の相関・一致度比較スクリプトへ修正 <!-- id: 23 -->
- [x] 5.2 `scripts/run_production_reruns.sh`: V1 Phase B の Instruct 追加、V2 RQ4 への `--device` 伝播、`.venv` 必須化 <!-- id: 24 -->

## 6. 検証とレポート作成
- [x] 6.1 `walkthrough.md` の作成と完了報告 <!-- id: 28 -->
