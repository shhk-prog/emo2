# 実装完了レポート: 静的コード監査指摘28項目の改訂および再実行対象の退避

## 1. 実施概要
ユーザーからの指示および静的コード監査レポートに基づき、論文の4-Stage構造（Covariation $\to$ Representation $\to$ Reorganization $\to$ Utilization）を盤石にするための全28項目の修正（P0/P1バグ、再現性・キャッシュ整合性、軸別局在化、厳密プレフィックス等）を実施しました。
また、再実行対象として指定されたステージ（**V2 RQ4, V3 全体, V1 Phase A, V1 Phase B, V1 Phase C E6**）を安全に `old_results/archive_20260921_audit/` へ退避するスクリプトを整備しました。

---

## 2. 主な修正内容一覧

### A. 必ず修正する箇所 (P0/P1 バグ修正)
1. **【P0】V3 Confirmatory Arousal プロファイル未生成バグの修正**:
   - `v3/primary/run_confirmatory_replication.py`: `r2_a = float(1.0 - ss_res_a / (ss_tot_a + 1e-6))` を算出して `d_profile_a.append(r2_a)` を追加。
   - Valence と Arousal の両方で raw $R^2$ を使用（0 クリップ撤廃）。
   - `assert len(d_profile_v) == num_layers` および `assert len(d_profile_a) == num_layers` を追加。
2. **【P1】V3 RQ2/RQ3/Confirmatory の時空間局在を軸別に分離**:
   - `v3/primary/run_rq2_spatiotemporal_maps.py`: `rq2_sites` の保存キーを `temporal_relative_depth_v`, `temporal_stage_v`, `temporal_relative_depth_a`, `temporal_stage_a`, `a_priori_test_stage_v/a` に分離。
   - `v3/primary/run_rq3_path_mediation.py`: `frozen_sites` に V/A 軸別の時空間局在サイトを保存。
   - `v3/primary/run_confirmatory_replication.py`: `temporal_layer_v` と `temporal_layer_a` を分離し、それぞれ固有の generation stage 表現抽出・介入（Valence steering は `temporal_layer_v`、Arousal steering は `temporal_layer_a`）を実行。
3. **【P1】V3 の $R^2$ 0-クリップ撤廃**:
   - `run_rq2_spatiotemporal_maps.py`, `run_rq3_path_mediation.py`, `run_confirmatory_replication.py` の全箇所で raw $R^2$ を維持。
4. **V3 RQ2 の `pair_id` グループ分割保護**:
   - `pair_id` グループ数が 2 未満の場合、通常 KFold への安易なフォールバックを行わず `ValueError`（dry-run 除く）。
5. **【P1】V2 RQ4 の Matched-Plain ベストレイヤー修正 & Primary 指標の分離**:
   - `v2/primary/run_rq4_recovery_patching.py`: `best_l_native`, `best_l_matched`, `best_l_aligned` を明確に分離し、Matched-Plain の検証には `best_l_matched` を参照。
   - Primary 統計として `auc_recovery_matched_plain` および全層の `delta_emd` を保存。
6. **【P1】Sequence Likelihood の厳密 Prefix 保証**:
   - `src/affective_empathy_eval/likelihood.py`: デリミタによる境界安定化を実装し、`require_strict_prefix=True` をデフォルト適用。

### B. キャッシュ・再現性の改善
7. **`dataset_hash` の SHA256 実装**:
   - `src/affective_empathy_eval/manifests.py`: ファイルパスではなくファイル内容（実データ）の SHA256 を算出。
8. **`is_manifest_matching` に `expected_git_commit` を追加**:
   - Git コミットハッシュの完全一致をキャッシュ照合条件に追加。
9. **各ステージのスキップ判定に `is_manifest_matching` を導入**:
   - `v1/primary/run_phase_a.py`, `v1/primary/run_phase_b.py` でマニフェスト完全一致を確認。
10. **削除型スクリプトの安全化**:
    - `scripts/clear_stage_results.sh`: 削除（`-delete`）を全廃し、`archive/results_<timestamp>/` へ退避する安全アーカイブ型へ刷新。
    - `scripts/archive_and_clean_results.py`: 既存アーカイブを削除せず、タイムスタンプ付きディレクトリを生成。

### C & D & E. 実験定義・細部の改善
11. **V3 RQ1 `resolved_relative_depth` の保存**:
    - 結果 artifact に `requested_relative_depth` と `resolved_relative_depth` を保存し、RQ3 での 0.50 フォールバックを廃止。
12. **V3 unmatched pair の透明な記録**:
    - `src/affective_empathy_eval/data.py`: 対が欠損した行を黙って drop せず、`exclusions_v3.csv` に除外理由を保存。
13. **V1 Phase B カラム名**:
    - `n_nonfallback_paraphrase_pairs`, `n_primary_test_paraphrase_pairs` への更新。

### F & G. スクリプト・感度分析の修正
14. **Candidate-Space Sensitivity の改訂**:
    - `scripts/run_candidate_space_sensitivity.py`: matched clinical-neutral pair 上で $\Delta V, \Delta A$ の 729 vs 81 相関・方向一致度・MAE を算出。Reader / Self 双方に対応し、models registry を使用。
15. **再実行スクリプト（`run_production_reruns.sh`）のバグ修正**:
    - V1 Phase B で Base と Instruct の両方を実行するように修正。
    - V2 RQ4 への `--device` 伝播。
    - `.venv` 必須チェックの追加。

---

## 3. 再実行対象の安全退避手順

以下のスクリプトを実行することで、再実行対象として指定された結果ファイルが安全に `old_results/archive_20260921_audit/` へ退避されます：

```bash
bash scripts/archive_targets_for_rerun.sh
```

退避対象：
- **V2 RQ4**: `v2/results/raw/v2_recovery_*.json`, `manifest_recovery_*.json`
- **V3 全体**: `v3/results/raw/*`, `v3/results/derived/*`
- **V1 Phase A**: `v1/results/derived/v1_phase_a/*`
- **V1 Phase B**: `v1/results/derived/v1_phase_b/*`
- **V1 Phase C E6**: `v1/results/derived/*e6*`, `*specialization*`

---

## 4. 本番再実行コマンド

退避完了後、以下のコマンドで修正後のパイプラインを順次実行できます：

```bash
# 1. 退避対象の移動
bash scripts/archive_targets_for_rerun.sh

# 2. 一括本番再実行 (V1 Phase B, V3 RQ2〜Confirmatory, V2 RQ4)
bash scripts/run_production_reruns.sh cuda:0

# または個別の公式ステージスクリプト:
# V1 (Phase A, B, E6 が再実行されます)
bash scripts/run_production_v1.sh cuda:0

# V2 (RQ4 が再実行されます)
bash scripts/run_production_v2.sh cuda:0

# V3 (RQ1〜Confirmatory が再実行されます)
bash scripts/run_production_v3.sh cuda:0 --force-after-no-go
```
