# 静的監査指摘28項目の包括的修正および再実行対象アーカイブ 実装計画

## 概要
最新の静的コード監査で指摘された全28項目（特に V3 Confirmatory の Arousal プロファイル未生成バグ、V3 RQ2/RQ3/Confirmatory の時空間局在軸別分離、V2 RQ4 のベストレイヤー指標修正、Likelihood の厳密プレフィックス保証、マニフェスト完全一致キャッシュ判定など）を包括的に修正し、指定された結果（V2 RQ4, V3 全体, V1 Phase A, V1 Phase B, V1 Phase C E6）をタイムスタンプ付きアーカイブへ退避して再実行可能状態を整えます。

---

## ユーザー確認事項
- **再実行対象ディレクトリの移動**:
  - `v2/results/raw/v2_recovery_*.json`
  - `v3/results/` (raw / derived 全体)
  - `v1/results/derived/v1_phase_a/`
  - `v1/results/derived/v1_phase_b/`
  - `v1/results/derived/v1_phase_c_e6/` および E6 関連ファイル
  これらを `old_results/archive_20260921_audit/` へ安全に退避（mv）します。

---

## 主な変更項目

### 1. 【P0/P1】V3 の時空間マッピング・Confirmatory 修正
- `v3/primary/run_confirmatory_replication.py`:
  - `d_profile_a` に `r2_a = float(1.0 - ss_res_a / (ss_tot_a + 1e-6))` を正しく追加。
  - $R^2$ を 0 でクリップせず raw $R^2$ を使用。
  - `assert len(d_profile_v) == num_layers` および `assert len(d_profile_a) == num_layers` を追加。
  - Valence と Arousal の temporal layer を分離 (`temporal_layer_v`, `temporal_layer_a`) し、それぞれの intervention layer に適用。
- `v3/primary/run_rq2_spatiotemporal_maps.py`:
  - `rq2_sites` の保存キーを軸別に分離 (`temporal_relative_depth_v/a`, `temporal_stage_v/a`, `a_priori_test_stage_v/a`)。
  - raw $R^2$ による peak/dissociation 計算。
  - `pair_id` グループ数が 2 未満の場合にフォールバックせず `ValueError`。
- `v3/primary/run_rq3_path_mediation.py`:
  - frozen site を V/A 軸別に分離保存。0.50 フォールバックの撤廃。

### 2. 【P1】V2 RQ4 のベストレイヤー指標と Primary 統計の分離
- `v2/primary/run_rq4_recovery_patching.py`:
  - `best_l` を `best_l_native`, `best_l_matched`, `best_l_aligned` に明確に分離。
  - sample record でも matched-plain の layer を参照。
  - Primary 指標として `auc_recovery_matched_plain` と `delta_emd` を保存。

### 3. 【P1】Sequence Likelihood の厳密 Prefix 保証
- `src/affective_empathy_eval/likelihood.py`:
  - prompt と candidate の間にデリミタを保証し、`require_strict_prefix=True` を production で適用。全ファミリーのトークナイザでテスト。

### 4. キャッシュ・マニフェスト完全一致照合と再現性
- `src/affective_empathy_eval/manifests.py`:
  - `dataset_hash` をファイル内容の SHA256 に変更。
  - `is_manifest_matching()` に `git_commit`、`model_revision` などの照合を追加。
- 各ステージの実行スクリプト（`run_phase_a.py`, `run_phase_b.py`, `run_behavioral_*.py` 等）にマニフェスト厳格照合を適用。
- `scripts/archive_and_clean_results.py` / `clear_stage_results.sh` の安全化（上書き・削除の禁止）。

### 5. 実行スクリプト等の修正
- `scripts/run_production_reruns.sh`:
  - V1 Phase B の Instruct 追加。
  - V2 RQ4 への `--device` 伝播。
  - `.venv` 必須化チェック。

---

## 検証計画
1. **ユニットテスト**:
   - `pytest` を実行し、既存テストおよび新規追加テスト（strict prefix, V3 V/A 分離など）が全てパスすることを確認。
2. **Dry-run 検証**:
   - V1 Phase A/B, V2 RQ4, V3 RQ1〜Confirmatory の `--dry-run` を実行し、エラーなく完了することを確認。
