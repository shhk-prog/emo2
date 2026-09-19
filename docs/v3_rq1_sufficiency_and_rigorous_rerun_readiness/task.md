# タスクリスト: V3 RQ1 Sufficiency修正および本番再実行盤石化（全18項目）

## ステータス概要

### 【P0: 最優先・科学的厳密化およびGate/成果物保護】
- [x] 1. V3 RQ1 の Sufficiency / Endogenous relevance 役割分離 (`v3/primary/run_rq1_state_induction.py`) <!-- id: 0 -->
  - Neutral + direction injection $\to$ Sufficiency / Dose-response / Specificity
  - Affective - affect subspace $\to$ Endogenous relevance / Subspace removal
  - シミュレーションおよび実モデル推論の両方で完全整合
- [x] 2. V3 RQ1 dry-run / pilot 出力先ディレクトリ完全分離 (`v3/primary/run_rq1_state_induction.py`) <!-- id: 1 -->
  - `args.dry_run` $\to$ `raw_dir / "dry_run"`, `derived_dir / "dry_run"`
  - `args.pilot` $\to$ `raw_dir / "pilot"`, `derived_dir / "pilot"`
  - 本番 `v3_gate_decision.json` の上書きを完全防止
- [x] 3. V3 pilot の production gate 上書き防止（上記2でディレクトリ分離） <!-- id: 2 -->
- [x] 4. Unified Runner (`src/affective_empathy_eval/run.py`) の gate path を dry-run 対応 <!-- id: 3 -->
  - dry-run 実行時は `v3/results/derived/dry_run/v3_gate_decision.json` を参照
- [x] 5. V2 RQ4 dry-run 出力先ディレクトリ完全分離 (`v2/primary/run_rq4_recovery_patching.py`) <!-- id: 4 -->
  - `raw_dir / "dry_run"`, `derived_dir / "dry_run"` へ完全分離

### 【P1: 統計・推定の頑健性と不整合解消】
- [x] 6. V2 RQ4 の Reader と Self で split seed を統一 (`v2/primary/run_rq4_recovery_patching.py`) <!-- id: 5 -->
  - 外側で一意に `train_indices, eval_indices` を生成し、両タスクへ同一 split を渡す
- [x] 7. V2 Confirmatory H1 で matched-plain から native-chat への自動 fallback を禁止 (`v2/primary/run_confirmatory_analysis.py`) <!-- id: 6 -->
  - Primary は厳格に matched-plain のみ（欠損時は `RuntimeError`）。Native-chat は Secondary に完全分離
- [x] 8. V2 Confirmatory H2 も native への fallback を Primary に使わない (`v2/primary/run_confirmatory_analysis.py`) <!-- id: 7 -->
  - Primary (Base plain vs Instruct matched-plain) と Secondary (Base plain vs Instruct native-chat) を完全分離
- [x] 9. V2 Confirmatory H1 に geometry 指標を併記 (`v2/primary/run_confirmatory_analysis.py`) <!-- id: 8 -->
  - `H1a_geometry_reorganization` (Procrustes distortion, RSA) と `H1b_decodability_profile_reorganization`
- [x] 10. V3 RQ3 Discovery で D profile も VA 両軸化 (`v3/primary/run_rq3_path_mediation.py`) <!-- id: 9 -->
  - `d_stim_v_profile`, `d_stim_a_profile`, `d_stim_joint_profile` を算出し、`stim_peak_layer = np.argmax(d_stim_joint_profile)`
- [x] 11. V3 RQ2 の 15件 causal sample も emotion-stratified に統一 (`v3/primary/run_rq2_spatiotemporal_maps.py`) <!-- id: 10 -->
  - 共通関数 `stratified_causal_subset` を実装し、各 emotion から均等抽出
- [x] 12. V3 RQ2 の β covariate fallback を one-hot 化 (`src/affective_empathy_eval/data.py`) <!-- id: 11 -->
  - `target_emotion`, `intensity`, `domain` を one-hot 化して重回帰へ投入
- [x] 13. V2 RQ4 の Primary に recovery AUC を設定し、`max_recovery_ratio` は Secondary に移行 <!-- id: 12 -->
  - `auc_recovery` を台形積分で算出し、Confirmatory H4 で Self vs Reader recovery AUC を比較
- [x] 14. Manifest validation 厳格化と RQ1 への RunManifest 追加 (`v3/primary/run_rq1_state_induction.py`) <!-- id: 13 -->
  - `config_hash`, `dataset_hash` 等の検証連携、`manifest_rq1_{family}.json` の出力
- [x] 15. V3 RQ2 の teacher-forced stage 明記 (`v3/primary/run_rq2_spatiotemporal_maps.py`) <!-- id: 14 -->
  - 出力 JSON に `"stage_evaluation_mode": "teacher_forced_joint_sequence"` を明記

### 【P2: 記述整合・共通化】
- [x] 16. V3 README の「QR」を「rank-aware SVD」に修正 (`v3/README.md`) <!-- id: 15 -->
- [x] 17. V2 cross-family bootstrap が N=4 であることを明記 (`v2/primary/run_confirmatory_analysis.py`) <!-- id: 16 -->
- [x] 18. `EXPECTED_DIRECTION` の共通モジュール化 (`src/affective_empathy_eval/affect_directions.py`) <!-- id: 17 -->

### 【検証・ドキュメント】
- [x] 19. 全テストスイート (`pytest`) による動作検証 <!-- id: 18 -->
- [x] 20. ドキュメント保存 (`docs/v3_rq1_sufficiency_and_rigorous_rerun_readiness/`) <!-- id: 19 -->
