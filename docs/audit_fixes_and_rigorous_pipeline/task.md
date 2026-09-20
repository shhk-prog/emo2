# タスクリスト: コード監査指摘対応（21項目）およびパイプライン厳密化

## 1. P1: 論文の結論に影響する重要修正 (Items 1-12)
- [x] 1. V1 Phase A キャッシュ判定修正 (`run_phase_a.py`): `--dataset both/aipsy` で AIPsy 3成果物と manifest ハッシュを確認 <!-- id: 1 -->
- [x] 2. V1 Phase B pair leakage fallback 削除 (`run_phase_b.py`): StratifiedKFold フォールバックと `test_pair_ids = train_pair_ids` を完全撤廃 <!-- id: 2 -->
- [x] 3. V1 Phase B semantic control 生成品質 (`prepare_v1_phase_b_controls.py`): fallback フラグ列追加、Primary では fallback 除外 <!-- id: 3 -->
- [x] 4. V1 Phase A Direct cross-decoding 負の R² 保持 (`run_phase_a.py`): `direct_transfer_score_raw` をそのまま保存し clip を撤廃 <!-- id: 4 -->
- [x] 5. V1 Phase A undefined metric NaN 化 (`run_phase_a.py`): 例外時の 0.5/0 代入を廃止し `np.nan` + `status`/`failure_reason` 記録 <!-- id: 5 -->
- [x] 6. V2 RQ1/RQ2 summary を matched-plain Primary へ修正 (`run_rq1_rq2_cross_decoding.py`): Primary/Secondary を明示分離し bootstrap も matched から算出 <!-- id: 6 -->
- [x] 7. V2 RQ4 eval=train fallback 削除 (`run_rq4_recovery_patching.py`): `eval_indices = train_indices` を例外化し `isdisjoint` アサーション追加 <!-- id: 7 -->
- [x] 8. V2 RQ4 の train_ratio (0.7) config 化 (`run_rq4_recovery_patching.py`): ハードコード `0.7` を config 読み込みへ統一 <!-- id: 8 -->
- [x] 9. V3 RQ2 の「いつ」の特定とアンカー明確化 (`run_rq2_spatiotemporal_maps.py`): `a_priori_test_stage` と全 stage 探索 peak の両方を明記 <!-- id: 9 -->
- [x] 10. V3 Confirmatory の stage を frozen artifact から読み込み (`run_confirmatory_replication.py`): ハードコード撤廃 <!-- id: 10 -->
- [x] 11. V3 RQ3 の split/seed config 化 (`run_rq3_path_mediation.py`): `seed`, `discovery_ratio` を config から取得し disjoint アサーション追加 <!-- id: 11 -->
- [x] 12. 全 Primary から hardcoded seed を排除: `base_seed = int(config["seed"])` からの一貫派生と manifest 記録 <!-- id: 12 -->

## 2. P2: 再現性・査読対応・データ整合性の強化 (Items 13-21)
- [x] 13. `results/raw` 上書き防止と `run_id` 体系導入: `YYYYMMDDTHHMMSSZ_<git>_<config>` 形式の run_id 体系と追記保存の確立 <!-- id: 13 -->
- [x] 14. Hugging Face モデル revision の固定 (`configs/models.yaml`): revision 定義と `from_pretrained(..., revision=...)` 対応 <!-- id: 14 -->
- [x] 15. Prompt hash の保存: prompt template 文字列の SHA256 ハッシュを manifest に記録 <!-- id: 15 -->
- [x] 16. Sequence Likelihood の token 境界厳密テスト: prompt 末尾と candidate 先頭のトークンマージ検証テスト追加 <!-- id: 16 -->
- [x] 17. V1 Phase C の resume/cache validation 強化 (`run_phase_c.py`): manifest のモデル・データ・シード・プロンプト一致検証 <!-- id: 17 -->
- [x] 18. Candidate-space sensitivity の README 記述修正: 過度な記述を改め、Supplementary での感度分析定義に更新 <!-- id: 18 -->
- [x] 19. Sequence likelihood の長さ感度と `joint conditional sequence log-likelihood` 表記徹底 <!-- id: 19 -->
- [x] 20. V1 Phase B の seed / train_ratio を config 化 (`configs/v1_experiments.yaml`) <!-- id: 20 -->
- [x] 21. V1 Phase C の split ratio を config 化 (`configs/v1_experiments.yaml` の `phase_c.discovery_ratio: 0.5`) <!-- id: 21 -->

## 3. 検証
- [x] EmoBank あり + AIPsy なしの cache skip 回避テストの実行
- [x] 全体 pytest テストスイートの実行 (95 passed)
- [x] V1, V2, V3 各 primary スクリプトの dry-run 実行確認 (全 11 スクリプト正常完了)
