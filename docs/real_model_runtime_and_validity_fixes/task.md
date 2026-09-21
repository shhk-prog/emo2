# Task: 実モデル実行経路 BLOCKER 解消および論文妥当性修正 (Real Model Runtime & Validity Fixes)

## 概要
ユーザーレビューで指摘された「実モデル実行経路でのみ発生する BLOCKER (NameError/未定義変数/未初期化等)」および「論文妥当性のための数理・統計修正 (V2 PCA-Procrustes, V3 H3 random subspace control, V1 E4 Permutation 等)」を網羅的に修正し、Qwen実モデルによる小規模スモークテストおよび全テストを完走させる。

## タスクリスト

### A. 実モデル実行 BLOCKER の修正
- [ ] 1. `v1/primary/run_phase_c.py`: `torch_dtype` を `actual_torch_dtype` に修正 (Line ~743)
- [ ] 2. `v1/primary/run_phase_c.py`: `e3_patching_records` を `e3_causal_records` に修正 (Line ~1726)
- [ ] 3. `v1/primary/run_phase_c.py`: データ読み込みを `data_file` (`args.data_path`) に一本化
- [ ] 4. `v3/primary/run_rq1_state_induction.py`: `run_real_state_induction()` のスコープ外 `v3_cfg` 参照を引数経由に修正
- [ ] 5. `v3/primary/run_rq2_spatiotemporal_maps.py`: `ref_alpha_idx = alpha_sweep.index(causal_reference_alpha)` を関数冒頭で算出し未定義エラー解消
- [ ] 6. `v3/primary/run_rq2_spatiotemporal_maps.py`: 実モデル関数内の未定義 `dry_run` 判定を削除し厳格な例外化へ
- [ ] 7. `v3/primary/run_rq3_path_mediation.py`: `ActivationHookManager`, `HookPoint` のインポートを追加
- [ ] 8. `v3/primary/run_rq3_path_mediation.py`: `residual_rand_v_list, residual_rand_a_list = [], []` の初期化を追加
- [ ] 9. `v3/primary/run_rq3_path_mediation.py`: return 変数名を `discovery_summary, confirmation_res` に修正

### B. 論文妥当性のための修正
- [ ] 10. `v2/primary/run_rq4_recovery_patching.py`: 高次元 rank-deficient 解消のため train-only PCA ($k=\min(64, n_{\mathrm{train}}-1, D)$) → Procrustes 回転に修正。補空間は identity 保持
- [ ] 11. `v3/primary/run_confirmatory_replication.py`: H3 に matched-rank random 2D subspace control (`Q_rand`) を導入し、`net_attenuation` の Bootstrap CI を算出・判定条件に追加
- [ ] 12. `v3/primary/run_rq3_path_mediation.py`: Net attenuation の Bootstrap CI (`mean`, `ci_lower`, `ci_upper`) を算出して JSON 保存
- [ ] 13. `src/affective_empathy_eval/statistics.py` & `v1/primary/run_phase_c.py`: V1 E4 に paired sign-flip permutation test (10,000回) を追加し、`aligned_permutation_p_V/A` を出力

### C. 再現性・QA・軽微修正
- [ ] 14. `src/affective_empathy_eval/data.py`: `from pathlib import Path` 追加 & 例外ログ出力
- [ ] 15. `v1/primary/phase_c/run_e6_specialization.py`: `logger` 定義追加
- [ ] 16. `v1/primary/run_phase_c.py`: `compute_cache_metadata()` の `transformers` import を safe import 化
- [ ] 17. `src/affective_empathy_eval/likelihood.py`: `temperature` パラメータを softmax 計算に反映
- [ ] 18. manifest config / hash に `sequence_likelihood` 設定を明示
- [ ] 19. `README.md`: V2 の統計検定名を `paired Wilcoxon signed-rank test` に修正
- [ ] 20. `configs/v3_experiments.yaml` & `run_confirmatory_replication.py`: confirmatory の fold 当たり介入サンプル数 (`n_intervention_samples_per_fold: 5`) を config 化
- [ ] 21. `v1/primary/run_phase_c.py`: `ci_95_high_V` 重複キーを削除
- [ ] 22. `likelihood.py`: docstring のスコアリング説明を token mean に更新

### D. 検証と本番前準備
- [ ] 23. `pytest -q` 全件合格の確認
- [ ] 24. Qwen による実モデル小規模スモークテストの実行（NameError が解消されていることの直接検証）
- [ ] 25. 全 Stage dry-run 完走確認
- [ ] 26. `walkthrough.md` の作成と結果報告
