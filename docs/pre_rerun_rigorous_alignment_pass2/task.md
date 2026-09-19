# タスクリスト: 本番再実行前の最終厳密化 (第2パス)

## 背景と目的
論文ストーリー（**Behavioral Covariation $\rightarrow$ V1 Representation/Causal Overlap $\rightarrow$ V2 Post-training-Associated Reorganization $\rightarrow$ V3 Localized Causal Leverage**）を厳密に維持しつつ、残存する定義のズレ・キー不整合・座標系不整合・キャッシュ検証の強化（全19項目）を実施する。

---

### 【P0: 本番推定値・主主張に直結する重要点】
- [x] 1. V2 RQ3 の `post_training_comparison` キーバグ修正 (`v2/primary/run_rq3_causal_map.py`) <!-- id: 0 -->
  - `base_self` vs `inst_matched_self` を用いて `post_training_comparison_matched` を Primary として保存
  - `base_self` vs `inst_native_self` を用いて `post_training_comparison_native` を Secondary として保存
- [x] 2. V2 RQ3 の family summary を matched-plain 主体に修正 (`v2/primary/run_rq3_causal_map.py`) <!-- id: 1 -->
  - Primary: `inst_matched_reader_c_v_peak`, `inst_matched_self_c_v_peak`
  - Secondary: `inst_native_reader_c_v_peak`, `inst_native_self_c_v_peak`
  - `inst_reader_c_v_peak` 等の曖昧な generic キーは削除または `deprecated_native_alias` と明記
- [x] 3. V2 RQ3 の LMM で matched-plain と native-chat を分離 (`v2/primary/run_rq3_causal_map.py`) <!-- id: 2 -->
  - Primary LMM: `Base plain` vs `Instruct matched-plain`
  - Secondary LMM: `Base plain` vs `Instruct native-chat`
  - 出力 JSON を `primary_matched_plain` と `secondary_native_chat` に分離
- [x] 4. V2 Confirmatory H4 で native への fallback を完全削除 (`v2/primary/run_confirmatory_analysis.py`) <!-- id: 3 -->
  - `auc_recovery_matched_plain`, `max_recovery_ratio_matched_plain` 欠損時は `RuntimeError` を送出
- [x] 5. V3 Confirmatory H4 で temporal-layer local direction を fit (`v3/primary/run_confirmatory_replication.py`) <!-- id: 4 -->
  - `temporal_map_layer`（depth ≈ 0.65）で抽出した表現から、CV fold の train split で個別に Ridge を fit して `d_temp_v, d_temp_a` を算出し注入

### 【P1: 統計・推定の頑健性と不整合解消】
- [x] 6. V2 RQ4 cross-family summary を matched-primary 化 (`v2/primary/run_rq4_recovery_patching.py`) <!-- id: 5 -->
  - 出力構造を `primary_matched_plain`, `secondary_native_chat`, `mechanistic_control_procrustes_aligned` に整理
- [x] 7. Confirmatory のレイヤー相対深度 hard-code を config 化 (`configs/v3_experiments.yaml`, `v3/primary/run_confirmatory_replication.py`) <!-- id: 6 -->
  - `sufficiency_relative_depth: 0.5`, `temporal_relative_depth: 0.65`, `mediation_relative_depth: 0.65`
- [x] 8. RQ3 で選んだ mediator 層の relative depth を summary に出力 (`v3/primary/run_rq3_path_mediation.py`) <!-- id: 7 -->
  - summary に `"mediator_relative_depth"` を追加
- [x] 9. V3 RQ1 Specificity で reference $\alpha=1.0$ を明示 (`configs/v3_experiments.yaml`, `v3/primary/run_rq1_state_induction.py`) <!-- id: 8 -->
  - `specificity_reference_alpha: 1.0` を設定し、末尾仮定を排除
- [x] 10. RQ2 `causal_reference_alpha` を完全一致チェック (`v3/primary/run_rq2_spatiotemporal_maps.py`) <!-- id: 9 -->
  - `causal_reference_alpha not in alpha_sweep` で `ValueError`
- [x] 11. V3 README の介入オペレータ表現修正 (`v3/primary/README.md`, `README.md`) <!-- id: 10 -->
  - 方向注入 (Additive) と部分空間除去 (Subspace removal) を明確に区別
- [x] 12. V3 の「necessity」表現の精緻化 (`v3/primary/README.md`, `README.md`) <!-- id: 11 -->
  - 主表現を `endogenous relevance` / `necessity-style evidence` に統一
- [x] 13. Manifest / Cache 検証の強化 (`src/affective_empathy_eval/manifests.py`, 各 primary スクリプト) <!-- id: 12 -->
  - config 全体辞書・dataset_path・seed による厳密な hash 検証
- [x] 14. model revision の manifest 記録 (`src/affective_empathy_eval/manifests.py`, 各 primary スクリプト) <!-- id: 13 -->
  - `base_model_id`, `instruct_model_id` を manifest config に記録
- [x] 15. V1 の論文上の位置づけ明記 (`v1/primary/README.md`, `README.md`) <!-- id: 14 -->
  - V1 Primary: Base models における Reader-Self overlap、V1 Secondary: Instruct models での within-model 解析
- [x] 16. V2 H1a の RSA 表現修正 (`v2/primary/run_confirmatory_analysis.py`) <!-- id: 15 -->
  - interpretation を representational similarity under matched-plain conditions に修正
- [x] 17. root README / 論文での teacher-forced 明示 (`README.md`, `v3/primary/README.md`) <!-- id: 16 -->
  - teacher-forced candidate sequence 上の段階であることを明示

### 【P2: 記述整合・テスト修正】
- [x] 18. V3 Confirmatory の重複行削除 (`v3/primary/run_confirmatory_replication.py`) <!-- id: 17 -->
- [x] 19. Production test の軽量化 (`tests/test_production_entrypoints.py`, `pyproject.toml`) <!-- id: 18 -->
  - `--help` test に `@pytest.mark.slow` を付与

### 【検証・ドキュメント】
- [x] 20. 全 pytest テストスイートの実行 (`pytest tests/ -q -m "not slow"`) <!-- id: 19 -->
- [x] 21. V3 テストハーネス実行 (`python tests/run_all_v3_dryruns.py`) <!-- id: 20 -->
- [x] 22. 統合ランナー dry-run 全ステージ実行検証 <!-- id: 21 -->
- [x] 23. ドキュメント保存 (`docs/pre_rerun_rigorous_alignment_pass2/`) <!-- id: 22 -->
