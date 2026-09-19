# 変更内容の確認 (Walkthrough): V1/V2/V3・Behavioral 実装上の厳密化・バグ修正

本作業では、中心ストーリー（**Behavioral Covariation $\rightarrow$ V1 Representation/Causal Overlap $\rightarrow$ V2 Post-training-Associated Reorganization $\rightarrow$ V3 Causal Leverage**）を一切変更せず、本番再実行において科学的妥当性・再現性・データ完全性を保証するための全25項目におよぶ実装上の問題・バグを修正・検証しました。

---

## 1. 実施した修正の総括（全25項目）

### 【P0: 最優先・結果に直接影響する不一致・バグ】
1. **V2 Confirmatory H1/H2 の JSON Schema 追従** ([`v2/primary/run_confirmatory_analysis.py`](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_confirmatory_analysis.py))
   - `run_rq1_rq2_cross_decoding.py` の実出力構造である `rq2_sharing[axis]` 配下の `base_r2_reader`, `base_r2_self`, `inst_matched_r2_reader`, `inst_matched_r2_self` からピーク深度差を算出する形に修正。
   - 出力構造を `valence.reader.shift`, `valence.self.shift`, `arousal.reader.shift`, `arousal.self.shift` に厳密化。
2. **V2 H1 判定基準の双方向化と効果量・CI 中心への移行** ([`v2/primary/run_confirmatory_analysis.py`](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_confirmatory_analysis.py))
   - 一方向の事前想定（instruct のみが深層化する）から、双方向の再編検定 `(ci_low > 0.0 or ci_high < 0.0)` へ変更。
   - 主出力を `mean shift`, `95% bootstrap CI`, `per_family_shifts` に整理。
3. **V2 H2 での `delta_sharing_matched` 直接参照** ([`v2/primary/run_confirmatory_analysis.py`](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_confirmatory_analysis.py))
   - フォーマット統制された Matched-Plain 条件の $\Delta\text{Sharing}$ を Primary 指標として直接参照（Native-Chat は Secondary として分離）。
4. **`v2_geometry` 返り値の修正** ([`v2/primary/run_rq1_rq2_cross_decoding.py`](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq1_rq2_cross_decoding.py))
   - 返り値辞書に `relative_depths` と `num_layers` を追加し、cache hit 判定および後続処理でのキー欠損を解消。
5. **V2 RQ3 cache hit 時の pair-level CSV 消失防止** ([`v2/primary/run_rq3_causal_map.py`](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq3_causal_map.py))
   - family ごとに `v2_causal_pair_level_{fam_id}.csv` を保存し、cache hit 判定時にも pair-level CSV の存在を検証してメモリ上に再構築。欠損時は安全に再計算。
6. **V1 E4 target direction aligned effect による相殺防止** ([`v1/primary/run_phase_c.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_c.py))
   - ペアごとに感情の期待方向（`EXPECTED_DIRECTION`）の符号を乗じた `aligned_*` を Primary 指標に設定。ポジティブ感情とネガティブ感情による介入効果の相殺を防止。
8. **V3 RQ3 mediator layer 選択の Stratified Sampling 化** ([`v3/primary/run_rq3_path_mediation.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq3_path_mediation.py))
   - 特定感情（rage等）への偏りを排除し、`target_emotion` による層化均等サンプリング（seed 42、最大16件）により mediator 候補層を選定。
9. **V3 RQ3 mediator selection の V/A 両軸実測と $C_{\text{joint}}$ 評価** ([`v3/primary/run_rq3_path_mediation.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq3_path_mediation.py))
   - Valence と Arousal の両介入を実測し、$C_{\text{joint}}(l) = (C_V(l) + C_A(l)) / 2$ の最大値層を中継層として客観選定。
12. **V3 Confirmatory の matched-neutral fallback 削除** ([`v3/primary/run_confirmatory_replication.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_confirmatory_replication.py))
   - 欠損時に自己プロンプトを中立扱いする fallback を削除し、データ欠損時は明示的に `ValueError` を送出。
16. **全 Stage (V2, V3) の dry-run と real 結果出力先ディレクトリ完全分離**
   - dry-run 実行時は `v2/results/raw/dry_run/`, `v3/results/raw/dry_run/` 等に完全分離し、本番のクリーンな results（`.gitkeep` のみ）を一切汚染しないよう防護。

---

### 【P1: 統計・推定の頑健性向上】
7. **V1 E4 Transfer Ratio のゼロ除算ガード** ([`v1/primary/run_phase_c.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_c.py))
   - 分母が微小（$< 0.05$）な場合は `np.nan` とし、異常値・外れ値の発生を防止。
10, 11. **V3 RQ3 および Confirmatory の attenuation ratio ゼロ近傍安定化**
   - `MIN_NATURAL_SHIFT = 0.05` を設定し、Primary を絶対減衰量 `mediated_attenuation`、Secondary を `attenuation_ratio` に整理。
13. **V3 RQ3 の `mu_neu` fallback 削除** ([`v3/primary/run_rq3_path_mediation.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq3_path_mediation.py))
   - 中立表現が空の場合は明示的に `ValueError` を送出し、意図しないゼロベクトル減衰を防止。
14. **V3 RQ2 意味段階トークン位置不変性検証の共通化** ([`src/affective_empathy_eval/prompts.py`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/prompts.py))
   - `validate_stage_index_invariance()` を共通モジュール化し、81 候補の評価開始前にトークン位置の同一性を厳密検証。
15. **V3 RQ2 の `C_A` 重複代入バグ削除** ([`v3/primary/run_rq2_spatiotemporal_maps.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq2_spatiotemporal_maps.py))
17. **Manifest validation の厳格化** ([`src/affective_empathy_eval/manifests.py`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/manifests.py))
   - dry-run 出力との混同防止フラグ照合を徹底。
18. **Cache hit 時の manifest 上書き防止ガード**
   - 既存の妥当な結果がロードされた際、新規 manifest で上書きしないよう安全にガード。
19. **V2 Confirmatory H3 名称修正** (`H3_causal_profile_reorganization_lmm`)
   - 科学的測定対象（因果プロファイルの再編）に即した名称へ更新（後方互換キーも保持）。
20. **V2 H3 LMM への `C(family)` 固定効果追加**
   - モデルファミリー間の baseline 差を吸収する `C(family)` を明示的にモデル式へ導入。
21. **V2 H3 主 FDR family の事前固定**
   - 事後学習関連交互作用項（Primary terms）のみを主 FDR 補正の対象に限定。
22. **V3 Confirmatory summary への H4 定量値追加**
   - `contrast_v`, `contrast_a` の平均値および 95% bootstrap CI を `summary["primary_effect_estimates"]["H4_temporal_contrast"]` に収録。

---

### 【P2: 記述整合・文言厳密化】
23, 24. **Behavioral README および root README 統計記述修正** ([`behavioral/README.md`](file:///mnt/nas/home/hiromi/src/emo2/behavioral/README.md), [`README.md`](file:///mnt/nas/home/hiromi/src/emo2/README.md))
   - Jonckheere-Terpstra 表記を廃止し、現在の実装である「Direction-aligned two-step monotonic contrast with an intersection-union test (IUT)」へ修正。
   - Sensitivity の主指標を「Direction-aligned paired difference に対する 1 標本 $t$ 検定」へ統一。
25. **V2 コードコメント・解釈表現の厳密化**
   - 単なる「純粋post-training効果」という過大表現を排し、「Prompt-format-controlled Base–Instruct comparison」「Post-training-associated difference」に統一。

---

## 2. 検証結果

テストスイートを実行し、修正後のパイプライン整合性と動作を確認しました。

### 自動テスト実行結果
```text
====================================================================================== test session starts =======================================================================================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /mnt/nas/home/hiromi/src/emo2/.venv/bin/python
cachedir: .pytest_cache
rootdir: /mnt/nas/home/hiromi/src/emo2
configfile: pyproject.toml
plugins: anyio-4.15.1
collected 12 items                                                                                                                                                                               

tests/test_confirmatory_pipeline.py::test_validate_stage_index_invariance PASSED                                                                                                           [  8%]
tests/test_confirmatory_pipeline.py::test_behavioral_aipsy_expected_direction_alignment PASSED                                                                                             [ 16%]
tests/test_confirmatory_pipeline.py::test_v2_confirmatory_analysis_dry_run PASSED                                                                                                          [ 25%]
tests/test_production_entrypoints.py::test_all_primary_entrypoints_exist PASSED                                                                                                            [ 33%]
tests/test_production_entrypoints.py::test_production_dry_run_dispatch PASSED                                                                                                              [ 41%]
tests/test_production_entrypoints.py::test_all_dispatched_commands_argparse_compatibility PASSED                                                                                           [ 50%]
tests/test_refinement_suite.py::test_python_syntax_and_core_imports PASSED                                                                                                                 [ 58%]
tests/test_refinement_suite.py::test_layer_index_mapping PASSED                                                                                                                            [ 66%]
tests/test_refinement_suite.py::test_candidate_space_sizes PASSED                                                                                                                          [ 75%]
tests/test_refinement_suite.py::test_pair_id_train_test_leakage_assertion PASSED                                                                                                           [ 83%]
tests/test_refinement_suite.py::test_cache_manifest_validation_and_mismatch PASSED                                                                                                         [ 91%]
tests/test_refinement_suite.py::test_config_propagation PASSED                                                                                                                             [100%]

=========================================================================== 12 passed, 4 warnings in 83.16s (0:01:23) ============================================================================
```

- 全 12 件の主要パイプラインテスト・CLI 互換性テスト・確証的統計解析テストがすべて **PASSED**。
- `results/` ディレクトリは汚染されておらず、本番実行のためのクリーンな状態が維持されています。
