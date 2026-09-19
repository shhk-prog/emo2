# コードフリーズ前 最終厳格化（Pass 3）完了報告書

本セッションでは、論文本体のストーリー（**Covariation $\rightarrow$ Representation / Causal Overlap $\rightarrow$ Post-training-Associated Reorganization $\rightarrow$ Localized Causal Leverage**）を一切変更せず、実験コードフリーズ前の全15項目の不整合・バグ・命名の曖昧性を完全に解消しました。

---

## 1. 実施した修正項目と成果（全15項目）

### 【P0】重要不具合・測定仕様の是正
1. **V3 RQ2 `is_manifest_matching` import漏れの解消 & キャッシュ検証テスト新設**
   - [run_rq2_spatiotemporal_maps.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq2_spatiotemporal_maps.py) で `is_manifest_matching`, `compute_string_or_dict_hash`, `DEFAULT_CODE_VERSION` をインポート。
   - キャッシュ存在時に再計算へ入らず正しくキャッシュが読み出される単体テスト `test_rq2_cache_hit_does_not_recompute` を [tests/test_refinement_suite.py](file:///mnt/nas/home/hiromi/src/emo2/tests/test_refinement_suite.py) に追加（通過確認）。
2. **V2 RQ4 `task_comparison` の matched-plain Primary 化**
   - [run_rq4_recovery_patching.py](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq4_recovery_patching.py) の `task_comparison` を `primary_matched_plain` と `secondary_native_chat` に構造化。
   - トップレベルの generic key（`diff_max_recovery_self_vs_reader`, `diff_auc_recovery_self_vs_reader`）も Primary である matched-plain の差分値を指すよう保証。

### 【P1】命名・概念・キャッシュハッシュの厳格化
3. **V2 RQ3 native-chat alias の完全統一**
   - [run_rq3_causal_map.py](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq3_causal_map.py) で `fam_causal["inst_reader"]` / `fam_causal["inst_self"]` という危険な別名を新規コードから完全排除し、`inst_native_reader` / `inst_native_self` に統一（`deprecated_inst_*_native_alias` として互換性維持）。
4. **V2 RQ3 `post_training_comparison` generic alias の整理**
   - `post_training_comparison_matched` と `post_training_comparison_native` を明示出力とし、generic alias は `{"deprecated_alias_of": "post_training_comparison_matched"}` として曖昧性を排除。
5. **V3 Confirmatory の層指定由来を config & manifest に明記**
   - [configs/v3_experiments.yaml](file:///mnt/nas/home/hiromi/src/emo2/configs/v3_experiments.yaml) に `selection_source: "qwen_discovery_frozen"` を明記。
   - [run_confirmatory_replication.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_confirmatory_replication.py) の manifest metadata に `"confirmatory_site_selection_source": "qwen_discovery_frozen"` を記録。
6. **V3 Confirmatory H3 QC 判定の Primary absolute attenuation 準拠化**
   - H3 QC 判定を Secondary ratio ではなく、Primary 指標である `atten_v_low > min_mediated_attenuation_ci_lower` (0.0) に変更。
7. **`all_confirmed` / `CONFIRMED` の論文主結果化防止**
   - 判定フラグを `auxiliary_qc_all_pass: bool(...)` に変更。
   - 実実行時の status を `"status": "EFFECT_ESTIMATES_AVAILABLE"` に整理し、1bit判定ゲームに見えないよう改善。
8. **V3 RQ2 cache manifest の厳格化**
   - 実行パラメータ（family, model_id, dataset_path, semantic_stages, alpha_sweep, causal_reference_alpha, n_causal_samples, seed, subsample）を辞書化し、`expected_config_hash`, `expected_dataset_hash`, `expected_code_version` による厳格なキャッシュ検証を実装。
9. **V3 RQ3 / Confirmatory cache manifest の厳格化**
   - RQ3 および Confirmatory でも config 全体ハッシュおよびデータセットハッシュによるキャッシュ照合を実装。
10. **V2 RQ3 / RQ4 cache manifest の full config hash 化**
    - `v2_config` 全体を含む `manifest_config` を生成し、ハイパーパラメータ変更時に確実にキャッシュが無効化されるよう修正。
11. **V3 README の介入手法記述の正確化**
    - [v3/README.md](file:///mnt/nas/home/hiromi/src/emo2/v3/README.md) および [v3/primary/README.md](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/README.md) で、加算方向注入 ($h'=h+\alpha\sigma_h\hat{d}$) と中心化部分空間除去 ($h'=h-QQ^\top(h-\mu_{\text{neu}})$) を明確に区別して記述。
12. **V3 Primary 用語の `Endogenous relevance` 統一**
    - [run_rq1_state_induction.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq1_state_induction.py) 等で `endogenous_relevance_*_pass` を Primary 出力キーにし、`necessity_*` は後方互換用 alias として保持。
13. **Root README の teacher-forced candidate sequence 明記**
    - [README.md](file:///mnt/nas/home/hiromi/src/emo2/README.md) で "causal leverage is concentrated at particular layers and stages along the teacher-forced candidate sequence" と記述を整合。

### 【P2】dry-run シミュレーションと閾値設定の整理
14. **V3 RQ3 dry-run simulation の ratio 定義整合**
    - [run_rq3_path_mediation.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq3_path_mediation.py) の `simulate_path_mediation_confirmation` 内でも `MIN_NATURAL_SHIFT = 0.05` によるフィルタリングを適用し、real path と一致。
15. **V3 Confirmatory の hard-coded threshold を config `confirmatory.qc` に集約**
    - `configs/v3_experiments.yaml` に `min_sufficiency_slope`, `min_mediated_attenuation_ci_lower`, `min_temporal_contrast` を定義し、スクリプトから動的参照。

---

## 2. 検証結果

### 自動テスト（pytest）
```text
.venv/bin/python -m pytest tests/ -m "not slow" -v
===================================== 79 passed, 1 deselected, 4 warnings in 9.22s ======================================
```
- 新設した `test_rq2_cache_hit_does_not_recompute` を含む全79件の単体・結合テストが成功。

### 構文チェック（py_compile）
- V1, V2, V3 の全 entrypoint スクリプト13件がエラーなくコンパイル通過。

### V3 Unified Dry-run パイプライン
```text
.venv/bin/python tests/run_all_v3_dryruns.py
2026-09-19 16:42:18,479 [INFO] V3-RQ1 State Induction dry-run passed.
2026-09-19 16:42:23,913 [INFO] V3-RQ2 Spatiotemporal Maps dry-run passed.
2026-09-19 16:42:29,419 [INFO] V3-RQ3 Path Mediation dry-run passed.
2026-09-19 16:42:35,254 [INFO] V3 Confirmatory Replication dry-run passed.
2026-09-19 16:42:35,254 [INFO] All V3 dry-run tests successfully passed!
```

### V2 Dry-run パイプライン
- `run_rq3_causal_map.py --dry-run` および `run_rq4_recovery_patching.py --dry-run` が完走。
- 出力 JSON `v2_recovery_qwen.json` において `primary_matched_plain` と `secondary_native_chat` の完全分離を確認。
- `v3_cross_model_replication_summary.json` において `auxiliary_qc_all_pass: true`, `status: "MOCK_SIMULATION..."` を確認。

---

## 3. 結論
全ての残存指摘事項が解消され、キャッシュ検証・条件命名・QC判定・数理ドキュメントの整合性が完全に取れました。これで安心して本番実験を実行し、コードベースを freeze していただける状態です。
