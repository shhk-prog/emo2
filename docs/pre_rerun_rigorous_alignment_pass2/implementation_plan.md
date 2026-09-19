# 実装計画: 本番再実行前の最終厳密化 (第2パス: 全19項目)

本計画は、ユーザーからの詳細なレビュー指摘に基づき、論文の科学的ストーリー（**Covariation $\rightarrow$ Representation/Causal Overlap $\rightarrow$ Post-training-Associated Reorganization $\rightarrow$ Localized Causal Leverage**）を一切変更することなく、残存する定義のズレ・キー不整合・層別座標系の不整合・キャッシュ検証の強化（全19項目）を包括的に解決するためのものです。

---

## ユーザー確認・承認事項

> [!IMPORTANT]
> - 本修正では、論文ストーリーの変更や新しいRQの追加は一切行いません。
> - V2 においては Primary を厳格に **Base (plain) vs Instruct (matched-plain)** に固定し、native-chat は完全に Secondary として分離します。
> - V3 においては、介入層ごとの内部情動方向の座標系を厳密にローカル fit させ、レイヤー間の不当な移植を排除します。
> - 修正完了後は、全テスト通過および統合ランナー dry-run 実行を確認し、本番再実行が完全にクリーンかつ再現可能な状態にします。

---

## 変更内容詳細

### 1. 【P0】V2 RQ3 の `post_training_comparison` キーバグ修正
- **対象**: `v2/primary/run_rq3_causal_map.py`
- **問題**: `dissoc_results[axis]` のキーは `inst_matched_self` / `inst_native_self` であるにもかかわらず、`inst_self` を探しているため `post_training_comparison` が生成されていませんでした。
- **修正**:
  - Primary: `post_training_comparison_matched`（`base_self` vs `inst_matched_self`, `base_reader` vs `inst_matched_reader`）。
  - Secondary: `post_training_comparison_native`（`base_self` vs `inst_native_self`, `base_reader` vs `inst_native_reader`）。

### 2. 【P0】V2 RQ3 の family summary を matched-plain 主体に修正
- **対象**: `v2/primary/run_rq3_causal_map.py`
- **修正**:
  - Primary: `inst_matched_reader_c_v_peak`, `inst_matched_self_c_v_peak`。
  - Secondary: `inst_native_reader_c_v_peak`, `inst_native_self_c_v_peak`。
  - 曖昧な `inst_reader_c_v_peak` などの generic キーは削除し、明確に条件を分離。

### 3. 【P0】V2 RQ3 の LMM で matched-plain と native-chat を分離
- **対象**: `v2/primary/run_rq3_causal_map.py`
- **問題**: `df_pair` 全体に fit していたため、`alignment="inst"` に matched-plain と native-chat の2条件が混在していました。
- **修正**:
  - Primary LMM:
    ```python
    df_primary = df_pair[
        ((df_pair["alignment"] == "base") & (df_pair["format_condition"] == "plain"))
        | ((df_pair["alignment"] == "inst") & (df_pair["format_condition"] == "matched_plain"))
    ].copy()
    formula = "c ~ C(family) + C(alignment) * C(task) * relative_depth"
    ```
  - Secondary LMM:
    ```python
    df_native = df_pair[
        ((df_pair["alignment"] == "base") & (df_pair["format_condition"] == "plain"))
        | ((df_pair["alignment"] == "inst") & (df_pair["format_condition"] == "native_chat"))
    ].copy()
    ```
  - 出力 JSON も `primary_matched_plain` と `secondary_native_chat` に分離。

### 4. 【P0】V2 Confirmatory H4 で native への fallback を完全削除
- **対象**: `v2/primary/run_confirmatory_analysis.py`
- **修正**:
  - `s_auc_m = s_dat.get("auc_recovery_matched_plain")`, `r_auc_m = r_dat.get("auc_recovery_matched_plain")`。
  - どちらかが欠損している場合は `RuntimeError(f"Matched-plain recovery AUC missing in {rf}")` を送出し、native へのサイレントフォールバックを排除。
  - `max_recovery_ratio_matched_plain` も同様に欠損時 `RuntimeError`。

### 5. 【P0】V3 Confirmatory H4 で temporal-layer local direction を fit
- **対象**: `v3/primary/run_confirmatory_replication.py`
- **問題**: `sufficiency_layer` (depth ≈ 0.5) で学習した方向 `d_v, d_a` を `temporal_map_layer` (depth ≈ 0.65) にそのまま注入していました。
- **修正**:
  - 各 fold の train split で `H_temporal = all_H[temporal_map_layer][train_idx]` に対し個別に Ridge を fit:
    ```python
    ridge_temp_v = Ridge(alpha=10.0).fit(H_temporal, y_v[train_idx])
    d_temp_v = ridge_temp_v.coef_ / (np.linalg.norm(ridge_temp_v.coef_) + 1e-6)
    h_std_temp_v = float(np.std(H_temporal @ d_temp_v))
    ```
  - H4 では `d_temp_v, h_std_temp_v`, `d_temp_a, h_std_temp_a` を使用。H2 は `d_v, d_a`（sufficiency layer local）を使用。

### 6. 【P1】V2 RQ4 cross-family summary を matched-primary 化
- **対象**: `v2/primary/run_rq4_recovery_patching.py`
- **修正**:
  - Primary: `self_auc_matched`, `reader_auc_matched`
  - Secondary: `self_auc_native`, `reader_auc_native`
  - 出力 JSON: `primary_matched_plain`, `secondary_native_chat`, `mechanistic_control_procrustes_aligned` に構造化。

### 7. 【P1】Confirmatory レイヤー相対深度の config 化
- **対象**: `configs/v3_experiments.yaml`, `v3/primary/run_confirmatory_replication.py`
- **修正**:
  - config に `sufficiency_relative_depth: 0.5`, `temporal_relative_depth: 0.65`, `mediation_relative_depth: 0.65` を設定。
  - コード側でハードコードを排除し、config から読み出して `round(rel_depth * (num_layers - 1))` で算出。

### 8. 【P1】RQ3 mediator 相対深度の summary 出力
- **対象**: `v3/primary/run_rq3_path_mediation.py`
- **修正**:
  - summary に `"mediator_relative_depth": float(mediator_layer / (num_layers - 1))` を追加。

### 9. 【P1】V3 RQ1 Specificity で reference $\alpha=1.0$ を明示
- **対象**: `configs/v3_experiments.yaml`, `v3/primary/run_rq1_state_induction.py`
- **修正**:
  - config に `specificity_reference_alpha: 1.0` を設定。
  - `ref_alpha not in alpha_grid` で例外送出し、`alpha_grid.index(ref_alpha)` から確実にインデックスを取得。

### 10. 【P1】RQ2 `causal_reference_alpha` を完全一致チェック
- **対象**: `v3/primary/run_rq2_spatiotemporal_maps.py`
- **修正**:
  - `if causal_reference_alpha not in alpha_sweep: raise ValueError(...)` を追加。silent nearest を廃止。

### 11. 【P1】V3 README の介入オペレータ表現修正
- **対象**: `v3/primary/README.md`, `README.md`
- **修正**:
  - 方向注入: Standardized additive operator ($h' = h + \alpha \sigma_h \hat{d}$)
  - 除去: Centered orthogonal subspace removal ($h' = h - QQ^\top(h - \mu_{\text{neu}})$)

### 12. 【P1】V3 の「necessity」表現の精緻化
- **対象**: `v3/primary/README.md`, `README.md`
- **修正**:
  - 主表現を `endogenous relevance` / `necessity-style evidence` に統一。

### 13. 【P1】Manifest / Cache 検証の強化
- **対象**: `src/affective_empathy_eval/manifests.py`, 各 primary スクリプト
- **修正**:
  - V3 RQ1, RQ2, RQ3, Confirmatory, V2 RQ3, RQ4 で、`manifest_config`（実験辞書全体）、`dataset_path`、`seed` を渡し、`config_hash`, `dataset_hash` の完全一致を検証。

### 14. 【P1】model revision の manifest 記録
- **対象**: `src/affective_empathy_eval/manifests.py`, 各 primary スクリプト
- **修正**:
  - `base_model_id`, `instruct_model_id` を manifest config に記録。

### 15. 【P1】V1 の論文上の位置づけ明記
- **対象**: `v1/primary/README.md`, `README.md`
- **修正**:
  - V1 Primary: Base models における Reader-Self overlap
  - V1 Secondary replication: Instruct models での within-model 解析
  - Base vs Instruct difference 自体は V1 では解釈せず、V2 のみで扱うことを明記。

### 16. 【P1】V2 H1a の RSA 表現修正
- **対象**: `v2/primary/run_confirmatory_analysis.py`
- **修正**:
  - `interpretation`: "post-training-associated geometric distortion and representational similarity under matched-plain conditions" に修正。

### 17. 【P1】root README / 論文での teacher-forced 明示
- **対象**: `README.md`, `v3/primary/README.md`
- **修正**:
  - "particular layers and stages along the teacher-forced candidate sequence" と明記。

### 18. 【P2】V3 Confirmatory の重複行削除
- **対象**: `v3/primary/run_confirmatory_replication.py`
- **修正**:
  - 重複代入があれば完全にクリーンアップ。

### 19. 【P2】Production test の軽量化
- **対象**: `tests/test_production_entrypoints.py`, `pyproject.toml`
- **修正**:
  - `test_all_dispatched_commands_argparse_compatibility` に `@pytest.mark.slow` を付与し、`pyproject.toml` に marker を登録。

---

## 検証計画

### 1. 構文・単体テスト検証
```bash
.venv/bin/python -m py_compile \
  v1/primary/run_phase_a.py v1/primary/run_phase_b.py v1/primary/run_phase_c.py \
  v2/primary/run_rq1_rq2_cross_decoding.py v2/primary/run_rq3_causal_map.py \
  v2/primary/run_rq4_recovery_patching.py v2/primary/run_confirmatory_analysis.py \
  v3/primary/run_rq1_state_induction.py v3/primary/run_rq2_spatiotemporal_maps.py \
  v3/primary/run_rq3_path_mediation.py v3/primary/run_confirmatory_replication.py

.venv/bin/python -m pytest tests/ -m "not slow" -q
```

### 2. V3 テストハーネス実行
```bash
.venv/bin/python tests/run_all_v3_dryruns.py
```

### 3. 統合ランナー dry-run フル実行
```bash
.venv/bin/python -m affective_empathy_eval.run --stage all --dry-run --max-samples 2 --model-set primary_small --force-after-no-go
```

### 4. Results クリーン性確認
```bash
git status --short
```
本番 `results/` ディレクトリにコミット対象外ファイルが一切生成されないことを確認。
