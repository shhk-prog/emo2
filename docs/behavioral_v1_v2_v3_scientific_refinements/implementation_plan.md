# 実装計画: P0〜P2 科学的・実験的完全化（本実験実行前リファインメント）

本計画は、本実験の本格的な再実行前に残された実装上の不備・統計的落とし穴（未定義変数、cross-fitting未統合、尤度API不整合、Behavioral感情極性相殺、Dose-Response中間条件不使用、介入スケール不統一、キャッシュ検証未接続等）を完全に解消するための作業計画です。

---

## 修正対象と実施内容

### 1. P0-1, P0-2, P0-3, P0-4: V3 Confirmatory の実モデル実行パイプラインの全面修正
- **対象**: `v3/primary/run_confirmatory_replication.py`
- **問題点**:
  - `run_real_model_confirmatory()` において、`causal_sub_df`, `opt_layer`, `batch_size`, `evaluate_candidate_likelihoods` 等の未定義変数が存在し、実モデル実行時に即時クラッシュする。
  - H3（必然性/subspace ablation）および H4（時間的局在/token stage）が cross-fitting ループの外に置かれ、最後の fold の `d_v, d_a` を全データに流用している。
  - 共通尤度 API ではなく未定義の `evaluate_candidate_likelihoods()` を呼んでいる。
  - トークン位置計算が RQ2 の Joint Tokenization 方式と乖離し、古い方式になっている。
- **改修方針**:
  - H2（十分性/injection）, H3（必然性/subspace removal）, H4（時間的局在/stage injection）を**同一の fold ループ (`for fold_idx, (train_idx, test_idx) in enumerate(splits_list):`) 内に完全に統合**。
  - `train_idx` のみから $d_V, d_A$, $\sigma_h$, 直交部分空間 $Q$, および中立平均 $\mu_{\text{neu}}$ を推定。
  - `test_idx`（held-out サンプル）のみで H2, H3, H4 を評価し、全 fold の held-out 結果のみを集約。
  - 尤度評価を共通 API `compute_sequence_likelihoods_for_candidates` および `compute_expected_va` に統一。
  - トークン位置計算を RQ2 と同一の `prepare_joint_sequence_with_boundary`, `get_generation_stage_tokens`, `resolve_joint_stage_index` に統一。
  - 81候補すべてで semantic stage の絶対位置が同一であるかを検証する `validate_stage_index_invariance` アサーションを追加。

### 2. P0-5: V3 RQ3 のキャッシュロード時 `full_output` 未定義バグ修正
- **対象**: `v3/primary/run_rq3_path_mediation.py`
- **問題点**: キャッシュヒット時に `full_output` が未定義となり `UnboundLocalError` が発生する。
- **改修方針**: `full_output = None` を関数冒頭で初期化し、キャッシュ読み込み時に `full_output = cached` を代入する。

### 3. P1-1, P1-2, P1-3: Behavioral 解析の統計的妥当性向上（符号相殺防止と真のDose-Response）
- **対象**: `behavioral/analysis/summarize_behavioral_aipsy.py`
- **改修方針**:
  - **P1-1 (Sensitivity)**: `EXPECTED_DIRECTION` 定義（grief: V-, terror: V- A+, ecstasy: V+ A+ 等）に基づき、各感情の期待方向に符号整列した `direction_aligned_mean_diff = sign * (clinical - neutral)` を Primary metric とする。方向が未定義の次元は除外し、`raw_mean_diff`, `direction_aligned_mean_diff`, `n_direction_defined` を共に出力。
  - **P1-2 (Dose-Response)**: 単純な端点差分 `(clinical - neutral)/2` を Secondary に降格。
    - `step_1 = moderate - neutral`
    - `step_2 = clinical - moderate`
    - 期待符号で整列した `aligned_step_1 = sign * step_1`, `aligned_step_2 = sign * step_2`
    - 単調性判定: `monotonic = (aligned_step_1 > 0 and aligned_step_2 > 0)` による `monotonicity_rate` の算出
    - `midpoint_deviation = moderate - (neutral + clinical) / 2` を算出。
  - **P1-3 (Specificity)**: 平均 Valence の生差分ではなく、中立参照点からの感情変位量（Affective Displacement: $D(x) = |E(x) - \mu_{\text{neutral}}|$）の比較、または期待方向整列効果を用いて正負相殺を防止。

### 4. P1-4: V3 RQ3 Discovery の $C(l)$ 介入スケール統一
- **対象**: `v3/primary/run_rq3_path_mediation.py`
- **改修方針**: Discovery の mediator layer 選択における $h + d_l$ を廃止し、共通 API `register_direction_intervention_hook(mode="inject", alpha=1.0, hidden_std=h_std_l)` による $h + \alpha \sigma_h \hat{d}$ に完全統一。

### 5. P1-5: V3 RQ2 Direction 推定の Cross-Fitting 化
- **対象**: `v3/primary/run_rq2_spatiotemporal_maps.py`
- **改修方針**: 全データ fit probe を廃止し、既存の CV split を再利用して train fold で方向と分散を推定し、held-out test fold（かつ `sub_eval_idx` に含まれるサンプル）に対してのみ時空間介入を実施。

### 6. P1-6: V2 RQ3 を Affect-Specific な因果マップに刷新
- **対象**: `v2/primary/run_rq3_causal_map.py`
- **改修方針**: 全隠れ状態のゼロパッチングは言語生成全般への破壊度（General site sensitivity）を測ってしまうため、Primary に感情方向（$d_V, d_A$）に対する介入・直交射影除去による $C_V(l), C_A(l)$ を実装。ゼロアブレーションは Secondary/統制条件として保持。

### 7. P1-7, P1-8: V1 Phase C E4 の Same-Task 統制追加と危険な Fallback 削除
- **対象**: `v1/primary/run_phase_c.py`
- **改修方針**:
  - E4 に Same-task controls (`Reader -> Reader`, `Self -> Self`, `Self -> Reader`) を追加し、正規化転移率 $TransferRatio_{R \to S} = |R \to S| / (|R \to R| + \epsilon)$ を算出。
  - `if len(conf_indices) < 5: conf_indices = list(range(n_pairs))` の危険な Discovery 再利用 fallback を完全削除し、確認ペア不足時は明示的エラーまたは `NO_GO` レポートを出力。

### 8. P1-9: V2 RQ4 Procrustes 分割の Seeded / Group Split 化
- **対象**: `v2/primary/run_rq4_recovery_patching.py`
- **改修方針**: 先頭70%/末尾30%の単純スライスを廃止し、`pair_id` に基づく Group split またはシード固定ランダムパーミュテーションによる分割へ移行。

### 9. P1-10, P1-11: Manifest 検証の実キャッシュ接続とリビジョン管理
- **対象**: `src/affective_empathy_eval/manifests.py`, V2/V3 各実行スクリプト
- **改修方針**:
  - `is_manifest_matching` に `config_hash`, `dataset_hash`, `code_version`, `git_commit` 等の検証項目を追加。
  - V2/V3 各スクリプトのキャッシュロード箇所で、ファイルの存在確認だけでなく `is_manifest_matching` の通過を必須化。
  - `configs/models.yaml` に `revision` を明示定義可能にし、ハードコード `"main"` を解消。

### 10. P2-1, P2-2, P2-3: ディレクトリ、README、Family キーの整合性整理
- **改修方針**:
  - Behavioral のデフォルトパスを `behavioral/results/raw/` および `behavioral/results/derived/` に完全統一。
  - V2 README から未実装の `run_confirmatory_analysis.py` 記述を整理し、実在するスクリプト構成と完全に一致させる（または最小限の統合スクリプトを作成）。
  - Confirmatory の Family 名をシステム内部キー（`llama`, `gemma`, `olmo`）と表示名（`Llama 3.2` 等）に分離し、シード値やファイル名への不正文字混入を防止。

---

## 検証計画

### 自動テスト
- `tests/test_v3_interventions_sanity.py`: 介入オペレータの正確性。
- `tests/test_behavioral_coupling.py`: 符号整列 Sensitivity、Dose-Response の単調性指標のテスト追加。
- `tests/test_confirmatory_pipeline.py` (新規作成): Confirmatory の cross-fitting ループ、H2/H3/H4 の held-out 評価、Joint Tokenization ステージインデックス不変性アサーション、未定義変数がないことのドライラン検証。
- `tests/test_refinement_suite.py`: キャッシュマニフェスト検証の厳密化テスト。
- 全 CPU テストスイートの実行 (`.venv/bin/python -m pytest tests/ -k "not test_integration and not test_gpu" -v`) で全件パスを確認。
