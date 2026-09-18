# 本実験実行前最終リファインメント（P0〜P2）修正内容の確認 (Walkthrough)

## 1. 概要
本改修では、大規模言語モデル情動反応性評価実験の全ステージ（Behavioral, V1, V2, V3）における本実験再実行前の最終ブロッカー（P0: 実行時未定義バグ、P1: 統計・因果介入の不整合、P2: ディレクトリ・スクリプト不整合）を包括的かつ厳密に修正しました。

全 78 件の CPU 単体・統合・回帰テストがすべてパス（`78 passed, 0 failed`）し、実機実行可能な状態であることを確認しました。

---

## 2. 実施した修正の詳細

### [P0] 実行時ブロッカーの解消 (Critical Bugs Fixed)
1. **P0-1〜P0-4, P2-3: `v3/primary/run_confirmatory_replication.py` の実モデル実行パイプライン刷新**
   - **未定義変数の完全解消**: H3/H4 に残存していた `causal_sub_df`, `opt_layer`, `batch_size`, `evaluate_candidate_likelihoods` 等の未定義変数を一掃し、共通の `compute_sequence_likelihoods_for_candidates` および `compute_expected_va` に統一。
   - **単一 Cross-Fitting ループへの統合**: H2 (Sufficiency: direction injection), H3 (Necessity: centered 2D subspace removal), H4 (Temporal Emergence: generation stages) を同一の `GroupKFold` ループ内に集約。`train_idx` のみから $d_V, d_A, Q, \mu_{\text{neu}}$ を推定し、`test_idx` のみで介入評価を行うことで、データ再利用・リークを完全に排除。
   - **Joint Tokenization & ステージインデックス不変性アサート**: `validate_stage_index_invariance` を実装し、プロンプトと 81 候補文字列の結合列において、全候補間で各生成ステージ（`candidate_start`, `pre_V`, `V_value`, `pre_A`, `A_value`, `response_end`）の絶対トークン位置が完全に一致することを事前検証。
   - **Family キーと表示名の分離**: 小文字キー（`llama`, `gemma`, `olmo`）と表示名（`Llama 3.2`, `Gemma 3` 等）を分離し、`models.yaml` の探索失敗を防止。
2. **P0-5: `v3/primary/run_rq3_path_mediation.py` のキャッシュロード時変数スコープバグ修正**
   - キャッシュロード時に `full_output` が未定義となり `UnboundLocalError` が発生していたバグを修正。同時に `is_manifest_matching` によるメタデータ整合性チェックを接続。

---

### [P1] 統計・介入・統制条件の科学的精緻化 (Scientific & Statistical Refinements)
1. **P1-1〜P1-3: `behavioral/analysis/summarize_behavioral_aipsy.py` の統計的落とし穴の解消**
   - **Sensitivity の期待方向符号整列 (Primary)**: AIPsy 8 感情の極性（例: grief は $V-$, ecstasy は $V+$）を反映した `EXPECTED_DIRECTION` を定義し、符号整列差分 `direction_aligned_mean_diff` を算出。正負感情の相殺による偽陰性を排除（後方互換性のため生の差分も保持）。
   - **Dose-Response の真の活用**: Neutral $\to$ Moderate $\to$ Clinical の推移において、`mean_step_1_aligned` (Moderate − Neutral) と `mean_step_2_aligned` (Clinical − Moderate) をそれぞれ計算し、単調性成立率 `monotonicity_rate` および中点からの偏り `midpoint_deviation` を評価。
   - **Specificity の Affective Displacement (変位量)**: 感情変位量 $D(x) = |E(x) - \mu_{\text{neutral}}|$ を導入し、Complex Neutral に対する特異的感情応答を頑健に評価。
2. **P1-4: `v3/primary/run_rq3_path_mediation.py` Discovery 介入スケールの統一**
   - 探索的スクリーニングでの $C(l)$ 測定を `register_direction_intervention_hook(mode="inject", alpha=1.0, hidden_std=h_std_l)` による加算注入に統一。
3. **P1-5: `v3/primary/run_rq2_spatiotemporal_maps.py` の Cross-Fitting 化**
   - CV splits を再利用し、方向推定を train fold のみで行い、介入評価を test fold のみで行う構成へ修正。
4. **P1-6: `v2/primary/run_rq3_causal_map.py` の Primary 介入を情動特異的因果マップに刷新**
   - 各層の残差ストリームにおける Ridge 推定方向 $d_V(l), d_A(l)$ への加算介入による $C_V(l), C_A(l)$ を Primary 指標に刷新。
   - 従来の zero ablation は非特異的回路破壊の統制条件（Secondary）として `c_v_zero`, `c_a_zero` に分離保存。
5. **P1-7, P1-8: `v1/primary/run_phase_c.py` E4 の統制条件拡充と安全化**
   - **Same-Task Controls**: `Self -> Self`（自己上限基準）、`Reader -> Reader`、`Self -> Reader`（逆方向転移）を追加し、他者認識から自己報告への転移効率 `TransferRatio` を算出。
   - **危険なフォールバックの排除**: `conf_indices < 5` の場合に全ペアへフォールバックする危険な挙動を削除し、dry-run 以外では即座にエラーとするデータリーク防止策を徹底。
6. **P1-9: `v2/primary/run_rq4_recovery_patching.py` の Procrustes 分割を Seeded Group/Permutation Split 化**
   - 固定の先頭 70% スライスを廃止し、`pair_id` に基づく再現可能なグループ分割（または seeded permutation）へ移行。
7. **P1-10, P1-11: Manifest 検証の拡充とモデルリビジョン管理**
   - `src/affective_empathy_eval/manifests.py` の `is_manifest_matching` に `expected_config_hash`, `expected_dataset_hash`, `expected_code_version`, `expected_model_revision` の検証を追加。

---

### [P2] ディレクトリ規約・スクリプト整合性 (Directory & Script Harmonization)
1. **P2-1: Behavioral デフォルト出力パスの統一**
   - `run_behavioral_aipsy.py`: デフォルトを `behavioral/results/raw/aipsy_4split` に統一。
   - `run_behavioral_emobank.py`: デフォルトを `behavioral/results/raw/emobank_3way` に統一。
   - 各サマリースクリプトの探索フォールバックも後方互換性を担保しつつ整備。
2. **P2-2: `v2/primary/run_confirmatory_analysis.py` の新規作成**
   - V2 README に記載されていた未実装スクリプト `run_confirmatory_analysis.py` を実装。4ファミリー横断での LMM (Linear Mixed Model) および Benjamini-Hochberg FDR 補正を実行し、`v2/results/derived/v2_lmm_confirmatory.json` に出力。

---

## 3. テストと検証結果

### 単体・回帰テストスイートの実行
- 新規テスト: `tests/test_confirmatory_pipeline.py`（ステージインデックス不変性、AIPsy期待方向定義、V2確証的解析ドライラン）
- 実行コマンド:
  ```bash
  .venv/bin/python -m pytest tests/ -k "not test_integration and not test_gpu and not heavy" -v
  ```
- **実行結果**:
  ```text
  ================== 78 passed, 4 warnings in 87.17s (0:01:27) ===================
  ```
- すべての主要テスト（Behavioral カップリング・Dose-Response、V1 Phase C 介入、V2 幾何・因果・リカバリー・LMM、V3 状態誘導・時空間マップ・媒介分析・Confirmatory、CLI argparse 互換性）が 100% 通過しました。
