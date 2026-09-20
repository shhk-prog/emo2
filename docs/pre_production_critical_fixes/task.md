# Task: 本番実行前必須修正 (Pre-Production Critical Fixes)

## 概要
研究コードの妥当性・厳密性を担保するため、本番実行前に以下の25項目（特に最重要・必須の14項目＋因果主張・再現性項目15〜25）の修正および対応する単体テストの実装を行う。

## 主な修正タスク

### 最重要・必須修正（項目1〜14）
- [x] 1. V3 RQ2: $\beta$ の回帰目的変数を `y_v_self` / `y_a_self` (Self-report) に修正し、pair bootstrap CI を保存
- [x] 2. V3: `response_start` の stage index を `cand_start - 1` (prompt_end) に修正（Causal LM の自己回帰的因果関係に整合）
- [x] 3. V3: `estimate_interventional_slope` の引数を `dose_grid`, `report_shift` に整理、1-SD 正規化介入ドーズあたりの report 変位量として統一
- [x] 4. V3 RQ1: `configs/v3_experiments.yaml` の `num_random_controls: 5` を実装へ接続（K本の random & orthogonal コントロール生成・評価と CI 記録）
- [x] 5. V3 Confirmatory: データ空時の `1.0 / 0.5` 架空フォールバックを排除し `RuntimeError` を送出、最低サンプル数アサーションを追加
- [x] 6. V3 Confirmatory: `non_uniform_leverage` を `bool(h4_pass)` に連動
- [x] 7. V1 Phase C: `--alphas` および `--split-seed` を YAML (`configs/v1_experiments.yaml`) から正しく読み込むように修正
- [x] 8. V1 Phase C: E3/E4 途中 CSV 再開時に `checkpoint_manifest.json` を照合し、不一致時は再計算
- [x] 9. V1 Phase C: activation cache の metadata に model_revision, tokenizer_revision, dtype, versions を追加、全プロンプトから hash 生成
- [x] 10. V1 Phase A/B/C: batch 最終トークン抽出を left/right padding の双方に対応する `valid_pos[-1]` ロジックに統一
- [x] 11. V1 Phase A: single-class fold スキップ時に 0 予測が混ざる不具合を修正（StratifiedGroupKFold の活用および evaluated_mask の導入）
- [x] 12. Behavioral: `--dry-run` 時の出力先を `.../dry_run` に隔離し本番成果物を保護、要約スクリプトも連動
- [x] 13. Behavioral: checkpoint 再開時に metadata が存在しない／不一致の場合は resume せず破棄・退避
- [x] 14. V2 RQ1/RQ2: manifest の config_payload に `v2_config` 全体を含め、YAML 設定変更時のキャッシュ無効化を保証

### 因果主張・再現性強化（項目15〜25）
- [x] 15. V2 RQ3: random & orthogonal control directions を追加し、Primary causal metric を $C_{\text{affect}} - C_{\text{random}}$, $C_{\text{affect}} - C_{\perp}$ に拡張
- [x] 16. V3 RQ3: subspace removal に matched-rank random 2D subspace control を追加
- [x] 17. V1 E4: random donor を 1 回から 20 回の固定 seed derangements に拡張し、分布と CI を保存
- [x] 18. V1 E3/E4: 表現・アブレーションの解釈・命名の厳密化（doc/コメント）
- [x] 19. V1 E6: task-specific causal site sensitivity としての明記
- [x] 20. V3 Confirmatory: H1/H2/H4 に pair-bootstrap CI を追加
- [x] 21. V3 RQ2: 因果介入サンプル数・安定性に関する設定・ドキュメント整備
- [x] 22. Sequence likelihood: 候補トークン長の均一性を検証する preflight テストを追加
- [x] 23. Prompt truncation: トークン長が max_length=1024 を超えて切り詰められた場合の検出・警告/エラー機構
- [x] 24. Behavioral RQ3: specificity 解析の統制に関するドキュメント・感度分析整備
- [x] 25. V2 RQ4: raw Base→Instruct patch の off-manifold に関する位置づけ整理

### テストスイート拡充
- [x] テスト実装:
  - `test_v3_beta_targets_self_report`
  - `test_response_start_is_prompt_end`
  - `test_response_end_is_negative_control`
  - `test_v3_num_random_controls_is_honored`
  - `test_confirmatory_empty_effects_raise`
  - `test_non_uniform_leverage_matches_h4`
  - `test_phase_c_alphas_loaded_from_yaml`
  - `test_phase_c_resume_rejected_on_manifest_mismatch`
  - `test_phase_c_cache_invalidated_on_revision_change`
  - `test_hidden_extraction_left_and_right_padding`
  - `test_classification_skipped_fold_not_scored_as_zero`
  - `test_behavioral_dry_run_never_touches_production_outputs`
  - `test_checkpoint_without_metadata_is_not_resumed`
  - `test_v2_geometry_manifest_changes_with_v2_config`
  - `test_candidate_token_lengths_primary_models`
- [x] 回帰テスト全件通過 (`pytest`)、`compileall`、`ruff` の確認
