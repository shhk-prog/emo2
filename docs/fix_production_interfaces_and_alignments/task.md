# タスク: 本番実行インターフェース不整合・位置交絡の解消とプロトコル統一

## 背景・目的
前回のレビューで特定された、dry-run や mock テストをすり抜けていた本番実行（real run）時の致命的なバグ、インターフェース不整合、位置交絡（DとCのtoken position不一致）、未初期化変数、および集計不足を根本的に修正し、fresh rerun を確実に成功させる基盤を整える。

## タスクリスト

### 1. Behavioral Stage の修復
- [x] `behavioral/primary/run_behavioral_emobank.py` に `--device` 引数を追加し、個別 GPU（`cuda:0` 等）へ正しく配置
- [x] `behavioral/primary/run_behavioral_aipsy.py` の `device_map` ロジックを修正し、`cuda:0` 等で確実に GPU ロード
- [x] `behavioral/analysis/summarize_behavioral_aipsy.py` に `pair_id` 照合、Dose-Response（`neutral`→`moderate`→`clinical`）、Specificity（`clinical` vs `complex_neutral`）集計を追加

### 2. V1 Stage の補完・整合
- [x] `v1/primary/run_phase_a.py` に Arousal 幾何、AIPsy 分類プローブ、Secondary model-output target を追加
- [x] `v1/primary/run_phase_b.py` の `extract_single_layer_hidden_states` 呼び出しに `task_type` と `is_instruct` を明示
- [x] `v1/primary/run_phase_c.py` および `affective_empathy_eval/run.py` で `--all-layers` が本番パイプラインで渡るように整備

### 3. V2 Stage の位置交絡解消とバグ修正
- [x] `v2/primary/run_rq1_rq2_cross_decoding.py` の抽出アンカー位置を `prompt_end` に統一（RQ3 の因果介入位置と揃え、DとCの位置交絡を解消）
- [x] `v2/primary/run_rq4_recovery_patching.py` で未初期化だった `layer_mean_ratios_plain` を初期化し、matched_plain の分母を matched_plain 自身の初期 EMD に修正
- [x] `v2/primary/run_confirmatory_analysis.py` を `v2/legacy/` へ退避

### 4. V3 Stage のインターフェース不一致・致命的バグの解消
- [x] `src/affective_empathy_eval/interventions.py` の `extract_conditional_directions` と `compute_orthonormal_subspace` の引数柔軟性を拡張
- [x] `v3/primary/run_confirmatory_replication.py` のインポート欠落（`get_model_adapter`）、未初期化変数（`all_H`）、不正なフック引数（`register_patch_hook`）を修正
- [x] `scripts/run_production_v3.sh` で `--force-after-no-go` を含む追加引数が CLI に正しく渡るよう修正
- [x] `v3/primary/run_rq2_spatiotemporal_maps.py` の因果介入サンプル選択を再現性のある seeded random subset に変更

### 5. 論文ドキュメント・Methods プロトコル表の整備
- [x] root `README.md` に共通 protocol table を追加し、主張のトーンを適切に調整
- [x] テストスイートの実行確認と `walkthrough.md` の作成
