# タスクリスト: V1 / V3 パイプライン修正および旧結果退避

## 1. 再実行対象の旧結果の退避 (old_results/)
- [x] `old_results/` ディレクトリ作成 <!-- id: 1.1 -->
- [x] V1 Phase A: 各モデルディレクトリ配下の `e1_aipsy_classification.csv` を `old_results/v1_phase_a/` に退避 <!-- id: 1.2 -->
- [x] V1 Phase B: `v1/results/derived/v1_phase_b/` の既存出力を `old_results/v1_phase_b/` に退避 <!-- id: 1.3 -->
- [x] V1 E6: 既存の E6 関連結果があれば `old_results/v1_e6/` に退避 <!-- id: 1.4 -->
- [x] V3: `v3_rq1_results.json`, `manifest_rq1_qwen.json`, `v3_gate_decision.json` 等を `old_results/v3/` に退避 <!-- id: 1.5 -->

## 2. V1 Phase A 修正 (`v1/primary/run_phase_a.py`)
- [x] Direct cross-decoding の scaler 修正 (Reader train fit scaler を Self test に適用、Self train fit scaler を Reader test に適用、task-specific は secondary 指標として保存) <!-- id: 2.1 -->
- [x] AIPsy Primary 解析を `clinical vs matched neutral` (group=`pair_id`) に変更 <!-- id: 2.2 -->
- [x] AIPsy 強度解析を `triplet_id` 存在サンプルによる `none < moderate < peak` 解析として分離出力 <!-- id: 2.3 -->
- [x] AIPsy emotion-category decoding を Secondary 解析へ移動 <!-- id: 2.4 -->

## 3. V1 Phase B 修正 (`v1/primary/run_phase_b.py`, `src/affective_empathy_eval/run.py`)
- [x] `run_phase_b.py`: 出力先を `--task-type` に応じて `v1/results/derived/v1_phase_b/{task_type}/{model_prefix}` に分離 <!-- id: 3.1 -->
- [x] `run.py`: Phase B 実行部で Reader と Self の両条件を順次実行するように修正 <!-- id: 3.2 -->

## 4. V1 E6 修正 (`v1/primary/phase_c/run_e6_specialization.py`)
- [x] `impact_v`, `impact_a`, `impact_va = sqrt(delta_v^2 + delta_a^2)` を保存 <!-- id: 4.1 -->
- [x] Primary metric を `impact_va` に統一、LMM 交互作用検定を `impact_va` を主として実行 <!-- id: 4.2 -->
- [x] Secondary として `impact_v`, `impact_a` の LMM 結果も辞書に記録 <!-- id: 4.3 -->

## 5. V1 E4 多重比較整理 (`v1/primary/run_phase_c.py`)
- [x] Primary confirmatory condition (`E3 peak layer × alpha=1.0`) のフラグ付け (`is_confirmatory`) <!-- id: 5.1 -->
- [x] その他の layer/alpha/V/A 条件に対する Benjamini-Hochberg FDR 補正後 p 値の算出と列追加 <!-- id: 5.2 -->

## 6. V3 RQ1 & Gate 修正 (`configs/v3_experiments.yaml`, `v3/primary/run_rq1_state_induction.py`)
- [x] `configs/v3_experiments.yaml`: `gate_criteria` に `min_sufficiency_slope: 0.1` を追加 <!-- id: 6.1 -->
- [x] `run_rq1_state_induction.py`: `half_pairs = len(unique_pairs) // 2` を削除し、`train_ratio: 0.7` を使用 <!-- id: 6.2 -->
- [x] `run_rq1_state_induction.py`: Gate 判定の slope 閾値 0.1 を config から取得するように修正 <!-- id: 6.3 -->

## 7. V3 Confirmatory site 生成・読込修正
- [x] `run_rq2_spatiotemporal_maps.py`: `frozen_confirmatory_sites.json` の確定出力を廃止 <!-- id: 7.1 -->
- [x] `run_rq3_path_mediation.py`: RQ3 終了時に RQ1 Gate site、RQ2 Causal site、RQ3 Mediator site を統合した `frozen_confirmatory_sites.json` を確定生成 <!-- id: 7.2 -->
- [x] `run_confirmatory_replication.py`: 本番実行時に frozen site artifact が無ければエラーにする <!-- id: 7.3 -->

## 8. Likelihood 記述と Legacy コードの整理
- [x] raw sequence log-likelihood の記述を `joint conditional sequence log-likelihood` に統一 <!-- id: 8.1 -->
- [x] `v3/scripts/legacy/README.md` に `DO NOT USE FOR PAPER RESULTS` と明記 <!-- id: 8.2 -->
- [x] Legacy コード内の `compute_expected_va(probs, ...)` 呼び出しを是正・削除 <!-- id: 8.3 -->

## 9. 検証
- [x] テストスイートの実行 (`test_production_entrypoints.py`, `test_v1_token_and_probe_alignment.py` 等) <!-- id: 9.1 -->
- [x] V1 各フェーズ (A, B reader/self, C, E6) の dry-run 実行確認 <!-- id: 9.2 -->
- [x] V3 各フェーズ (RQ1, RQ2, RQ3, Confirmatory) の dry-run 実行確認 <!-- id: 9.3 -->
- [x] `old_results/` の退避内容確認 <!-- id: 9.4 -->
