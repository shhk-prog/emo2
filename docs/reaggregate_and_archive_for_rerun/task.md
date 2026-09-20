# タスクリスト: 再集計および再実行用結果退避・実行手順の整備

## 1. 準備・移動（再実行対象の既存結果を `old_results` へ退避）
- [x] 1.1 `old_results/` 配下にタイムスタンプ付き退避ディレクトリ（`archive_20260921/`）を作成 <!-- id: 1 -->
- [x] 1.2 **V1 Phase B** の既存結果を `old_results/archive_20260921/v1_phase_b/` へ移動 <!-- id: 2 -->
- [x] 1.3 **V3 (RQ2, RQ3, Confirmatory)** の既存結果を `old_results/archive_20260921/v3/` へ移動 <!-- id: 3 -->
- [x] 1.4 **V2 RQ4** の既存結果（`v2_causal_map_*`, `v2_recovery_*`, `pair_level/` 等）を `old_results/archive_20260921/v2_rq4/` へ移動 <!-- id: 4 -->
- [x] 1.5 結果ディレクトリの `.gitkeep` を保持・確認 <!-- id: 5 -->

## 2. 再集計の実行
- [x] 2.1 **V2 RQ1/RQ2**: 既存の4ファミリー生結果（`v2_geometry_*.json`）から、改訂後の Primary (matched-plain) / Secondary (native-chat) / Format Effect 分離集計を行い `v2/results/derived/v2_cross_family_summary.json` を再生成 <!-- id: 6 -->
- [x] 2.2 **V1 Phase A**: `e2_emobank_geometry.csv` の生転移スコア保持（負の $R^2$ クリップ撤廃）を反映する再集計スクリプト（`scripts/reaggregate_v1_phase_a.py`）を配備 <!-- id: 7 -->

## 3. 再実行手順の提供（ユーザー実行用）
- [x] 3.1 ユーザーが GPU 環境等で再実行するための実行スクリプト・コマンド（`scripts/run_production_reruns.sh`）を作成・提示 <!-- id: 8 -->
- [x] 3.2 Walkthrough レポートの作成 <!-- id: 9 -->
