# タスクリスト: Primary スクリプト完全復元・エントリポイントテスト追加・V3 Confirmatory 統制精緻化

## 1. 課題と背景
- アーカイブ作業時に消失していた V1 / V2 Primary ソースコード（8ファイル）をコミット `277e362` から完全復元。
- Git 管理から外れていた `.gitignore` を再整備し、キャッシュや実験結果の一括追跡除外を設定。
- 本番実行時にスクリプト欠落やファンアウト漏れを即座に検知する回帰テスト `tests/test_production_entrypoints.py` を追加。
- V3 Confirmatory Replication において、中間層 $d_V$ の全層注入による幾何不適合の交絡を排除するため、各層 $l$ での局所情動ベクトル $d_V^{(l)}$ 推定・注入に layer-specific 化。
- V3 Confirmatory H4 (Temporal Emergence) において、Valence に加えて Arousal の時間的出現 ($d_A$) も独立して検証・保存するよう拡張。

## 2. タスク一覧
- [x] Primary スクリプト（V1/V2）の Git 履歴からの復元
  - [x] `v1/primary/prepare_v1_phase_b_controls.py`
  - [x] `v1/primary/run_phase_a.py`
  - [x] `v1/primary/run_phase_b.py`
  - [x] `v1/primary/run_phase_c.py`
  - [x] `v1/primary/phase_c/run_e6_specialization.py`
  - [x] `v2/primary/run_rq1_rq2_cross_decoding.py`
  - [x] `v2/primary/run_rq3_causal_map.py`
  - [x] `v2/primary/run_rq4_recovery_patching.py`
- [x] `.gitignore` の再整備とバイトコード・キャッシュ削除
- [x] 本番ランナー実体存在・ドライラン検証テスト `tests/test_production_entrypoints.py` の追加
- [x] V3 Confirmatory Replication (`v3/primary/run_confirmatory_replication.py`) の統制精緻化
  - [x] Causal Profile $C(l)$ の layer-specific 化（各層 $l$ での $d_V^{(l)}$ 推定とスケール適応）
  - [x] Temporal Emergence (H4) の Arousal 拡張（Valence/Arousal の独立出現検証・記録）
  - [x] Simulation ロジックとの整合性確保
- [x] テストスイート検証
  - [x] `pytest tests/test_production_entrypoints.py` (2 passed)
  - [x] `pytest -q` (60 passed)
- [x] 実装・検証ドキュメント（`task.md`, `implementation_plan.md`, `walkthrough.md`）の保存
