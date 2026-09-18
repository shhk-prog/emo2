# タスクリスト: 全再実行直前の最終4点リファインメント

## 概要
全再実行前の最終確認において挙がった以下の4点を修正し、実験パイプラインの測定厳密性・整合性を完全に確保する。

## タスク一覧
- [x] **1. V3 RQ2 metadata の修正 (`v3/primary/run_rq2_spatiotemporal_maps.py`)**
  - 実際の介入サンプル数（最大5件: `min(5, N)`）とマップ算出サンプル数（$N$）を分離
  - `"n_map_samples": N`
  - `"n_intervene_samples": min(5, N)`
  - `"n_causal_intervention_samples": min(5, N)`
- [x] **2. V3 RQ2 Decodability CV の GroupKFold 統一 (`v3/primary/run_rq2_spatiotemporal_maps.py`)**
  - AIPsy の `pair_id` を用いて、V1/V3 Path Mediation と同様に `GroupKFold(n_splits=min(3, n_groups))` による交差検証へ変更（pair leakage 防止方針の全Stage統一）
- [x] **3. E6 の `num_layers=28` silent fallback の完全削除 (`v1/primary/phase_c/run_e6_specialization.py`)**
  - `resolve_architecture_dims()` 失敗時の `num_layers = 28` 静的フォールバックを削除し、例外をそのまま `raise` して安全に失敗させる
- [x] **4. E6 の Double Dissociation 旧名称の完全統一 (`v1/primary/phase_c/run_e6_specialization.py`)**
  - docstring の名称を `Task-Specific Causal Specialization` に統一
  - legacy 出力（`e6_double_dissociation_trials.csv`）の二重出力を削除し、Primary 出力の `e6_specialization_trials.csv` のみに一本化
- [x] **ドキュメントの保存**
  - `docs/final_pre_rerun_4points_refinement/` 配下に `task.md`, `implementation_plan.md`, `walkthrough.md` を保存
