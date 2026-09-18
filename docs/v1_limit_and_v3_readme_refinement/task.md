# タスクリスト: V1 Phase B --limit 整合性確保・CLI引数互換性テスト追加・V3 README 更新

## 1. 課題と背景
- **V1 Phase B `--limit` 欠落問題**:
  - `run_v1()` は `--max-samples` 指定時に全 Phase に `--limit <N>` を渡すが、`v1/primary/run_phase_b.py` に `--limit` 引数が無いため `unrecognized arguments: --limit 2` で停止していた。
  - `run_phase_b.py` に `--limit` 引数を追加し、データサブセット処理を実装。
- **CLI 引数互換性回帰テストの追加**:
  - `tests/test_production_entrypoints.py` において、統合 CLI がディスパッチする全コマンドの全引数フラグが各サブスクリプトの parser で正しく認識されるかを検証する `test_all_dispatched_commands_argparse_compatibility` を追加。
- **V3 README の更新**:
  - RQ1: Primary 情動方向が Reader Prediction（$H_{\text{Self}} \rightarrow (V_R, A_R)$）由来であり、Self-derived は Secondary であることを正本 README に反映。
  - RQ2: 因果介入サンプル数を 5 件から 15 件（`spatiotemporal.n_causal_samples: 15` 連動）へ更新し、全マップが Discovery であり最終推論は独立 Confirmation であることを明記。

## 2. タスク進捗
- [x] `v1/primary/run_phase_b.py` への `--limit` 追加とデータ切り出し実装
- [x] `tests/test_production_entrypoints.py` への CLI 引数パース互換性テスト追加 (`test_all_dispatched_commands_argparse_compatibility`)
- [x] `v3/README.md` の Primary 情動方向および RQ2 介入サンプル数の更新
- [x] テスト実行（`pytest tests/test_production_entrypoints.py -v` 3 passed, `pytest -q` 61 passed）
- [x] 統合 CLI スモークテスト（`--stage v1 --dry-run --max-samples 2`）の正常完走確認
- [x] 結果ディレクトリ初期化確認 (`clear_stage_results.sh`)
- [x] ドキュメント（`task.md`, `implementation_plan.md`, `walkthrough.md`）の保存
