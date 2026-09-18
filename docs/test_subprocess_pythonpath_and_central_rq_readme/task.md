# タスク: テスト環境非依存化 (PYTHONPATH) および root README への Central RQ / 4-Stage 統合

## 背景・目的
ユーザーフィードバックに基づき、全再実行（Behavioral → V1 → V2 → V3）前の最後の2点の修正を実施する。
1. `test_all_dispatched_commands_argparse_compatibility` が clone 直後や非 editable install 環境で実行された際、subprocess 内で `src/` が見つからず `ModuleNotFoundError: affective_empathy_eval` となる問題を修正（subprocess 呼び出し時に `PYTHONPATH=<repo>/src` を確実に渡す）。
2. root `README.md` の冒頭に、今回提示された Central RQ、4-Stage の概念対応表、論文 Section 構造、3本の Contribution、および中心的主張（Core Thesis）を明記し、リポジトリ全体の学術的意図を冒頭で瞬時に把握できるようにする。

## タスクリスト
- [x] 計画策定と合意形成 (`docs/test_subprocess_pythonpath_and_central_rq_readme/implementation_plan.md`)
- [x] `tests/test_production_entrypoints.py` の修正（subprocess 実行時に `PYTHONPATH` と `cwd` を確実に設定）
- [x] `src/affective_empathy_eval/run.py` の `run_command` の堅牢化（念のため subprocess に `PYTHONPATH` を補完）
- [x] root `README.md` 冒頭への Central RQ・4-Stage 概念表・論文構成・Contribution・Core Thesis の追記整理
- [x] テスト実行による検証（`pytest -q` 全通過、および PYTHONPATH 未設定下での subprocess 挙動確認）
- [x] 変更内容の確認書 (`walkthrough.md`) 作成
