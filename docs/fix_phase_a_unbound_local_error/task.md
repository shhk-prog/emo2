# タスクリスト: v1 Phase A の UnboundLocalError 修正

- [x] エラー原因の特定 (`v1/primary/run_phase_a.py` の `main()` 内のローカル import による `UnboundLocalError`)
- [x] `v1/primary/run_phase_a.py` のトップレベル import 統一および関数内冗長 import の削除
- [x] `v1/primary/phase_c/run_e6_specialization.py` の同様の冗長局所 import の整理
- [x] ruff check (`--select F`) による未定義変数の不在検証
- [x] pytest による単体テスト通過検証 (`tests/test_v1_token_and_probe_alignment.py` 等)
- [x] `run_phase_a.py --dry-run` の正常終了確認
- [ ] ユーザーへの修正報告と本番パイプライン再開の案内
