# 修正内容の確認 (Walkthrough): v1 Phase A の UnboundLocalError 修正

## 1. 問題の概要
本番実行コマンド `bash scripts/run_production_v1.sh cuda:0 --family olmo` の実行中、OLMo Base の順伝播完了直後に以下の例外で停止しました：
```text
Traceback (most recent call last):
  File "/mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_a.py", line 1341, in <module>
    main()
  File "/mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_a.py", line 1102, in main
    save_experiment_result(
UnboundLocalError: cannot access local variable 'save_experiment_result' where it is not associated with a value
```

## 2. 変更内容
### [v1/primary/run_phase_a.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_a.py)
- モジュール先頭のトップレベル import に `from affective_empathy_eval.io import is_experiment_completed, save_experiment_result` を確保。
- 関数 `main()` 内にあった以下の局所 import を削除し、スコープ解決でローカル変数扱いされる問題を解消：
  - line 721: `from affective_empathy_eval.io import is_experiment_completed`
  - line 792: `from affective_empathy_eval.io import save_experiment_result`（`if args.dry_run:` 節内）
  - line 1309: `from affective_empathy_eval.io import save_experiment_result`（関数終盤）

### [v1/primary/phase_c/run_e6_specialization.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/phase_c/run_e6_specialization.py)
- モジュール先頭に `from affective_empathy_eval.io import is_experiment_completed, save_experiment_result` を集約。
- 関数内の局所 import（line 402, 461）を削除。

## 3. 検証結果
1. **未定義シンボル検証 (`ruff check --select F`)**:
   - `F821 (Undefined name)` は 0 件であり、インポートエラーや未定義変数参照がないことを確認。
2. **単体テスト (`pytest`)**:
   - `tests/test_v1_token_and_probe_alignment.py`
   - `tests/test_production_entrypoints.py`
   - `tests/test_io_modular.py`
   - 全て 8 passed (100%)。
3. **Dry-run 動作検証**:
   - `v1/primary/run_phase_a.py --model-id allenai/OLMo-2-0425-1B --model-prefix olmo_base --dry-run --device cpu` を実行し、正常終了を確認。
4. **シンボル参照検証**:
   - `python -c "import v1.primary.run_phase_a as pa; assert hasattr(pa, 'save_experiment_result')"` により、トップレベルから正常に参照可能であることを確認。

## 4. 次の手順
本修正により、本番実行時に `save_experiment_result` が正常に呼び出されます。
ターミナルにて再度コマンドを実行してください：
```bash
bash scripts/run_production_v1.sh cuda:0 --family olmo
```
（または全体本番実行：`bash scripts/run_production_v1.sh cuda:0 --force`）
