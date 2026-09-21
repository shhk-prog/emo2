# 実装計画: v1 Phase A の UnboundLocalError 修正

## 概要
`bash scripts/run_production_v1.sh cuda:0 --family olmo` 実行時に `v1/primary/run_phase_a.py` の line 1102 で発生した以下のエラーを修正する：
```text
UnboundLocalError: cannot access local variable 'save_experiment_result' where it is not associated with a value
```

## 原因分析
Python では、関数スコープ内のどこかに `from ... import name` が記述されていると、その関数内での `name` はローカル変数としてバインドされる。
`v1/primary/run_phase_a.py` では：
- line 792 の `if args.dry_run:` 節内
- line 1309 の末尾の `if e1_payload:` 直前
に `from affective_empathy_eval.io import save_experiment_result` が存在した。
本番実行時 (`args.dry_run=False`) は line 792 を通過せず、line 1102 で `save_experiment_result(...)` を最初に呼び出した時点でローカル変数 `save_experiment_result` が未代入となり、`UnboundLocalError` が発生した。

## 修正方針
1. `v1/primary/run_phase_a.py`
   - モジュール最上部（line 53付近）に既に `from affective_empathy_eval.io import is_experiment_completed, save_experiment_result` を配置済み。
   - `main()` 関数内部にある冗長な局所 import 文（line 721の `is_experiment_completed`、line 792の `save_experiment_result`、line 1309の `save_experiment_result`）を削除。
2. `v1/primary/phase_c/run_e6_specialization.py`
   - 同様の潜在的不具合（line 402, 461 の局所 import）を解消し、モジュール最上部の import に統合。

## 検証計画
- `ruff check --select F` で未定義シンボルがないことを確認。
- 関連する単体テスト (`pytest -q tests/test_v1_token_and_probe_alignment.py` 等) を実行して通過を確認。
- `--dry-run` で `run_phase_a.py` が正常終了することを確認。
- モジュールインポート時に `save_experiment_result` が正しく参照できることを確認。
