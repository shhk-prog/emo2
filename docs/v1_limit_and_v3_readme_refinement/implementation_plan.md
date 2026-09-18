# 実装計画書: V1 Phase B --limit 整合性確保・CLI引数互換性テスト追加・V3 README 更新

## 1. 概要
本改修では、スモークテストおよび小サンプル動作確認時に発生する `run_phase_b.py: error: unrecognized arguments: --limit` を解消し、統合 CLI から全ステージ（Behavioral, V1, V2, V3）が引数の齟齬なく実行可能であることを自動テストで保証します。
あわせて、直前の改修で実装された V3 の「Reader-Grounded 情動方向（Primary）」「RQ2 介入サンプル数 15件」を正本 README に完全に整合させます。

## 2. 変更対象と実装内容

### 2.1 `v1/primary/run_phase_b.py`
- `argparse` に `--limit` 引数を追加（`type=int, default=0, help="Sample limit (0 for full dataset)"`）。
- データ読み込み後、`if args.limit and args.limit > 0: df = df.head(args.limit).copy().reset_index(drop=True)` により、スモークテストでの迅速な実行を可能にする。

### 2.2 `tests/test_production_entrypoints.py`
- `test_dispatched_commands_argparse_compatibility()` を追加。
- 統合 CLI `affective_empathy_eval.run` が `--dry-run --max-samples 2` で生成する全コマンドについて、各スクリプトが引数を正常にパースできること（`unrecognized arguments` が発生しないこと）を自動検証。

### 2.3 `v3/README.md`
- **RQ1**: Primary 情動方向が $H_{\mathrm{Self}} \rightarrow (V_R, A_R)$ の Reader Prediction から推定され、Self-derived direction は Secondary analysis として保持・比較される旨を明記。
- **RQ2**: 実介入サンプル数を $\min(5, N)$ から `n_causal_samples: 15`（既定 $\min(15, N)$）に更新。全マップが Discovery であり、最終推論は独立 Confirmation で行う旨を明記。

## 3. 検証計画
- `pytest tests/test_production_entrypoints.py -v`:
  エントリポイント存在、ドライランディスパッチ、CLI 引数互換性テストの通過を確認。
- `pytest -q`:
  全体テストスイートの通過を確認。
- 統合 CLI スモークテスト:
  `python -m affective_empathy_eval.run --stage v1 --dry-run --max-samples 2`
  がエラーなく完走することを確認。
