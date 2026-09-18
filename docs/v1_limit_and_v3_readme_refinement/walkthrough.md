# 変更内容の確認 (Walkthrough): V1 Phase B --limit 整合性確保・CLI引数互換性テスト追加・V3 README 更新

## 1. 概要
スモークテストおよび小サンプル実行時（`--max-samples <N>`）に発生していた `run_phase_b.py: error: unrecognized arguments: --limit` を解消し、統合 CLI から全ステージが引数の齟齬なく実行可能であることを保証する回帰テストを追加しました。
あわせて、直前にコード側で刷新された V3 の「Reader-Grounded 情動方向（Primary）」「RQ2 介入サンプル数 15件」を正本 README に完全に整合させました。

---

## 2. 実施内容と検証結果

### 2.1 V1 Phase B への `--limit` 引数追加 (`v1/primary/run_phase_b.py`)
- `parser.add_argument("--limit", type=int, default=0, help="Sample limit (0 for full dataset)")` を追加。
- データフレーム読み込み後、`if args.limit and args.limit > 0: df = df.head(args.limit).copy().reset_index(drop=True)` により、スモークテストでの迅速な切り出し実行を可能にしました。

### 2.2 CLI 引数パース互換性テストの追加 (`tests/test_production_entrypoints.py`)
- `test_all_dispatched_commands_argparse_compatibility()` を新設。
- 統合 CLI `affective_empathy_eval.run` が全 4 ステージ（Behavioral, V1, V2, V3）で `--dry-run --max-samples 2` を指定して生成する全ディスパッチコマンドについて、渡された各フラグ（`--limit`, `--is-instruct`, `--device` 等）が対象スクリプトの parser で正しく定義・認識されているかを自動検証します。

### 2.3 V3 正本 README の更新 (`v3/README.md`)
- **RQ1**: Primary 情動方向が、同じ刺激に対するモデル自身の感情認識予測値（Reader Prediction）への回帰 $H_{\mathrm{Self}} \rightarrow (V_R, A_R)$ から同定される $d_V^R, d_A^R$ であり、自己報告自身から学習する方向は Secondary analysis として保持・比較される旨を明記。
- **RQ2**: 因果介入サンプル数を $\min(5, N)$ から設定連動の `spatiotemporal.n_causal_samples: 15`（既定 $\min(15, N)$）に更新。時空間マップ全体が Discovery であり、最終的な因果推論は独立 Confirmation 評価で行う旨を明記。

---

## 3. テストと動作検証

### 3.1 単体・統合テスト
```bash
$ .venv/bin/pytest tests/test_production_entrypoints.py -v
tests/test_production_entrypoints.py::test_all_primary_entrypoints_exist PASSED            [ 33%]
tests/test_production_entrypoints.py::test_production_dry_run_dispatch PASSED              [ 66%]
tests/test_production_entrypoints.py::test_all_dispatched_commands_argparse_compatibility PASSED [100%]
====================================== 3 passed in 81.19s ======================================

$ .venv/bin/pytest -q
.............................................................                              [100%]
61 passed in 85.42s (0:01:25)
```

### 3.2 統合 CLI スモークテスト
```bash
$ .venv/bin/python -m affective_empathy_eval.run --stage v1 --dry-run --max-samples 2 --model-set primary_small
...
All requested stages completed successfully!
```
全 4 ファミリー（Qwen, Llama, Gemma, OLMo）の Base / Instruct において、Phase A, Phase B, Phase C, E6, Summarize が一切のエラーなく完全に完走しました。

### 3.3 結果ディレクトリの初期化
- `bash scripts/clear_stage_results.sh v1` を実行し、全 results ディレクトリが `.gitkeep` のみのクリーンな状態に保たれていることを確認。

---

## 4. 本番全再実行コマンド

本番のフルデータ（4小型ファミリー: Qwen, Llama, Gemma, OLMo）に対する実行コマンドは以下の通りです：

```bash
# 1. Behavioral Stage（EmoBank & AIPsy）
bash scripts/run_production_behavioral.sh

# 2. V1 Stage（Phase A / B / C E3, E4, E6 / Summarize）
bash scripts/run_production_v1.sh

# 3. V2 Stage（RQ1/RQ2, RQ3 Causal Map, RQ4 Recovery Patching, Confirmatory）
bash scripts/run_production_v2.sh

# 4. V3 Stage（RQ1 State Induction, RQ2 Spatiotemporal, RQ3 Mediation, Confirmatory Replication）
bash scripts/run_production_v3.sh
```
