# 再実行前7点修正 — Walkthrough

## 結果

レビュー指定の7点を直し、正式再実行に進める状態にした。
`pytest`: **46 passed / 1 skipped**（新規4件を含む）。Primary 対象ファイルの `py_compile` は成功。

プロジェクト `.venv` は Linux 上の `/usr/bin/python3.12` を指すため、この macOS 作業環境では実行できない。検証は既存のローカル Python 3.13 で `PYTHONPATH=src` を付けて行った。パッケージの追加インストールはしていない。`ruff` はこの環境に無く、未実行。

## 7点の対応

### 1. `--stage all` の順序
`src/affective_empathy_eval/run.py` の `--stage all` を `PRODUCTION_STAGE_ORDER = (behavioral, v1, v2, v3)` に固定した。`run_production_all.sh` と同じ順になる。

### 2. 所要時間
「Behavioral 30分〜1時間」等の固定目安を production scripts と統合 runner のログから削除した。未計測と明記。実測は Qwen 1 family の benchmark 後に \(T_{\mathrm{total}} \approx \sum T_{\mathrm{family}}\) で更新する。

### 3. V1 Qwen default の廃止
Phase A/B/C/E6 の `--model-id` default を削除した。`--model-id` または `--family` が無い単独実行は `ValueError`。統合 runner は従来どおり明示的に `--model-id` を渡す。

### 4. V3 silent fallback の例外化
RQ1/RQ2/RQ3 の `else "Qwen/Qwen2.5-1.5B-Instruct"` を廃止し、`resolve_instruct_target_from_args()` で registry に無い family は `KeyError`。Confirmatory の YAML 後方互換フォールバックも `KeyError` にした。

### 5. モデル正本の一本化
`configs/scale_validation.yaml` から `base` / `instruct` を削除し、`model_set: scale_validation` のみ残した。ID は `configs/models.yaml` のみ。

### 6. V2 コホート表記
`v2/primary/README.md` と RQ1–RQ3 docstring を Qwen / Llama / Gemma 3 / OLMo 2 に更新した。

### 7. results クリア
`scripts/clear_stage_results.sh` を追加し、実行済み。
`behavioral/results/`, `v1/results/`, `v2/results/`, `v3/results/` は `.gitkeep` のみ。
Git 管理されていた V1 Phase A/B/C 成果物は作業ツリー上で削除されている（未コミット）。

## 関連整理

- Path Mediation の現行 docs を **mediated attenuation** に統一。NDE/NIE は現行導線から外した。
- Topic control は Topic TVD が小さいことを確認する非特異的統制。Self − Control は補助記録。ゲートは `max_topic_tvd`。
- Phase B は \(d=0.5\) 事前固定、\(l=\operatorname{round}(d(L-1))\)。Phase A peak は使わない。`v1/docs/protocol.md` に記載。
- V2 の重複 import を整理。
- データ件数は実 CSV ロード数をログする。shell コメントの「1,000 / 480 / 400」は削除。

## 本番の進め方

一括 `run_production_all.sh` より、stage 分割を推奨する。

```bash
bash scripts/run_production_behavioral.sh cuda:0
bash scripts/run_production_v1.sh cuda:0
bash scripts/run_production_v2.sh cuda:0
bash scripts/run_production_v3.sh cuda:0
```

先に Qwen 1 family で benchmark を取り、所要時間を更新してから 4 family フルに進むのが安全。
