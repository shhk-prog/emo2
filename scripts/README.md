# Production Scripts

リポジトリルートの `scripts/` は本番ラッパである。実験本体は `python -m affective_empathy_eval.run` と各 `*/primary/` にある。

## 本番ランナー

すべて `.venv` を activate し、`results/logs/production_<stage>_TIMESTAMP.log` に tee する。第1引数が device（既定 `cuda:0`）。第2引数以降は `EXTRA_ARGS` として統合 CLI へそのまま転送する。

| Script | 固定で付く CLI 引数 | 備考 |
|---|---|---|
| `run_production_behavioral.sh` | `--stage behavioral --model-set primary_small` | 完了後の要約は CLI 側が自動実行 |
| `run_production_v1.sh` | `--stage v1 --model-set primary_small --all-layers` | Phase B 統制 CSV が無ければ先に生成 |
| `run_production_v2.sh` | `--stage v2 --model-set primary_small` | family 未指定なので confirmatory も走る |
| `run_production_v3.sh` | `--stage v3 --model-set primary_small` | RQ1 が完全一致 `GO` でないと停止 |
| `run_production_all.sh` | 上記 4 本を Behavioral → V1 → V2 → V3 | 同じ `EXTRA_ARGS` を全 Stage へ転送 |

例:

```bash
bash scripts/run_production_behavioral.sh cuda:0
bash scripts/run_production_v1.sh cuda:0 --force
bash scripts/run_production_v3.sh cuda:0 --force-after-no-go
```

`--force` はキャッシュ再計算、`--force-after-no-go` は V3 ゲート継続である。混ぜない。

## その他

| Script | 役割 |
|---|---|
| `run_candidate_space_sensitivity.py` | 同一刺激での 729 VAD vs 81 VA 感度分析。`--model-revision` 未指定なら registry から解決 |
| `run_scale_validation.py` | Mistral 7B。統合 CLI の `--stage scale_validation` からも呼ばれる |
| `run_production_reruns.sh` | 既存成果物の部分再計算用。主本番経路ではない |

探索用の旧スクリプトは各 Stage の `*/scripts/legacy/` にある。主解析に使わない。
