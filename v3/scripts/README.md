# V3 Scripts

本ディレクトリの探索用スクリプトは `v3/scripts/legacy/` にある。主解析には使わない。旧 NDE/NIE や気分一致実験も legacy である。

正本は [`v3/primary/`](../primary/README.md) である。設計の本文は [`v3/README.md`](../README.md)。

| 正本 | 内容 |
|---|---|
| `v3/primary/run_rq1_state_induction.py` | 状態誘導と Go/No-Go。matched-neutral 必須 |
| `v3/primary/run_rq2_spatiotemporal_maps.py` | 4-Map。joint sequence patch。V/A 別介入 |
| `v3/primary/run_rq3_path_mediation.py` | mediated attenuation |
| `v3/primary/run_confirmatory_replication.py` | 他 family 追試。Sufficiency も V/A 別 |

統合 CLI は RQ1 が完全一致の `GO` のときだけ後続へ進む。本番ゲートは `v3/results/derived/v3_gate_decision.json`、dry-run は `v3/results/derived/dry_run/v3_gate_decision.json`。`--force` は再計算、`--force-after-no-go` はゲート継続。

データは AIPsy 192 pair。81 VA。YAML `response_start` は実行時 `candidate_start` に正規化する。

```bash
python -m affective_empathy_eval.run --stage v3 --model-set primary_small --device cuda:0
```
