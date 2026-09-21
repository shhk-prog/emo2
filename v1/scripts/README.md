# V1 Scripts

本ディレクトリの探索用スクリプトは `v1/scripts/legacy/` にある。主解析には使わない。

正本は [`v1/primary/`](../primary/README.md) である。設計の本文は [`v1/README.md`](../README.md)。

| 正本 | 内容 |
|---|---|
| `v1/primary/run_phase_a.py` | E1 decodability、E2 geometry |
| `v1/primary/run_phase_b.py` | rule-based controlled perturbation。`--task-type reader` と `self` を別実行 |
| `v1/primary/run_phase_c.py` | E3 causal map、E4 interchangeability。本番相当は `--all-layers` |
| `v1/primary/phase_c/run_e6_specialization.py` | Task-Specific Causal Specialization。distinct site が無ければ No-Go |
| `v1/primary/phase_c/summarize_phase_c.py` | Phase C 横断要約 |

`--model-id` または `--family` が必須。比較軸は Reader ↔ Self。Base/Instruct 差の解釈は V2。

```bash
python -m affective_empathy_eval.run --stage v1 --model-set primary_small --device cuda:0 --all-layers
```

`scripts/run_production_v1.sh` は常に `--all-layers` を付ける。
