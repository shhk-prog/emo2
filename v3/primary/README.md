# V3 Primary

V3 の正式実行面。設計・指標・解釈の本文は親の [`v3/README.md`](../README.md) を正本とする。

## スクリプト

| ファイル | 内容 |
|---|---|
| `run_rq1_state_induction.py` | 状態誘導と Go/No-Go。Topic は非特異的摂動の確認統制 |
| `run_rq2_spatiotemporal_maps.py` | 層 × 意味段階の 4-Map（$D$, $\beta$, $\gamma$, $C$） |
| `run_rq3_path_mediation.py` | mediated attenuation（NDE/NIE は使わない） |
| `run_confirmatory_replication.py` | Llama / Gemma 3 / OLMo 2 での追試 |

モデルは `configs/models.yaml`。未知 family は `KeyError`。層は指定が無ければ $d=0.5$ から $l=\operatorname{round}(d(L-1))$。

## 実行

```bash
python -m affective_empathy_eval.run --stage v3 --model-set primary_small --device cuda:0

python v3/primary/run_rq1_state_induction.py \
    --config configs/v3_experiments.yaml \
    --models-config configs/models.yaml \
    --family qwen --device cuda:0
```
