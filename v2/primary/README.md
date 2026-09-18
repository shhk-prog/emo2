# V2 Primary

V2 の正式実行面。設計・指標・解釈の本文は親の [`v2/README.md`](../README.md) を正本とする。

## スクリプト

| ファイル | 内容 |
|---|---|
| `run_rq1_rq2_cross_decoding.py` | 表現幾何と Reader–Self 共有（held-out, RSA, Procrustes） |
| `run_rq3_causal_map.py` | 因果マップと decodability / causal ピーク解離 |
| `run_rq4_recovery_patching.py` | 2D Wasserstein による分布回復 |
| `run_confirmatory_analysis.py` | LMM と FDR |

対象は Qwen 2.5 / Llama 3.2 / Gemma 3 / OLMo 2 の Base と Instruct。モデル ID は `configs/models.yaml`。

## 実行

```bash
python -m affective_empathy_eval.run --stage v2 --model-set primary_small --device cuda:0

python v2/primary/run_rq1_rq2_cross_decoding.py \
    --config configs/v2_experiments.yaml \
    --models-config configs/models.yaml \
    --family qwen --device cuda:0
```

相対深度は $d=l/(L-1)$。件数はロード時にログする。
