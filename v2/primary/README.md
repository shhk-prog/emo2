# V2 Primary

V2 の正式実行面。設計・指標・解釈の本文は親の [`v2/README.md`](../README.md) を正本とする。

比較軸は同一ファミリーの **Base ↔ Instruct**。Reader ↔ Self の解釈は V1、状態誘導の時空間は V3。候補空間は 81 VA。データ既定は EmoBank test1k（`configs/v2_experiments.yaml`）。

## 実行順

統合 CLI / `run_production_v2.sh` は次の順である。

1. `run_rq1_rq2_cross_decoding.py`
2. `run_rq3_causal_map.py`
3. `run_rq4_recovery_patching.py`

確証的 LMM は `run_confirmatory_analysis.py` を別途実行する。scale validation（Mistral 7B）は `--model-set scale_validation` または `--stage scale_validation` で Primary と分離する。

## スクリプト

| ファイル | 問い | 要点 |
|---|---|---|
| `run_rq1_rq2_cross_decoding.py` | 幾何と sharing はどう変わるか | 人間 `reader_V/A`、held-out Ridge、RSA、PCA→Procrustes、`native` / `matched_plain` |
| `run_rq3_causal_map.py` | $d_C$ と $d_D$ は同じか | $C(l)$ は実介入。Family × (Base, Instruct) × (Reader, Self) |
| `run_rq4_recovery_patching.py` | Instruct 分布を Base へ戻せるか | `Recovery = (W1(clean,target)-W1(patch,target))/W1(clean,target)`。`emd_va` |
| `run_confirmatory_analysis.py` | 横断は頑健か | LMM と FDR。7B を主表に混ぜない |

`--dry-run` は固定 fixture ラベル。乱数ラベルは使わない。`--max-samples` は確認用。

## 実行

```bash
bash scripts/run_production_v2.sh cuda:0

python -m affective_empathy_eval.run --stage v2 --model-set primary_small --device cuda:0
python -m affective_empathy_eval.run --stage scale_validation --device cuda:0

python v2/primary/run_rq1_rq2_cross_decoding.py \
    --config configs/v2_experiments.yaml \
    --models-config configs/models.yaml \
    --family qwen --device cuda:0

python v2/primary/run_rq1_rq2_cross_decoding.py --dry-run --family qwen
```

`v2/scripts/` と `v2/scripts/legacy/` は主解析に使わない。
