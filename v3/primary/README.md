# V3 Primary

V3 の正式実行面。設計・指標・解釈の本文は親の [`v3/README.md`](../README.md) を正本とする。

中心の問い: **When/where does information acquire causal leverage?**  
データ: AIPsy clinical–neutral 192 pair（`load_v3_matched_pair_table`）。EmoBank 3-way は使わない。  
候補空間: 81 VA。

## 実行順とゲート

統合 CLI / `run_production_v3.sh` は次の順である。

1. `run_rq1_state_induction.py` → `v3/results/derived/v3_gate_decision.json`
2. `decision == "GO"` のときだけ RQ2 → RQ3 → Confirmatory
3. `NO_GO` / 軸片方の GO は終了コード 2
4. `--force-after-no-go` のときだけ 2 を強制する

単独で RQ2 を呼ぶとゲートは見ない。本番経路では見ないといけない。

## スクリプト

| ファイル | 問い | 実装上の固定点 |
|---|---|---|
| `run_rq1_state_induction.py` | 方向注入は特異的に自己報告を動かすか | pair Group split。Primary 方向は人間 reader またはモデル Reader Prediction。$d_V$ / $d_A$ 別 sweep。matched-neutral 必須。層は $d=0.5$ |
| `run_rq2_spatiotemporal_maps.py` | $D,\beta,\gamma,C$ のピークはどこか | joint sequence patch。`n_map_samples` と `n_intervene_samples`（既定 15）を分離。`analysis_role: discovery`。符号付き $\beta$ と `abs_beta_*` |
| `run_rq3_path_mediation.py` | 部分空間遮断で変位は減衰するか | Discovery / Confirmation 50:50。NDE/NIE とは呼ばない |
| `run_confirmatory_replication.py` | 他 family でも同じか | Llama / Gemma 3 / OLMo 2。Sufficiency も V/A 別 sweep |

未知 family は `KeyError`。`--pilot` は RQ1 の 50 行。`--subsample` は RQ2/RQ3/Confirmatory の確認用。

## 実行

```bash
bash scripts/run_production_v3.sh cuda:0
# bash scripts/run_production_v3.sh cuda:0 --force-after-no-go

python -m affective_empathy_eval.run --stage v3 --model-set primary_small --device cuda:0

python v3/primary/run_rq1_state_induction.py \
    --config configs/v3_experiments.yaml \
    --models-config configs/models.yaml \
    --family qwen --device cuda:0

python v3/primary/run_rq1_state_induction.py --dry-run --family qwen
```

`v3/scripts/legacy/` は主解析に使わない。
