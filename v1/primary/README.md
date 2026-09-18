# V1 Primary

V1 の正式実行面。設計・指標・解釈の本文は親の [`v1/README.md`](../README.md) を正本とする。

比較軸は同一モデル内の **Reader ↔ Self**。Base / Instruct 8 条件は各モデル内の再現であり、差の解釈は V2。候補空間は 729 VAD。

## 実行順

統合 CLI / `run_production_v1.sh` は次の順である。

1. Phase B 統制 CSV が無ければ `prepare_v1_phase_b_controls.py`
2. 各 family × Base/Instruct について Phase A → B → C → E6
3. 最後に `summarize_phase_c.py`

`--model-id` または `--family` が必須。Qwen への暗黙 default は禁止。層は指定が無ければ $d=0.5$ から $l=\operatorname{round}(d(L-1))$。

## スクリプト

| ファイル | 内容 | 既定データ |
|---|---|---|
| `run_phase_a.py` | E1 decodability、E2 geometry | EmoBank test1k と AIPsy |
| `prepare_v1_phase_b_controls.py` | rule-based 統制文の生成 | 出力: `v1_e5_semantic_controls.csv` |
| `run_phase_b.py` | 語彙監査と統制後 decodability | Phase B CSV。層は a priori $d=0.5$ |
| `run_phase_c.py` | E3 / E4（内部で `phase_c/` を呼ぶ） | AIPsy。Discovery / Confirmation 50:50 |
| `phase_c/run_e6_specialization.py` | タスク選択性サイト + Confirmation LMM | E3 Discovery CSV 必須 |
| `phase_c/summarize_phase_c.py` | 8 条件の横断要約 | `v1/results/derived/` |

E6: $S_R(l)=C_R(l)-C_S(l)$。同一層または選択性が正でなければ No-Go。heuristic 層（$0.5(L-1)$ など）へ落とさない。

## 実行

```bash
bash scripts/run_production_v1.sh cuda:0

python -m affective_empathy_eval.run --stage v1 --model-set primary_small --device cuda:0
python -m affective_empathy_eval.run --stage v1 --model-set primary_small --family qwen --device cuda:0

python v1/primary/run_phase_a.py --family qwen --is-instruct --dataset both --device cuda:0
python v1/primary/run_phase_b.py --family qwen --is-instruct --relative-depth 0.5 --device cuda:0
python v1/primary/run_phase_c.py --family qwen --is-instruct --device cuda:0
python v1/primary/phase_c/run_e6_specialization.py --family qwen --is-instruct --device cuda:0
```

`--limit` / `--dry-run` は確認用。`v1/scripts/legacy/` は主解析に使わない。
