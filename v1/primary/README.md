# V1 Primary
## 論文対応: §5 Internal Representation and Causal Sharing

V1 の正式実行面。設計・指標・解釈の本文は親の [`v1/README.md`](../README.md) を正本とする。

- **科学的問い (RQ)**: *To what extent do Reader and Self share representational and causal structure internally?*（同一モデル内で、Reader と Self は表現と因果サイトをどこまで共有するか）
- **Primary Metric**:
  - **E1 Shared Decodability**: 外部刺激ラベル（EmoBank 人間評定 / AIPsy 条件）に対する層別線形プローブ性能（GroupKFold による held-out stimulus で評価、$R^2$, Pearson $r$, ROC-AUC）。ピーク相対深度 $d^*_R, d^*_S$。
  - **E2 Shared Geometry**: Direct cross-decoding transfer performance および Procrustes 変換幾何類似度。
  - **Phase B Semantic Transformation Sensitivity**: 制御された意味・統語摂動（否定、中立化、強調等）に対する表現感度（held-out stimulus pairs での decodability 低下度。pair_id リーク完全排除）。
  - **E3/E4 Causal Overlap & Interchangeability**: Activation patching による介入効果 $C(l)$ および相互置換時の出力回復率。
  - **E6 Task-Specific Specialization**: 特異化指標 $S_R(l) = C_R(l) - C_S(l)$。
- **統計単位**:
  - E1 / E2: Stimulus split（GroupKFold）
  - Phase B: Matched pair split（Pair-Aware held-out test）
  - E3 / E4 / E6: Matched pair（Discovery / Confirmation 50:50 独立分割）
- **統制条件**:
  - Activation shuffling / derangement controls
  - Rule-based semantic controlled perturbations
- **出力成果物**: `v1/results/derived/`

統合 CLI は `primary_small` の Base と Instruct を両方走らせる。比較軸は各モデル内の Reader ↔ Self である。Base↔Instruct の差そのものの解釈は V2 に置く。候補空間は 729 VAD。

## 実行順

統合 CLI / `run_production_v1.sh` は次の順である。

1. Phase B 統制 CSV が無ければ `prepare_v1_phase_b_controls.py`
2. 各 family × Base/Instruct について Phase A → Phase B `--task-type reader` → Phase B `--task-type self` → Phase C → E6
3. 最後に `summarize_phase_c.py`

`run_production_v1.sh` は常に `--all-layers` を付ける。統合 CLI は `--all-layers` を明示したときだけ Phase C 全層になる。`--force` は各 Phase のキャッシュを無視する。`--model-revision` は registry の pinned SHA。

`--model-id` または `--family` が必須。Qwen への暗黙 default は禁止。層は指定が無ければ $d=0.5$ から $l=\operatorname{round}(d(L-1))$。設定正本は `configs/v1_experiments.yaml`（`normalize_length: true`）。

## スクリプト

| ファイル | 内容 | 既定データ |
|---|---|---|
| `run_phase_a.py` | E1 decodability、E2 geometry | EmoBank test1k と AIPsy |
| `prepare_v1_phase_b_controls.py` | rule-based 統制文の生成 | 出力: `v1_e5_semantic_controls.csv` |
| `run_phase_b.py` | 語彙監査と統制後 decodability | Phase B CSV。層は a priori $d=0.5$。`--task-type {reader,self}`。出力は `v1_phase_b/{task_type}/{prefix}/` |
| `run_phase_c.py` | E3 / E4（内部で `phase_c/` を呼ぶ） | AIPsy。Discovery / Confirmation 50:50 |
| `phase_c/run_e6_specialization.py` | タスク選択性サイト + Confirmation LMM | E3 Discovery CSV 必須 |
| `phase_c/summarize_phase_c.py` | 8 条件の横断要約 | `v1/results/derived/` |

E6: $S_R(l)=C_R(l)-C_S(l)$。同一層または選択性が正でなければ No-Go。heuristic 層（$0.5(L-1)$ など）へ落とさない。

## 実行

```bash
bash scripts/run_production_v1.sh cuda:0

python -m affective_empathy_eval.run --stage v1 --model-set primary_small --device cuda:0 --all-layers
python -m affective_empathy_eval.run --stage v1 --model-set primary_small --family qwen --device cuda:0 --all-layers --force

python v1/primary/run_phase_a.py --family qwen --is-instruct --dataset both --device cuda:0
python v1/primary/run_phase_b.py --family qwen --is-instruct --task-type reader --relative-depth 0.5 --device cuda:0
python v1/primary/run_phase_b.py --family qwen --is-instruct --task-type self --relative-depth 0.5 --device cuda:0
python v1/primary/run_phase_c.py --family qwen --is-instruct --all-layers --device cuda:0
python v1/primary/phase_c/run_e6_specialization.py --family qwen --is-instruct --device cuda:0
```

`--limit` / `--dry-run` は確認用。`v1/scripts/legacy/` は主解析に使わない。
