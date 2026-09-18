# V1 Primary
## 論文対応: Section 4. Shared Representation and Causal Overlap in Base Models

V1 の正式実行面。設計・指標・解釈の本文は親の [`v1/README.md`](../README.md) を正本とする。

- **科学的問い (RQ)**: *What affect-relevant representations and causal mechanisms are shared or overlap between Reader and Self perspectives in base models?*（基底モデルにおいて、読者認識 (Reader) と自己報告 (Self) の背後に共通の情動内部表現および部分的に重複した因果機構・介入感受性部位が存在するか？）
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
