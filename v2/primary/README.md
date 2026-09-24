# V2 Primary
## 論文対応: §6 Post-training-Associated Reorganization of Affect-Relevant Computations

V2 の正式実行面。設計・指標・解釈の本文は親の [`v2/README.md`](../README.md) を正本とする。

> **重要な解釈規約 (Non-Causal Interpretation of Post-Training)**:  
> 本ステージにおける Base と Instruct の比較は、モデルファミリー間の事前学習後アーティファクト比較（observational comparison across model artifacts）であり、直接的な訓練介入実験ではありません。したがって、post-training の影響を「因果効果 (causal effect)」と過大解釈・表現してはならず、**「事後学習に伴う再編 (post-training-associated reorganization)」** として記述します。

- **科学的問い (RQ)**: *How does post-training associate with the reorganization of affect-relevant representations, reader-self geometry, and causal readouts?*（事後学習に伴い、情動関連表現の幾何、Reader/Self 間の共有構造、および自己報告への因果的読み出し機構はどのように再編されるか？）
- **Primary Metric**:
  - **RQ1/RQ2 Cross-Decoding & Representation Shift**: Base / Instruct における decodability ピーク層シフト ($\Delta l^*, \Delta d^*$)、Native テンプレートと Matched-Plain プロンプト間での Procrustes 幾何不一致度。
  - **RQ3 Causal Map Dissociation**: 因果的寄与ピーク層 $d_C$ と表現 decodability ピーク層 $d_D$ の解離（$\Delta d = d_C - d_D$）。
  - **RQ4 Distributional Recovery**: Instruct 表現を Base 表現へ差し戻した際の出力分布回復率（Wasserstein-1 距離 $W_1$ に基づく Recovery 指標）。直接パッチ（Condition A）と直交 Procrustes アラインメントパッチ（Condition B）の比較。
- **統計単位**: Stimulus (EmoBank test1k) / Model family
- **統制条件**:
  - Native chat template vs matched plain prompt（プロンプト形式交絡の統制）
  - Aligned Procrustes transformation control in activation patching（表現空間アラインメント統制）
  - 線形混合効果モデル (LMM) によるファミリー横断効果の頑健性検証
- **出力成果物**: `primary_small` は `v2/results/raw/` と `v2/results/derived/`。それ以外の `--model-set` は `results/ablation/{model_set}/`。ファイル名は `v2_geometry_{family}.json`, `v2_causal_map_{family}.json`, `v2_recovery_{family}.json`, 横断 summary と LMM JSON

比較軸は同一ファミリーの **Base ↔ Instruct**。Reader ↔ Self の解釈は V1、状態誘導の時空間は V3。候補空間は 81 VA。データ既定は EmoBank test1k（`configs/v2_experiments.yaml`）。

## 実行順

統合 CLI / `run_production_v2.sh` は次の順である。

1. `run_rq1_rq2_cross_decoding.py`
2. `run_rq3_causal_map.py`
3. `run_rq4_recovery_patching.py`
4. `--family` / `--base-model` / `--instruct-model` が無いときだけ `run_confirmatory_analysis.py`

1 family 実行では 4 を省略する。`--stage v2 --model-set scale_validation|scale_3b|scale_7b` は同じ 4 段で、成果物だけ `results/ablation/` に分かれる。`--stage scale_validation` は `scripts/run_scale_validation.py` 経由で RQ4 までであり、confirmatory も `--force` も `--model-set` の上書きも転送しない。`--force` が各 RQ と confirmatory に届くのは `--stage v2` だけである。設定正本は `configs/v2_experiments.yaml`（`normalize_length: true`）。

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
bash scripts/run_production_v2.sh cuda:0 --force

python -m affective_empathy_eval.run --stage v2 --model-set primary_small --device cuda:0
python -m affective_empathy_eval.run --stage v2 --model-set primary_small --family qwen --device cuda:0
python -m affective_empathy_eval.run --stage scale_validation --device cuda:0

python v2/primary/run_rq1_rq2_cross_decoding.py \
    --config configs/v2_experiments.yaml \
    --models-config configs/models.yaml \
    --family qwen --device cuda:0

python v2/primary/run_rq1_rq2_cross_decoding.py --dry-run --family qwen
```

`v2/scripts/legacy/` と、`v2/scripts/` 直下の旧 plot / extract は主解析に使わない。`v2/scripts/build_paper_summary.py` だけは derived から論文 19 列を作る presentation である。
