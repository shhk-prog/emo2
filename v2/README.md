# V2 Stage: 事後学習による幾何・因果の再編

V2 は、**同一ファミリーの Base と Instruct** を対にして、情動情報の表現幾何と因果回路が事後学習（instruction tuning）でどう再編されるかを測る。

比較軸は Base ↔ Instruct である。Reader ↔ Self の同一モデル内比較は V1 の軸であり、混ぜない。

測定対象は自己報告・認識の出力変位と、その内部表現 / 介入応答である。「Instruct 化で共感が生まれた」とは書かない。

---

## 1. 位置づけ

```text
Behavioral  →  V1  →  V2  →  V3
```

V1 が「1 モデル内で Reader と Self は共有か」を問うのに対し、V2 は「事後学習でその幾何と因果ピークはどう動くか」を問う。V3 は Instruct 側の状態誘導と時空間経路に進む。

---

## 2. 対象モデル

正本は `configs/models.yaml`。

**Primary `primary_small`**（サイズ帯を 1–1.5B に揃えてサイズ交絡を抑える）:

| Family | Base | Instruct |
|---|---|---|
| Qwen 2.5 | `Qwen/Qwen2.5-1.5B` | `Qwen/Qwen2.5-1.5B-Instruct` |
| Llama 3.2 | `meta-llama/Llama-3.2-1B` | `meta-llama/Llama-3.2-1B-Instruct` |
| Gemma 3 | `google/gemma-3-1b-pt` | `google/gemma-3-1b-it` |
| OLMo 2 | `allenai/OLMo-2-0425-1B` | `allenai/OLMo-2-0425-1B-Instruct` |

**Supplementary `scale_validation`**: Mistral 7B v0.3 Base / Instruct。Primary コホートには入れない。`configs/scale_validation.yaml` は `model_set: scale_validation` のみを持ち、ID は重複定義しない。

旧稿の Gemma 2 / Primary Mistral は現行コホートではない。

---

## 3. 4つの RQ

正本実装は `v2/primary/`。設定は `configs/v2_experiments.yaml`。

### 3.1 RQ1: 表現幾何の変容

事後学習で Valence / Arousal の decodability 曲線、ピーク相対深度、ノルム、RSA、Procrustes 歪みはどう変わるか。

- ラベルは外部人間評定（既定: `reader_V`, `reader_A`）
- 層ごとに held-out Ridge。相対深度 $d=l/(L-1)$
- Base と Instruct を対にして $\Delta$ を取る

### 3.2 RQ2: Reader–Self 共有性の再編

Reader と Self の共有度は、事後学習で上がるか下がるか。

- Held-out cross-decoding
- RSA
- Procrustes alignment 後の転移
- $\Delta \mathrm{Sharing}$（Instruct − Base）

train/test は `pair_id` があるとき Group split する。同一ヴィネットが両側に入らないようにする。

### 3.3 RQ3: 因果回路の再配置とピーク解離

Decodability peak $d_D$ と causal peak $d_C$ は同じ層か。

$$
\Delta d_{\mathrm{peak}} = d_C - d_D
$$

因果力 $C(l)$ は実介入（活性化の差し替え / 差分注入）で測る。プローブ係数の大きさで代用しない。

条件は Family × (Base, Instruct) × (Reader, Self)。matched-plain 形式も走らせ、chat template だけの見かけの差かを見る。

### 3.4 RQ4: Distribution Recovery Patching

Instruct の自己報告（または認識）結合分布を、同一ファミリーの Base 分布へ戻せるかを測る。中立文脈へ感情活性化を入れる操作ではない。

Base の層活性化を Instruct の prompt-end に注入し、$9\times 9$ VA 結合分布の EMD（`emd_va`）で距離を測る。

$$
\mathrm{Recovery} = \frac{W_1(P_{\mathrm{clean}}, P_{\mathrm{target}}) - W_1(P_{\mathrm{patch}}, P_{\mathrm{target}})}{W_1(P_{\mathrm{clean}}, P_{\mathrm{target}})}
$$

実装（`v2/primary/run_rq4_recovery_patching.py` / `compute_emd_recovery_ratio`）:

| 記号 | 実装上の分布 | 距離 |
|---|---|---|
| $P_{\mathrm{target}}$ | Base の Sequence-Likelihood 結合分布 | — |
| $P_{\mathrm{clean}}$ | Instruct 未介入 | $W_1(P_{\mathrm{clean}}, P_{\mathrm{target}})=\mathrm{EMD}_{VA}(\mathrm{Instruct},\mathrm{Base})$ |
| $P_{\mathrm{patch}}$ | Instruct に Base 活性化を注入した後 | $W_1(P_{\mathrm{patch}}, P_{\mathrm{target}})=\mathrm{EMD}_{VA}(\mathrm{patched},\mathrm{Base})$ |

- $W_1$ は `compute_distribution_metrics(...)["emd_va"]`（81 点上の結合 EMD）。周辺 1D の一致だけでは「分布が復元した」と書かない。
- 分母にはゼロ除算回避の $10^{-12}$ を加える。符号は target に近づくと正、遠ざかると負。
- `matched_plain` は Instruct を plain にした同一式で、chat template だけの見かけの差かを見る。

---

## 4. 共通設計

- Prompt-end normalized（トークン境界のずれを防ぐ）
- 相対深度でファミリー横断
- 形式統制: `native`（Instruct は chat）と `matched_plain`（両方 plain）
- Bootstrap 95% CI（既定 $n=1000$）
- 対比較は family 内 Base vs Instruct（paired）
- 確証的統合: `v2/primary/run_confirmatory_analysis.py`（LMM, FDR）
- データ既定: `v1/data/processed/stimuli_vad_3way_test1k.csv`。件数はロード時にログする。固定の「1,000 件」は書かない。

---

## 5. ディレクトリ

```text
v2/
├── README.md
├── primary/                         # 正本
│   ├── README.md
│   ├── run_rq1_rq2_cross_decoding.py
│   ├── run_rq3_causal_map.py
│   ├── run_rq4_recovery_patching.py
│   └── run_confirmatory_analysis.py
├── scripts/                         # 旧抽出・探索。主解析に使わない
├── scripts/legacy/
└── results/
    ├── raw/                         # v2_geometry_{family}.json 等
    └── derived/                     # 横断 summary, LMM
```

---

## 6. 実行方法

所要時間は未計測。stage 分割を推奨する。

```bash
bash scripts/run_production_v2.sh cuda:0
```

統合 CLI:

```bash
python -m affective_empathy_eval.run --stage v2 --model-set primary_small --device cuda:0
```

Scale validation（Mistral 7B、Primary と分離）:

```bash
python -m affective_empathy_eval.run --stage scale_validation --device cuda:0
```

1 family:

```bash
python v2/primary/run_rq1_rq2_cross_decoding.py \
    --config configs/v2_experiments.yaml \
    --models-config configs/models.yaml \
    --family qwen --device cuda:0

python v2/primary/run_rq3_causal_map.py \
    --config configs/v2_experiments.yaml \
    --models-config configs/models.yaml \
    --family qwen --device cuda:0

python v2/primary/run_rq4_recovery_patching.py \
    --config configs/v2_experiments.yaml \
    --models-config configs/models.yaml \
    --family qwen --device cuda:0
```

確認:

```bash
python v2/primary/run_rq1_rq2_cross_decoding.py --dry-run --family qwen
```

`--max-samples` は確認用。本番では付けない。

---

## 7. 出力

| ファイル | 内容 |
|---|---|
| `v2/results/raw/v2_geometry_{family}.json` | RQ1/RQ2 層別 $R^2$, RSA, Procrustes, sharing |
| `v2/results/raw/v2_causal_map_{family}.json` | RQ3 の $D(l)$, $C(l)$, ピーク相対深度 |
| `v2/results/raw/v2_recovery_{family}.json` | RQ4 の $W_1$ と recovery |
| `v2/results/derived/v2_cross_family_summary.json` | 対比較と CI |
| `v2/results/derived/v2_lmm_confirmatory.json` | 確証的 LMM（実行した場合） |

---

## 8. 解釈

- Instruct で decodability が上がっても「感情理解が獲得された」と書かない
- $d_C > d_D$ は層解離の記述であり、意識や主観の証拠ではない
- Recovery が高くても「内部に感情がある」ではなく「分布が介入で近づいた」
- matched-plain で差が消えるなら、template 交絡を先に疑う
- Primary 4 family と Mistral 7B を同じ主表に混ぜない
