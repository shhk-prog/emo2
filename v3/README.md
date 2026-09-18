# V3 Stage: 時空間経路と mediated attenuation

V3 は、刺激提示で作られた内部情動方向が、**どの層・どの生成段階を経て自己報告分布を動かすか**を測る。主指標は Pearl 流の NDE/NIE ではなく **mediated attenuation**（媒介減衰）である。

測定対象は介入前後の自己報告 VA 変位と、統制課題への非特異的摂動である。「内部に感情状態が宿る」「媒介因果が証明された」とは書かない。

---

## 1. 位置づけ

```text
Behavioral  →  V1  →  V2  →  V3
```

- V1: 同一モデル内の表現と因果サイト
- V2: Base ↔ Instruct の再編
- V3: Instruct（主に target family）で、状態誘導 → 時空間 map → 部分空間遮断による減衰 → 他 family での確認

Target family の既定は `configs/v3_experiments.yaml` の `target_family: qwen`。ID は `configs/models.yaml` から解決する。registry に無い family は Qwen 文字列へ落とさず `KeyError` にする。

---

## 2. 正本と legacy

正本は `v3/primary/` である。`v3/scripts/legacy/` は旧稿・探索であり、主解析に使わない。気分一致実験や旧 NDE/NIE スクリプトはここに属する。

```text
v3/
├── README.md
├── primary/
│   ├── README.md
│   ├── run_rq1_state_induction.py
│   ├── run_rq2_spatiotemporal_maps.py
│   ├── run_rq3_path_mediation.py
│   └── run_confirmatory_replication.py
├── docs/                    # 現行導線。legacy/ は旧ドラフト
├── scripts/legacy/
└── results/
    ├── raw/
    └── derived/
```

---

## 3. RQ1: State Induction と Go/No-Go

正本: `v3/primary/run_rq1_state_induction.py`

**問い**: 外部ラベルから推定した方向 $d_V, d_A$ を入れたとき、自己報告は単調かつ特異的に動くか。動かなければ RQ2/RQ3 の主解釈を進めるゲートを通さない。

### 3.1 手順

1. pair_id Group split。train で人間 `reader_V`, `reader_A` に対する方向を推定。
2. 中立平均 $\mu_{\mathrm{neu}}$ を matched-neutral から取る。固定 $5.0$ は使わない。
3. QR で 2D 情動部分空間 $Q$ を作る。
4. test で次を測る。
   - $\alpha$ sweep（用量反応）
   - centered projection removal: $h' = h - QQ^\top(h-\mu_{\mathrm{neu}})$
   - random control
   - orthogonal / perpendicular control
   - Topic control

層は `--layer` が無ければ相対深度 $d=0.5$ から $l=\operatorname{round}(d(L-1))$ で決める（Qwen の 14 層固定ではない）。

### 3.2 ゲート（Bootstrap 95% CI）

`configs/v3_experiments.yaml` の `gate_criteria`:

| 判定 | 内容 |
|---|---|
| Sufficiency / dose-response | slope の CI 下限が正 |
| Specificity | matched 効果が random / orthogonal を上回る |
| Necessity | 射影除去による attenuation の CI 下限が閾値超 |
| Topic control | Topic 課題の TVD 上限が `max_topic_tvd` 未満 |

### 3.3 Topic control の位置づけ

Self は正規化 VA shift、Topic は分類確率の TVD である。どちらも $[0,1]$ だが **同じ構成概念ではない**。

「Self effect − Control effect」を主効果量にしない。Topic は、**非特異的な課題崩壊が小さいこと**を見る統制である。差は補助記録に留める。

---

## 4. RQ2: Spatiotemporal 4-Maps

正本: `v3/primary/run_rq2_spatiotemporal_maps.py`

**問い**: デコード可能性と因果応答は、層 × 意味段階の格子上でどこにピークを持つか。

Primary は **4-Map × 2軸（V, A）** を同じ格子で出す。旧称のまま 4-Map を使う。本文のピーク解離は主に $D$ と $C$ を読む。

| Map | 記号 | 定義 |
|---|---|---|
| Decodability | $D_V, D_A$ | held-out Ridge $R^2$ |
| Partial association | $\beta_V, \beta_A$ | 刺激共変量を統制した internal score → report の偏回帰 |
| Interventional slope | $\gamma_V, \gamma_A$ | $\alpha$ sweep による因果応答の傾き |
| Causal displacement | $C_V, C_A$ | 介入による自己報告分布の変位 |

出力キーは `D_V`, `D_A`, `beta_V`, `beta_A`, `gamma_V`, `gamma_A`, `C_V`, `C_A`。$D$ と $C$ だけを走らせる縮小版ではない。

意味段階（JSON 自己報告）:

1. `response_start`（内部では `candidate_start` に正規化）
2. `pre_V`
3. `V_value`
4. `pre_A`
5. `A_value`
6. `response_end`

$C(l,t)$ と $\gamma(l,t)$ は実介入の $\alpha$ sweep から測る。プローブ係数で代用しない。

記述上の見込み（仮説であり結果ではない）: 刺激提示時の $D$ は中間層、生成時の $C$ は後期の pre-value トークンに寄る。これを時空間ピーク解離と呼ぶ。

本番は全層探索。`--subsample` は確認用。

---

## 5. RQ3: Mediated attenuation

正本: `v3/primary/run_rq3_path_mediation.py`

**問い**: 刺激提示時にできた情動部分空間を中間層で遮断すると、自己報告シフトはどれだけ残るか。

使わない用語: Natural Direct Effect (NDE), Natural Indirect Effect (NIE)。

使う量:

| 名前 | 定義の要点 |
|---|---|
| Total affective shift | matched-neutral 基準の $|E[V]_{\mathrm{aff}}-E[V]_{\mathrm{neu}}|$（$A$ も同様） |
| Residual shift after blocking | 部分空間除去後の同じ量 |
| Mediated attenuation | Total − Residual（または比） |
| Attenuation ratio | 減衰の割合。Bootstrap CI |

手順:

1. pair_id で Discovery 50% / Confirmation 50%
2. Discovery で mediator 層 $l_{\mathrm{med}}^*$ を実介入プロファイルから選ぶ
3. Confirmation でその層だけを固定して遮断する
4. matched-neutral baseline 必須
5. 探索で見た split を確認に再利用しない

「完全な因果媒介が証明された」ではなく、「遮断後に自己報告変位が減衰した」と書く。

---

## 6. Confirmatory replication

正本: `v3/primary/run_confirmatory_replication.py`

Target（Qwen）で立てた 4 仮説を、Llama 3.2 / Gemma 3 / OLMo 2 の Instruct で追試する。family が registry に無ければ落とす。

仮説の骨格:

1. Dissociation: $d_C$ が $d_D$ より深い
2. Sufficiency: 介入 slope が正
3. Necessity: attenuation が閾値を超える
4. Temporal emergence: `pre_V` 付近の因果が `response_start` より大きい

確認側でも GroupKFold / pair split を保つ。Discovery の数字を確認に再利用しない。

---

## 7. データと設定

| 項目 | 値 |
|---|---|
| データ | `v1/data/processed/stimuli_vad_3way_test1k.csv`（実件数はログ） |
| 実験設定 | `configs/v3_experiments.yaml` |
| モデル | `configs/models.yaml` |
| $\alpha$ grid（RQ1） | $-1.0,-0.5,0.0,0.5,1.0$ |
| $\alpha$ sweep（RQ2） | $-2.0$ から $2.0$ |
| Confirmatory families | `llama`, `gemma`, `olmo` |

---

## 8. 実行方法

所要時間は未計測。V3 は探索が重いので、一括 `all` より単独 stage を推奨する。

```bash
bash scripts/run_production_v3.sh cuda:0
```

統合 CLI:

```bash
python -m affective_empathy_eval.run --stage v3 --model-set primary_small --device cuda:0
```

個別:

```bash
python v3/primary/run_rq1_state_induction.py \
    --config configs/v3_experiments.yaml \
    --models-config configs/models.yaml \
    --family qwen --device cuda:0

python v3/primary/run_rq2_spatiotemporal_maps.py \
    --config configs/v3_experiments.yaml \
    --models-config configs/models.yaml \
    --family qwen --device cuda:0

python v3/primary/run_rq3_path_mediation.py \
    --config configs/v3_experiments.yaml \
    --models-config configs/models.yaml \
    --family qwen --device cuda:0

python v3/primary/run_confirmatory_replication.py \
    --config configs/v3_experiments.yaml \
    --models-config configs/models.yaml \
    --device cuda:0
```

確認:

```bash
python v3/primary/run_rq1_state_induction.py --dry-run --family qwen
```

`--pilot` は 50 行スモーク。`--subsample` は RQ2/RQ3/Confirmatory の確認用。本番では 0（全件）。

---

## 9. 出力

| ファイル | 内容 |
|---|---|
| `v3/results/raw/v3_rq1_results.json` | 用量反応、specificity、attenuation、Topic TVD、ゲート |
| `v3/results/derived/v3_gate_decision.json` | GO / NO_GO |
| `v3/results/raw/v3_spatiotemporal_maps_{family}.json` | $D,C$ の層×段階格子 |
| `v3/results/raw/v3_path_mediation_{family}.json` | Discovery 層、Confirmation の attenuation |
| `v3/results/raw/v3_confirmatory_{family}.json` | 4 仮説の合否 |
| `v3/results/derived/v3_*_summary.json` | 横断要約 |

---

## 10. 解釈

- ゲート GO は「主観が確認された」ではなく「誘導が特異的で、Topic 崩壊が小さい」
- 時空間解離は記述であり、二過程心理理論の証明ではない
- mediated attenuation は遮断後の減衰であり、NDE/NIE ではない
- Topic TVD と Self shift を引き算して主効果にしない
- Confirmatory 不成立を「Qwen だけが感情を持つ」と読まない
- `v3/docs/legacy/` の neutralization / greedy collapse ストーリーは主筋ではない
