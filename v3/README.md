# V3 Stage: 時空間経路と mediated attenuation
## 論文対応: §7 From Representation to Causal Utilization

V3 は、内部情動表現が**どの層・どの生成段階で自己報告分布に対して因果的影響力（causal leverage）を行使するか**を測る。中心の問いは **Where does affect-relevant information exert measurable causal leverage over self-report?** である。主指標は Pearl 流の NDE/NIE ではなく **mediated attenuation**（媒介減衰）である。

本ステージでは、介入の役割を概念的・数理的に明確化する：
- **Direction Injection**: 中立文への方向加算注入による因果的影響力・十分性（*sufficiency / causal leverage*）
  $$h' = h + \alpha \sigma_h \hat{d}$$
- **Subspace Removal**: 情動文からの内部部分空間除去による内生的関連性（*endogenous relevance*）
  $$h' = h - Q Q^\top (h - \mu_{\text{neu}})$$

自己報告に対する測定可能な因果的影響力はモデル全体に均一に分布するのではなく、**特定の層、および teacher-forced candidate sequence 上の特定の計算段階に集中する（causal leverage is concentrated at particular layers and stages along the teacher-forced candidate sequence）**という仮説を時空間マップにより検証する。

**方向注入実験**では $h' = h + \alpha \sigma_h \hat{d}$ の加算介入を用いる。一方、**内生的関連性の検証**では $h' = h - Q Q^\top (h - \mu_{\text{neu}})$ の中心化直交部分空間除去を用いる。両者はそれぞれ十分性と内生的関連性を検証する異なる介入であり、同一の操作として扱わない。Confirmatory 再現性評価はサンプル単位の完全独立分割（Sample-level holdout cross-fitting）によりプローブ推定と介入評価のデータ重複リークを排除して実施される。

測定対象は介入前後の自己報告 VA 変位と、統制課題への非特異的摂動である。「内部に感情状態が宿る」「媒介因果が証明された」とは書かない。Self は操作的な自己報告課題であり、情動的共感と同定しない。

Sequence-Likelihood は **81 VA** 候補である。Behavioral / V1 の 729 VAD 空間とは混ぜない。Stage 間では絶対値を直接比較せず、各 Stage 内の contrast と relative pattern を主たる推論対象とする。

---

## 1. 位置づけ

```text
Behavioral  →  V1  →  V2  →  V3
```

- V1: 同一モデル内の表現と因果サイト
- V2: Base ↔ Instruct の再編
- V3: Instruct（主に target family）で、状態誘導 → 時空間 map → 部分空間遮断による減衰 → 他 family での確認

Target family の既定は `configs/v3_experiments.yaml` の `target_family: qwen`（Instruct）。ID・pinned revision・`inference_dtype: bfloat16` は `configs/models.yaml` から解決する。registry に無い family は Qwen 文字列へ落とさず `KeyError` にする。

| 役割 | Family | スクリプト |
|---|---|---|
| Discovery / ゲート | Qwen 2.5 1.5B Instruct | RQ1 → RQ2 → RQ3 |
| Confirmation | Llama 3.2 / Gemma 3 / OLMo 2 Instruct | `run_confirmatory_replication.py` |

Base は V3 Primary に入れない。Base↔Instruct 差は V2 の軸である。

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
    ├── raw/                 # dry-run 時は raw/dry_run/
    └── derived/             # 本番ゲート: derived/v3_gate_decision.json
                             # dry-run ゲート: derived/dry_run/v3_gate_decision.json
```

---

## 3. RQ1: State Induction と Go/No-Go

正本: `v3/primary/run_rq1_state_induction.py`

**問い**: 推定した方向 $d_V, d_A$ を入れたとき、自己報告は単調かつ特異的に動くか。動かなければ RQ2/RQ3 の主解釈を進めるゲートを通さない。

### 3.1 手順

1. pair_id Group split。方向は人間 VA があればそれを使い、AIPsy 既定では同じ刺激に対するモデルの感情認識予測値（Reader Prediction）への回帰 $H_{\mathrm{Self}} \rightarrow (V_R, A_R)$ から Primary 情動方向 $d_V^R, d_A^R$ を推定する。モデル自己報告自身から同定する方向は Secondary analysis として保持し、方向アライメント $\cos(d^R, d^S)$ を記録する。
2. 中立平均 $\mu_{\mathrm{neu}}$ と necessity baseline は、同一 `pair_id` の matched-neutral 文を通した自己報告から取る。固定 $5.0$ も人工中立文も使わない。
3. rank-aware SVD により有効 rank を判定し、直交情動部分空間 $Q$ を構成する。
4. test で次を測る。
   - $\alpha$ sweep は軸を分ける。$d_V$ 注入 → Valence 用量反応、$d_A$ 注入 → Arousal 用量反応
   - centered projection removal: $h' = h - QQ^\top(h-\mu_{\text{neu}})$
   - random control: `num_random_controls: 5` 設定に基づき各軸 $K=5$ 本のランダム方向を評価し、平均効果 `mean_eff_rand_v`, `mean_eff_rand_a` および matched − mean(random) 差分を記録
   - orthogonal / perpendicular control: 同様に各軸 $K=5$ 本の直交方向を評価
   - Topic control: 非特異的課題崩壊の検証

層は `--layer` が無ければ相対深度 $d=0.5$ から $l=\operatorname{round}(d(L-1))$ で決める（Qwen の 14 層固定ではない）。

### 3.2 ゲート（Bootstrap 95% CI）

`configs/v3_experiments.yaml` の `gate_criteria`:

| 判定 | 内容 |
|---|---|
| Sufficiency / dose-response | 各軸の slope CI 下限が `min_sufficiency_slope`（現行 `0.1`）超（$d_V$ と $d_A$ を別 sweep） |
| Specificity | matched 効果が $K=5$ 本の random / orthogonal controls の平均を `min_specificity_diff`（現行 `0.05`）上回る |
| Necessity | 射影除去による attenuation の CI 下限が `min_necessity_attenuation`（現行 `0.05`）超 |
| Topic control | Topic 課題の TVD 上限が `max_topic_tvd`（現行 `0.15`）未満 |

統合 CLI / `run_production_v3.sh` は RQ1 のあと gate を読む。

- 本番: `v3/results/derived/v3_gate_decision.json`
- `--dry-run`: `v3/results/derived/dry_run/v3_gate_decision.json`

`decision` が完全一致の `GO` のときだけ RQ2 以降へ進む。`NO_GO` および `GO (Valence-only)` / `GO (Arousal-only)` では終了コード 2。継続は `--force-after-no-go` のみ。`--force` は各 RQ のキャッシュ再計算であり、ゲート継続とは別フラグである。

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
| Decodability | $D_V, D_A$ | held-out Ridge $R^2$（Primary は Reader Prediction、Secondary は Self-report） |
| Partial association | $\beta_V, \beta_A$ | 刺激共変量を統制した内部 Reader 予測スコア $\rightarrow$ **Self-report ($y_{\text{self}}$)** の偏回帰係数（pair bootstrap 95% CI 併記） |
| Interventional slope | $\gamma_V, \gamma_A$ | $\gamma = \frac{\Delta\text{Report}}{\Delta\alpha}$（1 SD 正規化介入用量あたりの自己報告変化率） |
| Causal displacement | $C_V, C_A$ | 介入による自己報告分布の変位 |

出力キーは `D_V`, `D_A`, `beta_V`, `beta_A`, `beta_V_ci`, `beta_A_ci`, `abs_beta_V`, `abs_beta_A`, `gamma_V`, `gamma_A`, `C_V`, `C_A`。$D$ と $C$ だけを走らせる縮小版ではない。

意味段階（Teacher-forced candidate sequence 上の計算段階）。`configs/v3_experiments.yaml` の `semantic_stages` は次の 6 段である。

1. `response_start`（YAML 名）
2. `pre_V`: Valence トークン生成直前
3. `V_value`: Valence トークン処理後
4. `pre_A`: Arousal トークン生成直前
5. `A_value`: Arousal トークン処理後
6. `response_end`: 最終候補 token 処理後（因果効果が原理上消失する **Negative control** として保持）

RQ2 / Confirmatory は実行キーを **`response_start` → `candidate_start` に正規化**してから patch する。`resolve_joint_stage_index` 自体は `response_start` を `cand_start - 1`（prompt_end）と解釈するが、正規化後の本番経路では第 1 段階は `candidate_start`（`cand_start + 0`、候補先頭 token）になる。YAML に `candidate_start` を別エントリとしては持たない。

$C(l,t)$ と $\gamma(l,t)$ は実介入の $\alpha$ sweep から測る。プローブ係数で代用しない。生成段階の patch は joint sequence 上の token であり、`prompt_end` に丸めない。

Valence 方向 $d_V$ の注入が $\gamma_V,C_V$、Arousal 方向 $d_A$ の注入が $\gamma_A,C_A$ である。片方の注入で両軸を同時に主張しない。

$\beta$ は符号付き偏回帰係数を Primary に残し、`abs_beta_*` および pair bootstrap 95% CI を併記する。

記述上の見込み（仮説であり結果ではない）: 刺激提示時の $D$ は中間層、生成時の $C$ は後期の pre-value トークンに寄る。これを時空間ピーク解離と呼ぶ。

RQ2 は全マップの探索的導出（Discovery）であり、出力に `analysis_role: discovery` を付ける。マップ用サンプル数 `n_map_samples` と実因果介入サンプル数 `n_intervene_samples` / `n_causal_intervention_samples`（設定 `spatiotemporal.n_causal_samples: 15` に連動、既定は $\min(15,N)$）は別記録する。なお、RQ2 の時空間マップ全体は Discovery であり、最終的な因果推論は独立した Confirmation 評価で行う。

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
| Mediated attenuation | Total − Residual（または比）。Bootstrap CI |
| Attenuation ratio | 減衰の割合。Bootstrap CI |
| Random Subspace Control | **matched-rank random 2D subspace removal ($Q_{\text{rand}}$)**。任意の 2D 破壊による非特異的変位減少と、情動部分空間の特異的減衰を分離 |

手順:

1. pair_id で Discovery 50% / Confirmation 50%
2. Discovery で mediator 層 $l_{\mathrm{med}}^*$ を実介入プロファイルから選ぶ
3. Confirmation でその層だけを固定して遮断する
4. matched-neutral baseline 必須
5. matched-rank random 2D subspace removal による統制効果との差分（Net attenuation）を算出して内生特異性を検証
6. 探索で見た split を確認に再利用しない

「完全な因果媒介が証明された」ではなく、「遮断後に自己報告変位が減衰した」と書く。

---

## 6. Confirmatory replication

正本: `v3/primary/run_confirmatory_replication.py`

Target（Qwen）で立てた 4 仮説を、Llama 3.2 / Gemma 3 / OLMo 2 の Instruct で追試する。family が registry に無ければ落とす。データ空時の架空値フォールバック（旧 1.0/0.5 等）は排除され、例外を送出する。

仮説の骨格と厳密判定基準（**全仮説について CI lower bound > preregistered threshold で判定**）:

1. **H1 Dissociation**: $d_C$ が $d_D$ より深い（$\Delta d_{\text{peak}}, \Delta d_{\text{center}}$ の bootstrap 95% CI 下限 $> 0$）
2. **H2 Sufficiency**: Valence/Arousal 方向注入の dose-response slope（$\gamma_V, \gamma_A$ の pair-bootstrap 95% CI 下限 $> \text{min\_slope}$）
3. **H3 Endogenous Relevance**: absolute mediated attenuation のbootstrap 95% CI下限が閾値を超え、かつ matched-rank random 2D subspace controlに対するNet attenuationの95% CI下限が0を超える
4. **H4 Temporal emergence**: `pre_V`/`pre_A` 付近の因果応答と `candidate_start` のコントラスト（temporal contrast の pair-bootstrap 95% CI 下限 $> \text{min\_temporal\_contrast}$）

確認側でも GroupKFold / pair split を保つ。Discovery の数字を確認に再利用しない。Sufficiency で片方の注入から両軸を同時に主張しない。

凍結パラメータは YAML の `confirmatory` ブロック（`selection_source: qwen_discovery_frozen`）。Llama / Gemma / OLMo の結果を見て再選定しない。

| キー | 現行値 |
|---|---|
| `n_intervention_samples_per_fold` | 5 |
| `sufficiency_relative_depth` | 0.5 |
| `temporal_relative_depth_v` / `_a` | 0.65 |
| `temporal_stage_v` / `_a` | `pre_V` / `pre_A` |
| `mediation_relative_depth` | 0.65 |
| `qc.min_sufficiency_slope` | 0.1 |
| `qc.min_mediated_attenuation_ci_lower` | 0.0 |
| `qc.min_temporal_contrast` | 0.0 |

---

## 7. データと設定

| 項目 | 値 |
|---|---|
| データ | `v1/data/processed/aipsy_4split_all.csv`。ローダー `load_v3_matched_pair_table` が clinical–neutral 192 pair を wide 化し、`neutral_text` を付ける |
| 除外 | EmoBank 3-way（`pair_id` なし）。AIPsy の `moderate` / `complex_neutral`（対が揃わない行） |
| 禁止 | 人工中立文 `"This is a neutral and ordinary statement."`、固定 $5.0$ fallback |
| 候補空間 | 81 VA。729 VAD の期待値と直接比較しない |
| 実験設定 | `configs/v3_experiments.yaml`。`sequence_likelihood.normalize_length: true`、`temperature: 1.0` |
| モデル | `configs/models.yaml`。target 既定 `qwen` Instruct。pinned revision + `bfloat16` |
| 層（RQ1） | `--layer` が無ければ $d=0.5$ から $l=\operatorname{round}(d(L-1))$ |
| $\alpha$ grid（RQ1） | $-1.0,-0.5,0.0,0.5,1.0$ |
| $\alpha$ sweep（RQ2） | $-2.0,-1.0,-0.5,0.0,0.5,1.0,2.0$ |
| 因果介入件数（RQ2） | `spatiotemporal.n_causal_samples: 15`（マップ件数 `N` とは別） |
| ゲート | `min_sufficiency_slope: 0.1`、specificity 差 $0.05$、necessity $0.05$、Topic TVD $0.15$、bootstrap $n=1000$ |
| Confirmatory families | `llama`, `gemma`, `olmo` |

### 7.1 方向推定（RQ1）

隠れ状態は Self プロンプトの prompt-end から取る。回帰ターゲットは次の優先順である。

1. **Primary**: 人間 `reader_V`, `reader_A` があればそれを使う。AIPsy 既定には無いので、同じ刺激に対するモデルの Reader Prediction（`TaskType.READER` の 81 VA 期待値）を使う
2. **Secondary**: モデル Self-report から同定した $d_V^S, d_A^S$。$\cos(d^R, d^S)$ を記録するだけであり、ゲートの主方向には使わない

介入は Primary 方向で行う。$d_V$ と $d_A$ は別 sweep する。

### 7.2 生成段階の位置（RQ2）

1. `prepare_joint_sequence_with_boundary` で prompt + candidate を joint tokenize する
2. YAML の `response_start` は実行キー `candidate_start` に正規化する
3. candidate 内オフセットは `get_generation_stage_tokens`（`candidate_start=0`, `pre_V`, `V_value` など。`response_start` は offset 0 のエイリアス）
4. 絶対位置は `resolve_joint_stage_index`:
   - `response_start` → `cand_start - 1`（prompt_end）
   - それ以外（正規化後の `candidate_start` を含む）→ `cand_start + offset`
5. `prompt_end` への `min` はしない。範囲外はエラー
6. 尤度計算は `generation_patch` で同じ絶対位置に hook する

現行本番経路は 2 の正規化を先に行うため、第 1 段階の patch 位置は候補先頭 token である。

### 7.3 ゲートファイル

`run.py` が読むパス:

- 本番: `v3/results/derived/v3_gate_decision.json`
- dry-run: `v3/results/derived/dry_run/v3_gate_decision.json`

`decision` が完全一致の `GO` のときだけ RQ2 以降を呼ぶ。`NO_GO` / `GO (Valence-only)` / `GO (Arousal-only)` は終了コード 2。`--force-after-no-go` のみ継続。`--force` はキャッシュ再計算であり、ゲート継続ではない。

---

## 8. 実行方法

所要時間は未計測。V3 は探索が重いので、一括 `all` より単独 stage を推奨する。

```bash
bash scripts/run_production_v3.sh cuda:0
# キャッシュ無視
# bash scripts/run_production_v3.sh cuda:0 --force
# RQ1 が GO でない場合に明示継続するときだけ
# bash scripts/run_production_v3.sh cuda:0 --force-after-no-go
```

`run_production_v3.sh` は第2引数以降を `EXTRA_ARGS` として統合 CLI へ転送する。

統合 CLI:

```bash
python -m affective_empathy_eval.run --stage v3 --model-set primary_small --device cuda:0
# python -m affective_empathy_eval.run --stage v3 --model-set primary_small --force
# python -m affective_empathy_eval.run --stage v3 --model-set primary_small --force-after-no-go
python -m affective_empathy_eval.run --stage v3 --model-set primary_small --dry-run
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
| `v3/results/derived/v3_gate_decision.json` | 本番ゲート。`GO` / `NO_GO` / 軸片方の GO。pipeline 継続は完全一致の `GO` のみ |
| `v3/results/derived/dry_run/v3_gate_decision.json` | `--dry-run` 時のゲート。本番ファイルを上書きしない |
| `v3/results/raw/v3_spatiotemporal_maps_{family}.json` | $D,\beta,\gamma,C$ と `abs_beta_*`。`analysis_role=discovery` |
| `v3/results/raw/v3_path_mediation_{family}.json` | Discovery 層、Confirmation の attenuation |
| `v3/results/raw/v3_confirmatory_{family}.json` | 4 仮説の合否 |
| `v3/results/derived/v3_*_summary.json` | 横断要約 |

---

## 10. 論文 19 列

`v3/scripts/build_paper_summary.py` は介入を再実行しない。`v3/results/derived/` を先に探し、無ければ `raw/` を読む。`v3/scripts/legacy/` とは別物である。

| ファイル | 内容 |
|---|---|
| `tables/table_v3_1_gate.csv` | `overall_decision` の文字列と、pipeline を続けるかのフラグ |
| `tables/table_v3_2_spatiotemporal_summary.csv` | 4-Map。`analysis_role=discovery`。`response_end` は negative control として残す |
| `tables/table_v3_3_mediated_attenuation.csv` | Confirmation の減衰。`analysis_role=confirmatory` |
| `tables/table_v3_4_confirmatory.csv` | Llama / Gemma 3 / OLMo 2 の H1〜H4 |
| `tables/table_v3_confirmatory_matrix.csv` | family × 仮説の行列 |
| `stage_summaries/v3/v3_paper_results.csv` | 19 列 |

`figure_data/` は作る。現行の `build_v3_summary` は figure CSV を書き出さない。

```bash
python v3/scripts/build_paper_summary.py --strict
```

LaTeX は `scripts/summarize_v3_causal_utilization.py`。

---

## 11. 解釈

- ゲート GO は「主観が確認された」ではなく「誘導が特異的で、Topic 崩壊が小さい」
- 時空間解離は記述であり、二過程心理理論の証明ではない
- mediated attenuation は遮断後の減衰であり、NDE/NIE ではない
- Topic TVD と Self shift を引き算して主効果にしない
- $\gamma_A, C_A$ を $d_V$ 注入時の Arousal 変化として読まない
- Confirmatory 不成立を「Qwen だけが感情を持つ」と読まない
- 81 VA の期待値を Behavioral / V1 の 729 VAD 期待値と直接比較しない
- `v3/docs/legacy/` の neutralization / greedy collapse ストーリーは主筋ではない
