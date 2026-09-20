# V1 Stage: 表現幾何と因果的オーバーラップ
## 論文対応: §5 Internal Representation and Causal Sharing

V1 は、Behavioral Stage で観測される Reader–Self の行動連動の背後に、**共有・整列された内部表現**と**部分的に重複した因果関連表現および介入感受性サイト（partially overlapping causally relevant representations and intervention-sensitive sites）**が存在するかを問う。

中心の問いは次である。

$$
\text{Does similar behavior imply shared representation and causal overlap?}
$$

測定対象はモデル出力と内部活性化の変位である。「モデルが感情を経験する」とは書かない。また、同一の因果機構を完全に証明するのではなく、因果関連表現および介入感受性部位の部分的重複（partial causal overlap）の検証を主眼とする。

Reader は操作的な認識課題、Self は操作的な自己報告課題である。認知的共感 / 情動的共感との一対一対応はしない。仮説は一直線ではなく、情動関連表現から Reader 関連計算と Self 自己報告関連計算が分岐する形である。

---

## 1. 位置づけ

```text
Behavioral  →  V1  →  V2  →  V3
```

- Behavioral は出力レベルの連動を測る。V1 は「同じ場所から読めるか」「同じ幾何か」「同じ因果サイトか」を測る。
- V1 の比較軸は **Reader ↔ Self**（同一モデル内）。Base / Instruct の 8 条件は「各モデル内で V1 を再現する」ためのものであり、Base↔Instruct 差そのものの解釈は V2 でのみ行う。
- 認識用プローブの出力を post 自己報告の代替にしてはならない。

---

## 2. 検証の階段（Phase A → B → C）

```text
Behavioral Coupling（出力の連動）
        │
        ▼
Phase A  存在と形式
  E1 Shared Decodability
  E2 Shared Geometry
        │
        ▼
Phase B  Semantic / contextual validity controls
  rule-based controlled perturbation
        │
        ▼
Phase C  因果
  E3 Shared Causal Map
  E4 Causal Interchangeability
  E6 Task-Specific Specialization
```

旧計画の「LLM Judge による自由生成評価（旧 E5）」は未実装で、主解析から除外する。V1 の主筋は E1, E2, Phase B, E3, E4, E6 である。

---

## 3. 共通プロトコル

- **Sequence-Likelihood**: 主測定は 729 候補（$V,A,D \in \{1..9\}^3$）の条件付き対数尤度と $E[V], E[A]$。V2 / V3 の 81 VA 空間とは混ぜない。
- **Prompt-End Normalized**: `add_special_tokens=False`、介入・抽出位置は left/right padding 双対応の `valid_pos[-1]`（`torch.nonzero(attention_mask)` 由来）で統一。
- **Truncation ガード**: Phase A/B において `max_length=1024` によるサイレントな切り捨てを監視し、切り詰め発生時は例外を送出。
- **相対深度**: $0 \le l < L$ に対し $d = l / (L-1)$（0-based）。論文・表は層番号ではなく $d$ で横断比較する。
- **独立セッション**: Reader と Self は別フォワードパス。
- **モデル正本**: `configs/models.yaml` の `primary_small`。
  - Qwen 2.5 1.5B / Llama 3.2 1B / Gemma 3 1B / OLMo 2 1B × Base / Instruct
- **単独実行のモデル指定**: `--model-id` または `--family` が必須。Qwen ID への暗黙 default は禁止。
- **データ役割**: EmoBank test1k は連続 VA ラベル（Phase A）。AIPsy は条件ラベルと matched-neutral（Phase A / C / E4）。Phase B は専用統制 CSV。V3 と同じ AIPsy ファイルを使っても、V1 は Reader↔Self の内部比較であり、V3 の状態誘導ゲートとは指標を混ぜない。
- **Phase A 評価マスク**: `evaluated_mask` を適用し、スキップされた fold のサンプルが暗黙の 0 予測としてメトリクスに混入しないよう評価。
- **厳密な中断再開とキャッシュ**: E3/E4 の途中 CSV は `checkpoint_manifest.json`（config_hash, model_revision, prompt_hash 等）が完全一致する場合のみ再開。activation cache は全プロンプトの完全ハッシュおよび環境情報と照合。
- **出力先**: `v1/results/derived/`。raw を上書きせず、`{prefix}` ごとに分ける。

---

## 4. Phase A: 存在と形式

正本: `v1/primary/run_phase_a.py`

### 4.1 E1 Shared Decodability

**問い**: 外部の感情情報が、Reader 表現と Self 表現の同じ層から読めるか。

Primary ラベルは外部客観指標 $Y_{\mathrm{stimulus}}$ である。

| データ | ラベル | モデル | 分割 |
|---|---|---|---|
| EmoBank | 人間 `reader_V`, `reader_A` | Ridge。$R^2$, Pearson $r$, Spearman $\rho$, MSE | GroupKFold（stimulus id） |
| AIPsy | 感情条件（clinical/moderate vs neutral） | Logistic。ROC-AUC, Balanced Accuracy, Macro F1 | StratifiedGroupKFold |
| AIPsy 強度 | None < Moderate < Clinical | 順序相関（Spearman $\rho$） | 同上 |

Secondary はモデル自身の $E[V], E[A]$ への回帰であり、主結論には使わない。

手順:

1. 各刺激で Reader / Self プロンプトを独立に通し、全層の prompt-end 隠れ状態を取る。
2. 層ごとに PCA（上限 50）+ StandardScaler + Ridge/Logistic を交差検証。
3. 層別 decodability 曲線、ピーク層 $l^*_R, l^*_S$、相対深度、層間距離を記録する。

解釈: 「同じ層で高い」は「同じ領域から decodable」までであり、因果実装の証拠ではない。$\mathrm{Decodability} \neq \mathrm{Causal\ Implementation}$。

### 4.2 E2 Shared Geometry

**問い**: ピークが近くても、同じ線形座標系か、回転で揃うだけか、線形には揃わないか。

比較は Reader ↔ Self のみ。

1. **Direct Cross-Decoding**: $W_R(H_S)$ と $W_S(H_R)$
2. **RSA**: 相関距離 RDM の Spearman 相関
3. **Orthogonal Procrustes**: $SVD(H_S^\top H_R)=U\Sigma V^\top$、$Q=UV^\top$、$W_R(H_S Q)$

| パターン | 目安 | 解釈 |
|---|---|---|
| Shared Geometry | Direct が高い | 共通の線形読み出し |
| Alignable Geometry | Direct が低く Aligned が高い | 回転で揃う幾何 |
| Poorly alignable | どちらも低い | 線形転移・整列が困難。非線形の余地は残す |

閾値 $0.3$–$0.5$ は記述用の目安であり、結果を見て動かさない。

既定データ: `v1/data/processed/stimuli_vad_3way_test1k.csv` および AIPsy。件数はロード時にログする。

---

## 5. Phase B: Semantic / contextual validity controls

正本: `v1/primary/run_phase_b.py`  
データ: `v1/data/processed/v1_e5_semantic_controls.csv`（実件数はログ）  
生成: `v1/primary/prepare_v1_phase_b_controls.py`

**問い**: Phase A で読めた表現は、語彙ショートカットではなく文脈・構成に追従するか。

統制は **rule-based controlled perturbation** である。LLM による言い換えや大規模な意味空間摂動（semantic perturbation）ではない。論文では後者の語を使わない。

1. Lexical confound audit（Jaccard, Levenshtein, 語長、感情語重複）
2. Lexically matched minimal pairs
3. Outcome reversal（規則置換で極性を反転。置換が当たらない場合は固定文を追記する fallback）
4. Paraphrase invariance（事前に固定した表層置換。当たらなければ文頭に定型句を付ける）
5. Word shuffle（語彙を保ち語順を壊す。精度低下を確認）

Outcome Reversal の fallback

```text
Fortunately, everything was completely resolved without any harm.
```

は語彙追加量が大きい。これを Outcome Reversal の強い証拠として過大解釈しない。該当ペアは補助記録に留める。

### 層選択（論文に明記すること）

Primary は Phase A の peak を使わない。相対深度 $d=0.5$ を事前固定し、

$$
l = \operatorname{round}\bigl(d(L-1)\bigr)
$$

でモデルごとに層を決める。中間深度の a priori 設計である。`--layer` を付けた場合だけ上書きし、感度分析として記録する。詳細は `v1/docs/protocol.md`。

---

## 6. Phase C: 因果

正本:

- `v1/primary/run_phase_c.py`（E3 / E4）
- `v1/primary/phase_c/run_e6_specialization.py`
- `v1/primary/phase_c/summarize_phase_c.py`

補助モジュール: `run_e3_causal_map.py`, `select_e4_sites.py`, `run_e4_interchangeability.py`（`run_phase_c.py` から呼ぶ）。

Discovery / Confirmation は 50:50。pair の derangement を用い、探索で見たペアを確認に再使用しない。

### 6.1 E3 Shared Causal Map

**問い**: decodable な場所ではなく、出力を動かす因果サイトは同じか。

各層で Activation Patching / Ablation を行い、

- Magnitude: 分布変位（Wasserstein $W_1$ または JSD）
- Direction: $(\Delta E[V], \Delta E[A])$ と $\cos(C_R, C_S)$

オーバーラップ:

1. Spatial overlap（上位 $k$ サイトの Jaccard）
2. Rank similarity（Spearman）
3. Peak displacement $|l^*_R - l^*_S|$（相対深度でも報告）

「同じ場所が強い」だけでなく、「同じ方向に動かすか」まで見る。

### 6.2 E4 Causal Interchangeability

**問い**: Reader で作った感情差分を、Self の中立文脈へ移植できるか。

同一ペアの matched-neutral を基準にする。固定値 $5.0$ への差分は使わない。

$$
\Delta h_{R,i} = h_{R,i}^{\mathrm{aff}} - h_{R,i}^{\mathrm{matched\text{-}neutral}}
$$

$$
h_{S,i}^{\mathrm{neutral}} \leftarrow h_{S,i}^{\mathrm{neutral}} + \alpha\,\Delta h_{R,i}
$$

- $\alpha$ の設定値: `configs/v1_experiments.yaml` の `alphas: [0.0, 0.5, 1.0, 2.0]` を source of truth として優先適用。
- **解釈の境界**: $\Delta h$ は感情差に関連した隠れ状態変位（stimulus-pair specific context を含む **affect-manipulation-associated hidden-state difference**）であり、コンテキストから完全に遊離した「純粋な情動コード」とは主張しない。

必須統制:

1. Matched same-stimulus: 同一ペア $i$ の $\Delta h_{R,i}$
2. **Random-stimulus (20-Derangements Control)**: 固定シードによる **20 回の完全撹乱順列（$K=20$ derangements）** を反復実行し、単一ドナー配置に依存しない random donor distribution、平均効果、標準偏差、matched − mean(random) 差分、permutation p、bootstrap CI を算出
3. Same-task Reader: $\Delta h_{R,i}$ を Reader 自身の中立文へ注入
4. Same-task Self: $\Delta h_{S,i}$ を Self 自身の中立文へ注入（因果的影響力の上限）

$$
\mathrm{Specificity} = \mathrm{Effect}_{\mathrm{matched}} - \mathrm{Mean}(\mathrm{Effect}_{\mathrm{random\_k}})
$$

不成立でも「完全に別系統」とは書かない。「直接の cross-task interchangeability は確認されない」と書く。off-manifold 化の余地を残す。

### 6.3 E6 Task-Specific Causal Specialization

**問い**: Reader-site と Self-site に課題特異的な因果があるか。

※ **解釈上の留意点**: E6 の zero ablation 介入は対象層・トークンの residual stream 全体を 0 にするため、情動表現のみを特異的に除去したものではなく、「**task-specific causal site sensitivity (whole residual zeroing)**」を測るものである。

サイトは E3 Discovery の因果変位から、タスク選択性コントラストで選ぶ。

$$
S_R(l) = C_R(l) - C_S(l),\qquad S_S(l) = C_S(l) - C_R(l)
$$

- Reader site = $\arg\max_l S_R(l)$、Self site = $\arg\max_l S_S(l)$
- 同一層、または最大選択性が正でない場合は **No-Go**（`status: no_distinct_sites_identified`）。第2ピークへの差し替えはしない
- E3 Discovery CSV が無いときは heuristic fallback（例: $0.5(L-1)$）を使わずエラーにする
- Discovery / Confirmation を分ける。探索で見たペアを確認に再利用しない

Confirmation でそのサイトを targeted ablation し、

$$
\mathrm{Outcome} \sim \mathrm{Task} \times \mathrm{SiteType} + (1 \mid \mathrm{pair})
$$

交互作用を Primary 検定とする。有意でも「完全独立回路」ではなく、因果的特異化 / 部分解離と書く。LMM が singular なときは診断を残し、事前に決めた fallback（ペア差分の $t$ 検定）を明示する。旧称 Double Dissociation は使わない。

---

## 7. データと出力

件数は手書きしない。実行ログの実ロード数を正とする。

| 用途 | 既定パス |
|---|---|
| Phase A EmoBank | `v1/data/processed/stimuli_vad_3way_test1k.csv` |
| Phase A / C AIPsy | `v1/data/processed/aipsy_4split_all.csv` |
| Phase B | `v1/data/processed/v1_e5_semantic_controls.csv` |

```text
v1/
├── README.md
├── primary/
│   ├── README.md
│   ├── run_phase_a.py
│   ├── run_phase_b.py
│   ├── run_phase_c.py
│   └── phase_c/
│       ├── run_e3_causal_map.py
│       ├── select_e4_sites.py
│       ├── run_e4_interchangeability.py
│       ├── run_e6_specialization.py
│       └── summarize_phase_c.py
├── data/processed/          # 刺激（原データではない）
├── scripts/legacy/          # 旧実行系。主解析に使わない
└── results/                 # derived / cache。再実行前は .gitkeep 以外をクリア
    └── derived/
        ├── v1_phase_a/{prefix}/
        ├── v1_phase_b/{prefix}/
        ├── v1_phase_c_prompt_end/{prefix}/
        └── v1_phase_c_summary/
```

`{prefix}` は `qwen_base`, `llama_instruct` など。manifest に model_id, 相対深度, seed, commit を残す。

### 7.1 Phase ごとの主な成果物

| Phase | 典型ファイル | 内容 |
|---|---|---|
| A | `e1_*_decodability.csv`, `e2_*_geometry.csv`, `phase_a_summary.md` | 層別 $R^2$ / AUC、ピーク相対深度、cross-decoding / RSA / Procrustes |
| B | `e5_1_lexical_audit.csv`, `e5_semantic_controls_results.csv`, `e5_semantic_summary.md` | 語彙監査と rule-based 統制後の decodability |
| C E3 | `e3_causal_map.csv` | 層別 magnitude / direction / 相対深度 |
| C E4 | `e4_interchangeability_results.csv` | matched / random / same-task の効果 |
| C E6 | `e6_specialization_trials.csv`, `e6_lmm_results.json` | Confirmation 試行と交互作用。No-Go なら `status: no_distinct_sites_identified` |
| 横断 | `v1_phase_c_summary/` | 8 条件の要約 |

E6 は E3 Discovery CSV を読む。無いときは heuristic 層へ落とさずエラーにする。distinct site が取れなければ Negative Result として正常終了する。

---

## 8. 実行方法

`.venv` を使う。所要時間は未計測。本番は stage 分割を推奨する。

```bash
bash scripts/run_production_v1.sh cuda:0
```

統合 CLI（`primary_small` の 8 モデルを Phase A → B → C → E6 → summarize）:

```bash
python -m affective_empathy_eval.run --stage v1 --model-set primary_small --device cuda:0
```

1 family だけ:

```bash
python -m affective_empathy_eval.run --stage v1 --model-set primary_small --family qwen --device cuda:0
```

### 8.1 単独実行

`--model-id` か `--family` が無いと落ちる。

```bash
python v1/primary/run_phase_a.py \
    --family qwen --is-instruct \
    --dataset both --device cuda:0

python v1/primary/run_phase_b.py \
    --family qwen --is-instruct \
    --relative-depth 0.5 --device cuda:0

python v1/primary/run_phase_c.py \
    --family qwen --is-instruct \
    --device cuda:0

python v1/primary/phase_c/run_e6_specialization.py \
    --family qwen --is-instruct \
    --device cuda:0
```

`--limit` / `--dry-run` は確認用。本番では limit を付けない。

---

## 9. 解釈

- E1 の高い $R^2$ を「同じ回路」と読まない
- E4 不成立を「感情が無い」と読まない
- E6 No-Go を「回路が無い」ではなく「distinct site がこの選定規則では取れない」と書く
- Phase B で語彙統制が効いても、人間の意味理解と同定しない
- Behavioral の coupling $r$ を V1 指標に代入しない
- Base / Instruct 差を V1 の主解釈にしない（V2 の軸）
- 主仮説を結果の後から変えない。変更は `docs/decision_log.md` と別実験として残す
