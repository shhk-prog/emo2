# Affective Empathy Evaluation（LLM 情動反応性評価実験）

人間評定済みの感情刺激（EmoBank VAD、AIPsy 4-split）に対して、大規模言語モデルの **操作的な認識出力** と **操作的な自己報告出力** を測り、その内部表現と因果利用を一続きの証拠階層として段階的に解明する実験基盤である。

本研究は、モデルが主観的な感情を経験すること、あるいは認知的共感 / 情動的共感を持つことを検証しない。論文構成の正本は現行 README 群であり、旧 `iclr2027/iclr2027_conference2.tex` は旧稿である。構成メモは [`docs/v3_prerun_five_fixes/paper_outline.md`](docs/v3_prerun_five_fixes/paper_outline.md)。

---

## 1. 中心リサーチクエスチョン (Central RQ)

> **Central RQ**:  
> **How are affect-relevant internal representations coupled to LLM self-reports, and how does this relationship vary across tasks, post-training, and stages of computation?**  
> （LLMの情動関連内部表現は自己報告とどのように結びついており、その関係はタスク、事後学習（post-training）、および計算過程を通じてどのように変化するのか。）

### 中心的主張 (Core Thesis)

> **LLM self-reports are systematically related to affect-relevant internal representations, but this relationship is partial and task-dependent. Across Base–Instruct pairs, the representation–report relationship exhibits post-training-associated reorganization, and decodable affect-relevant information is not uniformly causally relevant: measurable causal leverage over self-report is concentrated at particular layers and stages of computation.**
> 
> （**LLMの自己報告は情動関連内部表現と系統的に結びついているが、その対応は部分的かつタスク依存である。Base–Instruct間では表現と自己報告の関係にpost-training-associatedな再編が観測され、さらに内部でデコード可能な情動情報が一様に自己報告へ因果的に寄与するわけではなく、その因果的影響（measurable causal leverage）は特定の層・計算段階に集中する。**）
>
> ※ LLMの自己報告は出力レベルの共変動（covariation）のみによって特徴づけられるものではなく、またデコード可能なすべての情動関連情報の均一な直接読み出し（direct readout）でもありません（*LLM self-reports cannot be characterized by output-level covariation alone, nor as a uniform direct readout of all decodable affect-relevant information*）。

---

## 2. 証拠階層と論文構造 (4-Stage Evidence Hierarchy)

本研究は、中心 RQ に対する一続きの証拠階層として 4 つの Stage を配備し、論文の主要章（Section 3〜6）に対応させています。

```text
Covariation  →  Representation & Overlap  →  Reorganization  →  Causal Leverage
(Behavioral)               (V1)                       (V2)              (V3)
```

| Stage | 概念的役割 | 論文 Section | 中心的な科学的問い | 主な検証・比較軸 |
|---|---|---|---|---|
| **Behavioral** | **Covariation** (相関・導入現象) | §3. Behavioral Characterization: Alignment and Coupling between Reader and Self Perspectives | *Do Reader and Self covary in their responses to controlled affective changes?* | 制御された情動変化に対する出力レベルの連動（$\Delta$ カップリング） |
| **V1** | **Representation & Overlap** (表現・因果重複) | §4. Shared Representation and Causal Overlap in Base Models | *What do Reader and Self share internally?* | 同一基底モデル内における因果関連表現と介入感受性サイトの部分的重複 |
| **V2** | **Reorganization** (事後学習関連再編) | §5. Post-training-Associated Reorganization of Affect-Relevant Computations | *How does post-training associate with computational reorganization?* | 同一ファミリーの Base ↔ Instruct 幾何・共有性・分布回復（モデル内は活性化因果介入、モデル間は観察的再編） |
| **V3** | **Causal Leverage** (因果的利用可能性と必然性) | §6. From Decodability to Causal Leverage: Sufficiency, Specificity, and Spatiotemporal Dynamics | *Where does affect-relevant information exert measurable causal leverage over self-report?* | Instruct 側の層 × 生成段階（十分性・特異性・内生関連性の局在） |

- **Behavioral** (`behavioral/`): EmoBank 3-Way と AIPsy 4-Split。4軸は correspondence, Sensitivity, Dose-response / Specificity, Reader–Self coupling（刺激変化に対する $\Delta$ カップリング $\text{corr}(\Delta_R, \Delta_S)$ を Primary 化）。[`behavioral/README.md`](behavioral/README.md)
- **V1** (`v1/`): decodability / 幾何（E1/E2）、意味統制感度（Phase B）、因果マップと交換可能性（E3/E4）、課題特異化（E6）。同一モデル内での部分的重複（partially overlapping causally relevant representations and intervention-sensitive sites）を検証。[`v1/README.md`](v1/README.md)
- **V2** (`v2/`): 幾何再編、ピーク解離、2D OT 分布回復（RQ1〜RQ4。直交 Procrustes アラインメント統制）。各モデル内部の活性化操作（RQ3/RQ4）は**モデル内因果介入（within-model causal characterization）**として同定し、Base と Instruct のモデル間比較は**事後学習に伴う再編（post-training-associated reorganization）**として観察的に帰属（訓練過程そのものへの直接的因果介入ではないため過大主張を避ける）。[`v2/README.md`](v2/README.md)
- **V3** (`v3/`): AIPsy matched-neutral 192 pair での状態誘導ゲート、時空間 4-Map、mediated attenuation、確証的再現。
  - **Direction Injection**: 中立文への方向加算注入（$h + \alpha \sigma_h \hat{d}$）による十分性と因果的影響力（*sufficiency / causal leverage*）
  - **Subspace Removal**: 情動文に対する中心化2D直交部分空間除去による内生的な関連性（*endogenous relevance*）
  - 因果的影響は特定の層、および teacher-forced candidate sequence 上の特定の計算段階に集中（*concentrated at particular layers and stages along the teacher-forced candidate sequence*）することを実証。[`v3/README.md`](v3/README.md)

### 3つの学術的貢献 (Main Contributions)

1. **Behavioral + V1 (Covariation & Representation Overlap)**:  
   制御された刺激の情動変化に対して Reader Prediction（読者感情の推定）と Self-Report（提示後自己報告）が行動レベルで covary すること（*Reader and Self covary in their responses to controlled affective changes*）を示し、その背後に部分的に重複した因果関連表現と介入感受性サイト（*partially overlapping causally relevant representations and intervention-sensitive sites*）が存在することを実証する。
2. **V2 (Post-training-Associated Reorganization)**:  
   Base–Instruct 比較は単純な情動情報の消去説（simple complete-erasure account）と整合せず、表現幾何、Reader–Self 共有性、および介入感受性回路の系統的な再編（*changes in representational geometry, Reader–Self sharing, and intervention-sensitive organization*）が post-training 条件間で生じていることを明らかにする。
3. **V3 (Distinguishing Decodability from Causal Leverage)**:  
   内部でデコード可能な情動情報（*decodable affect-relevant information*）と介入によって実証される因果的関連性（*interventionally demonstrated causal relevance*）を明確に区別し、自己報告に対する測定可能な因果的影響力（*causal leverage*）がどの層および teacher-forced candidate sequence 上の計算段階に集中しているかを時空間的に特定する。

---

## 3. 操作的定義とモデル・データ構成

### 操作定義
- **Reader**: 平均的読者の VA を推定する認識課題（Cognitive estimation）
- **Self**: 刺激提示後の自己報告 / 反応性課題（Self-reported affective response）

刺激から自己報告に至る計算モデルは、共有表現の存在をあらかじめ仮定せず、内部表現から各タスク計算への分岐として記述した上で、V1 でその共有度を検証します。

```text
Stimulus
  → affect-relevant internal representations
     ↘ Reader-related computation
     ↘ Self-report-related computation
```

### 候補空間とスケーリング（729 VAD vs 81 VA）
論文 Methods / Limitations における設計根拠と解釈境界：
- **Behavioral / V1 Primary: $9^3 = 729$ VAD**  
  先行研究および人間評価アノテーション（EmoBank 等）のプロトコルを忠実に保持。※ Dominance は Behavioral / V1 の補助次元とし、最終主張には直接寄与しないため Appendix で補足する。
- **V2 / V3 Primary: $9^2 = 81$ VA**  
  層 × 生成段階にわたる網羅的因果スイープの計算量を実行可能範囲に抑え、Primary endpoint を Valence / Arousal に限定。
- **Stage 間の比較境界**: 729 空間と 81 空間の絶対値を直接比較することはせず、**各 Stage 内の contrast と relative pattern を主たる推論対象**とします（Supplementary において 81 空間と 729 空間の小規模な感度分析を配置）。

### 対象モデル構成（コホート設計）
モデルサイズ差を抑えた複数ファミリーの Base–Instruct 対を用いて、事後学習に伴う差異（post-training-associated differences）の再現性を検証するため、狭い 1〜1.5B パラメータ帯の 4 大独立ファミリーを Primary コホートとしています：
- **Primary 1–1.5B Cohort (`primary_small`)**:  
  *four independently developed model families in the 1–1.5B regime*
  - **Qwen 2.5 (1.5B)**: `Qwen/Qwen2.5-1.5B` $\leftrightarrow$ `Qwen/Qwen2.5-1.5B-Instruct`
  - **Llama 3.2 (1.23B)**: `meta-llama/Llama-3.2-1B` $\leftrightarrow$ `meta-llama/Llama-3.2-1B-Instruct`
  - **Gemma 3 (1B)**: `google/gemma-3-1b-pt` $\leftrightarrow$ `google/gemma-3-1b-it`
  - **OLMo 2 (1B)**: `allenai/OLMo-2-0425-1B` $\leftrightarrow$ `allenai/OLMo-2-0425-1B-Instruct`
- **Supplementary Scale Validation (`scale_validation`)**:
  - **Mistral (7B)**: `mistralai/Mistral-7B-v0.3` $\leftrightarrow$ `mistralai/Mistral-7B-Instruct-v0.3`（大規模モデルでの頑健性・再現性検証）

---

## 4. 統一実験枠組み

全 Stage で共有する規則である。詳細は [`AGENTS.md`](AGENTS.md) と各 Stage README。

### 4.1 測定対象

測るのはプロンプトへのモデル出力と、その隠れ状態・介入応答である。主観的感情の有無は仮説にしない。

| 用語 | 意味 | 使わない読み |
|---|---|---|
| Reader | 平均的読者の VA を推定する認識課題 | 認知的共感 |
| Self | 刺激提示後の自己報告 / 反応性課題 | 情動的共感、主観的感情 |
| Writer | 書き手状態の推定（Behavioral 補助） | 作者共感 |

認識と反応は独立セッションである。同一会話に baseline / recognition / post を並べない。認識出力を post 自己報告の代替にしない。

### 4.2 Sequence-Likelihood

主測定は自由生成ではなく、候補 JSON の条件付き対数尤度と Softmax 期待値である。

- Behavioral / V1: $9^3=729$ VAD。JSON は `{"valence": int, "arousal": int, "dominance": int}`
- V2 / V3: $9^2=81$ VA。JSON は `{"valence": int, "arousal": int}`
- Valence / Arousal が主対象。Dominance は Behavioral / V1 の補助次元
- 両空間の $E[V], E[A]$ を同一尺度として比較しない

### 4.3 層と介入位置

- 相対深度 $d = l/(L-1)$（0-based）。論文・横断表は層番号ではなく $d$ で比較する
- V1 / V2 の標準介入位置は prompt-end（`add_special_tokens=False`、`prompt_end = len(prompt_ids)-1`）
- V3 RQ2 の生成段階は joint sequence 上の token。`prompt_end` に丸めない
- Phase B / V3 RQ1 の既定層は $d=0.5$ から $l=\operatorname{round}(d(L-1))$。Qwen 14 層固定ではない

### 4.4 モデル正本

ID の正本は [`configs/models.yaml`](configs/models.yaml) のみ。コードへ model ID をハードコードしない。未知 family は Qwen へ落とさず `KeyError` にする。`--model-id` / `--family` が必要な単独スクリプトで Qwen default は使わない。

### 4.5 データ

件数は README に固定せず、実行時に実 CSV を読んでログする。

| データ | 既定パス | 使う Stage |
|---|---|---|
| EmoBank 3-way（本番 Behavioral） | `v1/data/processed/stimuli_vad_3way.csv` | Behavioral |
| EmoBank 3-way test1k | `v1/data/processed/stimuli_vad_3way_test1k.csv` | V1 Phase A、V2 |
| AIPsy 4-split | `v1/data/processed/aipsy_4split_all.csv` | Behavioral、V1、V3 |
| Phase B 統制 | `v1/data/processed/v1_e5_semantic_controls.csv` | V1 Phase B |

V3 は AIPsy の clinical–neutral 192 pair だけを wide 化する。EmoBank 3-way は `pair_id` / matched-neutral が無いため V3 Primary に使わない。原データは読み取り専用。

### 4.6 結果の不変性

`**/results/raw/**` と `**/results/derived/**` は追記専用で Git 管理しない（`.gitkeep` のみ残す）。再実行は別 `run_id`。失敗・拒否・パース不能は削除せず理由とともに残す。

### 4.7 共通プロトコル対応表 (Cross-Stage Methodological Matrix)

論文の Methods 章で定義される、各ステージにおける統一的な実験仕様の対応表です：

| 項目 / 次元 | Behavioral (§3) | V1 Stage (§4) | V2 Stage (§5) | V3 Stage (§6) |
|---|---|---|---|---|
| **Candidate Space** | $9^3 = 729$ VAD | $9^3 = 729$ VAD | $9^2 = 81$ VA | $9^2 = 81$ VA |
| **Prompt Format** | Plain (Base) / Chat (Instruct) | Plain (Base) / Chat (Instruct) | Plain (Base) / Chat & Matched-Plain (Instruct) | Chat (Instruct primary) |
| **Token Position** | Sequence-end (log-likelihood) | `prompt_end` (Phase A/B/C) | `prompt_end` (D/C anchor 統一) | Spatiotemporal Grid (`prompt_end` + joint stage tokens) |
| **Layer Coordinate** | N/A (Black-box behavioral) | Relative depth $d = l / (L-1)$ | Relative depth $d = l / (L-1)$ | Relative depth $d = l / (L-1)$ |
| **Split Unit** | Pair-aware (`pair_id`) / Unpaired (EmoBank) | Stratified Group Split (`pair_id` 漏洩防止) | Stratified Group Split (`pair_id` 漏洩防止) | Matched-pair (192 clinical-neutral pairs) |
| **Primary Metric** | $E[V], E[A]$, Cohen's $d_z$, Spearman $\rho$, $r_{RS}$ | $R^2$, Balanced Acc, RSA, Causal Shift $\Delta V, \Delta A$ | Cross-decoding $\Delta\Delta_{\text{cross}}$, $\Delta d_{\text{peak}}$, 2D OT EMD Recovery | Interventional slope $\gamma$, Subspace Attenuation, Causal Leverage $C$ |
| **Statistical Test** | Direction-aligned IUT, 1-sample/paired $t$, FDR (BH), Bootstrap CI | 5-fold GroupKFold CV, FDR (BH), LMM (Phase C E6) | Bootstrap 95% CI, Permutation Test | Pre-registered Go/No-Go Gate, Bootstrap 95% CI, FDR |

---

## 5. 環境構築とセットアップ

AGENTS.md の規定に従い、必ずプロジェクト専用の仮想環境（`.venv`）を構築して実行してください。

### 自動セットアップ（推奨: 1コマンド）
```bash
./setup_env.sh
```
- `uv` がインストールされている環境では数秒で自動構築されます。
- `uv` がない場合も標準の `python3 -m venv` を用いて自動セットアップされます。

### 手動セットアップ (`uv`)
```bash
uv venv .venv --python python3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

---

## 6. テストの実行

仮想環境を有効化（または `.venv/bin/pytest` を直接指定）してテストを実行します：

```bash
.venv/bin/pytest -q
```

---

## 7. 実行方法

所要時間は未計測。本番は **stage 分割** を推奨する。`run_production_all.sh` より、Behavioral → V1 を完走してから V2 → V3 の方が障害切り分けと resume が容易である。

### 7.1 production bash と統合 CLI の違い

| 項目 | `scripts/run_production_*.sh` | `python -m affective_empathy_eval.run` |
|---|---|---|
| 仮想環境 | `.venv` を activate する | 呼ばれた python をそのまま使う |
| device 既定 | `cuda:0`（第1引数） | `cpu` |
| ログ | `results/logs/production_<stage>_TIMESTAMP.log` に tee | 標準出力のみ |
| モデル集合 | `primary_small` 固定 | `--model-set` で切替 |
| `--family` / `--max-samples` | 基本なし | あり |
| V3 ゲート継続 | 第2/第3引数に `--force-after-no-go` | `--force-after-no-go` |

本番 GPU では bash script を使う。確認や 1 family だけなら統合 CLI。

### 7.2 本番（4 family）

```bash
source .venv/bin/activate
bash scripts/run_production_behavioral.sh cuda:0
bash scripts/run_production_v1.sh cuda:0
bash scripts/run_production_v2.sh cuda:0
bash scripts/run_production_v3.sh cuda:0
```

V3 の RQ1 が完全一致の `GO` でないと RQ2 以降は走らない。明示継続だけ:

```bash
bash scripts/run_production_v3.sh cuda:0 --force-after-no-go
```

### 7.3 統合 CLI

`--stage all` も Behavioral → V1 → V2 → V3 の順（`PRODUCTION_STAGE_ORDER`）。

```bash
python -m affective_empathy_eval.run --stage behavioral --model-set primary_small --device cuda:0
python -m affective_empathy_eval.run --stage v1 --model-set primary_small --device cuda:0
python -m affective_empathy_eval.run --stage v2 --model-set primary_small --device cuda:0
python -m affective_empathy_eval.run --stage v3 --model-set primary_small --device cuda:0

python -m affective_empathy_eval.run --stage v2 --model-set scale_validation --device cuda:0
python -m affective_empathy_eval.run --stage v3 --model-set primary_small --dry-run
```

`--dry-run` はモデル重みを載せない。transformers 未導入でも Primary は起動する。

### 7.4 論文

現行 README 群が設計の正本である。`iclr2027/iclr2027_conference.tex` はテンプレート、`iclr2027_conference2.tex` は旧稿である。構成メモは [`docs/v3_prerun_five_fixes/paper_outline.md`](docs/v3_prerun_five_fixes/paper_outline.md)。再実行前の数値を本文に入れない。

---

## 8. ディレクトリ構成

```text
├── README.md                      # 本ドキュメント
├── pyproject.toml                 # プロジェクト共通依存関係およびテスト設定定義（正本）
├── setup_env.sh                   # 仮想環境自動構築スクリプト
├── src/
│   └── affective_empathy_eval/    # 共通基盤ライブラリ (唯一の正本パッケージ)
│       ├── run.py                 # 統合実験実行 CLI
│       ├── models/                # 共通 ModelRegistry & ModelAdapters
│       └── ...
├── tests/                         # 共通テストスイート (All tests should pass)
├── configs/                       # 共通設定ファイル (models.yaml: 唯一のモデル定義正本)
├── behavioral/                    # 行動実験 Primary パイプライン & 分析
├── v1/                            # V1 表現幾何・Prompt-End 因果パッチング実験
├── v2/                            # V2 幾何・因果結合・分布回復実験 (Primary: v2/primary/)
├── v3/                            # V3 時空間ダイナミクス・媒介分析実験 (Primary: v3/primary/)
└── docs/                          # 実験記録・仕様書・決定ログ
```

各 Stage の本文は `behavioral/README.md`, `v1/README.md`, `v2/README.md`, `v3/README.md`。実行面の短い案内はそれぞれの `primary/README.md`。モデル ID は `configs/models.yaml` のみ。決定の記録は `docs/decision_log.md`。
