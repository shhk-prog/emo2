# Behavioral Stage: 行動レベル情動反応性評価

Behavioral Stage は、内部表現や因果介入の前に、**モデル出力として得られる自己報告 VA（Valence–Arousal）および認識 VA が、人間評定済み刺激に対してどう変位するか**を測定する独立パイプラインである。

本研究は、LLM が主観的な感情を経験していることを検証しない。測定対象は次に限る。

- `self-reported affective state`（自己報告情動状態）
- `elicited affective response`（刺激誘発性情動反応）
- `affective reactivity profile`（情動反応プロファイル）

「モデルが悲しみを感じる」「共感する」といった断定は、本文・図表・変数名でも用いない。

---

## 1. この Stage の位置づけ

全体パイプラインは次の順で進む。

```text
Behavioral  →  V1  →  V2  →  V3
```

- **Behavioral**: 出力分布の行動的変位（認識と自己報告を独立セッションで測定）
- **V1**: その行動が、共有表現・文脈統制（rule-based perturbation）・共有因果実装で支えられるか
- **V2**: Base ↔ Instruct の事後学習で幾何と因果回路がどう再編されるか
- **V3**: 内部状態から自己報告へ至る時空間経路と mediated attenuation

Behavioral の結果を、認識測定の代替値として post 自己報告に流用してはならない。Recognition と Reactivity はデータ・指標・表を分離する。

---

## 2. 測定プロトコル

### 2.1 3課題・独立セッション

各刺激について、次の 3 課題を **独立したフォワードパス** として実行する。同一会話履歴に混ぜない。

| 課題 | 記号 | 問い | 人間参照ラベル |
|---|---|---|---|
| Writer-State Estimation | $W$ | 書き手はどう感じていたか | EmoBank writer VAD / AIPsy 条件 |
| Reader-Response Prediction | $R$ | 平均的読者はどう感じるか | EmoBank reader VAD / AIPsy 条件 |
| Self-Report | $S$ | あなた自身はどう感じるか | 人間 Self-report の直接 GT ではない。EmoBank では reader を correspondence 参照として用いる |

主解析の perspective は reader を用いる。writer は補助分析であり、主解析と混ぜない。

### 2.2 Sequence-Likelihood（729 候補）

Greedy / Sampling 生成は中立 JSON への縮約を起こしやすい。主測定は生成文ではなく、

$$
(V, A, D) \in \{1,\ldots,9\}^3
$$

の **729 個の JSON 候補** に対する条件付き対数尤度である。これから連続期待値を出す。

$$
E[V] = \sum_{v=1}^{9} v\, P(V=v),\qquad
E[A] = \sum_{a=1}^{9} a\, P(A=a),\qquad
E[D] = \sum_{d=1}^{9} d\, P(D=d)
$$

- Valence / Arousal が主対象。Dominance は補助次元。
- 許容範囲は整数 $[1, 9]$。解析時に $[-1, 1]$ へ変換する場合の式は共通プロトコルに従う。
- `exact_neutral_argmax`（例: $(5,5,5)$ への縮約率）は主指標ではなく補助診断である。

### 2.3 プロンプト形式

- **Instruct**: chat template（system + user、`add_generation_prompt=True`）
- **Base**: plain text の Task / Text / Output 形式
- JSON スキーマは `{"valence": int, "arousal": int, "dominance": int}`

---

## 3. 4大行動評価軸

### 3.1 Human-Affect Correspondence（人間評定対応度）

モデル期待値と人間 VAD の Pearson $r$ / Spearman $\rho$、および MAE / RMSE。

- Writer: $W \leftrightarrow W_{\mathrm{human}}$
- Reader: $R \leftrightarrow R_{\mathrm{human}}$
- Self: $S \leftrightarrow R_{\mathrm{human}}$（人間 Reader は Self の直接 GT ではない。correspondence 参照）

### 3.2 Sensitivity（感度）

AIPsy の Clinical と Neutral のペア差分。Cohen's $d_z$、対応 $t$ 検定、Bootstrap 95% CI、BH-FDR。

### 3.3 Dose-Response & Specificity（用量反応と特異性）

- Dose-response: Neutral → Moderate → Clinical の単調変位
- Specificity: Complex Neutral（難解だが感情を含まない統制）に対する誤反応が小さいこと

### 3.4 Reader–Self Coupling（認識–自己報告連動）

刺激ごとの $\mathrm{Corr}(R, S)$。高い相関は「刺激間での共変動」であり、「認識が自己報告へ因果伝播した」証拠ではない。因果は V1 Phase C 以降で別測定する。

---

## 4. データセット

件数は README に固定しない。実行時に実 CSV の行数・`pair_id` 数をログする。

| データ | 既定パス | 役割 |
|---|---|---|
| EmoBank 3-Way VAD | `v1/data/processed/stimuli_vad_3way.csv` | Writer / Reader 人間評定付きテキスト。3課題 correspondence |
| AIPsy-Affect 4-Split | `v1/data/processed/aipsy_4split_all.csv` | `neutral` / `moderate` / `clinical` / `complex_neutral`。Sensitivity, dose-response, specificity, coupling |

原データ `data/raw/` は読み取り専用。加工は `data/processed/` または `v1/data/processed/` へ新規出力する。

---

## 5. 対象モデル

モデル ID の正本は `configs/models.yaml` のみ。

**Primary (`primary_small`)**: Qwen 2.5 1.5B / Llama 3.2 1B / Gemma 3 1B / OLMo 2 1B の Base と Instruct（8 モデル）。

**Supplementary (`scale_validation`)**: Mistral 7B v0.3（V2 外部スケール検証。Behavioral 主解析のコホートではない）。

単独実行でモデル ID を書く場合も、YAML の ID と一致させる。Qwen への暗黙 default は使わない。

---

## 6. ディレクトリ構成

```text
behavioral/
├── README.md
├── primary/
│   ├── run_behavioral_emobank.py   # EmoBank 3-Way VAD
│   └── run_behavioral_aipsy.py     # AIPsy 4-Split
├── analysis/
│   ├── summarize_behavioral_emobank.py
│   └── summarize_behavioral_aipsy.py
└── results/                        # 再実行前は .gitkeep 以外をクリアしてよい
    ├── emobank_3way/
    ├── emobank_3way_summary/
    ├── aipsy_4split/
    └── aipsy_4split_summary/
```

---

## 7. 実行方法

仮想環境 `.venv` を有効化し、プロジェクトルートから実行する。所要時間は未計測。Qwen 1 family の benchmark 後に更新する。

本番は stage 分割を推奨する。

```bash
bash scripts/run_production_behavioral.sh cuda:0
```

統合 CLI:

```bash
python -m affective_empathy_eval.run --stage behavioral --model-set primary_small --device cuda:0
```

Dry-run（モデル重みを載せないスモークテスト）:

```bash
python -m affective_empathy_eval.run --stage behavioral --model-set primary_small --dry-run
```

### 7.1 単独実行（1 モデル）

```bash
python behavioral/primary/run_behavioral_emobank.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --is_instruct \
    --tag qwen_instruct \
    --stimuli-path v1/data/processed/stimuli_vad_3way.csv \
    --out-dir behavioral/results/emobank_3way \
    --device cuda:0

python behavioral/primary/run_behavioral_aipsy.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --is-instruct \
    --tag qwen_instruct \
    --stimuli-path v1/data/processed/aipsy_4split_all.csv \
    --out-dir behavioral/results/aipsy_4split \
    --device cuda:0
```

`--model` と `--tag` は必須。`--limit` は動作確認用。本番では付けない。

### 7.2 集計

```bash
python behavioral/analysis/summarize_behavioral_emobank.py \
    --input-dir behavioral/results/emobank_3way \
    --out-dir behavioral/results/emobank_3way_summary

python behavioral/analysis/summarize_behavioral_aipsy.py \
    --input-dir behavioral/results/aipsy_4split \
    --out-dir behavioral/results/aipsy_4split_summary
```

---

## 8. 出力

| 出力 | 内容 |
|---|---|
| `{tag}_3way_vad.csv` | 刺激ごと $E[V], E[A], E[D]$、greedy argmax、人間参照 |
| `{tag}_3way_vad_summary.json` | 課題×次元の相関、$(5,5,5)$ 率、manifest |
| `{tag}_aipsy_4split.csv` | split 付き刺激ごとの期待値 |
| `behavioral_*_summary.csv` | 4 軸のモデル横断表 |

生応答は CSV に埋め込みすぎず、manifest（`run_id`、model_id、commit、設定）を残す。失敗・パース不能は削除せず理由コードとともに保存する。

---

## 9. 解釈上の禁止事項

- 高い $r$ を「共感」や「主観的感情」と読まない
- Recognition（$W$, $R$）と Reactivity（$S$）を同じ列に混ぜない
- 失敗応答を無記録で落とさない
- 結果を見てから除外閾値を動かさない
- Behavioral の数字を V1 の内部表現指標の代替にしない
