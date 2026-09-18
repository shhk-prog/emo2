# Behavioral Primary

Behavioral の正式実行面。設計・指標・解釈の本文は親の [`behavioral/README.md`](../README.md) を正本とする。

中心の問い: **Do Reader and Self covary?**  
候補空間: 729 VAD。V2 / V3 の 81 VA 期待値と直接比較しない。

## スクリプト

| ファイル | データ既定 | 独立セッション | 主な指標 |
|---|---|---|---|
| `run_behavioral_emobank.py` | `v1/data/processed/stimuli_vad_3way.csv` | Writer / Reader / Self | 人間 VAD との $r$, $\rho$, MAE。$(5,5,5)$ 率は補助 |
| `run_behavioral_aipsy.py` | `v1/data/processed/aipsy_4split_all.csv` | 同上 | Sensitivity, dose-response, specificity, $R$–$S$ coupling |

集計:

| ファイル | 入力 | 出力 |
|---|---|---|
| `behavioral/analysis/summarize_behavioral_emobank.py` | `behavioral/results/emobank_3way/` | `emobank_3way_summary/` |
| `behavioral/analysis/summarize_behavioral_aipsy.py` | `behavioral/results/aipsy_4split/` | `aipsy_4split_summary/` |

## 引数

両スクリプト共通の要点:

- `--model`: `configs/models.yaml` の ID と一致させる。暗黙 Qwen default は無い
- `--tag`: 出力接頭辞（例: `qwen_instruct`）
- Instruct は EmoBank が `--is_instruct`、AIPsy が `--is-instruct`（ハイフンの有無が違う）
- `--device`: 本番は `cuda:0`
- `--limit`: 確認用。本番では付けない
- `--stimuli-path` / `--out-dir`: 既定は上表

統合 CLI は `--model` / `--tag` を registry から埋める。bash `run_production_behavioral.sh` は `.venv` を有効化し、device 既定 `cuda:0`、ログを `results/logs/` に残す。

## 実行

### 推奨: 統合 CLI（root configs/models.yaml から一括実行）

モデル ID は `configs/models.yaml` を正本とし、CLI が自動解決してディスパッチします。

```bash
# Primary 1-1.5B コホート全体を実行
python -m affective_empathy_eval.run --stage behavioral --model-set primary_small --device cuda:0

# 特定ファミリーのみ実行する場合
python -m affective_empathy_eval.run --stage behavioral --model-set primary_small --family qwen --device cuda:0
```

### 本番 Bash ランナー

環境構築済みの `.venv` を自動 activate し、ログを `results/logs/` に tee します。

```bash
bash scripts/run_production_behavioral.sh cuda:0
```

### 個別スクリプト実行例（低レイヤ確認・手動検証用）

```bash
python behavioral/primary/run_behavioral_emobank.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --is_instruct --tag qwen_instruct --device cuda:0

python behavioral/primary/run_behavioral_aipsy.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --is-instruct --tag qwen_instruct --device cuda:0
```

V3 は同じ AIPsy CSV から clinical–neutral pair だけを使う。Behavioral の 4-split 表を V3 指標の代わりにしない。
