# 実装計画: 正式再実行前の results クリアと再現性整理

## 方針

実験仕様は変えない。旧結果の混在を止め、README / dry-run / Recovery 定義を現行 Primary コードに揃える。

## 1. results の完全クリア

対象:

- `behavioral/results/`
- `v1/results/`
- `v2/results/`
- `v3/results/`

手順:

1. 追跡中の raw/derived ファイルを `.gitkeep` 以外 `git rm`
2. `scripts/clear_stage_results.sh` を、ネストした `.gitkeep` を残し `raw/` `derived/` プレースホルダを作り直す形に更新して実行
3. ルートの `results/logs/` も本番ログの混在を避けるため削除（gitignore 済み）

残すもの: 各 `results/.gitkeep` と `results/raw/.gitkeep`, `results/derived/.gitkeep`。

## 2. .gitignore

```
**/results/raw/**
**/results/derived/**
!**/results/raw/.gitkeep
!**/results/derived/.gitkeep
```

既存の cache/logs/mock archive ignore は維持する。

## 3. V2 RQ4 Recovery

現行コード:

```
ratio = (EMD(Instruct, Base) - EMD(patched Instruct, Base)) / (EMD(Instruct, Base) + 1e-12)
```

README の代数:

```
Recovery = (W1(P_clean, P_target) - W1(P_patch, P_target)) / W1(P_clean, P_target)
```

対応:

- `P_target` = Base の 9×9 Sequence-Likelihood 結合分布
- `P_clean` = Instruct 未介入
- `P_patch` = Instruct へ Base 活性化を注入した後
- `W1` = `compute_distribution_metrics(...)["emd_va"]`

`compute_emd_recovery_ratio` を `likelihood.py` に切り出し、unit test で符号・分母・完全回復 / 無変化を固定する。README の分布対応もこの定義に直す（中立文脈への感情注入ではない）。

## 4. V2 dry-run fixture

`reader_V` / `reader_A` 欠損時は `np.random.uniform` を使わず、`np.linspace(1, 9, n)` の決定論的 fixture を使う。本番経路は触らない。

## 5. 表現

- V1 Phase B: Semantic / contextual validity controls。実体は rule-based controlled perturbation。Outcome Reversal の fallback 文は語彙追加が大きいため過大解釈しない。
- V3 RQ2: 4-Map は D, β, γ, C（各 V/A）。本文のピーク解離は D と C を主に読む。
- 本番実行: `run_production_all.sh` より stage 別を推奨。順序は Behavioral → V1、問題なければ V2 → V3。

## 6. transformers 遅延 import

Primary ランナーの `from transformers import ...` を `ImportError` 時に None へフォールバックする。`--dry-run` はモデル重みも transformers も不要なスモークテストとして起動できるようにする。本番経路は従来どおり `from_pretrained` を使う。

## 検証

- `.venv` がこの macOS checkout で使えない場合は、利用可能な Python に `PYTHONPATH=src` を付けて pytest
- Primary スクリプトの compile
- bash syntax
- UTF-8
- results 配下に `.gitkeep` 以外が残っていないこと
