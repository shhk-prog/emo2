# Walkthrough：現行コードに合わせた README 修正

## 実施内容

論文用の階層説明は残し、運用正本を `run.py` / YAML / production bash に揃えた。数値結果は書いていない。GPU 本番は実行していない。

## 更新した README

- `README.md`
- `behavioral/README.md`, `behavioral/primary/README.md`
- `v1/README.md`, `v1/primary/README.md`, `v1/scripts/README.md`
- `v2/README.md`, `v2/primary/README.md`
- `v3/README.md`, `v3/primary/README.md`, `v3/scripts/README.md`
- `scripts/README.md`（新規）

## コードとの照合表

| コード上の事実 | README への反映 |
|---|---|
| `models.yaml` の pinned SHA と `inference_dtype: bfloat16` | ルート 4.4、各 Stage のモデル節 |
| `sequence_likelihood.normalize_length: true`, `temperature: 1.0` | ルート 4.2、V1/V2/V3 設定表 |
| `--force` / `--all-layers` / `--batch-size` | ルート 4.7、実行節 |
| `run_production_v1.sh` が常に `--all-layers` | ルート 7.1、V1、`scripts/README.md` |
| 全 production script が `EXTRA_ARGS` を転送 | ルート 7.1、`scripts/README.md` |
| V1 Phase B を reader → self の 2 回実行 | ルート 7.3、V1 README / primary |
| Phase B 出力 `v1_phase_b/{task_type}/{prefix}/` | V1 ディレクトリ図と表 |
| V2 は family 未指定時のみ confirmatory 自動実行 | ルート 7.3、V2 README / primary |
| Behavioral 完了後に summarize 自動実行 | ルート 7.3、Behavioral README / primary |
| Behavioral 既定パスが `results/raw/` と `results/derived/` | Behavioral 実行例と集計表 |
| 本番 `--model-revision` 必須（registry 解決） | Behavioral / V1 |
| V3 dry-run ゲート `derived/dry_run/v3_gate_decision.json` | V3 3.2 / 7.3 / 9、primary |
| `min_sufficiency_slope: 0.1` と confirmatory frozen ブロック | V3 ゲート表と §6 |
| YAML `response_start` を実行キー `candidate_start` に正規化 | ルート 4.3、V3 §4 / 7.2 |
| `resolve_joint_stage_index("response_start") = cand_start - 1` | ルート 4.3、V3 7.2 |
| `PYTHONPATH` に `src/` を注入 | ルート 4.7 |

## 意図的にコードのまま書いた点

V3 の YAML 先頭段階は `response_start` だが、RQ2 / Confirmatory は実行前に `candidate_start` へ正規化する。そのため現行本番経路の第 1 段階 patch 位置は prompt_end ではなく候補先頭 token である。README は関数の特殊ケースと本番正規化の両方を書いた。コード変更はしていない。

## 残していないもの

- 実験結果の捏造
- `conference2.tex` の本文改稿
- GPU 本番実行
- 既存 `docs/readme_*` 履歴フォルダの上書き
