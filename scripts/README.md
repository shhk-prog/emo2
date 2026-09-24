# Production Scripts

リポジトリルートの `scripts/` は、本番ラッパ・再集計・論文表の入口である。実験本体は `python -m affective_empathy_eval.run` と各 `*/primary/` にある。論文 19 列のスキーマは `src/affective_empathy_eval/paper_summary/`。

## 本番ランナー

Primary 4 本は `.venv` を activate し、`results/logs/production_<stage>_TIMESTAMP.log` に tee する。第1引数が device（既定 `cuda:0`）。第2引数以降は `EXTRA_ARGS` として統合 CLI へそのまま転送する。

統合 CLI 自身も `resolve_log_dir` へ `run_{stage}_TIMESTAMP.log` を書く。`primary_small` は `results/logs/`、それ以外の model set は `results/ablation/{model_set}/logs/`。`--dry-run` はそれぞれの `dry_run/`。

| Script | 固定で付く CLI 引数 | 備考 |
|---|---|---|
| `run_production_behavioral.sh` | `--stage behavioral --model-set primary_small` | 完了後の Behavioral 要約は CLI 側が自動実行。19 列集計は自動では走らない |
| `run_production_v1.sh` | `--stage v1 --model-set primary_small --all-layers` | Phase B 統制 CSV が無ければ先に生成 |
| `run_production_v2.sh` | `--stage v2 --model-set primary_small` | family 未指定なので confirmatory も走る。成果物は `v2/results/` |
| `run_production_v3.sh` | `--stage v3 --model-set primary_small` | RQ1 の `decision` が完全一致の `GO` でないと終了コード 2 |
| `run_production_all.sh` | 上記 4 本を Behavioral → V1 → V2 → V3 | 同じ `EXTRA_ARGS` を全 Stage へ転送 |
| `run_production_scale_ablation.sh` | `--stage v2 --model-set <第1引数>` | 第1引数は `scale_3b` または `scale_7b`。第2引数が device。ログは `results/ablation/${MODEL_SET}/logs/production_v2_TIMESTAMP.log`。成果物は `results/ablation/${MODEL_SET}/` |

例:

```bash
bash scripts/run_production_behavioral.sh cuda:0
bash scripts/run_production_v1.sh cuda:0 --force
bash scripts/run_production_v3.sh cuda:0 --force-after-no-go
bash scripts/run_production_scale_ablation.sh scale_3b cuda:0
bash scripts/run_production_scale_ablation.sh scale_7b cuda:0 --family qwen
```

`--force` はキャッシュ再計算、`--force-after-no-go` は V3 ゲート継続である。混ぜない。

`--stage scale_validation` は `run_scale_validation.py` を呼ぶ。転送されるのは `--device` / `--dry-run` / `--max-samples` だけである。既定 model set は Mistral の `scale_validation` で、RQ4 まで。confirmatory は呼ばない。3B/7B や confirmatory 付きの外部コホートは `--stage v2 --model-set ...` を使う。

## 論文集計

実験を再実行しない。既存 derived を読む。

| Script | 役割 | 既定出力 |
|---|---|---|
| `build_all_paper_summaries.py` | Behavioral → V1 → V2 → V3 の 19 列マスター。`--strict` で欠落・スキーマ失敗を落とす | `results/derived/paper_summary/` |
| `generate_paper_results_tables.py` | 下の 5 本を順に呼び、LaTeX と Markdown を書く | `iclr2027/tables` |
| `summarize_behavioral_emobank.py` | Table B1 と B5 から EmoBank 表 | `--out-dir`（マスター経由では `iclr2027/tables`） |
| `summarize_behavioral_aipsy.py` | Table B2〜B5 から AIPsy 表 | 同上 |
| `summarize_v1_internal_sharing.py` | Table V1-1〜V1-6 | 同上 |
| `summarize_v2_reorganization.py` | V2 confirmatory / relocation / controls / LMM / recovery | 同上 |
| `summarize_v3_causal_utilization.py` | V3 gate / 4-map / attenuation / confirmatory | 同上 |

```bash
python scripts/build_all_paper_summaries.py --strict
python scripts/generate_paper_results_tables.py --repo-root . --out-dir iclr2027/tables
```

`scripts/summarize_behavioral_*.py` と `behavioral/analysis/summarize_behavioral_*.py` は別名である。analysis 側は生 CSV の統計、scripts 側は論文 CSV からの LaTeX である。

## 感度分析とスケール検証

| Script | 役割 |
|---|---|
| `run_candidate_space_sensitivity.py` | 同一刺激での 729 VAD vs 81 VA。既定データは AIPsy、`--task {reader,self}`、`--n-samples 20`、出力 `results/derived/candidate_space_sensitivity`。`--model-revision` 未指定なら registry から解決 |
| `run_scale_validation.py` | Mistral 7B の V2 RQ1〜RQ4。既定 `--model-set scale_validation`。confirmatory は含まない。成果物は `resolve_output_dirs` により `results/ablation/scale_validation/` |
| `run_production_reruns.sh` | 監査後の部分再計算用。主本番経路ではない |

## 退避と再集計

| Script | 役割 |
|---|---|
| `clear_stage_results.sh` | 結果を削除せず `archive/<timestamp>/` へ移し、`.gitkeep` を残す |
| `archive_stage_results.sh` | 上記の別名 |
| `archive_targets_for_rerun.sh` | 指定ステージを `old_results/archive_20260921_audit/` へ移す |
| `restore_non_rerun_results.sh` | V2 RQ3 と V3 RQ1 を `old_results` から戻す |
| `reaggregate_all.sh` | GPU 推論なしで V1 Phase A と V2 RQ1/RQ2 を再集計 |
| `reaggregate_v1_phase_a.py` / `reaggregate_v2_summary.py` | 上記の個別再集計 |
| `repair_e3_causal_map.py` | E3 causal map の修復 |
| `archive_and_clean_results.py` / `cleanup_project_files.py` / `cleanup_redundant_files.py` | 退避と整理。主解析の入口ではない |

探索用の旧スクリプトは各 Stage の `*/scripts/legacy/` にある。主解析に使わない。各 Stage の `build_paper_summary.py` は legacy ではなく、19 列 presentation の正本である。
