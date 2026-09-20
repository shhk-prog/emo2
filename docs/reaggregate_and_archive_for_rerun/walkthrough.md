# 実装検証レポート: 再集計および各ステージのスキップ/再実行挙動の検証

## 1. 概要
ユーザーからの「`run_production_behavioral.sh`, `run_production_v1.sh`, `run_production_v2.sh`, `run_production_v3.sh` を実行すれば、既存結果があるものはスキップされ、再実行対象（`old_results` へ退避したもの）だけが再実行されるのではないか？」という質問に対し、各スクリプトのスキップ判定ロジックと現在のファイル配置を詳細に検証しました。

## 2. 各ステージのスキップ判定と実行挙動

| ステージ | スキップ判定の基準ファイル | 現状のファイル配置 | 実行時の挙動 |
|---|---|---|---|
| **Behavioral** | `behavioral/results/raw/emobank_3way/*_3way_vad.csv`<br>`behavioral/results/raw/aipsy_4split/*_aipsy_4split.csv` | 全8モデル分が完全存在 | **全件即座に `[SKIP]`**。<br>サマリー集計のみ実行されて完了（意図通り）。 |
| **V1** | Phase A: `v1/results/derived/v1_phase_a/*/manifest.json`<br>Phase B: `v1/results/derived/v1_phase_b/*/manifest.json`<br>Phase C: `v1/results/derived/v1_phase_c_prompt_end/*/manifest.json` | Phase B は退避済み（空）。<br>Phase A / C は既存4モデル分が存在。 | **Phase B は確実に再実行される（意図通り）**。<br>Phase C は既存モデルでスキップ。<br>※注意: 未実行モデルがある場合、Phase A も走る可能性があるため、Phase B 単独実行または全実行か選択。 |
| **V2** | RQ1/RQ2: `v2/results/raw/v2_geometry_*.json`<br>RQ3: `v2/results/raw/v2_causal_map_*.json`<br>RQ4: `v2/results/raw/v2_recovery_*.json` | RQ1/RQ2 は全4モデル存在。<br>RQ3 は復元が必要（復元後は存在）。<br>RQ4 は退避済み（空）。 | 復元スクリプト実行後：<br>**RQ1〜RQ3 は `[SKIP]` され、RQ4 のみ再実行される（意図通り）**。 |
| **V3** | RQ1: `v3/results/raw/v3_rq1_results.json`<br>`v3/results/derived/v3_gate_decision.json`<br>Gate 判定: `"GO"` か否か | RQ1 は復元が必要（復元後は存在）。<br>RQ2/RQ3/Confirmatory は退避済み。 | **超重要**: Gate 判定が `"NO_GO"` のため、そのままでは RQ1 終了時に `sys.exit(2)` で停止する。<br>**`--force-after-no-go` を付与することで、RQ1 がスキップされ RQ2/RQ3/Confirmatory が再実行される**。 |

## 3. 作成した補助スクリプト
- `scripts/restore_non_rerun_results.sh`:
  再実行が不要なファイル（V2 RQ3 の Causal Map、V3 RQ1 の State Induction / Gate Decision）を `old_results` から正規の results ディレクトリへ復元するスクリプト。
