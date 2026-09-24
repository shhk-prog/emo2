# タスク: production_v2_20260925_051237.log における RQ4 エラーの修正

## 状況・背景
`/mnt/nas/home/hiromi/src/emo2/results/logs/production_v2_20260925_051237.log` において、`v2/primary/run_rq4_recovery_patching.py` の実行終了直前に以下のエラーが発生して停止した。

```text
Traceback (most recent call last):
  File "/mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq4_recovery_patching.py", line 1016, in <module>
    main()
  File "/mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq4_recovery_patching.py", line 934, in main
    self_max_matched = [all_recovery_results[f]["self"]["max_recovery_ratio_matched_plain"] for f in all_recovery_results]
                        ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^
KeyError: 'self'
```

## 原因分析
- `v2/primary/run_rq4_recovery_patching.py` のキャッシュ読込処理（行 851–855）において、`is_experiment_completed` で既存結果を検証した後に読み込むファイルとして `modular_rq4_path`（`v2_rq4_recovery_patching_{fam_id}.json`）が優先されていた。
- `modular_rq4_path` は `affective_empathy_eval.io.save_experiment_result` によって保存されており、最上位階層が `{"execution_status": "success", "results": {...}}` というエンベロープ（メタデータラッパー）構造になっている。
- 一方、`all_recovery_results[fam_id]` には実験ペイロード（`{"family_id": ..., "self": ..., "reader": ...}`）が直接格納されていることを後続処理（行 934 以降のクロスファミリー統合集計）が前提としていた。
- キャッシュから読み込まれたファミリー（Qwen, Gemma, OLMo）はエンベロープ構造のまま格納されたため、`all_recovery_results[f]["self"]` で `KeyError: 'self'` が発生した。

## 目標
1. `v2/primary/run_rq4_recovery_patching.py` でキャッシュ読み込み時にエンベロープ（`"results"` キー）が存在する場合はアンラップして実験ペイロードを取得するように修正する。
2. 同様に `v2_recovery_{fam_id}.json`（生ペイロード保存版）からの読み込み優先、または `"results"` キー有無による安全なアンラップを実装する。
3. 修正後に RQ4 のサマリー集約処理が正常に完了し、`v2/results/derived/v2_distribution_recovery_summary.json` が生成されることを検証する。
4. 全体のパイプライン（または Unified Runner）が正常に通ることを確認する。

## タスクリスト
- [x] 原因箇所の特定と修正方針の策定（完了）
- [x] `task.md` および `implementation_plan.md` の作成（docs/fix_rq4_recovery_patching_keyerror/ に配置）
- [x] `v2/primary/run_rq4_recovery_patching.py` の修正
- [x] テスト・検証（キャッシュヒットにより再推論なしで高速に集約完了することを確認）
- [x] `walkthrough.md` の作成と結果報告
