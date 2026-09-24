# 修正内容確認レポート (Walkthrough) - production_v2_20260925_051237.log エラー修正

## 1. 概要
本レポートは、`/mnt/nas/home/hiromi/src/emo2/results/logs/production_v2_20260925_051237.log` で発生した以下のエラーの原因調査、コード修正、および再検証結果をまとめたものです。

```text
Traceback (most recent call last):
  File "/mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq4_recovery_patching.py", line 1016, in <module>
    main()
  File "/mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq4_recovery_patching.py", line 934, in main
    self_max_matched = [all_recovery_results[f]["self"]["max_recovery_ratio_matched_plain"] for f in all_recovery_results]
                        ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^
KeyError: 'self'
```

---

## 2. 根本原因の特定

1. **エンベロープ（メタデータラッパー）の存在**:
   `affective_empathy_eval.io.save_experiment_result` は、実験結果を以下のエンベロープ形式で保存します。
   ```json
   {
     "execution_status": "success",
     "execution_success": true,
     "stage": "v2",
     "experiment_id": "v2_rq4_recovery_patching",
     "completed_at": "...",
     "metadata": { ... },
     "results": {
       "family_id": "qwen",
       "self": { ... },
       "reader": { ... }
     }
   }
   ```
2. **キャッシュ読み込み時のアンラップ漏れ**:
   `v2/primary/run_rq4_recovery_patching.py` のキャッシュ読込部分（行 850–855）において、`modular_rq4_path`（`v2_rq4_recovery_patching_{fam_id}.json`）が存在する場合、そのエンベロープ形式の辞書をそのまま `all_recovery_results[fam_id]` に代入していました。
3. **集約処理での KeyError**:
   直近の実行で、Llama ファミリーの再計算が完了し `all_recovery_results["llama"]` には生の計算結果が入っていましたが、Qwen, Gemma, OLMo はキャッシュから読み込まれ、トップレベルに `"self"` や `"reader"` が存在せず `"results"` の下にあったため、行 934 で `all_recovery_results[f]["self"]` を参照した際に `KeyError: 'self'` が発生しました。

---

## 3. 修正内容

[v2/primary/run_rq4_recovery_patching.py](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq4_recovery_patching.py) のキャッシュ読込ブロックにアンラップ処理を追加しました。

```diff
                     try:
                         read_p = modular_rq4_path if modular_rq4_path.exists() else out_path
                         with open(read_p, "r", encoding="utf-8") as f:
                             cached = json.load(f)
+                        if isinstance(cached, dict) and "results" in cached:
+                            cached = cached["results"]
                         logger.info(f"Loaded existing results matching manifest for {fam_id} from {read_p}. Skipping computation.")
                         all_recovery_results[fam_id] = cached
```

---

## 4. 検証結果

1. **RQ4 単独実行での検証**:
   - コマンド: `.venv/bin/python v2/primary/run_rq4_recovery_patching.py --models-config configs/models.yaml --model-set primary_small --device cpu`
   - 4ファミリー（Qwen, Llama, Gemma, OLMo）すべてのキャッシュ検証がパスし、計算処理はすべてスキップ。
   - クロスファミリー統合サマリー（Primary: Matched-Plain, Secondary: Direct Native, Mechanistic Control: Aligned）の Bootstrap CI およびタスク間比較が正常に完了。
   - `v2/results/derived/v2_distribution_recovery_summary.json` が正常に生成されました。
2. **エンドツーエンド（E2E）検証**:
   - コマンド: `bash scripts/run_production_v2.sh cuda:0`
   - 全ステージ（RQ1/RQ2 Cross-decoding, RQ3 Causal Map, RQ4 Recovery Patching, Confirmatory Analysis）がキャッシュ検証をパスし、再推論なし（所要時間45秒）で正常完了：
     ```text
     2026-09-25 06:21:19,662 [INFO] All recovery experiments completed! Summary saved to v2/results/derived/v2_distribution_recovery_summary.json
     2026-09-25 06:21:30,861 [INFO] Saved V2 Confirmatory Analysis report to v2/results/derived/v2_lmm_confirmatory.json
     2026-09-25 06:21:31,408 [INFO] All requested stages completed successfully!
     ==================================================================
     V2 Pipeline Evaluation completed successfully!
     Elapsed Time: 45 seconds
     ==================================================================
     ```

## 5. 生成・更新された成果物
- [v2/results/derived/v2_distribution_recovery_summary.json](file:///mnt/nas/home/hiromi/src/emo2/v2/results/derived/v2_distribution_recovery_summary.json): 4ファミリーの分布回復パッチング統合サマリー（Primary / Secondary / Control Bootstrap CI および paired 検定結果）
- [v2/results/derived/v2_recovery_sample_level_all.csv](file:///mnt/nas/home/hiromi/src/emo2/v2/results/derived/v2_recovery_sample_level_all.csv): 全4ファミリーのサンプルレベル回復記録（N=2400）
- [v2/results/derived/v2_lmm_confirmatory.json](file:///mnt/nas/home/hiromi/src/emo2/v2/results/derived/v2_lmm_confirmatory.json): Confirmatory Analysis レポート
- 修正コード: [v2/primary/run_rq4_recovery_patching.py](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq4_recovery_patching.py)

