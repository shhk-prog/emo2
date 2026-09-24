# 実装計画: RQ4 Recovery Patching キャッシュ読込時のエンベロープ解除と統合集約エラー修正

## 1. 修正対象ファイル
- `v2/primary/run_rq4_recovery_patching.py`

## 2. 変更内容の詳細

### 2.1 キャッシュ読込処理の堅牢化 (行 850–863 付近)
**現状:**
```python
read_p = modular_rq4_path if modular_rq4_path.exists() else out_path
with open(read_p, "r", encoding="utf-8") as f:
    cached = json.load(f)
logger.info(f"Loaded existing results matching manifest for {fam_id} from {read_p}. Skipping computation.")
all_recovery_results[fam_id] = cached
```

**問題点:**
`modular_rq4_path`（`v2_rq4_recovery_patching_{fam_id}.json`）は `save_experiment_result()` で生成されており、構造が次のようになっている：
```json
{
  "execution_status": "success",
  "execution_success": true,
  "stage": "v2",
  "experiment_id": "v2_rq4_recovery_patching",
  "results": {
    "family_id": "...",
    "self": { ... },
    "reader": { ... }
  }
}
```
そのため、`cached` をそのまま `all_recovery_results[fam_id]` に入れると、`all_recovery_results[fam_id]["self"]` で `KeyError: 'self'` となる。

**修正案:**
`cached` が辞書型かつ `"results"` キーを持つ場合、`cached["results"]` を抽出して格納する。
```python
read_p = modular_rq4_path if modular_rq4_path.exists() else out_path
with open(read_p, "r", encoding="utf-8") as f:
    cached = json.load(f)
if isinstance(cached, dict) and "results" in cached:
    cached_payload = cached["results"]
else:
    cached_payload = cached
logger.info(f"Loaded existing results matching manifest for {fam_id} from {read_p}. Skipping computation.")
all_recovery_results[fam_id] = cached_payload
```

### 2.2 同様の箇所の確認
`run_rq3_causal_map.py` や他のスクリプトでも同様に `save_experiment_result` のエンベロープを扱う際に問題がないか点検する。
（RQ3 では `all_causal_results` の内部キーを直接参照していないが、一貫性のために確認）

## 3. 検証手順
1. `v2/primary/run_rq4_recovery_patching.py` のコード修正。
2. 仮想環境 (`.venv/bin/python`) を使用し、テスト実行：
   ```bash
   .venv/bin/python v2/primary/run_rq4_recovery_patching.py --models-config configs/models.yaml --model-set primary_small --device cpu
   ```
   ※ 4ファミリー（Qwen, Llama, Gemma, OLMo）すべての raw 結果および manifest が揃っているため、計算処理はすべてスキップされ、数秒でキャッシュ読み込みと集約処理（Bootstrap CI, paired t-test / wilcoxon）が完了する。GPU を使わずに CPU で安全に検証可能。
3. `v2/results/derived/v2_distribution_recovery_summary.json` が正常に出力されることを確認。
4. Unified runner またはパイプライン全体のステータス確認。
5. `walkthrough.md` の作成。
