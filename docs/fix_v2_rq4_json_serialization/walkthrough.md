# production_v2_20260922_210714.log エラー修正完了レポート (Walkthrough)

## 概要

`results/logs/production_v2_20260922_210714.log` で発生した以下のエラーに対する修正と検証を完了しました。

```text
TypeError: Object of type int64 is not JSON serializable
```

上流のデータ生成時における型正規化（`int()` キャスト）と、下流の JSON 境界（`io.py`, `manifests.py`, `run_rq4_recovery_patching.py`）でのフォールバックセーフガード（`json_serializable_default`）を網羅的に施すことで、同様のクラッシュを恒久的に防ぎました。

---

## 修正内容

### 1. `src/affective_empathy_eval/io.py`
- [io.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/io.py)
  - 保守的な `json_serializable_default(o)` を実装：
    - `np.integer` → `int(o)`
    - `np.floating` → `float(o)`
    - `np.bool_` → `bool(o)`
    - `np.ndarray` → `o.tolist()`
    - `Path` → `str(o)`
    - それ以外の未対応型は `TypeError` を送出
  - `save_experiment_result()` の `json.dump` に `default=json_serializable_default` を適用。
  - `record_latest_run()` の `json.dump` に `default=json_serializable_default` を適用。

### 2. `src/affective_empathy_eval/manifests.py`
- [manifests.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/manifests.py)
  - `ManifestManager.save()` の `json.dumps` に `default=json_serializable_default` を追加。
  - `compute_string_or_dict_hash()` の `json.dumps` に `default=json_serializable_default` を追加。
  - `RunManifest.save()` の `json.dump` に `default=json_serializable_default` を追加。

### 3. `v2/primary/run_rq4_recovery_patching.py`
- [run_rq4_recovery_patching.py](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq4_recovery_patching.py)
  - `permutation(N)` の明示的 `int()` キャスト（line 364 & line 638）：
    `perm = [int(x) for x in rng_split.permutation(N)]`
  - `"sample_idx": int(i)` の明示的キャスト（dry-run line 174 & full run line 525）
  - ファミリー結果保存（line 889）の `json.dump(res, f, indent=2, default=json_serializable_default)`
  - 統合サマリー保存（line 1009）の `json.dump(summary_data, f, indent=2, default=json_serializable_default)`

### 4. `tests/test_io_modular.py`
- [test_io_modular.py](file:///mnt/nas/home/hiromi/src/emo2/tests/test_io_modular.py)
  - `test_save_experiment_result_with_numpy_types` を追加：
    - `np.int64`, `np.float64`, `np.bool_`, `np.ndarray`
    - ネストした `dict` / `list` 内の NumPy 型
    - `metadata` 内の `Path` および `np.int64`
    - `save_experiment_result()` で保存後、`json.load()` で標準 Python 型として復元されることを検証。

---

## 検証結果

### 1. 単体テスト (pytest)
```bash
.venv/bin/python -m pytest tests/test_io_modular.py -v
```
**結果: PASS (2 passed in 11.15s)**
- `test_save_and_check_experiment_result` PASSED
- `test_save_experiment_result_with_numpy_types` PASSED

### 2. 回帰テスト (CPU Dry-run)
```bash
.venv/bin/python v2/primary/run_rq4_recovery_patching.py \
  --models-config configs/models.yaml \
  --model-set primary_small \
  --family olmo \
  --dry-run \
  --device cpu
```
**結果: 正常終了 (exit code 0)**
- `v2_recovery_samples_olmo.csv` 保存成功
- `v2_rq4_recovery_patching_olmo.json` 保存成功
- `v2_recovery_olmo.json` 保存成功
- `v2_recovery_sample_level_all.csv` 保存成功
- `v2_distribution_recovery_summary.json` 保存成功
- クラッシュ地点（CSV 保存直後の JSON dump）を完全に通過し、正常にパイプラインが完走することを確認。

### 3. 関連テストスイート実行
```bash
.venv/bin/python -m pytest tests/test_io_modular.py tests/test_execution_skip_caching.py tests/test_confirmatory_pipeline.py -q
```
**結果: PASS (10 passed in 6.36s)**

---

## 今後の実行手順（GPU 再計算）

今回のクラッシュは計算終了後の JSON 保存直前に発生したため、OLMo の計算済みインメモリデータは破棄されています。
本番の OLMo 結果を正式に得るためには、GPU 環境での再実行が必要です。

```bash
# OLMo の RQ4 再実行コマンド例
./scripts/run_production_v2.sh cuda:0 --family olmo
```

> [!NOTE]
> RQ1/RQ2 および RQ3 の結果・マニフェスト（`v2_rq1_decodability_preservation_olmo.json`, `v2_causal_map_olmo.json` 等）は既に正常保存されており、マニフェストキャッシュにより自動スキップ（数秒で通過）されるため、RQ4 のみ最初から実行されます。
