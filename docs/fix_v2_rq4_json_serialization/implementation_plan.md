# production_v2_20260922_210714.log エラー修正計画（最終確定版）

## 概要

`/mnt/nas/home/hiromi/src/emo2/results/logs/production_v2_20260922_210714.log` において、`OLMo` ファミリーに対する `v2/primary/run_rq4_recovery_patching.py`（分布回復パッチング）の計算終了後、結果を JSON ファイルへ保存する箇所で以下の例外が発生して異常終了しました。

```text
Traceback (most recent call last):
  File "/mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq4_recovery_patching.py", line 1014, in <module>
    main()
  File "/mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq4_recovery_patching.py", line 889, in main
    json.dump(res, f, indent=2)
...
  File "/usr/lib/python3.12/json/encoder.py", line 180, in default
    raise TypeError(f'Object of type {o.__class__.__name__} '
TypeError: Object of type int64 is not JSON serializable
```

本計画では、上流のデータ生成時における型正規化と、下流の JSON シリアライズ境界でのセーフガードを完全に整合させ、取りこぼしなく全箇所を網羅した実装を行います。研究ロジックや評価値計算には一切変更を加えず、入出力境界のシリアライズ安全性のみを高めます。

---

## 修正計画の詳細

### 1. `src/affective_empathy_eval/io.py`
1. **保守的な `json_serializable_default(o)` の定義**:
   ```python
   def json_serializable_default(o: Any) -> Any:
       if isinstance(o, np.integer):
           return int(o)
       if isinstance(o, np.floating):
           return float(o)
       if isinstance(o, np.bool_):
           return bool(o)
       if isinstance(o, np.ndarray):
           return o.tolist()
       if isinstance(o, Path):
           return str(o)
       raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
   ```
   - 意図しないオブジェクトを暗黙変換せず、NumPy スカラー/配列および `Path` のみに限定。
2. **`save_experiment_result()` への適用**:
   - `json.dump(envelope, tf, indent=indent, ensure_ascii=False, default=json_serializable_default)`
3. **`record_latest_run()` への適用**:
   - `json.dump(payload, f, indent=2, default=json_serializable_default)`

### 2. `src/affective_empathy_eval/manifests.py`
1. **`ManifestManager.save()` への適用**:
   ```python
   f.write(json.dumps(asdict(manifest), default=json_serializable_default) + "\n")
   ```
2. **`compute_string_or_dict_hash()` への適用**:
   ```python
   s = json.dumps(obj, sort_keys=True, default=json_serializable_default)
   ```
3. **`RunManifest.save()` への適用**:
   ```python
   json.dump(self.to_dict(), f, indent=2, default=json_serializable_default)
   ```

### 3. `v2/primary/run_rq4_recovery_patching.py`
1. **`permutation(N)` の明示的 `int()` キャスト（2箇所）**:
   - line 364 (`run_recovery_patching_for_task` 内):
     ```python
     perm = [int(x) for x in rng_split.permutation(N)]
     ```
   - line 638 (`run_recovery_patching_for_family` 内):
     ```python
     perm = [int(x) for x in rng_split.permutation(N)]
     ```
2. **`sample_idx` の明示的 `int(i)` キャスト（2箇所）**:
   - line 174 (dry-run 分岐):
     ```python
     "sample_idx": int(i),
     ```
   - line 525 (本番計算分岐):
     ```python
     "sample_idx": int(i),
     ```
3. **JSON 保存境界での防御（2箇所）**:
   - line 889 (`v2_recovery_{fam_id}.json`):
     ```python
     json.dump(res, f, indent=2, default=json_serializable_default)
     ```
   - line 1009 (`v2_distribution_recovery_summary.json`):
     ```python
     json.dump(summary_data, f, indent=2, default=json_serializable_default)
     ```

### 4. `tests/test_io_modular.py`
1. **NumPy型およびネスト構造の網羅的テスト**:
   - `np.int64`, `np.float64`, `np.bool_`, `np.ndarray`
   - ネストした `dict` / `list` 内に配置された NumPy 型
   - `save_experiment_result()` 経由で一時ディレクトリに保存し、`json.load()` 後に標準 Python 型（`int`, `float`, `bool`, `list`）として復元され完全一致することをアサート。

---

## 検証計画

### 1. ユニットテスト実行
- コマンド:
  ```bash
  python -m pytest tests/test_io_modular.py -v
  ```
- 期待結果: NumPy 型を含む全テストケースが PASS。

### 2. 回帰テスト（CPU Dry-run）
- コマンド:
  ```bash
  python v2/primary/run_rq4_recovery_patching.py \
    --models-config configs/models.yaml \
    --model-set primary_small \
    --family olmo \
    --dry-run \
    --device cpu
  ```
- 検証ポイント:
  - dry-run でも family JSON (`v2_recovery_olmo.json`)、modular JSON (`v2_rq4_recovery_patching_olmo.json`)、Manifest (`manifest_recovery_olmo.json`)、全ファミリー統合サマリー (`v2_distribution_recovery_summary.json`) の保存ステップが全て通過し、例外なく正常終了（exit code 0）すること。
  - GPU を一切使用しない。

---

## 実験結果の再計算について

- 今回のクラッシュは計算終了後の JSON 保存直前に発生したため、OLMo の計算メモリ（`res`）は保存されずに破棄されています。
- したがって、修正完了後に OLMo（および未実行のファミリー）の RQ4 を正式な結果として得るためには、GPU 環境での再計算が必要となります。
- CPU dry-run が exit code 0 で通過した段階でコード修正完了とし、GPU 再計算コマンドをご案内します。
