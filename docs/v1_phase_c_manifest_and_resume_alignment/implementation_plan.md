# V1 Phase C / E6 マニフェスト & 再開機構 整合化 実装計画

本番実行を完全に堅牢かつ再現可能とするため、ご指摘いただいた V1 Phase C および E6 の manifest / resume 周りの3点の不整合を解消します。

---

## ユーザー確認事項
- 統計解析、モデル比較、介入ロジック本体には一切手を加えず、マニフェストおよび途中チェックポイントのメタデータ整合化のみを実施します。
- これにより、途中再開（resume）およびキャッシュ整合性判定が完全一致するようになります。

---

## 提案する変更内容

### 1. V1 Phase C チェックポイントメタデータへの `code_version` 追加
#### [MODIFY] [run_phase_c.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_c.py)
- `from affective_empathy_eval.manifests import DEFAULT_CODE_VERSION` を使用。
- `make_phase_c_checkpoint_manifest()` の返り値辞書に `"code_version": DEFAULT_CODE_VERSION` を追加。
- `compute_cache_metadata()` の返り値辞書に `"code_version": DEFAULT_CODE_VERSION` を追加。

---

### 2. V1 Phase C マニフェスト情報の統一
#### [MODIFY] [run_phase_c.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_c.py)
- `manifest_config` の `candidate_schema` を `"VA_81"` から `"VAD_729"` に修正。
- `is_manifest_matching()` 呼び出し時に `expected_intervention_version="v1_phase_c_v2"` を指定。
- スクリプト末尾の `create_run_manifest()` の引数を以下のように統一：
  ```python
  manifest = create_run_manifest(
      run_type="v1_phase_c",
      model_name=args.model_id,
      model_revision=args.model_revision or "main",
      config=manifest_config,
      dataset_path=str(data_file),
      dataset_hash=dataset_hash,
      candidate_space="VAD_729",
      measurement_space="VA_expectation_from_VAD_729",
      intervention_version="v1_phase_c_v2",
      actual_dtype=str(actual_torch_dtype).replace("torch.", ""),
      run_id=args.run_id,
      dry_run=args.dry_run,
  )
  ```

---

### 3. V1 Phase C E6 (run_e6_specialization.py) のマニフェスト / 再開機構の統一
#### [MODIFY] [run_e6_specialization.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/phase_c/run_e6_specialization.py)
- E3 CSV の確認および Reader/Self レイヤー確定処理（`select_sites_from_e3()`）を early skip チェックより前に移動。
- `e3_hash = compute_file_hash(e3_csv_path) if os.path.exists(e3_csv_path) else "unknown"` を計算。
- レイヤーを確定させた上で、統一された `manifest_config` を一度だけ構築：
  ```python
  manifest_config = {
      "model_prefix": args.model_prefix,
      "model_id": args.model_id,
      "model_revision": args.model_revision or "main",
      "reader_layer": args.reader_layer,
      "self_layer": args.self_layer,
      "site_selection_method": site_selection_method,
      "ablation_type": args.ablation_type,
      "split_eval": args.split_eval,
      "split_seed": args.split_seed,
      "limit": args.limit,
      "dataset_hash": dataset_hash,
      "e3_hash": e3_hash,
  }
  ```
- early skip で `is_manifest_matching()` を判定。
- 末尾の `create_run_manifest()` で `config=manifest_config`, `dataset_hash=dataset_hash` を渡して保存。

---

### 4. 単体テストの追加
#### [MODIFY] [test_pre_production_fixes.py](file:///mnt/nas/home/hiromi/src/emo2/tests/test_pre_production_fixes.py)
- `test_v1_phase_c_checkpoint_metadata_code_version()`:
  - `make_phase_c_checkpoint_manifest` および `compute_cache_metadata` が `code_version == DEFAULT_CODE_VERSION` を含むことを検証。
- `test_v1_e6_manifest_config_consistency()`:
  - E6 の `manifest_config` が構築・保存されるキーの一貫性を検証。

---

## 検証手順

1. 単体テストの実行:
   ```bash
   PYTHONPATH=src:. pytest -q -k "phase_c or e6"
   ```
2. 全単体テストの実行:
   ```bash
   PYTHONPATH=src:. pytest -q
   ```
3. V1 dry-run の再確認:
   ```bash
   bash scripts/run_production_v1.sh cpu --dry-run
   ```
