# V1 Phase C / E6 マニフェスト & 再開機構 整合化 完了報告 (Walkthrough)

本番全実行を完全に堅牢かつ再現可能とするため、ご指摘いただいた V1 Phase C および E6 の manifest / resume 周りの3点の不整合を解消しました。

---

## 1. 実施した修正の詳細

### ① V1 Phase C の途中 checkpoint への `code_version` 追加
- **問題点**: E3/E4 の途中 checkpoint や cache metadata に `code_version` が記録されておらず、通常の `manifest.json` と異なりコードバージョン変更（2.2.0 → 2.3.0）を途中チェックポイント側で検知できない状態であった。
- **実施内容**:
  - [run_phase_c.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_c.py):
    - `DEFAULT_CODE_VERSION` をインポート。
    - `make_phase_c_checkpoint_manifest()` の返り値辞書に `"code_version": DEFAULT_CODE_VERSION` を追加。
    - `compute_cache_metadata()` の返り値辞書に `"code_version": DEFAULT_CODE_VERSION` を追加。

---

### ② V1 Phase C マニフェスト情報の統一
- **問題点**: 
  - `manifest_config` の `candidate_schema` が `"VA_81"` と記述されていたが、実際の Phase C は `build_vad_candidates()` による 729 候補（`"VAD_729"`）であった。
  - 末尾の `create_run_manifest()` が別の辞書を再構築していたため、`dataset_hash` や `intervention_version` が記録されず、dry-run 等で `"none"` / `"unknown"` になっていた。
  - `is_manifest_matching()` で `expected_intervention_version` が検証されていなかった。
- **実施内容**:
  - [run_phase_c.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_c.py):
    - `manifest_config` の `candidate_schema` を `"VAD_729"` に修正。
    - `is_manifest_matching()` に `expected_intervention_version="v1_phase_c_v2"` を追加。
    - 末尾の `create_run_manifest()` に `config=manifest_config`, `dataset_path=str(data_file)`, `dataset_hash=dataset_hash`, `intervention_version="v1_phase_c_v2"` を渡すように統一。

---

### ③ V1 Phase C E6 (run_e6_specialization.py) のマニフェスト / 再開機構の統一
- **問題点**: 
  - early cache check 用の `manifest_config`（先頭）と、末尾で保存する `config` のキー構成が不一致であった。
  - E3 CSV から動的決定された layer や `site_selection_method`、`e3_hash` が early check の `manifest_config` に含まれておらず、正常終了後でも次回の manifest 一致判定が失敗していた。
- **実施内容**:
  - [run_e6_specialization.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/phase_c/run_e6_specialization.py):
    - E3 CSV 読み込みおよび Reader/Self site 決定処理を early skip より前に移動。
    - `e3_hash = compute_file_hash(e3_csv_path) if os.path.exists(e3_csv_path) else "unknown"` を計算。
    - `args.reader_layer`, `args.self_layer` を確定させた上で、統一された `manifest_config` を一度だけ構築。
    - early skip（`is_manifest_matching`）と末尾の `create_run_manifest()` の両方で同一の `manifest_config` および `dataset_hash=dataset_hash` を使用するように統一。

---

### ④ 単体テストの追加
- **追加内容 ([test_pre_production_fixes.py](file:///mnt/nas/home/hiromi/src/emo2/tests/test_pre_production_fixes.py))**:
  - `test_v1_phase_c_checkpoint_metadata_code_version`: `make_phase_c_checkpoint_manifest` および `compute_cache_metadata` が `code_version == DEFAULT_CODE_VERSION`（`"2.3.0"`）を含むことを検証。
  - `test_v1_phase_c_manifest_config_schema`: Phase C の `candidate_schema` が `"VAD_729"` であること、および E6 の `manifest_config` が `e3_hash` を含み一貫して使用されていることを静的検証。

---

## 2. ターミナルでの確認手順

```bash
source .venv/bin/activate

# 1. 新規単体テストの実行
PYTHONPATH=src:. pytest -q -k "phase_c"

# 2. 全単体テストの実行
PYTHONPATH=src:. pytest -q

# 3. V1 dry-run の再実行（Phase A/B/C/E6 の完走確認）
bash scripts/run_production_v1.sh cpu --dry-run
```

確認完了後、本番全実行（GPU環境、初回 `--force` 付き）へ進んでください。
