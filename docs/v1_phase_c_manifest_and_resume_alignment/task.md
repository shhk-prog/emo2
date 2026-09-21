# タスクリスト: V1 Phase C / E6 マニフェスト & 再開機構の整合化 (V1 Phase C Manifest and Resume Alignment)

## 状況サマリー
科学的実験設計、統計モデル、全ステージの実行パスはすべて GO 判定を得ている。
本番フル実行前に、V1 Phase C および E6 における manifest / checkpoint / resume 関連の不整合（3点）を解消し、再現性と再開安全性を完全にする。

---

## タスク一覧

### 1. V1 Phase C チェックポイントメタデータへの `code_version` 追加
- [x] `v1/primary/run_phase_c.py`:
  - [x] `make_phase_c_checkpoint_manifest()` に `"code_version": DEFAULT_CODE_VERSION` を追加
  - [x] `compute_cache_metadata()` に `"code_version": DEFAULT_CODE_VERSION` を追加

### 2. V1 Phase C マニフェスト情報の統一
- [x] `v1/primary/run_phase_c.py`:
  - [x] `manifest_config` 内の `"candidate_schema": "VA_81"` を `"candidate_schema": "VAD_729"` に修正
  - [x] `is_manifest_matching()` に `expected_intervention_version="v1_phase_c_v2"` を追加
  - [x] 末尾の `create_run_manifest()` の引数を `config=manifest_config`, `dataset_path=str(data_file)`, `dataset_hash=dataset_hash`, `intervention_version="v1_phase_c_v2"` に統一

### 3. V1 Phase C E6 (run_e6_specialization.py) の マニフェスト / 再開機構の統一
- [x] `v1/primary/phase_c/run_e6_specialization.py`:
  - [x] E3 CSV 読み込みおよび Reader/Self site 決定（`r_l, s_l, site_selection_method`）の後に `e3_hash = compute_file_hash(e3_csv_path)` を算出
  - [x] `args.reader_layer = r_l`, `args.self_layer = s_l` を確定させた上で、統一された `manifest_config` を構築
  - [x] 先頭の early skip チェックと末尾の `create_run_manifest()` で同一の `manifest_config` および `dataset_hash=dataset_hash` を使用するよう統一

### 4. 単体テストの追加と検証
- [x] `tests/test_pre_production_fixes.py`:
  - [x] `make_phase_c_checkpoint_manifest` および `compute_cache_metadata` に `code_version` が含まれることを検証するテスト
  - [x] Phase C および E6 の manifest 生成スキーマ検証テスト
- [ ] `PYTHONPATH=src:. pytest -q`
- [ ] `docs/v1_phase_c_manifest_and_resume_alignment/walkthrough.md` の作成
