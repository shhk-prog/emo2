# 実装計画: 本番実行前監査指摘事項の完全修正 (Pre-run Audit Fixes)

ユーザー監査で指摘された P0（実行ブロック・重要バグ）、P1（論文品質・再現性・整合性）、P2（安全性・堅牢性）の計16項目を体系的に修正・検証します。

---

## 修正項目と対象ファイル

### 1. 【P0】`v1/primary/run_phase_b.py` の `--force` 重複削除 & テスト追加
- **対象**: `v1/primary/run_phase_b.py`, `tests/test_production_entrypoints.py`
- **内容**: 重複している2つ目の `--force` 引数定義を削除。`test_production_entrypoints.py` に `run_phase_b.py --help` が exit code 0 になるテストを追加。

### 2. 【P0】Behavioral / V1 への HF revision SHA 伝播 & スクリプト側での安全弁
- **対象**: `src/affective_empathy_eval/run.py`, `v1/primary/run_phase_a.py`, `v1/primary/run_phase_b.py`, `v1/primary/run_phase_c.py`, `v1/primary/phase_c/run_e6_specialization.py`
- **内容**: 
  - `run.py` の `run_behavioral` と `run_v1` で `variants` を `(model_spec, ...)` とし、`--model-revision model_spec.revision` を明示的に渡す。
  - V1 各スクリプトで、もし `model_revision is None and not args.dry_run` の場合、`registry.get_family_by_model_id()` から revision を逆引き補完し、見つからなければ例外送出。

### 3. 【P1】Behavioral EmoBank の dtype を `bfloat16` に統一
- **対象**: `behavioral/primary/run_behavioral_emobank.py`, `src/affective_empathy_eval/run.py`
- **内容**: `run_behavioral_emobank.py` の `--dtype` デフォルトを `bfloat16` に変更。`run.py` からも `--dtype cfg.inference_dtype` を明示的に渡す。

### 4. 【P1】V1/V2/V3 のキャッシュ判定における `is_manifest_matching` 厳密照合
- **対象**: `v1/primary/run_phase_c.py`, `v1/primary/phase_c/run_e6_specialization.py`, `v2/primary/run_rq1_rq2_cross_decoding.py`, `v2/primary/run_rq3_causal_map.py`, `v2/primary/run_rq4_recovery_patching.py`, `v3/primary/run_rq1_state_induction.py`, `v3/primary/run_rq2_spatiotemporal_maps.py`, `v3/primary/run_rq3_path_mediation.py`, `v3/primary/run_confirmatory_replication.py`
- **内容**: 単なる `is_experiment_completed` だけでなく、`is_manifest_matching` に `expected_config_hash`, `expected_dataset_hash`, `expected_model_revision` を渡して厳密照合。

### 5. 【P1】`create_run_manifest` への `dataset_hash` 引数追加
- **対象**: `src/affective_empathy_eval/manifests.py`, `v1/primary/run_phase_a.py`
- **内容**: `create_run_manifest()` に `dataset_hash: Optional[str] = None` を追加し、Phase A の EmoBank + AIPsy 複合ハッシュをそのまま manifest に保存・照合できるようにする。

### 6. 【P1】V2 manifest への Base/Instruct revision 保存とキャッシュハッシュ統合
- **対象**: `v2/primary/run_rq1_rq2_cross_decoding.py`, `v2/primary/run_rq3_causal_map.py`, `v2/primary/run_rq4_recovery_patching.py`
- **内容**: `manifest_config` に `base_model_id`, `base_revision`, `instruct_model_id`, `instruct_revision` を保存し、ハッシュ計算に含める。

### 7. 【P1】V3 manifest の candidate_space / measurement_space 表記を `VA_81` へ修正
- **対象**: `v3/primary/run_rq1_state_induction.py`, `v3/primary/run_rq2_spatiotemporal_maps.py`, `v3/primary/run_rq3_path_mediation.py`, `v3/primary/run_confirmatory_replication.py`
- **内容**: `candidate_space="VA_81"`, `measurement_space="VA_81"` に修正。

### 8. 【P1】V1 Phase C / E6 / Behavioral の measurement_space 表記適正化
- **対象**: `v1/primary/run_phase_c.py`, `v1/primary/phase_c/run_e6_specialization.py`, `behavioral/primary/run_behavioral_emobank.py`, `behavioral/primary/run_behavioral_aipsy.py`
- **内容**:
  - V1 Phase C / E6: `measurement_space="VA_expectation_from_VAD_729"`
  - Behavioral: `measurement_space="VAD_expectation_from_VAD_729"`

### 9. 【P1】AIPsy / EmoBank の candidate hash を 729 候補全体に修正
- **対象**: `behavioral/primary/run_behavioral_aipsy.py`, `behavioral/primary/run_behavioral_emobank.py`
- **内容**: `get_vad_candidates_and_triplets()` から取得した 729 候補すべての JSON 文字列リストから SHA256 を計算。

### 10. 【P1】`is_manifest_matching()` の `expected_intervention_version` デフォルトを `None` に変更
- **対象**: `src/affective_empathy_eval/manifests.py`
- **内容**: `expected_intervention_version: Optional[str] = None` とし、明示指定された場合のみ検証。

### 11. 【P1】V1 Phase B dry-run スキーマの `n_validated_*` 削除・統一
- **対象**: `v1/primary/run_phase_b.py`
- **内容**: `n_validated_*` を削除し、実 run と同様のキー（`n_nonfallback_*`）に統一。

### 12. 【P1】`configs/v3_experiments.yaml` の旧 `temporal_relative_depth` 削除・V/A 分離統一
- **対象**: `configs/v3_experiments.yaml`
- **内容**: 旧単一キーを削除し、`temporal_relative_depth_v: 0.65`, `temporal_relative_depth_a: 0.65`, `temporal_stage_v: "pre_V"`, `temporal_stage_a: "pre_A"` に統一。

### 13. 【P1】V2/V3 の dtype を registry (`fam_cfg.inference_dtype`) 参照に統一
- **対象**: `src/affective_empathy_eval/models/registry.py`, `v2/primary/run_rq1_rq2_cross_decoding.py`, `v2/primary/run_rq3_causal_map.py`, `v2/primary/run_rq4_recovery_patching.py`
- **内容**: `resolve_torch_dtype` ヘルパーを活用し、`fam_cfg.inference_dtype` を正しく参照。`getattr(fam_cfg, "dtype", ...)` を修正。

### 14. 【P2】`src/affective_empathy_eval/extraction.py` の mock extractor を決定論的 sha256 シード化
- **対象**: `src/affective_empathy_eval/extraction.py`
- **内容**: `abs(hash(...))` を `hashlib.sha256(text.encode()).digest()` から 4 バイト int に変換する形に修正。

### 15. 【P2】AIPsy の stimuli path 存在確認と明示的 FileNotFoundError 送出
- **対象**: `behavioral/primary/run_behavioral_aipsy.py`
- **内容**: EmoBank 同様、`stim_path` および `fallback` が存在しない場合は明示的に `FileNotFoundError` を raise。

### 16. 【P2】`v1/.env` の削除と漏洩防止確認
- **対象**: `v1/.env`
- **内容**: 存在する場合は削除。

---

## 検証計画
1. `python -m compileall behavioral v1 v2 v3 src scripts`
2. `pytest -q tests/test_production_entrypoints.py`
3. 重点 pytest群（`test_likelihood.py`, `test_v1_refinements.py`, `test_v3_prerun_fixes.py` など）
4. `python -m affective_empathy_eval.run --stage all --model-set primary_small --family qwen --device cpu --dry-run --max-samples 16 --force`
