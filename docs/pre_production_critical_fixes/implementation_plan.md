# 最終Production実行前の必須修正 実装計画 (Pre-production Critical Fixes)

本計画は、実験の最終production runを開始する前に残されている、測定妥当性・再現性・Provenance（追跡可能性）・データ完全性に関わる必須修正項目（1〜10）および推奨改善項目を漏れなく確実に解消するための設計・実装手順を定めたものです。

---

## ユーザー確認事項 (User Review Required)

> [!IMPORTANT]
> **1. モデルRevisionの指定方針**  
> `configs/models.yaml` の `base_revision` / `instruct_revision` に指定するコミットSHAについて：  
> オフラインまたは既存のローカルキャッシュ環境で安全にロードできるよう、各モデルの最新固定コミットSHA（または既にキャッシュされているスナップショットSHA）を反映します。各モデルのPrimary runner CLIにも `--model-revision` を追加し、指定がない場合は設定ファイルの値を継承します。
> 
> **2. Results ディレクトリと run_id 体系の扱い**  
> `AGENTS.md` の「results/rawは追記専用」に準拠し、各実験スクリプトで `--run-id` を受け入れ、指定時や production 実行時に `results/raw/<run_id>/` に直接出力できるように拡張します。同時に、既存のフラットなパス（`results/raw/foo.json`）を参照する既存ツールや可視化スクリプトとの後方互換性のため、`results/latest.json` や最新結果へのシンボリックリンク/コピーを維持します。また、既存結果の誤上書き防止のため、run開始時に既存結果がある場合はバックアップディレクトリへ自動退避（archive）する安全策を組み込みます。
> 
> **3. 推論 dtype の統一方針**  
> `configs/models.yaml` の `inference_dtype: "bfloat16"` に統一し、V1/V3 の `torch.float16` ハードコードを解消します。GPU環境で bfloat16 が利用可能な場合は bfloat16 を使用し、CPU等の環境では float32 へ安全にフォールバックさせ、manifest に `actual_dtype` を必ず保存します。

---

## 修正対象コンポーネントと詳細設計

---

### 1. V1 Hidden-State 異常値置換の削除 (項目 1)

#### [MODIFY] [run_phase_a.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_a.py)
- **activation抽出後 (line 158付近)**:
  `np.nan_to_num(arr, ...)` および `np.clip(arr, -1e4, 1e4)` を完全削除。
  非有限値（NaN, Inf）が含まれている場合は `FloatingPointError(f"Non-finite hidden states detected: {bad_count}")` を送出。
  最大絶対値をログ出力 (`logger.info("Hidden-state max abs = %.6f", max_abs)`) し、`arr = arr.astype(np.float32)` のみを行う。Primary で clip は行わない。
- **`evaluate_cross_decoding_and_geometry()` (line 405付近)**:
  `H_R, H_S` に対する `np.clip(np.nan_to_num(...))` を削除。
  各表現行列に対して `np.all(np.isfinite(H))` を検証し、非有限値があれば `FloatingPointError(f"{name} hidden states contain non-finite values.")` を送出。
  `H_R = H_R.astype(np.float32)`, `H_S = H_S.astype(np.float32)` とする。

---

### 2. Behavioral EmoBank fallback 実装ミスの修正 (項目 2)

#### [MODIFY] [run_behavioral_emobank.py](file:///mnt/nas/home/hiromi/src/emo2/behavioral/primary/run_behavioral_emobank.py)
- **fallback代入処理 (line 303付近)**:
  ```python
  stim_path = Path(args.stimuli_path)
  if not stim_path.exists():
      fallback = Path("data/processed/stimuli_vad_3way.csv")
      if fallback.exists():
          stim_path = fallback
      else:
          raise FileNotFoundError(
              f"Stimuli dataset not found: {args.stimuli_path} or {fallback}"
          )
  os.makedirs(args.out_dir, exist_ok=True)
  ```
  `stim_path` に `fallback` を確実に代入し、どちらも存在しない場合は即座に `FileNotFoundError` を送出する。

---

### 3. Behavioral Cache判定・Provenance の厳格化 (項目 3)

#### [MODIFY] [run_behavioral_aipsy.py](file:///mnt/nas/home/hiromi/src/emo2/behavioral/primary/run_behavioral_aipsy.py)
#### [MODIFY] [run_behavioral_emobank.py](file:///mnt/nas/home/hiromi/src/emo2/behavioral/primary/run_behavioral_emobank.py)
- **Early skip 条件の厳格化**:
  実行開始前に `manifest_config`（`model_id`, `model_revision`, `is_instruct`, `limit`, `dataset_hash`, `prompt_hash`, `candidate_space`, `actual_dtype`）を作成。
  `expected_config_hash = compute_string_or_dict_hash(manifest_config)` を算出。
  `is_manifest_matching` にて以下を完全一致照合：
  - `expected_model_name=args.model`
  - `expected_config_hash=expected_config_hash`
  - `expected_dataset_hash=compute_file_hash(stim_path)`
  - `expected_prompt_hash=prompt_hash`
  - `expected_model_revision=args.model_revision`
  - `expected_dry_run=args.dry_run`
- **Checkpoint expected_meta 拡張**:
  checkpoint 保存・読み込み時に `model_revision`, `dataset_hash`, `prompt_hash`, `candidate_hash`, `dtype` を格納し、不一致時は再開せず最初から実行する。

---

### 4. V1 Phase A Cache Hash 生成の完全一致 (項目 4)

#### [MODIFY] [run_phase_a.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_a.py)
- **同一の `manifest_config` の利用**:
  関数の冒頭で一意の `manifest_config` を定義：
  ```python
  manifest_config = {
      "model_prefix": args.model_prefix,
      "model_id": args.model_id,
      "model_revision": args.model_revision,
      "dataset": args.dataset,
      "dataset_hash": compute_file_hash(dataset_file_path),
      "limit": args.limit,
      "seed": phase_a_seed,
      "cv_folds": phase_a_cv,
      "ridge_alpha": phase_a_alpha,
  }
  ```
  Early skip 時の `expected_config_hash` 生成と、末尾の `create_run_manifest(..., config=manifest_config)` で全く同一の辞書を使用。
  `is_manifest_matching` で `expected_dataset_hash` も照合。

---

### 5. V1 Phase B Cache Provenance の強化 (項目 5)

#### [MODIFY] [run_phase_b.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_b.py)
- **詳細 manifest_config の照合**:
  `manifest_config` に `model_prefix`, `model_revision`, `task_type`, `relative_depth`, `target_layer`, `seed`, `train_ratio`, `cv_folds`, `dataset_hash`, `prompt_hash` を含める。
  単なる `is_experiment_completed` の成否だけでなく、`is_manifest_matching` で上記パラメータが完全一致しているかを検証してから skip する。

---

### 6. モデル Revision の固定と全 Stage への適用 (項目 6)

#### [MODIFY] [models.yaml](file:///mnt/nas/home/hiromi/src/emo2/configs/models.yaml)
- `base_revision` および `instruct_revision` を固定コミットSHA（または固定スナップショットタグ）に更新。
#### [MODIFY] 全 Primary runner スクリプト
- CLI 引数に `--model-revision`（デフォルト: `None` の場合は `models.yaml` の定義から解決）を追加。
- `AutoTokenizer.from_pretrained(..., revision=model_revision)`
- `AutoModelForCausalLM.from_pretrained(..., revision=model_revision)`
  対象:
  - `behavioral/primary/run_behavioral_aipsy.py`
  - `behavioral/primary/run_behavioral_emobank.py`
  - `v1/primary/run_phase_a.py`
  - `v1/primary/run_phase_b.py`
  - `v1/primary/run_phase_c.py`
  - `v1/primary/phase_c/run_e6_specialization.py`
  - `v2/primary/run_rq1_rq2_cross_decoding.py`
  - `v2/primary/run_rq3_causal_map.py`
  - `v2/primary/run_rq4_recovery_patching.py`
  - `v3/primary/run_rq1_state_induction.py`
  - `v3/primary/run_rq2_spatiotemporal_maps.py`
  - `v3/primary/run_rq3_path_mediation.py`
  - `v3/primary/run_confirmatory_replication.py`

---

### 7. Run ID 体系と結果ディレクトリ管理 (項目 7)

#### [MODIFY] [io.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/io.py)
#### [MODIFY] 各 Primary runner
- 各スクリプトに `--run-id` 引数を追加。
- 出力先ディレクトリとして、`--run-id` が指定された場合は `results/raw/<run_id>/` および `results/derived/<run_id>/` に出力。
- 既存ファイルの上書きを防止するため、上書きが要求された場合でも既存結果を `results/archive/<timestamp>_<old_run_id>/` へ安全に退避するヘルパー関数 `archive_existing_results()` を導入。
- `results/latest.json` に最新の `run_id` と結果ファイルパスを記録。

---

### 8. V2 RQ4 dry-run の sample-wise ΔEMD 計算修正 (項目 8)

#### [MODIFY] [run_rq4_recovery_patching.py](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq4_recovery_patching.py)
- **dry-run ブランチの ΔEMD 計算**:
  `initial_emd_mean * r` の近似を廃止。
  real-model ブランチと同一のロジックとして：
  - 各サンプル $i$、各層 $l$ について、`sample_delta_emds_plain[l][i] = sample_initial_emds[i] - sample_patched_emds_plain[l][i]` を計算。
  - 層ごとの平均 `layer_mean_delta_emds_plain` を集計。
  - 台形積分 `auc_delta_emd_plain = trapz(layer_mean_delta_emds_plain, depths)` を算出。
  これにより dry-run と本番コードの出力スキーマ・集計ロジックを100%一致させる。

---

### 9. 推論 dtype の統一 (bf16 / float32) (項目 9)

#### [MODIFY] [v1/primary/run_phase_a.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_a.py)
#### [MODIFY] [v1/primary/run_phase_b.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_b.py)
#### [MODIFY] [v3/primary/run_rq1_state_induction.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq1_state_induction.py)
#### [MODIFY] [v3/primary/run_rq2_spatiotemporal_maps.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq2_spatiotemporal_maps.py)
#### [MODIFY] [v3/primary/run_rq3_path_mediation.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq3_path_mediation.py)
#### [MODIFY] [v3/primary/run_confirmatory_replication.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_confirmatory_replication.py)
- `torch.float16` ハードコードを削除。
- `configs/models.yaml` の `inference_dtype`（bfloat16）を参照し、`torch.cuda.is_bf16_supported()` 等を確認した上で適切な dtype（`torch.bfloat16` または CPU/非対応時の `torch.float32`）を選択する共通リゾルバ `resolve_torch_dtype()` を適用。
- manifest の `actual_dtype` に実際にロードした dtype 文字列を記録。

---

### 10. Prompt Hash の SHA-256 完全化と推奨項目の適用 (項目 10〜13)

#### [MODIFY] [extraction.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/extraction.py)
- Python の組み込み `hash()`（プロセス毎に値が変わる）を `hashlib.sha256(text.encode("utf-8")).hexdigest()` に置換。
#### [MODIFY] 各 Primary runner
- `reader_prompt_hash`, `self_prompt_hash`, `chat_template_hash`, `template_mode`（`system_user` vs `user_only`）を manifest に記録。
- V3 runner (`run_rq1_state_induction.py`, `run_confirmatory_replication.py` 等) において、`n_input_pairs`, `n_matched_pairs`, `n_excluded_pairs` を manifest に保存。

---

## 検証計画 (Verification Plan)

### 自動検証
1. **構文チェック & コンパイル**:
   ```bash
   python -m compileall behavioral v1 v2 v3 src scripts
   ```
2. **高速単体テスト (Fast tests)**:
   ```bash
   pytest -q
   ```
3. **個別重要テスト**:
   ```bash
   pytest -q tests/test_likelihood.py
   pytest -q tests/test_v1_refinements.py
   pytest -q tests/test_v1_token_and_probe_alignment.py
   pytest -q tests/test_v3_prerun_fixes.py
   pytest -q tests/test_confirmatory_pipeline.py
   ```
4. **統合 Dry-Run**:
   ```bash
   python -m affective_empathy_eval.run \
     --stage all \
     --model-set primary_small \
     --family qwen \
     --device cpu \
     --dry-run \
     --max-samples 16
   ```

### 成果物作成
- `walkthrough.md` を作成し、各修正箇所の差分・実行結果・検証ログを報告。
- `docs/pre_production_critical_fixes/` に保存。
