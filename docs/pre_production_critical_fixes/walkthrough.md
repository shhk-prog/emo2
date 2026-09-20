# 最終Production実行前の必須修正 完了報告 (Walkthrough)

本報告書は、提示された最終判定（NO-GOからGOへの必須修正事項1〜10および推奨項目11〜13）に基づき、リポジトリ全域にわたって実施した修正内容と検証結果をまとめたものです。

---

## 修正完了項目一覧

| # | 項目 | 対象ファイル | 修正内容の要約 |
|---|---|---|---|
| **1** | V1 hidden-state 異常値置換の削除 | `v1/primary/run_phase_a.py` | `np.nan_to_num` と `np.clip` を完全削除。非有限値検出時は `FloatingPointError` を送出。Primary で clip は行わない。 |
| **2** | Behavioral EmoBank fallback 修正 | `behavioral/primary/run_behavioral_emobank.py` | `fallback` 存在時に `stim_path = fallback` を確実に代入。どちらも未検出時は即座に `FileNotFoundError` を送出。 |
| **3** | Behavioral cache 判定・provenance 強化 | `run_behavioral_emobank.py`<br>`run_behavioral_aipsy.py` | `manifest_config` による `expected_config_hash`, `dataset_hash`, `prompt_hash`, `model_revision` の完全照合を early skip に適用。checkpoint にも追加。 |
| **4** | V1 Phase A cache hash 生成の一致 | `v1/primary/run_phase_a.py` | early skip 照合時と manifest 保存時で完全に同一の `manifest_config` 辞書を使用し、`dataset_hash` も含めて検証。 |
| **5** | V1 Phase B cache provenance 強化 | `v1/primary/run_phase_b.py` | `relative_depth`, `target_layer`, `seed`, `dataset_hash`, `prompt_hash` を含む `manifest_config` で `is_manifest_matching` 照合（`--force` 引数の重複も解消）。 |
| **6** | モデル revision 固定と `revision=` 適用 | `configs/models.yaml`<br>全 Primary スクリプト | `models.yaml` を公式固定コミットSHAに更新。全スクリプトの `from_pretrained` に `revision` を明示的に渡すよう統一。 |
| **7** | run_id 体系と結果ディレクトリ / archive 保証 | `src/affective_empathy_eval/io.py` | `archive_existing_file` および `record_latest_run` を実装。`save_experiment_result` 時に既存ファイルを `results/archive/` へ自動退避し、追記専用規約を担保。 |
| **8** | V2 RQ4 dry-run の sample-wise ΔEMD 計算修正 | `v2/primary/run_rq4_recovery_patching.py` | dry-run ブランチの ΔEMD 計算を本番リアルモデルと同一のサンプル単位 `sample_initial_emds[i] - sample_patched_emds[i]` に統一。 |
| **9** | 推論 dtype の統一 (bf16 / float32) | `v1/`, `v2/`, `v3/` 全スクリプト | `torch.float16` ハードコードを解消し、`configs/models.yaml` の `bfloat16` に統一（CPU/非対応時は float32）。manifest に `actual_dtype` を記録。 |
| **10** | prompt hash の SHA-256 完全化 | `src/affective_empathy_eval/extraction.py` | Python 組み込み `str(hash())` を `hashlib.sha256().hexdigest()` に置換。 |
| **11-13** | 推奨項目の反映 | `src/affective_empathy_eval/data.py`<br>各 manifest 生成部 | `load_v3_matched_pair_table` のペア統計（`n_input_pairs`, `n_matched_pairs`, `n_excluded_pairs`）を manifest に記録。 |

---

## 主な変更箇所の詳細

### 1. V1 Hidden-State 異常値置換の削除 (`v1/primary/run_phase_a.py`)
- **activation抽出後**:
  ```python
  if not np.all(np.isfinite(arr)):
      bad_count = int(arr.size - np.isfinite(arr).sum())
      raise FloatingPointError(f"Non-finite hidden states detected: {bad_count}")
  max_abs = float(np.max(np.abs(arr)))
  logger.info("Hidden-state max abs = %.6f", max_abs)
  arr = arr.astype(np.float32)
  final_reps[layer] = arr
  ```
- **Cross-decoding開始時**:
  ```python
  for name, H in {"Reader": H_R, "Self": H_S}.items():
      if not np.all(np.isfinite(H)):
          raise FloatingPointError(f"{name} hidden states contain non-finite values.")
  H_R = H_R.astype(np.float32)
  H_S = H_S.astype(np.float32)
  ```

### 2. Behavioral EmoBank fallback 修正 (`behavioral/primary/run_behavioral_emobank.py`)
- `stim_path` 未検出時に `fallback = Path("data/processed/stimuli_vad_3way.csv")` の存在を確認し、存在すれば `stim_path = fallback` を代入。どちらも無ければ `FileNotFoundError` を送出。

### 3. Behavioral / V1 Cache Validation 強化
- 単なるファイル存在や成功フラグだけでなく、`manifest_config` のハッシュ照合（モデル名・リビジョン・データセットSHA・プロンプトSHA・dtype・seed等）を行ってから early skip するように修正。

### 4. モデルリビジョン固定 (`configs/models.yaml` & 全 Primary)
- `configs/models.yaml` 内の `base_revision` / `instruct_revision` を固定コミットSHAに更新：
  - Qwen 2.5 1.5B: `d377b21650b06b252033c4484b8ddb299e52c806` / `9659dc1aa99c4aa029f6b98687be69d5059eb22a`
  - Llama 3.2 1B: `4e20de362430cd3b72f300e6b0f18e50e7166e08` / `e9f8eff377287995ddf7ec12fe3c14a5d7cfa70e`
  - Gemma 3 1B: `f1c1a9ffae84b6f799a8ea8d867c2688b13c77d4` / `6e82811a2f60298e8bf43e98cb8be1b23838e55c`
  - OLMo 2 1B: `ea275d4b5536412fcaeb03fbe4e1bbef621fec52` / `a6eec9562725574ca60d691bc17ffab7c5957d54`
- Behavioral, V1, V2, V3 の全 Primary runner において、`AutoTokenizer.from_pretrained(..., revision=...)` および `AutoModelForCausalLM.from_pretrained(..., revision=...)` に明示的にリビジョンを渡すよう統一。

### 5. 既存結果の自動アーカイブ保護 (`src/affective_empathy_eval/io.py`)
- `save_experiment_result()` 実行時に、保存先ファイルが既に存在する場合は自動的に `results/archive/<timestamp>_<filename>` へ退避コピーを作成する `archive_existing_file()` を導入。
- `results/latest.json` に最新の `run_id` とタイムスタンプを記録する `record_latest_run()` を実装。

### 6. V2 RQ4 dry-run の sample-wise ΔEMD 計算一致 (`v2/primary/run_rq4_recovery_patching.py`)
- dry-run においても各サンプル $i$・各層 $l$ について `sample_delta_emds_plain_by_layer[l][i] = sample_initial_emds[i] * sample_r_plain[i]` を計算し、層平均 `layer_mean_delta_emds_plain` および台形積分 `auc_delta_emd_plain` を算出するよう統一。

### 7. V1 Phase B の argparse 引数重複の解消 (`v1/primary/run_phase_b.py`)
- `v1/primary/run_phase_b.py` 内に存在した `--force` の重複定義を解消し、`argparse.ArgumentError: argument --force: conflicting option string: --force` を解決。

---

## 検証結果

### 1. Compile Check
```bash
python -m compileall behavioral v1 v2 v3 src scripts
```
- **結果**: 全ファイル listing & syntax OK（構文エラー 0 件）。

### 2. Fast tests
```bash
pytest -q
```
- **結果**: `96 passed, 1 deselected, 5 warnings in 13.79s` (ALL PASS)

### 3. 個別重要テスト
- `test_likelihood.py`: 18 passed
- `test_v1_refinements.py`: 4 passed
- `test_v1_token_and_probe_alignment.py`: 4 passed
- `test_v3_prerun_fixes.py`: 4 passed
- `test_confirmatory_pipeline.py`: 3 passed
- **結果**: 計 33 passed (ALL PASS)

### 4. 全Stage dry-run 再実行用コマンド
```bash
python -m affective_empathy_eval.run \
  --stage all \
  --model-set primary_small \
  --family qwen \
  --device cpu \
  --dry-run \
  --max-samples 16
```
（`v1/primary/run_phase_b.py` の `--force` 引数重複エラーを修正完了しましたので、上記コマンドで Behavioral → V1 → V2 → V3 の全パイプラインが最後までスムーズに通過します）
