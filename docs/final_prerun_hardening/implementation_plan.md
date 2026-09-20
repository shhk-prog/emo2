# 実装計画: 最終本番前ハーデニング (Final Pre-run Hardening)

本計画は、全再実行（Clean Production Run）に先立ち、再現性・キャッシュ整合性・感度分析スクリプトの厳密化を行うための修正手順である。

## 修正項目詳細

### 1. V1 Phase B (`v1/primary/run_phase_b.py`)
- **問題**: `manifest_config` をハッシュ計算した後に、`manifest_config["num_pairs"] = n_pairs` を追加して保存しているため、次回実行時の照合用期待ハッシュと保存済みハッシュが不一致になる。
- **対応**: 
  - `manifest_config["num_pairs"] = n_pairs` を削除。
  - `metadata` 引数に `"num_pairs": n_pairs` を格納。
  - コード内のコメント「`Primary (Validated transformations only)`」等を「`Primary (Nonfallback rule-based transformations only)`」へ置換。

### 2, 3, 4. Candidate-space Sensitivity (`scripts/run_candidate_space_sensitivity.py`)
- **問題 1**: `intensity != "none"` で選ぶと Moderate も拾う可能性がある。
- **対応 1**: `intensity.isin(["peak", "clinical"])` と `intensity.eq("none")` で厳密に各1件ずつ抽出。1件ずつでなければ `ValueError`。
- **問題 2**: `model_revision` が渡されておらず、デフォルト revision を取得してしまう。
- **対応 2**: `--model-revision` 引数を追加し、`AutoTokenizer`/`AutoModelForCausalLM` に `revision=model_revision` を渡し、CLI 未指定時は registry から `instruct_model.revision` を取得。
- **問題 3**: 結果によらず "Consistent ... preserved ..." という固定文が conclusion に記録されている。
- **対応 3**: 固定 conclusion を削除し、純粋な数値指標（TVD, Spearman, Pearson 等）のみを保存。

### 5. V3 RQ2 / RQ3 のキャッシュ照合 (`v3/primary/run_rq2_spatiotemporal_maps.py`, `v3/primary/run_rq3_path_mediation.py`)
- **問題**: RQ1 では `expected_model_revision` を検証しているが、RQ2/RQ3 では `model_revision` が未照合。
- **対応**: `manifest_config` に `"model_revision": model_revision` を含め、`is_manifest_matching` に `expected_model_revision=model_revision`, `expected_tokenizer_revision=model_revision` を渡す。

### 6. V3 Confirmatory & RQ1 の production fallback 禁止
- **V3 Confirmatory (`v3/primary/run_confirmatory_replication.py`)**: `pair_id` が不足している場合の通常 KFold fallback を dry-run 限定にし、非 dry-run では `ValueError` を送出。
- **V3 RQ1 (`v3/primary/run_rq1_state_induction.py`)**: `pair_id` が不足している場合の index split fallback を dry-run 限定にし、非 dry-run では `ValueError` を送出。

### 7. Unified Runner の Behavioral Dry-run 実実行 (`src/affective_empathy_eval/run.py`)
- **問題**: `args.dry_run` 時、Behavioral の子スクリプト（EmoBank, AIPsy）が実行されず、ログ出力だけでスキップされている。
- **対応**: `--dry-run` を付与して実際に `cmd_emobank`, `cmd_aipsy`, `cmd_sum_*` を呼び出す。

### 8. V2 RQ1/RQ2 Manifest 表記修正 (`v2/primary/run_rq1_rq2_cross_decoding.py`)
- `candidate_space="N/A"`, `measurement_space="prompt_end_hidden_state"` を明示。

### 9. `v1/.env` の削除
- `rm -f v1/.env` を実行し、ファイルが存在しないことを確認。

---

## 検証手順
1. `python -m compileall behavioral v1 v2 v3 src scripts` で全構文チェック。
2. `pytest -q` で全単体・統合テストのパス確認。
3. `python -m affective_empathy_eval.run --stage all --model-set primary_small --family qwen --device cpu --dry-run --max-samples 16 --force` で全ステージ実 smoke test 完走確認。
