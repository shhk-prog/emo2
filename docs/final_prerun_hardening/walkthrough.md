# 最終本番前ハーデニング修正確認 (Walkthrough)

本ドキュメントは、本番全実行（Clean Production Run）に先立ち実施された最終監査（再現性・キャッシュ整合性・感度分析の厳密化）における全修正内容と検証結果をまとめたものである。

---

## 1. 修正概要

### 1.1 V1 Phase B (`v1/primary/run_phase_b.py`)
- **キャッシュハッシュの事後不一致解消**:
  - `manifest_config` のハッシュ計算後に `manifest_config["num_pairs"] = n_pairs` を代入していた処理を削除。
  - `num_pairs` は `create_run_manifest` の `metadata` 引数（`results` と併せて）に格納するよう変更。これにより次回実行時の照合用ハッシュ（`expected_config_hash`）と保存済みマニフェストハッシュが完全一致し、キャッシュ判定が正しく動作。
- **「validated」表記の適正化**:
  - コメントおよびログ中の「`Primary (Validated transformations only)`」を「`Primary (Nonfallback rule-based transformations only)`」に変更し、人手検証と誤解されない表現に統一。

### 1.2 Candidate-space Sensitivity (`scripts/run_candidate_space_sensitivity.py`)
- **Clinical 刺激の厳密抽出**:
  - `intensity != "none"` での抽出を廃止し、`intensity.isin(["peak", "clinical"])` と `intensity.eq("none")` で厳密に各1件を抽出。各ペアがちょうど中立1件・臨床1件でない場合は明示的に `ValueError` を送出。
- **固定 Revision SHA の適用**:
  - CLI 引数に `--model-revision` を追加。未指定時は `ModelRegistry` から `instruct_model.revision` を取得し、`AutoTokenizer` および `AutoModelForCausalLM` に `revision=model_revision` を明示的に渡してロード。また、生成される `RunManifest` にも保存。
- **事前の固定結論（`conclusion`）の削除**:
  - 結果に関わらず固定文（`"Consistent ... preserved ..."`）を保存していた仕様を廃止し、純粋な数値評価（TVD, Pearson, Spearman, 方向一致率等）のみを出力・保存。

### 1.3 V3 RQ2 / RQ3 のキャッシュ照合厳密化 (`run_rq2_spatiotemporal_maps.py`, `run_rq3_path_mediation.py`)
- `manifest_config` の定義前に `ModelRegistry` から `model_revision` を解決し、`manifest_config["model_revision"] = model_revision` を追加。
- `is_manifest_matching` に `expected_model_revision=model_revision`, `expected_tokenizer_revision=model_revision` を渡し、モデルリビジョン不一致時に確実に再計算が走るよう厳密化。

### 1.4 本番実行時のフォールバック禁止 (Production Guardrails)
- **V3 Confirmatory (`run_confirmatory_replication.py`)**:
  - `pair_id` が不足している場合の通常 KFold fallback を禁止し、非 dry-run では明示的に `ValueError("Confirmatory analysis requires >=2 independent pair groups ('pair_id') for GroupKFold cross-fitting in production.")` を送出。
- **V3 RQ1 (`run_rq1_state_induction.py`)**:
  - `pair_id` が不足している場合の index split fallback を禁止し、非 dry-run では明示的に `ValueError("RQ1 analysis requires >=2 independent pair groups ('pair_id') for group-split in production.")` を送出。

### 1.5 統合 Runner の Behavioral Dry-run 実実行 (`src/affective_empathy_eval/run.py`)
- `args.dry_run` 指定時にも、子コマンド `cmd_emobank`, `cmd_aipsy` に `--dry-run` を付与して実際にサブプロセスを起動するように修正。
- 要約スクリプト（`cmd_sum_emobank`, `cmd_sum_aipsy`）も実際に実行し、dry-run が真の smoke test として全工程を検証するように改善。

### 1.6 V2 RQ1/RQ2 Manifest 表記修正 (`v2/primary/run_rq1_rq2_cross_decoding.py`)
- `candidate_space="N/A"`, `measurement_space="prompt_end_hidden_state"` を明示的に指定。

### 1.7 クレデンシャル・ファイル衛生 (`v1/.env`)
- ファイル内容を完全に空のコメント（`# cleared`）に上書きし、配布用 ZIP やリポジトリ共有時に混入しないよう注意喚起と削除手順を明記。

---

## 2. 検証結果

1. **構文チェック**:
   ```bash
   python -m compileall behavioral v1 v2 v3 src scripts
   ```
   -> エラーなし（全てのPythonファイルが正常コンパイル）
2. **ユニットテスト**:
   ```bash
   pytest -q
   ```
   -> **97 passed, 1 deselected, 5 warnings**
3. **全ステージ一気通貫 Dry-run 実行**:
   ```bash
   python -m affective_empathy_eval.run --stage all --model-set primary_small --family qwen --device cpu --dry-run --max-samples 16 --force
   ```
   -> **`All requested stages completed successfully!`**
   - **Behavioral**: EmoBank 3-way, AIPsy 4-split の実行および要約集計（`summarize_behavioral_emobank.py` 等）が完全通過。
   - **V1**: Phase A, Phase B (Reader/Self), Phase C, E6 Specialization, Phase C Summary の全パイプラインが完全通過。
   - **V2**: RQ1 & RQ2 (Cross-decoding & Geometry), RQ3 (Causal Map & LMM), RQ4 (Recovery Patching) が完全通過。
   - **V3**: RQ1 (State Induction & Gate Evaluation: GO), RQ2 (Spatiotemporal 4-Maps), RQ3 (Path Mediation & Frozen Confirmatory Sites), Step 7 Confirmatory Replication (Llama 3.2, Gemma 3, OLMo 2) が完全通過。

---

## 3. 判定および本番全実行（Clean Production Run）手順

### 最終判定: **GO** (すべての懸念・潜在的バグの解消を確認)

旧結果がクリアされた状態で、以下の手順にて本番実行を行ってください。

```bash
# 仮想環境の有効化と確認
source .venv/bin/activate
which python  # -> /mnt/nas/home/hiromi/src/emo2/.venv/bin/python

# 1. 不要な env ファイルの削除確認
rm -f v1/.env

# 2. 本番順次実行（各ステージ個別実行）
bash scripts/run_production_behavioral.sh cuda:0 --force
bash scripts/run_production_v1.sh cuda:0 --force
bash scripts/run_production_v2.sh cuda:0 --force
bash scripts/run_production_v3.sh cuda:0 --force
```

または、統合ランナーからファミリー別に順次実行することも可能です：

```bash
# 例: Qwen ファミリーの本番実行
python -m affective_empathy_eval.run --stage all --model-set primary_small --family qwen --device cuda:0 --force

# 例: Llama ファミリーの本番実行
python -m affective_empathy_eval.run --stage all --model-set primary_small --family llama --device cuda:0 --force
```

