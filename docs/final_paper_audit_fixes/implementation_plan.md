# 実装計画: 最終論文用コード監査40項目の包括的修正

本ドキュメントは、論文の4-Stage構造（Covariation $\to$ Representation $\to$ Reorganization $\to$ Utilization）を最終論文の主結果として完全に固定可能とするため、最新の静的コード監査で指摘された全40項目を漏れなく改訂するための包括的実装計画です。

---

## ユーザーレビューが必要な事項

> [!IMPORTANT]
> **1. Model Revision の固定 (Item 19)**:
> 現在 `configs/models.yaml` の revision は `main` となっています。論文の完全な再現性のために、各モデル（Qwen 2.5 1.5B, Llama 3.2 1B, Gemma 3 1B, OLMo 2 1B 等）の Hugging Face コミット SHA を明示的に固定します。
> 
> **2. Scale 定義の統一 (Item 34)**:
> プロジェクト全体の Primary 測定空間は **1–9 raw integer / expected value scale** で一貫しており、Gate 閾値 0.05 もこの scale 上の値です。コードや文書内の古い `[-1, 1]` 関連の記述は Secondary/感度分析の位置づけに統一します。
>
> **3. 既存結果の完全な保護と run_id 分離 (Item 25, 27)**:
> 原データ（`data/raw/`）および既存のアーカイブ（`old_results/`, `archive/`）は一切上書きせず、結果ファイル保存構造を `results/raw/<run_id>/` に分離し、`latest.json` を生成して追跡可能性を高めます。

---

## 主な修正内容 (全40項目)

### 1. P1: 論文の結果・推論に影響する重要項目 (Item 1 - 15)

1. **【Item 1】V1 E4 の Confirmatory selection leakage 修正**:
   - `v1/primary/run_phase_c.py`: E4 の confirmatory condition 判定において、全データを使った `magnitude_reader` を廃止し、必ず `discovery_mag_reader` のピーク層から選択する。
   - `discovery_mag_reader` が存在しない場合は production (`not args.dry_run`) で `ValueError` を送出し、heuristic な first-layer fallback を削除。
2. **【Item 2】V1 Phase A 測定不能時の NaN 保持**:
   - `v1/primary/run_phase_a.py`: `evaluate_regression_probe()` で `len(X) < cv` や `isnan(y)` 時に 0.0 を返すのをやめ、`float("nan")`, `"status": "failed"`, `"failure_reason"`, `"n_valid_folds": 0` を返す。正常時も status/valid_folds を記録。
3. **【Item 3】V1 Intensity 解析での未知ラベル検査**:
   - `v1/primary/run_phase_a.py`: `.fillna(0.0)` を撤廃し、`intensity` または `split` に未知ラベルが含まれる場合は `ValueError` を送出。
4. **【Item 4】V1 Phase A/B での非有限 Hidden State の例外化**:
   - `v1/primary/run_phase_a.py`, `v1/primary/run_phase_b.py`: `np.nan_to_num` と `np.clip(..., -1e4, 1e4)` による暗黙の数値置換を廃止。production では非有限値検出時に `FloatingPointError` を送出し、`max_abs` を manifest に記録。
5. **【Item 5】V1 Phase B Primary N の修正**:
   - `v1/primary/run_phase_b.py`: `n_primary_test_paraphrase_pairs` を `int(np.sum(test_para_valid_mask))`、`n_primary_test_reversal_pairs` を `int(np.sum(test_rev_valid_mask))` に修正。さらに `n_train_pairs`, `n_test_pairs`, `n_nonfallback_*` を保存。
6. **【Item 6】Phase B `validated` 表現の改名**:
   - `n_validated_paraphrase_pairs` 等を `n_nonfallback_paraphrase_pairs` へ名称変更。
7. **【Item 7】Phase B セマンティックコントロールの品質監査列追加**:
   - 生成結果テーブルに `transformation_method`, `quality_status` を追加保存。
8. **【Item 8】V2 RQ3 seed の config 連動化**:
   - `v2/primary/run_rq3_causal_map.py`: ハードコードされた `seed = 42` を廃止し、`v2_config["seed"]` から `split_seed = seed`, `control_seed = seed + 1`, `bootstrap_seed = seed + 2` を生成・伝播。
9. **【Item 9】V2 RQ4 layer-level ΔEMD 計算のサンプル単位集計化**:
   - `v2/primary/run_rq4_recovery_patching.py`: $E[D] \times E[R]$ による近似を廃止。サンプル $i$ ごとに $\Delta EMD_i = EMD_i^{initial} - EMD_i^{patched}$ を計算し、その平均として層ごとの $\Delta EMD$ を算出。
10. **【Item 10】V2 RQ4 Primary 指標と Secondary (peak) の分離**:
    - Primary は `auc_recovery_matched_plain` と `delta_emd_matched_plain_auc` に固定し、evaluation set 上の `best_layer`, `max_recovery` 比較は `secondary_peak_localization` に分離。
11. **【Item 11】V3 RQ2 causal subset seed の連動化**:
    - `v3/primary/run_rq2_spatiotemporal_maps.py`: `seed=42` 固定を引数 `seed` 由来に変更。
12. **【Item 12】V3 RQ3 subset seed の連動化**:
    - `v3/primary/run_rq3_path_mediation.py`: `seed=42` 固定を引数 `seed` 由来に変更。
13. **【Item 13】V3 RQ3 group 不足時の例外化**:
    - `v3/primary/run_rq3_path_mediation.py`: `n_splits < 2` の場合に $R^2=0$ とするのを廃止し、production では `ValueError` を送出。
14. **【Item 14】V3 Confirmatory cache key の V/A 軸別分離 & frozen_sites_hash 追加**:
    - `v3/primary/run_confirmatory_replication.py`: manifest_config に `temporal_relative_depth_v/a`, `temporal_stage_v/a`, `mediation_relative_depth`, `sufficiency_relative_depth` を保存し、`frozen_confirmatory_sites.json` の SHA256 ハッシュを cache key に含める。単一値 `temporal_relative_depth` fallback を削除。
15. **【Item 15】V3 Confirmatory コメント・説明文の改訂**:
    - コード冒頭や docstring の「pre_V で causal peak」を「Discovery cohort で推定した layer × stage map に基づき Valence/Arousal 固有サイトを freeze し検証」に統一。

---

### 2. P2: 最終論文の再現性向上のための修正 (Item 16 - 40)

16. **【Item 16 & 37】Behavioral キャッシュ検証強化 & チェックポイント分離**:
    - `behavioral/primary/run_behavioral_aipsy.py`, `run_behavioral_emobank.py`: チェックポイントを `results/checkpoints/` へ分離。manifest 照合（model_id, dataset_hash, prompt_hash, candidate_space, git_commit）を行い、一致時のみ resume。`checkpoint_used`, `n_resumed_samples` を記録。
17. **【Item 17 & 18】V1 Phase A/B/C キャッシュ判定 & 内容 Hash 必須化**:
    - `v1/primary/run_phase_a.py`, `run_phase_b.py`, `run_phase_c.py`: `is_manifest_matching()` に `expected_config_hash`, `expected_dataset_hash`, `expected_prompt_hash`, `expected_git_commit` 等を厳格に照合。dataset_hash は実ファイル SHA256 を保証。
18. **【Item 19, 20, 21】HF モデル・Tokenizer revision 固定**:
    - `configs/models.yaml`: 各モデルの commit SHA を設定。
    - `src/affective_empathy_eval/models/registry.py`: `AutoConfig`, `AutoTokenizer`, `AutoModel` に `revision` を伝播。manifest に `model_revision`, `tokenizer_revision` を独立保存・照合。
19. **【Item 22, 23, 24】Prompt 共通化・SHA-256 Hash・ChatTemplate 判定**:
    - `src/affective_empathy_eval/prompts.py`: `build_reader_prompt_vad`, `build_self_prompt_vad`, `build_reader_prompt_va`, `build_self_prompt_va` を集約。
    - 組み込み `hash()` を全廃し SHA-256 化。
    - `apply_chat_template` の generic exception 捕捉をやめ、`template_mode` ("system_user" / "user_only") を manifest に記録。
20. **【Item 25 & 26】run_id 保存構造 & 衝突防止**:
    - `src/affective_empathy_eval/manifests.py`: `generate_run_id` にマイクロ秒と UUID 8 桁を追加し、同一秒衝突を防止。
    - 結果保存時に `results/raw/<run_id>/` をサポートし、最新 run のポインタ `latest.json` を生成。
21. **【Item 27】Archive tarball 名のタイムスタンプ化**:
    - `scripts/archive_and_clean_results.py`: 固定名 `results_pre_rerun_20260918.tar.gz` を廃止し、`results_archive_<timestamp>.tar.gz` を生成。
22. **【Item 28】`run_production_reruns.sh` の Confirmatory 重複呼び出し解消**:
    - family ループ内での重複呼び出しを廃止し、loop 外で 1 回だけ `run_confirmatory_replication.py` を実行。
23. **【Item 29】V1 Phase A config 読み込み対応**:
    - `v1/primary/run_phase_a.py`: `--config` 引数を追加し、`configs/v1_experiments.yaml` の `seed`, `cv_folds`, `ridge_alpha` を probe へ反映。
24. **【Item 30】V1 E3/E4 Discovery 失敗時の fallback 禁止**:
    - `v1/primary/run_phase_c.py`: E3 Discovery 欠損時に heuristic layer へフォールバックせず、production では `RuntimeError` を送出。
25. **【Item 31】Candidate-space sensitivity の実モデル matched Δ 実装**:
    - `scripts/run_candidate_space_sensitivity.py`: 単一テキストの E[V], E[A] 比較をやめ、AIPsy matched pair 上で $\Delta V_{729}$ vs $\Delta V_{81}$, $\Delta A_{729}$ vs $\Delta A_{81}$ の Pearson $r$, Spearman $\rho$, 方向一致度, MAE を算出。Reader/Self 両対応、model registry から revision 込みで読み込み。
26. **【Item 32】Phase A/B candidate_space manifest 名整理**:
    - 内部表現抽出段階の manifest では `measurement_space="prompt_end_hidden_state"`, `candidate_space="N/A"` と記録。
27. **【Item 33】Stage 間 dtype 設定 & manifest 記録**:
    - `configs/models.yaml` に `inference_dtype: "bfloat16"` を導入。manifest に `actual_dtype` を記録。
28. **【Item 34】Scale 定義の統一**:
    - Primary は 1–9 raw scale（期待値・差分）に統一し、関連文書・コメントを整合。
29. **【Item 35】V2 RQ1/RQ2 分割の非空アサーション**:
    - `v2/primary/run_rq1_rq2_cross_decoding.py`: `len(train_df) == 0 or len(test_df) == 0` で `ValueError` を送出。
30. **【Item 36】Phase A 分類での単一クラス fold 処理**:
    - `v1/primary/run_phase_a.py`: `np.unique(y_train).size < 2` を検知し、有効 fold 不足時は partial_failure / エラーとして処理。
31. **【Item 38】V3 unmatched pair の除外記録**:
    - `src/affective_empathy_eval/data.py`: `load_v3_matched_pair_table` で除外された pair を `v3_exclusions.csv` に記録。
32. **【Item 39 & 40】V3 RQ1 depth fallback 撤廃 & frozen 互換キー完全削除**:
    - `v3/primary/run_rq3_path_mediation.py`, `run_confirmatory_replication.py`: `resolved_relative_depth` を必須化し、旧 0.50 fallback および `temporal_relative_depth` 単一キーを削除。

---

## 変更対象ファイル一覧

- `configs/models.yaml` (Item 19, 33)
- `src/affective_empathy_eval/models/registry.py` (Item 19, 20, 33)
- `src/affective_empathy_eval/manifests.py` (Item 16, 17, 18, 21, 25, 26, 32)
- `src/affective_empathy_eval/prompts.py` (Item 22, 23, 24)
- `src/affective_empathy_eval/data.py` (Item 38)
- `behavioral/primary/run_behavioral_emobank.py` (Item 16, 37)
- `behavioral/primary/run_behavioral_aipsy.py` (Item 16, 37)
- `v1/primary/run_phase_a.py` (Item 2, 3, 4, 17, 29, 32, 36)
- `v1/primary/run_phase_b.py` (Item 4, 5, 6, 7, 17, 32)
- `v1/primary/run_phase_c.py` (Item 1, 17, 30)
- `v2/primary/run_rq1_rq2_cross_decoding.py` (Item 35)
- `v2/primary/run_rq3_causal_map.py` (Item 8)
- `v2/primary/run_rq4_recovery_patching.py` (Item 9, 10)
- `v3/primary/run_rq2_spatiotemporal_maps.py` (Item 11, 40)
- `v3/primary/run_rq3_path_mediation.py` (Item 12, 13, 39, 40)
- `v3/primary/run_confirmatory_replication.py` (Item 14, 15, 39, 40)
- `scripts/archive_and_clean_results.py` (Item 27)
- `scripts/run_production_reruns.sh` (Item 28)
- `scripts/run_candidate_space_sensitivity.py` (Item 31)
- `docs/protocol.md` (Item 34)

---

## 検証計画

### 1. 静的コンパイル・構文検証
```bash
.venv/bin/python -m compileall behavioral v1 v2 v3 src scripts
```

### 2. 単体テスト実行 (CPU)
```bash
.venv/bin/pytest -q tests/test_likelihood.py
.venv/bin/pytest -q tests/test_manifests.py
.venv/bin/pytest -q tests/test_v1*
.venv/bin/pytest -q tests/test_v2*
.venv/bin/pytest -q tests/test_v3*
.venv/bin/pytest -q -m "not slow"
```

### 3. 全ステージ Production Dry-run スモークテスト
```bash
.venv/bin/python -m affective_empathy_eval.run \
  --stage all \
  --model-set primary_small \
  --family qwen \
  --device cpu \
  --dry-run \
  --max-samples 16
```
