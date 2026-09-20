# 実装計画: コード監査指摘対応（21項目）および再現性・パイプライン厳密化

## 概要
最新の静的コード監査により指摘された21項目（P1: 1〜12項目、P2: 13〜21項目）に対応し、論文の4-Stage構造（Covariation $\to$ Representation $\to$ Reorganization $\to$ Utilization）を強固に支えるデータリーク防止、Primary/Secondary 指標の明確化、因果利用時点の定義、および再現性管理を実装します。

---

## ユーザー確認事項 (User Review Required)

> [!IMPORTANT]
> **V3 RQ2 の「いつ（When）」の設計方針について**
> 監査では「A: pre_V / pre_A を事前定義 anchor（a priori test stage）として扱い、変数名と主張を整合させる」または「B: Discovery Qwen で全 stage を探索して peak stage を特定し、frozen artifact に保存して Confirmatory に伝播する」の2択が提示されました。
> 
> **本計画の提案**:
> 1. 事前理論的仮説として設定されたアンカーについては `a_priori_test_stage_v: "pre_V"`, `a_priori_test_stage_a: "pre_A"` として明示。
> 2. 同時に Discovery Qwen の全 stage 計算結果から最大の因果効果を持つ `empirical_peak_stage_v`, `empirical_peak_stage_a` を同定し、`v3_rq2_causal_sites.json` / `frozen_confirmatory_sites.json` に両方を保存。
> 3. `run_confirmatory_replication.py` では、ハードコードではなく `frozen_confirmatory_sites.json` に記録された stage を動的に読み取って検証を実施。
> これにより、事前定義アンカーの比較検証とデータドリブンなピーク探索の両方を矛盾なく成立させます。

---

## 修正内容の詳細

### 1. P1: 論文の結論に影響する重要修正 (Items 1-12)

#### Item 1: V1 Phase A キャッシュ判定が AIPsy を見ていない ([`v1/primary/run_phase_a.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_a.py))
- `required_outputs` を `args.dataset` に応じて動的に定義：
  - `emobank` or `both`: `e1_emobank_decodability.csv`, `e2_emobank_geometry.csv`
  - `aipsy` or `both`: `e1_aipsy_classification.csv`, `e1_aipsy_intensity.csv`, `e1_aipsy_emotion_secondary.csv`
- `manifest.json` の存在と `is_manifest_matching(...)`（model, config_hash, dataset, dry_run）を確認。
- テスト: EmoBank のみ存在し AIPsy が欠落している場合、skip されず AIPsy が実行されることを検証。

#### Item 2: V1 Phase B pair leakage fallback 削除 ([`v1/primary/run_phase_b.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_b.py))
- `evaluate_probe_cross_validated`:
  - `groups` が与えられた場合、`n_groups = len(np.unique(groups))` が 2 未満なら `float("nan")` を返却。
  - `n_splits = min(cv, n_groups)` を設定し、`StratifiedGroupKFold` を使用。成立しない場合は `GroupKFold` へフォールバックし、決して同一 pair が train/test に跨ぐ `StratifiedKFold` にはフォールバックしない。
- `test_pair_ids = train_pair_ids` のハードコードを削除し、`len(train_pair_ids) == 0 or len(test_pair_ids) == 0` の場合は `raise ValueError("Independent train/test pair split cannot be constructed.")` とする。
- `assert len(train_pair_ids.intersection(test_pair_ids)) == 0` を確実に維持。

#### Item 3: V1 Phase B semantic control 生成品質 ([`v1/primary/prepare_v1_phase_b_controls.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/prepare_v1_phase_b_controls.py), [`v1/primary/run_phase_b.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_b.py))
- `prepare_semantic_controls` で各変換レコードに以下のメタデータを付与：
  - `paraphrase_method`: 具体的な置換ルール名、または `"rule_based_prefix"`
  - `reversal_method`: 具体的な反転ルール名、または `"fallback_clause"`
  - `paraphrase_fallback`: prefix 付加の場合は `True`、高品質語彙置換の場合は `False`
  - `reversal_fallback`: 汎用節付加の場合は `True`、語彙・構文反転の場合は `False`
- `run_phase_b.py` の Primary 解析では `(~df["paraphrase_fallback"]) & (~df["reversal_fallback"])` のみを使用し、fallback 含む全件は Secondary / Sensitivity として別集計。

#### Item 4: V1 Phase A Direct Cross-Decoding の負の R² を保持 ([`v1/primary/run_phase_a.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_a.py))
- `direct_transfer_score_raw = (r2_r_to_s + r2_s_to_r) / 2.0` をそのまま保存。
- 0 未満を 0 にクリップせず、負の値（平均予測器以下への悪化＝共有性の欠如の直接的証拠）をそのまま Primary 指標として記録。
- 可視化・参考用に `direct_transfer_score_clipped = max(0.0, direct_transfer_score_raw)` を別列として追加。
- `geometry_pattern` の判定も `direct_transfer_score_raw` を基準とする。

#### Item 5: V1 Phase A で undefined metric を 0.5/0 に置換しない ([`v1/primary/run_phase_a.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_a.py))
- `evaluate_classification_probe` において、単一クラスや例外等で計算不能な場合は 0.5 や 0.0 を代入せず `np.nan` を返却。
- 返却辞書に `status` ("success" / "failed"), `n_valid_folds`, `failure_reason` を含める。
- 集計処理は `np.nanmean()` を使用し、有効 fold 数や missing 数を記録。

#### Item 6: V2 RQ1/RQ2 summary を matched-plain Primary へ修正 ([`v2/primary/run_rq1_rq2_cross_decoding.py`](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq1_rq2_cross_decoding.py))
- `results["summary_metrics"]` を明示的に構造化：
  ```python
  results["summary_metrics"] = {
      "primary_matched_plain": {
          "com_distortion_reader": ...,
          "com_distortion_self": ...,
          "peak_depth_base_cross": ...,
          "peak_depth_inst_cross": ...,
          "peak_depth_delta_share": ...,
          "mean_delta_sharing": ...,
      },
      "secondary_native_chat": { ... },
      "format_effect": { ... }
  }
  ```
- `v2_cross_family_summary.json` のクロスモデル・ブートストラップも、matched-plain の指標から算出するように統一。

#### Item 7: V2 RQ4 eval=train fallback 削除 ([`v2/primary/run_rq4_recovery_patching.py`](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq4_recovery_patching.py))
- `if len(eval_indices) == 0: eval_indices = train_indices`（2箇所）を完全削除。
- `if len(eval_indices) == 0: raise ValueError("Independent evaluation split could not be constructed.")` に変更。
- `assert set(train_pairs).isdisjoint(eval_pairs)` を追加。

#### Item 8: V2 RQ4 の 0.7 ハードコードを config 読み込みへ統一 ([`v2/primary/run_rq4_recovery_patching.py`](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq4_recovery_patching.py))
- `configs/v2_experiments.yaml` の `dataset.train_ratio` (0.7) を読み込み、関数引数経由で伝播。コード内の `0.7` 直書きを排除。

#### Item 9: V3 RQ2 の「いつ」の探索とアンカーの明確化 ([`v3/primary/run_rq2_spatiotemporal_maps.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq2_spatiotemporal_maps.py))
- 事前定義 anchor: `a_priori_test_stage_v = "pre_V"`, `a_priori_test_stage_a = "pre_A"` と明記。
- データドリブン peak: $C(l, s)$ マップ全域（全 layer × 全 stage）の探索を行い、最大の因果変位を示す `empirical_peak_stage_v`, `empirical_peak_stage_a` を算出。
- `v3_rq2_causal_sites.json` に両方を保存。

#### Item 10: V3 Confirmatory も stage を frozen artifact から読み込み ([`v3/primary/run_confirmatory_replication.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_confirmatory_replication.py))
- `frozen_confirmatory_sites.json` に `temporal_stage_v`, `temporal_stage_a` を含める。
- `run_confirmatory_replication.py` はハードコードされた `"pre_V"`, `"pre_A"` ではなく、frozen artifact の stage を使用して介入を実施。

#### Item 11: V3 RQ3 の split/seed config 化 ([`v3/primary/run_rq3_path_mediation.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq3_path_mediation.py))
- `seed = int(v3_cfg.get("seed", 42))`
- `discovery_ratio = float(v3_cfg["path_mediation"]["discovery_ratio"])`
- `assert disc_pairs.isdisjoint(conf_pairs)` を追加。

#### Item 12: 全 Primary から hardcoded seed を排除
- 各 config の `seed` をルートとし、`split_seed = base_seed`, `control_seed = base_seed + 1`, `bootstrap_seed = base_seed + 2` などの決定論的派生ルールを統一。使用した全 seed を manifest に記録。

---

### 2. P2: 再現性・査読対応・データ整合性の強化 (Items 13-21)

#### Item 13: `results/raw` 上書き防止と `run_id` 導入
- `run_id`（`YYYYMMDDTHHMMSSZ_<git_short>_<config_short>`）を生成し、`results/raw/<run_id>/` または各 stage の出力パスに紐付け。
- 最新実行への pointer として `latest.json` を生成。`--force` 指定時も上書きではなく新しい `run_id` を発行。

#### Item 14: Hugging Face モデル revision の固定 ([`configs/models.yaml`](file:///mnt/nas/home/hiromi/src/emo2/configs/models.yaml))
- `configs/models.yaml` に `base_revision`, `instruct_revision` フィールドを追加。
- モデル読み込み時に `revision=revision` を渡し、resolved commit hash を manifest に保存。

#### Item 15: Prompt hash の保存
- prompt template 文字列の SHA256 ハッシュ（`reader_prompt_hash`, `self_prompt_hash` 等）を算出して manifest に保存。

#### Item 16: Sequence Likelihood の token 境界厳密テスト ([`tests/test_likelihood.py`](file:///mnt/nas/home/hiromi/src/emo2/tests/test_likelihood.py))
- 各 tokenizer において `tokens(prompt) == tokens(prompt + candidate)[:len(tokens(prompt))]` が成立することを検証するテストを追加。

#### Item 17: V1 Phase C の resume/cache validation 強化 ([`v1/primary/run_phase_c.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_c.py))
- manifest 内のモデル revision、データハッシュ、split seed、candidates、prompt hash が完全一致する場合のみ途中 resume を許可。

#### Item 18: Candidate-space sensitivity の README 記述修正 ([`README.md`](file:///mnt/nas/home/hiromi/src/emo2/README.md))
- 「主要傾向が完全に保たれる」という過度な表現を改め、Supplementary での感度分析の定義および Pearson r / Spearman rho / direction agreement 報告形式に改訂。

#### Item 19: Sequence likelihood の長さ感度と表記統一
- 論文表記を `joint conditional sequence log-likelihood` に統一。候補トークン長分布を記録。

#### Item 20 & 21: V1 Phase B / Phase C の split 設定を config 化 ([`configs/v1_experiments.yaml`](file:///mnt/nas/home/hiromi/src/emo2/configs/v1_experiments.yaml))
- `phase_b.seed: 42`, `phase_b.train_ratio: 0.7`, `phase_b.cv_folds: 5`
- `phase_c.discovery_ratio: 0.5`
- スクリプト側から config を参照し、pair IDs ハッシュを manifest に記録。

---

## 検証計画

### 自動テスト (pytest)
```bash
.venv/bin/pytest tests/test_production_entrypoints.py tests/test_v1_token_and_probe_alignment.py tests/test_likelihood.py tests/test_confirmatory_pipeline.py
```
- EmoBank 結果あり + AIPsy 結果なしの状態で Phase A を呼び出した際、early skip されずに AIPsy が実行されることを検証するテストケースを追加。
- Pair leakage が確実に防止され、不正な fallback が発生しないことを検証。
- トークン境界テストがパスすることを確認。

### 各フェーズの dry-run
```bash
.venv/bin/python v1/primary/run_phase_a.py --dry-run
.venv/bin/python v1/primary/run_phase_b.py --dry-run --task-type reader
.venv/bin/python v1/primary/run_phase_b.py --dry-run --task-type self
.venv/bin/python v1/primary/run_phase_c.py --dry-run
.venv/bin/python v2/primary/run_rq1_rq2_cross_decoding.py --dry-run
.venv/bin/python v2/primary/run_rq4_recovery_patching.py --dry-run
.venv/bin/python tests/run_all_v3_dryruns.py
```
