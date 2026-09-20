# 本番実行前必須修正 (Pre-Production Critical Fixes) 実装計画

## 概要
研究論文の主張と測定ロジックを完全に一致させ、成果物の再現性・完全性を保証するため、指摘された必須項目（項目1〜14）および因果主張・再現性強化項目（項目15〜25）、さらに対応するテストスイート15項目を実装・検証します。

---

## 修正方針と対象ファイル

### 1. 【最重要】V3 $\beta$ の目的変数を Self-report に修正
- **対象ファイル**: [`v3/primary/run_rq2_spatiotemporal_maps.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq2_spatiotemporal_maps.py)
- **変更内容**:
  - `reg_v = LinearRegression().fit(X_cov_v, y_v)` → `LinearRegression().fit(X_cov_v, y_v_self)` に修正（Valence/Arousal 共に）。
  - 刺激ラベル共変量（`covar_v`, `covar_a`）を統制した内部表現スコア $\rightarrow$ モデル自己報告（Self-report）への偏回帰係数 $\beta$ を計算。
  - pair bootstrap による $\beta$ の 95% 信頼区間（CI）を推定し、出力マップおよびサマリー辞書に記録。

### 2. 【最重要】V3 生成時刻 `response_start` のインデックス修正
- **対象ファイル**: [`src/affective_empathy_eval/likelihood.py`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/likelihood.py)
- **変更内容**:
  - `resolve_joint_stage_index()` において、Causal LM の自己回帰的因果効果（位置 $t$ の隠れ層は $t+1$ 以降のトークン生成に影響）に準拠。
  - `if stage_name == "response_start": t_idx = cand_start - 1` （プロンプト最終トークン＝最初の回答トークン生成直前）とする。
  - `response_end` については負の対照（negative control）として位置づけ、ドキュメント・コメントを整備。

### 3. V3 $\gamma$ の定義・実装・変数名の整理
- **対象ファイル**: [`src/affective_empathy_eval/interventions.py`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/interventions.py)
- **変更内容**:
  - `estimate_interventional_slope(dose_grid, report_shift)` に引数名をリネーム。
  - $\gamma$ を「1-SD 正規化介入ドーズ（$\alpha \times \mathrm{std}$）あたりの自己報告変位（Self-report shift）」として定義を統一。
  - `delta_z_list` などの古い名称を廃止。

### 4. V3 RQ1 コントロール方向（`num_random_controls: 5`）の適用
- **対象ファイル**: [`v3/primary/run_rq1_state_induction.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq1_state_induction.py), [`configs/v3_experiments.yaml`](file:///mnt/nas/home/hiromi/src/emo2/configs/v3_experiments.yaml)
- **変更内容**:
  - `K = v3_cfg.get("interventions", {}).get("num_random_controls", 5)` を読み込み。
  - V/A それぞれについて $K$ 本のランダム方向および直交方向を生成（`seed = base_seed + k`）。
  - 各コントロール方向に対する変位を収集し、平均変位、分布、ターゲットとの差（target − mean(random), target − mean(orthogonal)）、および bootstrap CI を保存。

### 5 & 6. V3 Confirmatory のフォールバック排除と H4 連動
- **対象ファイル**: [`v3/primary/run_confirmatory_replication.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_confirmatory_replication.py)
- **変更内容**:
  - `nat_shifts_v`, `att_shifts_v` が空の場合の `1.0 / 0.5` フォールバックを完全排除し、`RuntimeError("No valid shift samples collected...")` を送出。最低サンプル数のアサーションを追加。
  - `non_uniform_leverage` をハードコード `True` から `bool(h4_pass)` に修正。

### 7. V1 Phase C の $\alpha$ および split-seed を YAML から読み込む
- **対象ファイル**: [`v1/primary/run_phase_c.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_c.py)
- **変更内容**:
  - `--alphas` の CLI default を `None` に設定し、未指定時は `configs/v1_experiments.yaml` の `phase_c.alphas` (`[0.0, 0.5, 1.0, 2.0]`) を適用。
  - `--split-seed` も同様に config の `phase_c.seed` を優先。

### 8. V1 Phase C の E3/E4 チェックポイント再開時 manifest 検証
- **対象ファイル**: [`v1/primary/run_phase_c.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_c.py)
- **変更内容**:
  - E3/E4 の層別チェックポイント保存時に `e3_checkpoint_manifest.json` / `e4_checkpoint_manifest.json` を生成。
  - config hash, dataset hash, model revision, tokenizer revision, prompt hash, intervention version を記録し、完全一致する場合のみ resume。不一致時は安全に破棄・再計算。

### 9. V1 Phase C activation cache 識別情報の拡張
- **対象ファイル**: [`v1/primary/run_phase_c.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_c.py)
- **変更内容**:
  - `compute_cache_metadata()` において、一部サンプルではなく Reader/Self × affective/neutral の全プロンプトをハッシュ化。
  - `model_revision`, `tokenizer_revision`, `dtype`, `torch_version`, `transformers_version` をメタデータに含め、モデルリビジョン変更時に確実にキャッシュを無効化。

### 10. V1 Phase A/B/C のトークン抽出における left/right padding 対応
- **対象ファイル**: [`v1/primary/run_phase_a.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_a.py), [`v1/primary/run_phase_b.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_b.py), [`v1/primary/run_phase_c.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_c.py)
- **変更内容**:
  - `seq_lengths = attention_mask.sum(dim=1) - 1` に依存せず、
    `valid_pos = torch.nonzero(attention_mask[b_idx], as_tuple=False).flatten()`
    `last_pos = int(valid_pos[-1])` に統一。

### 11. V1 Phase A 分類におけるスキップ済み fold の評価除外
- **対象ファイル**: [`v1/primary/run_phase_a.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_a.py)
- **変更内容**:
  - `StratifiedGroupKFold` を使用して極力単一クラス fold の発生を防止。
  - `evaluated_mask = np.zeros(len(y_enc), dtype=bool)` を用意し、有効に予測されたサンプルのみを対象に `balanced_accuracy_score` を計算。スキップされたサンプルが暗黙に class 0 として計算されるのを防ぐ。

### 12 & 13. Behavioral の `--dry-run` 隔離とチェックポイント堅牢化
- **対象ファイル**: [`behavioral/primary/run_behavioral_aipsy.py`](file:///mnt/nas/home/hiromi/src/emo2/behavioral/primary/run_behavioral_aipsy.py), [`behavioral/primary/run_behavioral_emobank.py`](file:///mnt/nas/home/hiromi/src/emo2/behavioral/primary/run_behavioral_emobank.py), [`behavioral/analysis/summarize_behavioral_aipsy.py`](file:///mnt/nas/home/hiromi/src/emo2/behavioral/analysis/summarize_behavioral_aipsy.py), [`behavioral/analysis/summarize_behavioral_emobank.py`](file:///mnt/nas/home/hiromi/src/emo2/behavioral/analysis/summarize_behavioral_emobank.py), [`src/affective_empathy_eval/run.py`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/run.py)
- **変更内容**:
  - `--dry-run` 指定時は出力ディレクトリを `.../dry_run` に隔離。サマライザーも `--dry-run` フラグを受け取り、本番結果CSV・manifestの上書きを防止。
  - チェックポイント再開時、`checkpoint_meta_path` が存在し、かつ全メタデータキーが完全一致する場合のみ `can_resume = True` とする。メタデータ欠損・不一致時は再開せず新規開始。

### 14. V2 RQ1/RQ2 manifest への実験設定全体の包含
- **対象ファイル**: [`v2/primary/run_rq1_rq2_cross_decoding.py`](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq1_rq2_cross_decoding.py)
- **変更内容**:
  - `config_payload` に `"v2_config": v2_config` 全体を含め、`train_ratio`, `ridge_alpha` 等の YAML 設定変更時に確実にキャッシュが無効化されるように統一。

### 15〜25. 因果主張・再現性強化
- **V2 RQ3 コントロール**: ランダム方向・直交方向の介入変位を算出し、$C_{\text{affect}} - C_{\text{random}}$, $C_{\text{affect}} - C_{\perp}$ を出力。
- **V3 RQ3 部分空間コントロール**: matched-rank random 2D subspace removal を追加。
- **V1 E4 ドナー拡張**: 20回の固定シード derangements による random donor 分布と CI を算出。
- **ドキュメント・命名整理**: V1 E3/E4, E6, V2 RQ4 (off-manifold), V3 Confirmatory H1/H2/H4 CI, sequence likelihood トークン長検査などを実施。

---

## 検証計画

### 1. 新規単体テストの実装
`tests/test_pre_production_fixes.py` を新設し、以下のテストを実装・実行：
1. `test_v3_beta_targets_self_report`
2. `test_response_start_is_prompt_end`
3. `test_response_end_is_negative_control`
4. `test_v3_num_random_controls_is_honored`
5. `test_confirmatory_empty_effects_raise`
6. `test_non_uniform_leverage_matches_h4`
7. `test_phase_c_alphas_loaded_from_yaml`
8. `test_phase_c_resume_rejected_on_manifest_mismatch`
9. `test_phase_c_cache_invalidated_on_revision_change`
10. `test_hidden_extraction_left_and_right_padding`
11. `test_classification_skipped_fold_not_scored_as_zero`
12. `test_behavioral_dry_run_never_touches_production_outputs`
13. `test_checkpoint_without_metadata_is_not_resumed`
14. `test_v2_geometry_manifest_changes_with_v2_config`
15. `test_candidate_token_lengths_primary_models`

### 2. 回帰テスト・ビルド検証
- `pytest` の全テスト実行（既存97件＋新規テスト）
- `python -m compileall -q behavioral v1 v2 v3 src tests`
- `ruff check` の実行確認
