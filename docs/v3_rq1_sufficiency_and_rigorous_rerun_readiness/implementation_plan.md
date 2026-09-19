# 実装計画: V3 RQ1 Sufficiency修正および本番再実行盤石化（全18項目）

本計画は、論文の中心主張（**Behavioral Covariation $\rightarrow$ V1 Partial Representation/Causal Overlap $\rightarrow$ V2 Post-training-Associated Reorganization $\rightarrow$ V3 Localized Causal Leverage**）を一切変更せず、実験コードをこの科学的論理構造に厳密に一致させ、本番再実行を盤石にするための全18項目の修正方針を定めます。

---

## ユーザー確認事項 (User Review Required)

> [!IMPORTANT]
> **1. V3 RQ1 の因果役割の完全分離（最重要改善）**
> - **Sufficiency / Dose-response / Specificity**: **Matched-neutral 刺激**へ推定情動方向 $d_V, d_A$ を加算注入し、中立自己報告からの情動誘導を実測します。
>   - $\Delta V = \text{patched\_neu\_ev} - \text{clean\_neu\_ev}$
>   - Random / Orthogonal / Topic 統制も同様に中立プロンプト上で比較。
> - **Endogenous relevance / Necessity**: **Affective 刺激**から 2D 部分空間 $Q$ を除去（Centered subspace removal）し、自然変位の減衰を実測します。
>   - $\text{natural\_shift} = |\text{clean\_aff} - \text{clean\_neu}|$
>   - $\text{residual\_shift} = |\text{ablated\_aff} - \text{clean\_neu}|$
>   - $\text{mediated\_attenuation} = \text{natural\_shift} - \text{residual\_shift}$

> [!IMPORTANT]
> **2. Dry-run / Pilot 出力先の完全分離と Gate 保護**
> - `run_rq1_state_induction.py`、`run_rq4_recovery_patching.py` で `--dry-run` 時は `raw_dir / "dry_run"`, `derived_dir / "dry_run"`、`--pilot` 時は `raw_dir / "pilot"`, `derived_dir / "pilot"` へ分岐。
> - 本番の `v3_gate_decision.json` を mock や pilot で上書きすることを物理的に排除。
> - `affective_empathy_eval/run.py` の gate 判定も `--dry-run` 時は `dry_run/v3_gate_decision.json` を参照するように連動。

> [!NOTE]
> **3. V2 Confirmatory H1/H2 の Matched 厳格化と Recovery AUC の導入**
> - H1/H2 で Matched-Plain が無い場合に Native-Chat へ自動 fallback する処理を排除（Primary は厳格に matched のみ、欠損時は例外）。Native-Chat は Secondary として完全分離。
> - H1 に既存の幾何指標（Procrustes distortion, RSA）を `H1a_geometry_reorganization` として追加し、ピーク深度シフトを `H1b_decodability_profile_reorganization` と併記。
> - H4 の Primary を層選定バイアスのない「Recovery AUC」（台形積分）に移行し、`max_recovery_ratio` は Secondary として保持。

---

## 修正対象コンポーネントとファイル別変更計画

### 1. V3 Stage パイプライン

#### [MODIFY] [`v3/primary/run_rq1_state_induction.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq1_state_induction.py)
- **Sufficiency / Endogenous relevance の分離**:
  - `prompt_neu_self` を評価して `clean_neu_ev, clean_neu_ea` を取得。
  - Dose-response / Specificity / Controls を `prompt_neu_self` に $d_V, d_A$ を注入して測定。
  - Subspace removal は `prompt_aff_self` から $Q$ を除去し、`clean_neu` との差分から減衰を測定。
  - シミュレーション関数 `simulate_state_induction` もこの定義に合わせて整合。
- **出力先ディレクトリの完全分離**:
  - `args.dry_run` $\to$ `raw_dir / "dry_run"`, `derived_dir / "dry_run"`
  - `args.pilot` $\to$ `raw_dir / "pilot"`, `derived_dir / "pilot"`
- **RunManifest の出力**:
  - `manifest_rq1_{family}.json` を出力し、再現性と cache invalidation を担保。

#### [MODIFY] [`v3/primary/run_rq3_path_mediation.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq3_path_mediation.py)
- **Discovery D profile の VA 両軸化**:
  - GroupKFold で $y_v$ と $y_a$ の両方に対して OOF 予測を行い、$R^2_V, R^2_A$ を算出。
  - `d_stim_v_profile`, `d_stim_a_profile`, `d_stim_joint_profile` を保存し、`stimulus_peak_layer = np.argmax(d_stim_joint_profile)` で中継層候補を選定。

#### [MODIFY] [`v3/primary/run_rq2_spatiotemporal_maps.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq2_spatiotemporal_maps.py)
- **Causal サンプルの Stratified 抽出**:
  - 15サンプルの選定に共通関数 `stratified_causal_subset` を適用し、8感情カテゴリからの偏りを排除。
- **Teacher-forced Joint Sequence の明記**:
  - 出力 JSON に `"stage_evaluation_mode": "teacher_forced_joint_sequence"` を追加。

#### [MODIFY] [`v3/README.md`](file:///mnt/nas/home/hiromi/src/emo2/v3/README.md)
- RQ1 の記述を「QR分解」から「rank-aware SVDによる直交情動部分空間Qの構成」へ修正。
- RQ2 を「teacher-forced candidate generation stages」と明記。

---

### 2. V2 Stage パイプライン

#### [MODIFY] [`v2/primary/run_rq4_recovery_patching.py`](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq4_recovery_patching.py)
- **Dry-run 出力先分離**:
  - `args.dry_run` 時に `raw_dir / "dry_run"`, `derived_dir / "dry_run"` へ分岐。
- **Reader / Self の split seed 統一**:
  - 外側で一意に `train_indices, eval_indices` を分割し、Reader と Self に共通の split を渡す。
- **Recovery AUC の算出**:
  - 各タスクの回復曲線に対し、`auc_recovery = float(np.trapezoid(recovery_ratios, depths))`（または `np.trapz`）を算出し結果辞書へ格納。

#### [MODIFY] [`v2/primary/run_confirmatory_analysis.py`](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_confirmatory_analysis.py)
- **H1/H2 の Matched Primary 厳格化**:
  - Primary は matched-plain のみとし、欠損時は `RuntimeError`。Native-chat は Secondary として別キーに出力。
- **H1 への幾何指標併記**:
  - `rq1_geometry` から `reader_distortion_matched`, `self_distortion_matched`, `rsa_reader_matched`, `rsa_self_matched` を集約し、`H1a_geometry_reorganization` として追加（`H1b_decodability_profile_reorganization` と併記）。
- **H4 の Primary を Recovery AUC に移行**:
  - `diff_auc_self_minus_reader` を Primary とし、`diff_max_recovery` を Secondary に。
- **N=4 Descriptive Bootstrap の明記**:
  - 出力メタデータに `"n_families": 4`, `"inference_scope": "descriptive_cross_family_bootstrap"` を追加。

---

### 3. コアモジュール・共通ユーティリティ

#### [NEW] [`src/affective_empathy_eval/affect_directions.py`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/affect_directions.py)
- `AIPSY_EXPECTED_DIRECTION` を一元定義（grief, terror, rage, ecstasy 等の V, A 期待方向符号）。

#### [MODIFY] [`behavioral/analysis/summarize_behavioral_aipsy.py`](file:///mnt/nas/home/hiromi/src/emo2/behavioral/analysis/summarize_behavioral_aipsy.py)
- `EXPECTED_DIRECTION` を `AIPSY_EXPECTED_DIRECTION` からインポートするように共通化。

#### [MODIFY] [`v1/primary/run_phase_c.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_c.py)
- `EXPECTED_DIRECTION` を `AIPSY_EXPECTED_DIRECTION` からインポートするように共通化。

#### [MODIFY] [`src/affective_empathy_eval/data.py`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/data.py)
- `stratified_causal_subset(df, n_samples, seed, stratify_col="target_emotion")` を追加。
- `v3_stimulus_covariate`: 人間 VA が無い場合、`target_emotion`, `intensity`, `domain` を one-hot ダミー変数化して返すよう改修。

#### [MODIFY] [`src/affective_empathy_eval/run.py`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/run.py)
- `run_v3` 内の `v3_gate_allows_continuation` 呼び出しで、`args.dry_run` 時は `v3/results/derived/dry_run/v3_gate_decision.json` を参照するように修正。

---

## 検証計画 (Verification Plan)

### 自動テスト
- `tests/test_confirmatory_pipeline.py`: 新しい H1a/H1b、H4 AUC、V3 gate 分離に対応するテストを追加・更新。
- `tests/test_production_entrypoints.py`: 全 CLI 引数互換性と dry-run 分離を検証。
- `tests/test_refinement_suite.py`: 新しいモジュールインポートと共通関数のテスト。
- コマンド:
  ```bash
  .venv/bin/python -m pytest tests/test_confirmatory_pipeline.py tests/test_production_entrypoints.py tests/test_refinement_suite.py -v
  ```

### Dry-run 検証
- V3 RQ1 dry-run を実行し、`v3/results/raw/dry_run/` および `v3/results/derived/dry_run/` のみに出力され、本番ディレクトリが一切汚染されないことを確認。
- V2 RQ4 dry-run を実行し、`v2/results/raw/dry_run/` および `v2/results/derived/dry_run/` のみに出力されることを確認。
