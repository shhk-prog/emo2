# 実装計画: 本実験前 最終科学的妥当性・整合性修正 (全25項目)

本実験パイプラインの本格再実行に向け、論文のコアストーリー：
> **Behavioral: Covariation → V1: Representation / Causal Overlap → V2: Post-training-Associated Reorganization → V3: Causal Leverage**

を堅持しつつ、本番結果の信頼性と再現性を担保するために残る全25項目の修正を実施します。

---

## ユーザー確認事項 (User Review Required)

> [!IMPORTANT]
> - 本計画では論文のストーリーおよびリサーチクエスチョン（RQ）の追加・変更は行いません。
> - 主な修正は、実機実験成果物のスキーマ整合、正負感情の相殺防止、探索的サイト選定の標本化・両軸化、成果物のキャッシュ・dry-run分離、および統計記述の整合です。

---

## 実施計画 (Proposed Changes)

### 1. V2 Confirmatory & 幾何スキーマ修正 (項目 1〜5, 19〜21, 25)
- **`v2/primary/run_confirmatory_analysis.py`**:
  - H1/H2 スキーマ修正: `gdata["rq2_sharing"][axis]` の `base_r2_reader/self`, `inst_matched_r2_reader/self` を Primary として参照し、Valence/Arousal × Reader/Self の 4 系列のピークシフトを算出。
  - H1 判定緩和: 後方シフト固定ではなく系統的再編 `reorganization_supported = (ci_low > 0.0 or ci_high < 0.0)` とし、平均シフトと 95% CI を主出力とする。
  - H2: `delta_sharing_matched` を直接使用（Primary: Base plain vs Instruct matched-plain, Secondary: Base plain vs Instruct native-chat）。
  - H3: `H3_causal_profile_reorganization_lmm` に名称変更。LMM に family 固定効果 `c ~ C(family) + C(alignment) * C(task) * relative_depth` を導入。Primary FDR 対象項（`PRIMARY_TERMS`）を事前固定。
- **`v2/primary/run_rq1_rq2_cross_decoding.py`**:
  - `analyze_v2_geometry_and_sharing` の返り値に `relative_depths` と `num_layers` を保存。
  - コメント内の「純粋事後学習効果」等の表現を客観的な比較記述に修正。
- **`v2/primary/run_rq3_causal_map.py`**:
  - `v2/results/derived/pair_level/v2_causal_pair_level_{fam_id}.csv` を保存。cache hit 時には既存の family pair-level CSV を読み込んで `all_pair_level_records` を復元し、存在しない場合は再計算（CSV 上書き消失バグの完全防止）。
  - コメント内の表現修正。

### 2. V1 E4 感情方向アラインメント & 伝達比率 (項目 6〜7)
- **`v1/primary/run_phase_c.py`**:
  - AIPsy の 8 感情に対する `EXPECTED_DIRECTION` を導入。
  - 各刺激ペアの感情極性に基づき、`aligned_matched_shift = sign * shift` 等を計算。
  - Primary effect を `aligned_matched_shift`, `aligned_random_shift`, `aligned_specificity` とし、正負感情の相殺を解消（raw signed shift は Secondary に配置）。
  - `transfer_ratio = mean(aligned_m) / mean(aligned_ss)`。微小分母（`abs(mean_ss) < 0.05`）時は NaN とする。

### 3. V3 RQ3 Mediator 選定 & 減衰比率安定化 (項目 8〜11, 13)
- **`v3/primary/run_rq3_path_mediation.py`**:
  - Discovery での Mediator 層探索において、先頭 8 件抽出を廃止し、8 感情から各 1 件ずつ抽出する stratified sampling（シード 42 固定）を実装。
  - 各層で Valence / Arousal 別に $d_V(l), d_A(l)$ を fit し、実介入変位 $C_V(l), C_A(l)$ から $C_{\text{joint}}(l) = (C_V + C_A) / 2$ を計算して mediator layer を決定。
  - Attenuation ratio において `MIN_NATURAL_SHIFT = 0.05` のガードを適用。Primary を absolute mediated attenuation、Secondary を attenuation ratio とする。
  - `mu_neu` fallback を撤廃し、matched-neutral 表現が存在しない場合は例外を送出。
- **`v3/primary/run_confirmatory_replication.py`**:
  - H3 attenuation ratio において同様に `natural_shift > 0.05` ガードを適用。
  - `neu_text` / `train_neutral_reps` の fallback を撤廃し、matched-neutral 不在時は直ちに例外を送出。

### 4. V3 RQ2 トークン位置不変性 & バグ修正 (項目 12, 14, 15, 22)
- **`src/affective_empathy_eval/prompts.py`**:
  - `validate_stage_index_invariance` を共通関数として配置。
- **`v3/primary/run_rq2_spatiotemporal_maps.py`**:
  - 解析開始時に `validate_stage_index_invariance` を実行し、トークン位置の不変性を厳格に確認。
  - `C_A[l, s_idx]` の重複代入行を削除。
- **`v3/primary/run_confirmatory_replication.py`**:
  - Cross-family 主サマリーに H4 定量コントラスト（`stage_causal_v["pre_V"] - stage_causal_v["candidate_start"]` 等の mean + bootstrap CI）を集約。

### 5. キャッシュ・Manifest・Dry-run 出力分離 (項目 16〜18)
- **全ステージ (`v2/primary/*`, `v3/primary/*`)**:
  - `args.dry_run` 指定時は `raw_dir / "dry_run"`、`derived_dir / "dry_run"` に出力先を完全分離し、実実験の成果物を mock で汚染・上書きしない。
  - `is_manifest_matching` に `expected_config_hash`, `expected_dataset_hash`, `expected_code_version` を照合。
  - `cache_hit` 時は manifest を新規設定で上書き保存しない（洗浄防止）。

### 6. ドキュメント & README 統計記述修正 (項目 23〜24)
- **`behavioral/README.md`** & **`README.md`**:
  - Jonckheere-Terpstra の記述を削除し、現行の「Direction-aligned two-step monotonic contrast with an intersection-union test」に更新。
  - Sensitivity の「paired t-test」表記を「Direction-aligned paired clinical–neutral contrast against zero」に精密化。

---

## 検証計画 (Verification Plan)

### 自動テスト
1. **構文コンパイル**:
   ```bash
   .venv/bin/python -m compileall src behavioral v1 v2 v3 tests
   ```
2. **全ユニットテスト**:
   ```bash
   .venv/bin/python -m pytest -q
   ```
3. **ドライラン動作確認**:
   - `run_phase_c.py --dry-run`
   - `run_rq1_rq2_cross_decoding.py --dry-run`
   - `run_rq3_causal_map.py --dry-run`
   - `run_confirmatory_analysis.py --dry-run`
   - `run_rq2_spatiotemporal_maps.py --dry-run`
   - `run_rq3_path_mediation.py --dry-run`
   - `run_confirmatory_replication.py --dry-run`
   出力が `dry_run/` サブディレクトリに分離され、実本番ファイルを一切汚染しないことを確認。
