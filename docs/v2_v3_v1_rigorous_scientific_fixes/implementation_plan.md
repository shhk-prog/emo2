# 実装計画書: 全25項目 実装上の厳密化・バグ修正

本計画は、論文の中心ストーリー：
$$
\boxed{\text{Behavioral Covariation} \rightarrow \text{V1 Representation / Causal Overlap} \rightarrow \text{V2 Post-training-Associated Reorganization} \rightarrow \text{V3 Causal Leverage}}
$$
を完全に維持しながら、本番全再実行における測定妥当性・再現性・データ整合性を保証するための全25項目の修正計画です。

---

## ユーザー確認事項 (User Review Required)

> [!IMPORTANT]
> - 本計画により新しい RQ や仮説を増やすことはありません。既存の RQ1〜RQ3 (V2), RQ1〜RQ4 (V3), Phase A〜C (V1) の枠組みと測定論理を強固にします。
> - dry-run 実行結果の保存先を `results/dry_run/` 配下に完全分離するため、本番の `results/raw/` および `results/derived/` が dry-run で上書きされるリスクを完全に排除します。
> - V3 において matched-neutral が欠損している場合は警告ではなく `ValueError` を発生させ、実験の完全性を担保します。

---

## 主な修正内容と対象ファイル

### 1. V2 Confirmatory Analysis & Cross-decoding (項目 1, 2, 3, 4, 19, 20, 21, 25)
- **対象ファイル**:
  - `v2/primary/run_confirmatory_analysis.py`
  - `v2/primary/run_rq1_rq2_cross_decoding.py`
- **変更内容**:
  1. **H1/H2 Schema 追従**:
     - `gdata["rq2_sharing"][axis]` から `base_r2_reader`, `base_r2_self`, `inst_matched_r2_reader`, `inst_matched_r2_self` を取得。
     - `depths = np.linspace(0.0, 1.0, L)` を用いて各ピーク深度を `compute_peak_depth()` で算出。
     - 出力形式: `valence.reader.shift`, `valence.self.shift`, `arousal.reader.shift`, `arousal.self.shift`。
  2. **H1 判定修正**:
     - `reorganization_supported = (ci_low > 0.0 or ci_high < 0.0)` とし、片側 shift を強制しない。
     - 主出力を `mean shift`, `95% CI`, `per-family shift` とする。
  3. **H2 直接利用**:
     - `delta_sharing_matched` を直接読み込み、平均値を計算（Primary: Base plain vs Instruct matched-plain, Secondary: Base plain vs Instruct native-chat）。
  4. **Geometry 返り値の補完**:
     - `analyze_v2_geometry_and_sharing()` の返り値に `results["relative_depths"] = depths` と `results["num_layers"] = num_layers` を追加。
  5. **H3 LMM の整理**:
     - 検定名を `H3_causal_profile_reorganization_lmm` に変更。
     - モデル式に `C(family)` 固定効果を追加: `c_v ~ C(family) + C(alignment) * C(task) * relative_depth`。
     - Primary FDR 対象を `alignment × depth`, `alignment × task`, `alignment × task × depth` に限定。
  6. **表現の適正化**:
     - コメント等の「純粋事後学習効果」を「Prompt-format-controlled Base–Instruct comparison」に置換。

### 2. V2 Causal Map キャッシュと Pair-level CSV の保護 (項目 5)
- **対象ファイル**:
  - `v2/primary/run_rq3_causal_map.py`
- **変更内容**:
  - family ごとの pair-level CSV を `v2/results/derived/pair_level/v2_causal_pair_level_{fam}.csv` に保存。
  - cache hit 判定時に、JSON だけでなく上記 pair-level CSV が存在することを確認し、存在する場合はレコードを読み出して `all_pair_level_records.extend(...)` する。
  - CSV が欠損している場合は cache invalid として再計算を行い、空 DataFrame による上書きを完全に防ぐ。

### 3. V1 E4 Target Direction Aligned Effect & Transfer Ratio (項目 6, 7)
- **対象ファイル**:
  - `v1/primary/run_phase_c.py`
- **変更内容**:
  - `EXPECTED_DIRECTION` マッピング辞書（grief: V=-1, ecstasy: V=+1, A=+1等）を定義。
  - 各ペアの shift に対して `aligned_m_v = sign_v * m_sv` 等を計算。
  - Primary 指標を `aligned_matched_shift`, `aligned_random_shift`, `aligned_specificity` とし、raw signed shifts を Secondary とする。
  - Transfer Ratio を `mean(aligned_reader_to_self_v) / (mean(aligned_self_to_self_v) + eps)` とし、分母が `< 0.05` の場合は `np.nan` とする。

### 4. V3 RQ3 Path Mediation & Mediator Selection (項目 8, 9, 10, 13)
- **対象ファイル**:
  - `v3/primary/run_rq3_path_mediation.py`
- **変更内容**:
  - 先頭8件選択をやめ、`target_emotion` による層化抽出（seed 固定、均等サンプリング）を実装。
  - 各層で Valence/Arousal それぞれの介入方向 $d_V, d_A$ を fit し、介入実験を行って $C_V(l), C_A(l)$ を測定。
  - $C_{\text{joint}}(l) = (C_V(l) + C_A(l)) / 2$ の最大値をとる層を `mediator_layer = np.argmax(C_joint)` として選定。
  - `mu_neu` fallback（全体平均）を削除し、neutral 表現が存在しない場合は `ValueError` を送出。
  - attenuation ratio 計算時、`natural_shift > 0.05` のサンプルのみを対象とし、Primary を `absolute mediated attenuation`、Secondary を `attenuation ratio` とする。

### 5. V3 Confirmatory Replication & RQ2 (項目 11, 12, 14, 15, 22)
- **対象ファイル**:
  - `v3/primary/run_confirmatory_replication.py`
  - `v3/primary/run_rq2_spatiotemporal_maps.py`
  - `src/affective_empathy_eval/prompts.py`
- **変更内容**:
  - matched-neutral fallback（自己報告や全体平均）を削除し、欠損時は `ValueError`。
  - attenuation ratio のゼロ近傍除外（`natural_shift > 0.05`）。
  - `validate_stage_index_invariance()` を `src/affective_empathy_eval/prompts.py` に配置し、RQ2 開始時にも81候補での不変性を検証。
  - `run_rq2_spatiotemporal_maps.py` における `C_A[l, s_idx]` の重複代入を削除。
  - V3 Confirmatory サマリーに H4 定量値（temporal contrast $pre_V - candidate\_start$ の平均および 95% CI）を追加。

### 6. 成果物保護・実行分離・メタデータ検証 (項目 16, 17, 18)
- **対象ファイル**:
  - `v2/primary/run_rq1_rq2_cross_decoding.py`
  - `v2/primary/run_rq3_causal_map.py`
  - `v2/primary/run_confirmatory_analysis.py`
  - `v3/primary/run_rq2_spatiotemporal_maps.py`
  - `v3/primary/run_rq3_path_mediation.py`
  - `v3/primary/run_confirmatory_replication.py`
  - `src/affective_empathy_eval/validation.py`
- **変更内容**:
  - `if args.dry_run:` の場合は出力先を `results/dry_run/...` に完全ルーティングし、本番成果物を隔離。
  - manifest validation で `expected_config_hash`, `expected_dataset_hash`, `expected_code_version` 等を検証。
  - cache hit 時は新規 manifest での上書きを行わないガードを徹底。

### 7. ドキュメントおよび表記の整合化 (項目 23, 24)
- **対象ファイル**:
  - `README.md`
  - `docs/behavioral_v1_v2_v3_scientific_refinements/walkthrough.md` 等
- **変更内容**:
  - Behavioral の統計記述から古い検定名（Jonckheere-Terpstra）を排除し、現在の実装である「Direction-aligned two-step monotonic contrast with an intersection-union test (IUT)」に統一。
  - Sensitivity の paired 表現を「We first computed paired clinical–neutral differences, aligned them to the prespecified affective direction, and tested the aligned differences against zero.」に整合。

---

## 検証計画 (Verification Plan)

### 自動テスト (Automated Tests)
1. 構文およびコンパイル検証:
   ```bash
   python -m py_compile $(git ls-files "*.py")
   ```
2. 既存および更新後ユニットテスト:
   ```bash
   pytest tests/test_production_entrypoints.py -v
   pytest tests/test_confirmatory_pipeline.py -v
   pytest tests/ -v
   ```
3. Lint / Format チェック:
   ```bash
   ruff check .
   ```

### 動作検証 (Dry-run Execution)
- V1, V2, V3 の統合 CLI dry-run を実行し、ファイル出力先が `results/dry_run/` に分離されていること、キャッシュ機構が正しく動作すること、エラーなく完了することを確認する。

---

## ドキュメントの保存先
本チャットの成果物は規約に従い、以下に保存します：
- `docs/v2_v3_v1_rigorous_scientific_fixes/task.md`
- `docs/v2_v3_v1_rigorous_scientific_fixes/implementation_plan.md`
- `docs/v2_v3_v1_rigorous_scientific_fixes/walkthrough.md`
