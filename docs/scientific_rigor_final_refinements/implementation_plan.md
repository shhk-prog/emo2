# 科学的再現性・測定規約・スキーマの最終厳密化 実装計画

本計画は、論文の単一ストーリー（$\text{Covariation} \rightarrow \text{Representation / Causal Overlap} \rightarrow \text{Post-training-associated Reorganization} \rightarrow \text{Causal Utilization}$）を堅持した上で、本番集計時に停止する P0 バグ（`cn_vals` NameError）を即時修正し、テストスイートの API 不一致、戻り値スキーマ、E6 候補形式、Manifest メタデータ、dtype、感度分析、H4 サンプルレベル統計、frozen confirmatory sites 等の重要修正を完全に解消することを目的とします。

## User Review Required

> [!IMPORTANT]
> **729 VAD vs 81 VA 感度分析の実装方針**
> 論文中の「同一サブセットでの 729 vs 81 感度分析により傾向維持を確認」という主張を実証するため、[`scripts/run_candidate_space_sensitivity.py`](file:///mnt/nas/home/hiromi/src/emo2/scripts/run_candidate_space_sensitivity.py) を新規追加します。
> 同一刺激に対して 729 VAD と 81 VA の両空間で Sequence Likelihood を算出し、$\Delta V, \Delta A$ 相関、符号一致率（Direction Agreement）、順序一貫性（Rank Consistency）、Reader–Self カップリングの安定性を検証します。

> [!IMPORTANT]
> **V2 H4 統計のサンプル／ペアレベルへの強化**
> 4-family bootstrap（$N=4$）のみに依存する確証的統計の脆弱性を防ぐため、[`v2/primary/run_rq4_recovery_patching.py`](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq4_recovery_patching.py) において `family, pair_id, task, layer, relative_depth, recovery_ratio, recovery_emd` のサンプル・ペアレベルレコードを出力・集計し、多層的な統計報告を可能にします。

## Open Questions
現時点で未解決の疑問点はありません。提示された全13項目について、コードベースとの整合性を完全に確認済みです。

---

## Proposed Changes

### Component 1: Behavioral Analysis P0 Fix & Verification

#### [MODIFY] [summarize_behavioral_aipsy.py](file:///mnt/nas/home/hiromi/src/emo2/behavioral/analysis/summarize_behavioral_aipsy.py)
- line 272 付近で欠落していた `cn_vals = cneu_df[c_col].dropna().values` の定義を復元し、`NameError: name 'cn_vals' is not defined` を解消。

#### [NEW] [test_behavioral_specificity.py](file:///mnt/nas/home/hiromi/src/emo2/tests/test_behavioral_specificity.py)
- clinical, complex_neutral, neutral を含む最小ダミー DataFrame で RQ3 特異性集計関数が正常に完走することを検証する単体テストを追加。

---

### Component 2: V1 Test & API Alignment

#### [MODIFY] [run_phase_a.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_a.py)
- `evaluate_cross_decoding_and_geometry` において、グループ不足時（`n_splits < 2`）の戻り値辞書キーを通常時と完全に統一:
  - `"r2_within_r"`, `"r2_within_s"`, `"r2_cross_r_to_s"`, `"r2_cross_s_to_r"`, `"direct_transfer_score"`, `"rsa_correlation"`, `"r2_aligned_transfer"`, `"geometry_pattern": "insufficient_groups"`, `"is_held_out": True`

#### [MODIFY] [test_v1_token_and_probe_alignment.py](file:///mnt/nas/home/hiromi/src/emo2/tests/test_v1_token_and_probe_alignment.py)
- import 対象を `evaluate_cross_decoding_and_geometry` に修正し、統一されたスキーマのキーで NaN 返却をアサート。高速テストの failure を解消。

---

### Component 3: V1 E6 Candidate Builder Unification

#### [MODIFY] [run_e6_specialization.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/phase_c/run_e6_specialization.py)
- 独自のスペース入り `build_vad_candidates()` を完全削除。
- `from affective_empathy_eval.likelihood import build_vad_candidates` を使用し、全リポジトリで単一の compact JSON（`separators=(',', ':')`）に一本化。

---

### Component 4: Manifest Provenance & Default Settings

#### [MODIFY] [manifests.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/manifests.py)
- `DEFAULT_INTERVENTION_VERSION` を `"none"` に変更。

#### [MODIFY] [run_behavioral_emobank.py](file:///mnt/nas/home/hiromi/src/emo2/behavioral/primary/run_behavioral_emobank.py) & [run_behavioral_aipsy.py](file:///mnt/nas/home/hiromi/src/emo2/behavioral/primary/run_behavioral_aipsy.py)
- `create_run_manifest` 呼び出し時に `candidate_space="VAD_729"`, `dataset_path=str(stim_path)`, `intervention_version="none"` を明示。

#### [MODIFY] [run_phase_a.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_a.py) 他 V1 スクリプト
- `candidate_space="VAD_729"`, `intervention_version="none"` を明示。

---

### Component 5: dtype Unification (V2 Stage)

#### [MODIFY] [run_rq1_rq2_cross_decoding.py](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq1_rq2_cross_decoding.py)
- モデルロード時の `torch_dtype` を `torch.bfloat16` に変更し、V2（RQ1〜RQ4）の dtype を完全に統一。

---

### Component 6: Candidate-Space Sensitivity Analysis

#### [NEW] [run_candidate_space_sensitivity.py](file:///mnt/nas/home/hiromi/src/emo2/scripts/run_candidate_space_sensitivity.py)
- 同一のモデル・刺激・プロンプトに対して 729 VAD と 81 VA の両候補空間で評価を行い、相関（Pearson $r$, Spearman $\rho$）、符号一致率、順序一貫性を定量化して出力する解析スクリプトを実装。

---

### Component 7: V2 H4 Sample-Level Statistical Records

#### [MODIFY] [run_rq4_recovery_patching.py](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq4_recovery_patching.py) & [run_confirmatory_analysis.py](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_confirmatory_analysis.py)
- ペア／サンプルレベルの recovery 指標を出力・集計し、4-family bootstrap に加えた重層的統計を担保。

---

### Component 8: V3 Frozen Confirmatory Sites Artifact

#### [MODIFY] [run_rq2_spatiotemporal_maps.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq2_spatiotemporal_maps.py) & [run_confirmatory_replication.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_confirmatory_replication.py)
- Discovery 終了時に `frozen_confirmatory_sites.json` を保存し、Confirmatory 側で読み込んで使用する設計を確立。

---

### Component 9: Seed & Config Management

#### [MODIFY] V3 Primary Scripts
- 固定値 `42`, `43` を `config["seed"]` からの派生値に統一。

---

### Component 10: Documentation, Outlines & pytest Configuration

#### [MODIFY] [paper_outline.md](file:///mnt/nas/home/hiromi/src/emo2/docs/v3_prerun_five_fixes/paper_outline.md)
- V2 の内容を現行 production story（Held-out decodability & cross-decoding, RSA / Procrustes, causal peak relocation, distribution recovery）へ更新。

#### [MODIFY] 各 Stage README (`behavioral/README.md`, `v1/README.md`, `v2/README.md`, `v3/README.md`)
- Section 番号を §4〜§7 に統一し、V1 の `in Base Models` を削除。

#### [MODIFY] [README.md](file:///mnt/nas/home/hiromi/src/emo2/README.md)
- 表内の `\multicolumn` を段落注記へ整形。

#### [MODIFY] [pyproject.toml](file:///mnt/nas/home/hiromi/src/emo2/pyproject.toml)
- `addopts = "-m 'not slow'"` を追加し、通常 `pytest` を高速完走可能に整理。

---

## Verification Plan

### Automated Tests
1. **構文チェック (py_compile)**:
   修正スクリプト全件のコンパイル確認。
2. **高速テストスイート**:
   ```bash
   .venv/bin/python -m pytest -v
   ```
   （全テスト pass、0 failed を確認）
3. **新規感度分析の動作検証**:
   ```bash
   .venv/bin/python scripts/run_candidate_space_sensitivity.py --dry-run
   ```
4. **Behavioral 特異性テストの検証**:
   ```bash
   .venv/bin/python -m pytest tests/test_behavioral_specificity.py -v
   ```

### Manual Verification
- README および paper_outline の Section 番号とタイトルの目視確認。
