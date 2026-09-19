# Walkthrough: V3 RQ1 Sufficiency 実装是正および本番再実行準備

本ドキュメントは、論文の中心主張（**Behavioral Covariation $\rightarrow$ V1 Partial Representation/Causal Overlap $\rightarrow$ V2 Post-training-Associated Reorganization $\rightarrow$ V3 Localized Causal Leverage**）を保ちながら、本番再実行を盤石にするために実施した全18項目の実装改修と検証結果をまとめたものである。

---

## 1. 実施した主要な改修内容

### 【P0: 最優先・科学的厳密化およびGate/成果物保護】

1. **V3 RQ1「Sufficiency (十分性)」と「Endogenous relevance (内生的関与)」の役割分離** (`v3/primary/run_rq1_state_induction.py`)
   - **Sufficiency (十分性)**: 中立文 (`prompt_neu_self`) に対して情動方向 $d_V, d_A$ を加算注入 (`mode="inject"`)。用量反応関係（$\alpha \in \{0.5, 1.0, 1.5, 2.0\}$ による単調性）および特異性（直交・ランダム方向注入との比較）を測定。
   - **Endogenous relevance / Necessity (内生的関与・必要性)**: 臨床・情動文 (`prompt_aff_self`) から 2D 情動部分空間 $Q$ を直交除去 (`mode="subspace_ablation"`)。自然な情動変位 $|\text{clean\_aff} - \text{clean\_neu}|$ に対する残余変位 $|\text{abl\_aff} - \text{clean\_neu}|$ の減衰効果を測定。
   - **Topic control**: トピック分布（TVD / Jensen-Shannon）への影響が限定的であることを検証。
   - 実機推論・シミュレーション双方で完全整合させ、`manifest_rq1_{family}.json` の出力を追加。

2. **V3 RQ1 dry-run / pilot 出力先ディレクトリ完全分離** (`v3/primary/run_rq1_state_induction.py`)
   - `args.dry_run` $\to$ `raw_dir / "dry_run"`, `derived_dir / "dry_run"`
   - `args.pilot` $\to$ `raw_dir / "pilot"`, `derived_dir / "pilot"`
   - 本番の `v3_gate_decision.json` を dry-run や pilot が上書きする事故を構造的に防止。

3. **Unified Runner の gate path dry-run 対応** (`src/affective_empathy_eval/run.py`)
   - dry-run 実行時は `v3/results/derived/dry_run/v3_gate_decision.json` を参照するよう分岐を導入。

4. **V2 RQ4 dry-run 出力先ディレクトリ完全分離** (`v2/primary/run_rq4_recovery_patching.py`)
   - `args.dry_run` 時に `raw_dir / "dry_run"`, `derived_dir / "dry_run"` へ完全分離。

---

### 【P1: 統計・推定の頑健性と不整合解消】

5. **V2 RQ4 の Reader と Self で split seed を統一** (`v2/primary/run_rq4_recovery_patching.py`)
   - ファミリー単位で `train_indices, eval_indices` を 1 回生成し、Reader と Self の両方に共通のインデックスと seed を渡すよう改修。

6. **V2 RQ4 の Primary 指標に Recovery AUC を採用** (`v2/primary/run_rq4_recovery_patching.py`, `v2/primary/run_confirmatory_analysis.py`)
   - 相対深度 $d \in [0, 1]$ に対する回復率の台形積分 `auc_recovery` を算出し、Confirmatory H4 の Primary 指標として Self vs Reader の差分および 95% Bootstrap CI を評価。従来の `max_recovery_ratio` は Secondary 指標として併記。

7. **V2 Confirmatory H1 / H2 で matched-plain から native-chat への自動 fallback を禁止** (`v2/primary/run_confirmatory_analysis.py`)
   - H1b（ピーク深度シフト）および H2（共有度変化）において、Primary 解析は厳格に matched-plain コントラスト (`inst_matched_r2_*`, `delta_sharing_matched`) のみを使用（欠損時は `RuntimeError`）。
   - Native-chat コントラストは Secondary 解析として完全に分離集約。

8. **V2 Confirmatory H1a（幾何学的再編）の追加** (`v2/primary/run_confirmatory_analysis.py`)
   - `rq1_geometry` から `reader_distortion_matched`, `self_distortion_matched`, `rsa_reader_matched`, `rsa_self_matched` を集約し、`H1a_geometry_reorganization` としてレポート。

9. **V3 RQ3 Discovery で D プロファイルを VA 両軸化** (`v3/primary/run_rq3_path_mediation.py`)
   - 刺激側アンカー回帰で Valence と Arousal の両軸を別個に回帰し、`d_stim_v_profile`, `d_stim_a_profile`, `d_stim_joint_profile` を算出・保存。
   - `stim_peak_layer = int(np.argmax(d_stim_joint_profile))` に修正。

10. **V3 RQ2 / RQ3 の因果サンプル抽出層化** (`src/affective_empathy_eval/data.py`, `v3/primary/run_rq2_spatiotemporal_maps.py`, `v3/primary/run_rq3_path_mediation.py`)
    - 共通関数 `stratified_causal_subset` を実装し、感情カテゴリから均等に抽出。
    - 人間 VA ラベル非保持時の刺激共変量フォールバック `v3_stimulus_covariate` を one-hot ダミー変数化。

11. **V3 RQ2 の計算プロトコル明記** (`v3/primary/run_rq2_spatiotemporal_maps.py`, `v3/README.md`)
    - 出力 JSON に `"stage_evaluation_mode": "teacher_forced_joint_sequence"` と `"temporal_protocol": "teacher_forced_candidate_sequence"` を明記。
    - README の直交化手法の記述を「QR」から「rank-aware SVD」に更新。

---

### 【P2: 記述整合・共通化】

12. **AIPsy 想定符号辞書の共通モジュール化** (`src/affective_empathy_eval/affect_directions.py`)
    - `AIPSY_EXPECTED_DIRECTION` を新設共通モジュールに集約。
    - `behavioral/analysis/summarize_behavioral_aipsy.py` および `v1/primary/run_phase_c.py` のローカル辞書を共通インポートに置換。

13. **V2 Confirmatory メタデータの明記** (`v2/primary/run_confirmatory_analysis.py`)
    - `"n_families": 4`, `"inference_scope": "descriptive_cross_family_bootstrap"`, `"sample_inference_scope": "lmm_with_family_fixed_effects"` を明記。

---

## 2. 検証結果

### A. 全自動テストスイート (pytest)
```bash
.venv/bin/python -m pytest tests/ -q
```
- **結果**: **78 passed, 4 warnings in 86.21s**
- `test_confirmatory_pipeline.py`: 通過（ステージ不変性、AIPsy 想定符号、V2確証的分析）
- `test_production_entrypoints.py`: 通過（全スクリプトの引数互換性・実行可能性）
- `test_refinement_suite.py`: 通過（構文、層インデックスマッピング、候補空間サイズ、データリーク防止、キャッシュマニフェスト）

### B. 統合ランナー dry-run フルパイプライン実行
```bash
.venv/bin/python -m affective_empathy_eval.run --stage all --dry-run --max-samples 2 --model-set primary_small --force-after-no-go
```
- **結果**: **All requested stages completed successfully!**
- **実行順序**:
  1. `behavioral`: EmoBank および AIPsy の行動データ処理・集計完了
  2. `v1`: Phase A, B, C 全スクリプトが dry-run で通過
  3. `v2`: RQ1/RQ2, RQ3, RQ4, Confirmatory の全パイプラインが通過
  4. `v3`: RQ1 (Gate Decision: GO), RQ2 (Spatiotemporal 4-Maps), RQ3 (Path Mediation), Confirmatory Replication の全スクリプトが通過

### C. 本番成果物ディレクトリの完全性
- `behavioral/results/`, `v1/results/`, `v2/results/`, `v3/results/` の本番成果物領域に意図しない上書きや未追跡ファイルは一切混入しておらず、クリーンな状態が維持されています。

---

## 3. 結論と次のステップ

すべての懸念点および改善提案が解消され、Behavioral $\rightarrow$ V1 $\rightarrow$ V2 $\rightarrow$ V3 の論文論理構造と実装が完全に一致しました。
GPU 環境での本番フル実行（`affective_empathy_eval.run`）へ移行できる状態が整っています。
