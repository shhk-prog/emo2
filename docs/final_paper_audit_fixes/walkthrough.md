# 最終論文用コード監査40項目 包括的修正 完了報告書 (Walkthrough)

## 概要
最新版の静的コード監査で指摘された全40項目（P1: 15項目、P2: 25項目）をすべて改修し、プロジェクト専用仮想環境（`.venv`）において構文コンパイル・単体テストスイート（95 tests）・全ステージ dry-run（Behavioral $\to$ V1 $\to$ V2 $\to$ V3）を完全に通過させました。

これにより、論文の4-Stage構造（Covariation $\to$ Representation $\to$ Reorganization $\to$ Utilization）が完全な整合性と厳密な再現性を備えた状態で固定されました。

---

## 修正内容一覧（全40項目）

### グループ1: 共通基盤・設定・プロンプト・マニフェスト
- **Item 19, 20, 21, 33 (モデル設定・リビジョン・dtype固定)**:
  - [configs/models.yaml](file:///mnt/nas/home/hiromi/src/emo2/configs/models.yaml): 各モデルおよびモデルセットに `revision`（コミットSHA）と `inference_dtype: "bfloat16"` を明記。
  - [src/affective_empathy_eval/models/registry.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/models/registry.py): `resolve_architecture_dims` に `revision` を伝播し、`ModelFamilyConfig` に `inference_dtype` を追加。
- **Item 22, 23, 24 (プロンプト共通化・SHA-256・chat template判定)**:
  - [src/affective_empathy_eval/prompts.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/prompts.py): `INSTRUCTION_READER_VAD`, `INSTRUCTION_SELF_VAD` を新設し、`build_reader_prompt_vad`, `build_self_prompt_vad`, `build_reader_prompt_va`, `build_self_prompt_va` を提供。
  - 組み込み `hash()` を廃止し `compute_prompt_hash()` (SHA-256) を実装。
- **Item 17, 18, 25, 26, 32 (マニフェスト完全性・run_id衝突防止)**:
  - [src/affective_empathy_eval/manifests.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/manifests.py): `generate_run_id` にマイクロ秒と UUID 8桁を追加し同一秒衝突を完全防止。
  - `RunManifest` に `tokenizer_revision`, `measurement_space`, `actual_dtype`, `template_mode` を追加。`is_manifest_matching` に厳格照合を実装。
- **Item 34 (Scale 定義の統一)**:
  - [AGENTS.md](file:///mnt/nas/home/hiromi/src/emo2/AGENTS.md): セクション 6.3 に Primary は 1–9 raw integer / expected value scale（期待値・差分）であり、`[-1, 1]` は Secondary であることを明記。
- **Item 38 (V3 unmatched pair 除外記録)**:
  - [src/affective_empathy_eval/data.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/data.py): 除外ペアを理由付きで `v3/results/derived/exclusions_v3.csv` および `v3_exclusions.csv` に保存。

### グループ2: Behavioral (振る舞い解析)
- **Item 16, 37 (チェックポイント分離と厳格キャッシュ検証)**:
  - [behavioral/primary/run_behavioral_emobank.py](file:///mnt/nas/home/hiromi/src/emo2/behavioral/primary/run_behavioral_emobank.py) & [behavioral/primary/run_behavioral_aipsy.py](file:///mnt/nas/home/hiromi/src/emo2/behavioral/primary/run_behavioral_aipsy.py): チェックポイントを `results/checkpoints/` へ分離し、モデル・プロンプトハッシュ・データセットハッシュの一致を検証した上で再開（resume）する安全設計を導入。`checkpoint_used` を manifest に記録。

### グループ3: V1 (表現解析・幾何構造)
- **Item 2, 3, 4, 17, 29, 32, 36 (Phase A 改修)**:
  - [v1/primary/run_phase_a.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_a.py):
    - `evaluate_regression_probe` で測定不能時は 0.0 置換せず `NaN` / `status="failed"` を返却。
    - AIPsy intensity 解析で未知ラベル検出時に `ValueError` を送出。
    - hidden state の非有限値検出時に `FloatingPointError` を送出。
    - 分類プローブで単一クラス fold を安全に検出・ハンドリング。
    - `--config` 引数を追加し、YAML 設定（`seed`, `cv_folds`, `alpha`）を probe に反映。
    - manifest の `candidate_space="N/A"`, `measurement_space="prompt_end_hidden_state"` を設定。
- **Item 4, 5, 6, 7, 17, 32 (Phase B 改修)**:
  - [v1/primary/run_phase_b.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_b.py):
    - 非有限 hidden state の例外化。
    - held-out test の valid ペア数（`n_primary_test_paraphrase_pairs`, `n_primary_test_reversal_pairs`）および `n_train_pairs`, `n_test_pairs` を記録。
    - `n_validated_*` 表現を `n_nonfallback_*` へ完全改名。
    - `phase_b_pairs_quality_audit.csv` に `transformation_method`, `quality_status` を保存。
    - `--limit` 引数をサポート。
- **Item 1, 17, 30 (Phase C 改修)**:
  - [v1/primary/run_phase_c.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_c.py):
    - E4 の confirmatory condition を `discovery_mag_reader` から選択（全データ `magnitude_reader` を撤廃し、selection leakage を完全排除）。
    - E3 未実行時および Discovery 欠損時の heuristic fallback を撤廃し、本番環境で `RuntimeError` を送出。

### グループ4: V2 (再編成解析・因果マッピング)
- **Item 35 (RQ1/RQ2 分割)**:
  - [v2/primary/run_rq1_rq2_cross_decoding.py](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq1_rq2_cross_decoding.py): train/test 分割が空の場合に `ValueError` を送出。
- **Item 8 (RQ3 seed 連動)**:
  - [v2/primary/run_rq3_causal_map.py](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq3_causal_map.py): ハードコードされた `seed=42` を撤廃し、`v2_config["seed"]` から引数伝播。
- **Item 9, 10 (RQ4 ΔEMD & 指標分離)**:
  - [v2/primary/run_rq4_recovery_patching.py](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq4_recovery_patching.py):
    - サンプル単位差分 $\Delta EMD_i = EMD_i^{init} - EMD_i^{patch}$ の平均値として $\Delta EMD$ を正確に算出（$E[D]\times E[R]$ 近似を撤廃）。
    - 戻り値 JSON を `primary_matched_plain`（AUC）と `secondary_peak_localization`（best_layer, max_recovery）に明確に分離。

### グループ5: V3 (時空間4-Map・媒介推論・追試検証)
- **Item 11, 40 (RQ2 spatiotemporal maps)**:
  - [v3/primary/run_rq2_spatiotemporal_maps.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq2_spatiotemporal_maps.py):
    - `stratified_causal_subset` の `seed=42` を引数 `seed` に変更。
    - 旧互換キー `temporal_relative_depth` の単一キー出力を削除し、`temporal_relative_depth_v`, `temporal_relative_depth_a` に完全統一。
- **Item 12, 13, 39, 40 (RQ3 path mediation)**:
  - [v3/primary/run_rq3_path_mediation.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq3_path_mediation.py):
    - `stratified_causal_subset` の `seed=42` を引数 `seed` に変更。
    - `n_splits < 2` の場合に $R^2=0$ のサイレントフォールバックを廃止し、production 時に `ValueError` を送出。
    - `resolved_relative_depth` を必須化（フォールバック廃止）。
    - RQ2 sites から `temporal_relative_depth_v`, `temporal_relative_depth_a` の両方を必須取得。
- **Item 14, 15, 39, 40 (Confirmatory replication)**:
  - [v3/primary/run_confirmatory_replication.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_confirmatory_replication.py):
    - manifest_config に `temporal_relative_depth_v/a`, `temporal_stage_v/a`, `mediation_relative_depth`, `sufficiency_relative_depth`, `frozen_sites_hash` を設定しキャッシュキーへ含める。
    - docstring / コメント内の「pre_V で因果ピーク」表現を「Discovery で同定された frozen site を追試検証」に統一。

### グループ6: スクリプト・感度分析・アーカイブ
- **Item 27 (Archive スクリプト安全化)**:
  - [scripts/archive_and_clean_results.py](file:///mnt/nas/home/hiromi/src/emo2/scripts/archive_and_clean_results.py): 固定 tarball 名を `results_archive_<timestamp>.tar.gz` に変更し、既存アーカイブの上書き・削除を禁止。
- **Item 28 (本番スクリプト重複解消)**:
  - [scripts/run_production_reruns.sh](file:///mnt/nas/home/hiromi/src/emo2/scripts/run_production_reruns.sh): family ループ内の Confirmatory 呼び出しを loop 外で 1 回のみ実行に修正。
- **Item 31 (Candidate Space 感度分析の matched pair 実装)**:
  - [scripts/run_candidate_space_sensitivity.py](file:///mnt/nas/home/hiromi/src/emo2/scripts/run_candidate_space_sensitivity.py):
    - AIPsy matched pair 上で臨床刺激と対照 neutral の差分 $\Delta V_{729}$ vs $\Delta V_{81}$, $\Delta A_{729}$ vs $\Delta A_{81}$ の相関・方向一致度・MAE を算出する設計に完全改修。
    - デフォルトパスを `v1/data/processed/aipsy_4split_all.csv` に設定し、モデルレジストリと連携。

---

## 検証結果

### 1. 構文コンパイル (`compileall`)
全ステージおよび共通ライブラリをプロジェクト専用仮想環境でコンパイルし、エラーがないことを確認しました。
```bash
.venv/bin/python -m compileall behavioral v1 v2 v3 src scripts
# -> 終了コード 0 (全ファイル正常コンパイル完了)
```

### 2. 単体・統合テストスイート (`pytest`)
テストスイート全体を実行し、全テストの通過を確認しました。
```bash
.venv/bin/pytest -q tests/
# -> 95 passed, 1 deselected, 5 warnings in 10.28s (終了コード 0)
```

### 3. Stage All Dry-run スモークテスト
4-Stage パイプライン全体（Behavioral $\to$ V1 $\to$ V2 $\to$ V3）を結合実行し、各ステージ間の成果物伝播と manifest 生成が正常に完走することを確認しました。
```bash
.venv/bin/python -m affective_empathy_eval.run --stage all --model-set primary_small --family qwen --device cpu --dry-run --max-samples 16
# -> All requested stages completed successfully! (終了コード 0)
```

### 4. Candidate Space 感度分析テスト
```bash
.venv/bin/python scripts/run_candidate_space_sensitivity.py --dry-run
# -> Valence: Pearson r = 0.9987, Dir Agreement = 100.00%
# -> Arousal: Pearson r = 0.9971, Dir Agreement = 100.00%
# -> 終了コード 0
```

---

## 本番実行への手順

すべてのコードおよび設定が最新の監査要件に合致したため、ユーザー環境での本番実行（GPU環境）は以下のスクリプトを順次実行することで完了します：

```bash
# 仮想環境のアクティベート
source .venv/bin/activate

# 1. 監査反映後の本番再実行ターゲット（V1 Phase B, V3 RQ2/RQ3/Confirmatory, V2 RQ4）
bash scripts/run_production_reruns.sh cuda:0

# 2. 再集計スクリプトの実行（既存結果から最新サマリーを再生成）
python scripts/reaggregate_v1_phase_a.py
python scripts/reaggregate_v2_summary.py
```
