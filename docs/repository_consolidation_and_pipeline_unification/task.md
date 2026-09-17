# タスク定義: リポジトリ正本一本化・Primary Pipeline 確立・構成整理

## 目的
リポジトリ全体の構造を「Behavioral → V1 → V2 → V3」という論文の論理的ストーリーに完全に一致させ、重複パッケージの解消、Primary / Exploratory / Legacy の物理的完全分離、キャッシュ・メタデータの厳密化、相対深度の数学的定義統一を行い、第三者が完全に再現・検証可能な公開リサーチコードベースを完成させる。

## 主要タスク項目と進捗状況

1. **共通コードの一本化 (Root `src/affective_empathy_eval` への集約)**: [完了]
   - `v1/src/affective_empathy_eval` の全モジュール（`data.py`, `evaluation.py`, `extraction.py`, `intervention.py`, `manifests.py`, `metrics.py`, `probing.py`, `schemas.py`, `splits.py`, `controls.py`, `prompts.py`, `models/`）を root `src/` へ統合完了。
   - `v3/src/ot_utils.py` を `src/affective_empathy_eval/optimal_transport.py`、`v3/src/diagnostics.py` を `src/affective_empathy_eval/diagnostics.py` として統合完了。
   - `src/affective_empathy_eval/__init__.py` を更新し全モジュールを export。
   - `v1/src/affective_empathy_eval/__init__.py` を root パッケージへの自動リダイレクトおよび DeprecationWarning 出力仕様に変更し、同名重複によるバージョン齟齬を完全解消。

2. **ステージごとの Primary / Exploratory / Legacy 物理的分離**: [完了]
   - **Behavioral (`behavioral/`)**:
     - `behavioral/primary/run_behavioral_emobank.py`（EmoBank 3-Way VAD 729候補評価）
     - `behavioral/primary/run_behavioral_aipsy.py`（AIPsy-Affect 4-Split 729候補評価）
     - `behavioral/analysis/summarize_behavioral_emobank.py`（EmoBank 結果集計・4指標分析）
     - `behavioral/analysis/summarize_behavioral_aipsy.py`（AIPsy 結果集計・Cohen's d_z, FDR補正）
     - `behavioral/README.md`（4大指標: Human Grounding, Sensitivity, Dose-Response & Specificity, Reader-Self Coupling を明記）
   - **V1 (`v1/`)**:
     - `v1/primary/run_phase_a.py`（E1 Decodability & E2 Geometry）
     - `v1/primary/run_phase_b.py`（Phase B Semantic Controls Audit）
     - `v1/primary/run_phase_c.py`（E3 Causal Map & E4 Interchangeability）
     - `v1/primary/phase_c/select_e4_sites.py`（Discovery スプリットに基づく候補層選定）
     - `v1/primary/phase_c/run_e3_causal_map.py`（E3 実行ラッパー）
     - `v1/primary/phase_c/run_e4_interchangeability.py`（E4 実行ラッパー）
     - `v1/primary/phase_c/run_e6_specialization.py`（E6 標的消去・LMM交互作用検定）
     - `v1/primary/phase_c/summarize_phase_c.py`（Phase C 統合レポート生成）
     - `v1/README.md` 更新（5大実験体系 E1, E2, E3, E4, E6 への整理、未実装E5除外の明記、Prompt-End Normalized）
   - **V2 (`v2/`)**:
     - `v2/primary/run_rq1_rq2_cross_decoding.py`（RQ1/RQ2 クロスデコード・幾何解析）
     - `v2/primary/run_rq3_causal_map.py`（RQ3 因果回路再配置・ピーク解離）
     - `v2/primary/run_rq4_recovery_patching.py`（RQ4 復元パッチング）
     - `v2/primary/run_confirmatory_analysis.py`（確証的仮説検証 LMM）
     - `v2/primary/README.md`（RQ1〜RQ4 の明確な研究体系と実行方法）
   - **V3 (`v3/`)**:
     - `v3/primary/run_rq1_state_induction.py`（RQ1 状態誘発と部分空間幾何）
     - `v3/primary/run_rq2_spatiotemporal_maps.py`（RQ2 4-Map 時空間マッピングとピーク解離）
     - `v3/primary/run_rq3_path_mediation.py`（RQ3 因果媒介解析）
     - `v3/primary/run_confirmatory_replication.py`（確証的追試・反証実験）
     - `v3/docs/legacy/README.md`（旧論文ドラフト退避・隔離）
     - `v3/primary/README.md`（3大 RQ + Confirmatory の研究体系と実行方法）

3. **研究仕様・数学的定義の全ステージ統一**: [完了]
   - 相対深度の計算式を全スクリプトで `d = l / (num_layers - 1) if num_layers > 1 else 0.0` に置換・完全統一（旧 `(l + 1) / num_layers` を全廃）。
   - 介入位置を Prompt-End Normalized (`pos = len(prompt_ids) - 1`, `add_special_tokens=False`) に固定。
   - `create_run_manifest` を Primary スクリプトに統合し、git commit, config, metadata を保存。

4. **文書・設定・成果物管理の整理**: [完了]
   - `.gitignore` の厳格化（`.DS_Store`, `._*`, `*.npz`, `**/results/cache/`, `**/results/logs/`, `*.pt`, `.env` 等を除外）。
   - `pyproject.toml` を最新の依存関係（`statsmodels`, `tqdm`, `POT` 等）および pytest 設定の単一正本へ更新。
   - `tests/test_v3_causal_extensions.py` を新設し、root `tests/` で全共通機能および拡張機能のテストを一括実行可能に統合。

5. **旧実験結果のアーカイブ退避と results のクリーン初期化**: [完了]
   - `.gitignore` に `archive/` を追加し Git 管理対象外に設定。
   - `v2/results/prompt_hashes.json` を `v2/configs/prompt_hashes.json` に安全退避。
   - `scripts/archive_and_clean_results.py` を作成し、全ステージ（behavioral, v1, v2, v3）の旧 results を `archive/results_pre_rerun_20260918/` および `archive/results_pre_rerun_20260918.tar.gz` へ一括退避し、各 `results/` ディレクトリを `.gitkeep` のみで初期化するパイプラインを確立。
