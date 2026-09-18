# 全体改修タスクリスト (10項目の完全修正)

- [x] **Task 1: パッケージ一本化と壊れたexportの修正 (最優先・pytest 100% pass)**
  - [x] `src/affective_empathy_eval/data.py` に `load_emobank_csv = load_emobank`, `load_aipsy_csv = load_aipsy_affect` エイリアスを追加
  - [x] `src/affective_empathy_eval/evaluation.py` に `Evaluator = InterventionEvaluator` エイリアスを追加
  - [x] `src/affective_empathy_eval/extraction.py` に `extract_activations_batch` ヘルパーを追加
  - [x] `src/affective_empathy_eval/intervention.py` に `HookManager = ActivationPatcher`, `RepresentationSteering = SteeringController` エイリアスを追加
  - [x] `src/affective_empathy_eval/statistics.py` に `apply_benjamini_hochberg = apply_fdr_correction`, `compute_d_z = compute_paired_cohen_dz` エイリアスを追加
  - [x] `src/affective_empathy_eval/geometry.py` の export 整理
  - [x] `src/affective_empathy_eval/__init__.py` の import / export を実在する全シンボルに完全整合
  - [x] `v1/src/` を物理削除して一本化
  - [x] `pytest.ini` を削除し `pyproject.toml` に設定一本化 (`testpaths = ["tests"]`, `pythonpath = ["src"]`)
  - [x] `v1/pyproject.toml` を削除
  - [x] `requirements.txt` のヘッダーに pyproject.toml 正本と明記
  - [x] pytest テストスイートの実行確認（40/40 100% pass）

- [x] **Task 2: V3 Path Mediation の重大人工計算排除と実介入化**
  - [x] `v3/primary/run_rq3_path_mediation.py`（および `v3/scripts/run_v3_path_mediation.py`）の Discovery 内 `Ridge` を 5-fold held-out CV に修正
  - [x] 人工ガウス関数 `c_score` を完全撤去し、各層 $l$ での実 activation intervention（$\alpha=1.0$）による出力変位 $C(l) = \sqrt{\Delta V(l)^2 + \Delta A(l)^2}$ の実測コードへ置換
  - [x] TE / NDE の計算を 5.0 基準から matched-neutral 基準（$|E[V]_{\rm aff} - E[V]_{\rm neu}|$）へ修正

- [x] **Task 3: V3 State Induction の 5.0 fallback 禁止**
  - [x] `v3/primary/run_rq1_state_induction.py`（および `v3/scripts/run_v3_state_induction.py`）の 5.0 fallback を禁止し、matched-neutral 欠損時は明示的エラー（`ValueError`）を発生させる

- [x] **Task 4: V2 / V3 Primary スクリプトの wrapper 解消（実装本体配置）**
  - [x] `v2/primary/run_rq3_causal_map.py` に `v2/scripts/run_v2_2x2_causal_map.py` の実装本体を統合し `sys.path.insert` を撤廃
  - [x] `v2/primary/run_rq4_recovery_patching.py` に `v2/scripts/run_v2_recovery_patching.py` の実装本体を統合
  - [x] `v2/primary/run_confirmatory_analysis.py` に `v2/scripts/run_v2_confirmatory_analysis.py` の実装本体を統合
  - [x] `v3/primary/run_rq1_state_induction.py` に `v3/scripts/run_v3_state_induction.py` の実装本体を統合
  - [x] `v3/primary/run_rq2_spatiotemporal_maps.py` に `v3/scripts/run_v3_spatiotemporal_maps.py` の実装本体を統合
  - [x] `v3/primary/run_rq3_path_mediation.py` に `v3/scripts/run_v3_path_mediation.py` の実装本体を統合
  - [x] `v3/primary/run_confirmatory_replication.py` に `v3/scripts/run_v3_confirmatory_replication.py` の実装本体を統合

- [x] **Task 5: 全 README の記述・config パス修正**
  - [x] root `README.md`: Behavioral Stage を `behavioral/` に、V1 を Prompt-End Normalized に、最新ディレクトリ構造に更新
  - [x] `v1/README.md`: Prompt-End 記述に修正、Behavioral の詳細を `behavioral/README.md` へ委譲、Phase B の LLM Judge 矛盾を解消、過去の数値に (previous run) と注記
  - [x] `behavioral/README.md`: 「接地度」を human-affect correspondence に統一
  - [x] `v2/README.md` & `v2/primary/README.md`: config パスを root `configs/` に修正
  - [x] `v3/README.md` & `v3/primary/README.md`: config パスを root `configs/` に修正

- [x] **Task 6: 旧 results のアーカイブとディレクトリ初期化**
  - [x] `scripts/archive_and_clean_results.py` を実行して `archive/results_pre_rerun_20260918/` へ退避し、`behavioral/results/`, `v1/results/`, `v2/results/`, `v3/results/` を空化

- [x] **Task 7: 不要ファイル・ディレクトリの整理**
  - [x] `.DS_Store` および `._*` ファイルの全削除
  - [x] `scratch/` の消去
  - [x] ルート直下の `get_stats.py`, `get_md_tables.py`, `get_arousal_stats.py` を `tools/reporting/` へ移動
  - [x] `fairshare_gpu/` を `docs/infrastructure/` へ移動
  - [x] `docs/` 配下の古い draft を `docs/archive/` に退避・整理
