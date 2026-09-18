# 全体コード修正・再実行前整備 完了確認書 (Walkthrough)

全実験をクリーン再実行するにあたり、リクエストされた重要10項目（パッケージ一本化、pytest 100% pass、V3 Path Mediation の実介入化・held-out化・matched-neutral化、V3 State Induction の 5.0 fallback 禁止、Primary の wrapper 解消、全 README 更新、旧 results アーカイブと初期化、不要ファイル整理）の改修がすべて完了しました。

---

## 1. 完了した改修内容の総括

### ① 共通パッケージの一本化 (`src/affective_empathy_eval/`)
- `v1/src/affective_empathy_eval/` を物理削除し、ルート直下の `src/affective_empathy_eval/` を唯一の正本パッケージとして確定。
- `data.py`, `evaluation.py`, `extraction.py`, `intervention.py`, `statistics.py` に不足していた後方互換エイリアスおよびヘルパー関数を定義。
- `__init__.py` の import / export を整理し、実在しないシンボル（`InterchangeHook`, `InterventionController`, `SubspaceDeflectionHook`, `PromptFormat`, `PromptTemplateManager`）を完全排除。

### ② pytest テストスイート 100% Pass 達成
- 設定ファイルの二重化を解消：`pytest.ini` および `v1/pyproject.toml` を削除し、ルートの `pyproject.toml` に設定を一本化。
- `uv pip install statsmodels` により環境依存関係を完全に同期。
- `apply_benjamini_hochberg` を Pure NumPy フォールバック付きの `apply_fdr_correction` を用いる実装に更新。
- **検証結果**:
  ```text
  .venv/bin/pytest -q tests
  ........................................ [100%]
  40 passed in 5.88s
  ```
  全40件のテストがエラーゼロで 100% pass することを確認。

### ③ V3 Path Mediation の重大人工計算排除と実介入化
- **Held-out CV Probe**:
  Discovery 内での Ridge プローブ学習を、同一データでの in-sample 評価から 5-fold held-out Cross-Validation による $R^2$ 評価に変更。
- **実 Activation Intervention による $C(l)$ 実測**:
  人工的なガウス関数 `exp(-((relative_depth - 0.68)**2) / 0.04)` を完全撤去。各層 $l$ において、Discovery 代表サンプルに対する $\alpha=1.0$ のステアリング介入を行い、自己報告ロジットの変位 $C(l) = |\Delta \text{Report}(l)|$ を実測して mediator layer を同定する実装へ刷新。
- **Matched-Neutral 基準の TE / NDE**:
  固定値 `5.0` からの差分を撤廃し、各刺激ペアに対応する matched-neutral プロンプトに対するモデルの自己報告期待値との差分：
  $$TE = |E[V]_{\rm clean} - E[V]_{\rm neu}|, \quad NDE = |E[V]_{\rm abl} - E[V]_{\rm neu}|$$
  として算出するよう修正。

### ④ V3 State Induction の 5.0 Fallback 禁止
- `v3/primary/run_rq1_state_induction.py`（および `v3/scripts/run_v3_state_induction.py`）において、matched-neutral 列が存在しない場合に `5.0` へサイレント fallback していた箇所を禁止し、明示的に `ValueError` を送出するよう変更。

### ⑤ V2 / V3 Primary スクリプトの自立化（Wrapper 解消）
- `v2/primary/run_rq3_causal_map.py`
- `v2/primary/run_rq4_recovery_patching.py`
- `v2/primary/run_confirmatory_analysis.py`
- `v3/primary/run_rq1_state_induction.py`
- `v3/primary/run_rq2_spatiotemporal_maps.py`
- `v3/primary/run_rq3_path_mediation.py`
- `v3/primary/run_confirmatory_replication.py`
各スクリプトに実装本体を配置し、`sys.path.insert(0, str(scripts_dir))` による wrapper 依存を完全撤廃。`python -m compileall` により全 Primary スクリプトの構文整合性を検証。

### ⑥ 全 README の記述・Config パス修正
- **root `README.md`**: Behavioral Stage を `behavioral/` に、V1 を Prompt-End Normalized に、最新のディレクトリ構造に更新。
- **`v1/README.md`**: Prompt-End Normalized に修正、行動実験の詳細を `behavioral/README.md` へ委譲し前提知見 (previous run) として短く引用、Phase B の「独立Judge検証付き」記述を削除。
- **`behavioral/README.md`**: 接地度を `human-affect correspondence` に統一。
- **`v2/primary/README.md` & `v3/primary/README.md`**: 誤っていた `v2/configs/...`, `v3/configs/...` の実行例を root `configs/...` に統一。

### ⑦ 旧 Results のアーカイブ退避と Results のクリーン初期化
- `scripts/archive_and_clean_results.py` により旧結果を `archive/results_pre_rerun_20260918/` に完全退避。
- `archive/results_pre_rerun_20260918.tar.gz` (55.27 MB) を生成。
- `behavioral/results/`, `v1/results/`, `v2/results/`, `v3/results/` をクリーン・空化（`.gitkeep` のみ配置）。

### ⑧ 不要ファイル・ディレクトリの整理
- 全ディレクトリから `.DS_Store` および `._*`（計22ファイル）を完全削除。
- `scratch/` を初期化。
- ルート直下の分析スクリプト（`get_stats.py`, `get_md_tables.py`, `get_arousal_stats.py`）を `tools/reporting/` へ移動。
- `fairshare_gpu/` を `docs/infrastructure/fairshare_gpu/` へ移動。
- `docs/` 配下の古い過去ファイル（71アイテム）を `docs/archive/` へ退避・整理。

---

## 2. 変更・移動ファイル一覧

| 区分 | ファイル / パス | 変更内容 |
|---|---|---|
| **正本パッケージ** | `src/affective_empathy_eval/__init__.py` | 実在APIのみへの export 整理、存在しないクラスの import 削除 |
| | `src/affective_empathy_eval/data.py` | `load_emobank_csv`, `load_aipsy_csv` エイリアス追加 |
| | `src/affective_empathy_eval/evaluation.py` | `Evaluator = InterventionEvaluator` エイリアス追加 |
| | `src/affective_empathy_eval/extraction.py` | `extract_activations_batch`, `RepresentationExtractor` エイリアス追加 |
| | `src/affective_empathy_eval/intervention.py` | `HookManager`, `RepresentationSteering` エイリアス追加 |
| | `src/affective_empathy_eval/statistics.py` | `apply_benjamini_hochberg`, `compute_correlation_with_ci`, `compute_d_z` 追加 |
| **環境・設定** | `pyproject.toml` | pytest 設定に一本化 (`norecursedirs`, `pythonpath`) |
| | `requirements.txt` | pyproject.toml が正本である旨の注記追加 |
| | `pytest.ini`, `v1/pyproject.toml`, `v1/src/` | 削除（重複解消） |
| **V3 実装** | `v3/primary/run_rq3_path_mediation.py` | held-out 5-fold CV、実介入 $C(l)$ 測定、matched-neutral TE/NDE |
| | `v3/primary/run_rq1_state_induction.py` | 5.0 fallback の禁止と例外送出 |
| **Primary 統合** | `v2/primary/run_*.py`, `v3/primary/run_*.py` | 実装本体の統合（`sys.path.insert` 撤廃） |
| **ドキュメント** | 各 `README.md` (root, v1, v2, v3, behavioral) | 最新仕様・config パス・用語に統一 |
| **アーカイブ** | `archive/results_pre_rerun_20260918/` | 旧 results の退避（tar.gz バックアップ作成済） |
| | `tools/reporting/` | ルート直下の集計スクリプトを移動 |
| | `docs/infrastructure/fairshare_gpu/` | インフラツールの移動 |
| | `docs/archive/` | 過去のドキュメントの退避 |

---

## 3. 次のステップ（全実験再実行の推奨順序）

リポジトリ全体が極めてクリーンになり、テストも 100% 通過する状態が整いました。以下の順序で全実験の再実行を開始することを推奨します：

1. **Behavioral Stage の再実行**:
   ```bash
   python behavioral/primary/run_behavioral_emobank.py --model Qwen/Qwen2.5-0.5B-Instruct --is_instruct --tag qwen05b_instruct --stimuli-path data/processed/stimuli_vad_3way.csv --out-dir behavioral/results/emobank_3way
   python behavioral/primary/run_behavioral_aipsy.py --model Qwen/Qwen2.5-0.5B-Instruct --is_instruct --tag qwen05b_instruct --stimuli-path data/processed/aipsy_affect_4split.csv --out-dir behavioral/results/aipsy_4split
   python behavioral/analysis/summarize_behavioral_emobank.py
   python behavioral/analysis/summarize_behavioral_aipsy.py
   ```
2. **V1 Stage の再実行**:
   ```bash
   python v1/primary/run_phase_a.py --dry-run
   python v1/primary/run_phase_b.py --dry-run
   python v1/primary/run_phase_c.py --dry-run
   ```
3. **V2 Stage の再実行**:
   ```bash
   python v2/primary/run_rq1_rq2_cross_decoding.py --dry-run
   python v2/primary/run_rq3_causal_map.py --dry-run
   python v2/primary/run_rq4_recovery_patching.py --dry-run
   ```
4. **V3 Stage の再実行**:
   ```bash
   python v3/primary/run_rq1_state_induction.py --dry-run
   python v3/primary/run_rq2_spatiotemporal_maps.py --dry-run
   python v3/primary/run_rq3_path_mediation.py --dry-run
   python v3/primary/run_confirmatory_replication.py --dry-run
   ```
