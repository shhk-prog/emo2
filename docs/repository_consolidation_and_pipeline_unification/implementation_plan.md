# 全体コード修正・再実行前整備実装計画 (Implementation Plan)

本計画は、全実験を再実行する前に必要な10項目の改善を完全・確実に実行し、リポジトリの正本一本化、テスト全通過、Primary パイプラインの完全自立、研究仕様の厳密化を完了させるための詳細設計である。

---

## 1. 概要と修正項目

| 項目 | 現状の問題 | 修正方針 |
|---|---|---|
| **1. 共通パッケージ一本化** | `src/` と `v1/src/` の二重化 | `root src/` を唯一の正本とし、`v1/src/` を完全削除 |
| **2. pytest 収集エラー** | `__init__.py` の不整合 export による collection error | 実在する API への export 修正および必要な互換エイリアス整備、`pytest 100% pass` |
| **3. 設定ファイルの二重化** | `pytest.ini` と `pyproject.toml`、`v1/pyproject.toml` の重複 | `pyproject.toml` に一本化し他を削除 |
| **4. V3 Path Mediation の人工計算** | in-sample probe & ガウス関数 `c_score` | 5-fold held-out CV probe、実 activation intervention による $C(l)$ 測定 |
| **5. V3 TE/NDE の 5.0 基準** | 固定値 5.0 からの差分 | matched-neutral 基準（$\|E[V]_{\rm aff} - E[V]_{\rm neu}\|$）へ変更 |
| **6. V3 State Induction の fallback** | matched-neutral 欠損時に 5.0 へ fallback | Primary 解析では fallback を禁止し例外を送出 |
| **7. V2/V3 Primary の wrapper 依存** | `sys.path.insert` で `scripts/` を import | 実装本体を `primary/` へ統合し自立化 |
| **8. 各 README の古い記述・パス** | `v2/configs/...` の誤パス、Response-Onset、接地度表記 | root `configs/` 基準への統一、Prompt-End Normalized、human-affect correspondence |
| **9. 旧 results のクリア** | 古いキャッシュや結果が大量に残存 | `archive/results_pre_rerun_20260918/` へ退避し `results/` を空化 |
| **10. ゴミファイル・スクリプト整理** | `.DS_Store`, `._*`, `scratch/`, ルート直下の分析スクリプト | 削除、`tools/reporting/`、`docs/infrastructure/`、`docs/archive/` への配置 |

---

## 2. 詳細実装ステップ

### Step 1: パッケージ一本化と壊れた export の修正 (`pytest 100% pass`)
1. `src/affective_empathy_eval/` 各モジュールの整合化:
   - `data.py`: `load_emobank_csv = load_emobank`, `load_aipsy_csv = load_aipsy_affect`
   - `evaluation.py`: `Evaluator = InterventionEvaluator`
   - `extraction.py`: `extract_activations_batch` ヘルパー
   - `intervention.py`: `HookManager = ActivationPatcher`, `RepresentationSteering = SteeringController`
   - `statistics.py`: `apply_benjamini_hochberg = apply_fdr_correction`, `compute_d_z = compute_paired_cohen_dz`
   - `__init__.py`: 実在するシンボルのみを明示的に export
2. 重複ファイルの削除:
   - `v1/src/` の削除
   - `v1/pyproject.toml` の削除
   - `pytest.ini` の削除（`pyproject.toml` に設定一本化）
3. pytest の全テスト実行確認（テスト 100% pass）

### Step 2: V3 Path Mediation の実介入化・held-out 化・matched-neutral 化
1. **Held-out CV Probe**:
   - `Ridge(alpha=10.0)` の fit/predict を Discovery セット内 5-fold CV または held-out split による $R^2$ 評価に変更。
2. **人工ガウス関数の完全撤去と実介入**:
   - `c_score = exp(-((relative_depths[l] - 0.68)**2) / 0.04) * ...` を撤去。
   - Discovery サンプルに対し、各層 $l$ で $\alpha=1.0$ の activation intervention を適用し、自己報告ロジットの変位 $C(l) = \sqrt{\Delta V(l)^2 + \Delta A(l)^2}$ を実測して mediator layer を同定。
3. **Matched-neutral 基準の TE / NDE**:
   - $TE = |E[V]_{\rm aff} - E[V]_{\rm neu}|$, $NDE = |E[V]_{\rm abl} - E[V]_{\rm neu}|$ に修正。

### Step 3: V3 State Induction の 5.0 fallback 禁止
- `v3/primary/run_rq1_state_induction.py`:
  - `neutral_base = float(row.get("neutral_expected_v", row.get("reader_V_neutral", 5.0)))` を修正し、欠損時は `ValueError` を投げる。

### Step 4: V2 / V3 Primary スクリプトの wrapper 解消
- `v2/primary/run_rq3_causal_map.py`: `v2/scripts/run_v2_2x2_causal_map.py` の実装を統合
- `v2/primary/run_rq4_recovery_patching.py`: `v2/scripts/run_v2_recovery_patching.py` の実装を統合
- `v2/primary/run_confirmatory_analysis.py`: `v2/scripts/run_v2_confirmatory_analysis.py` の実装を統合
- `v3/primary/run_rq1_state_induction.py`: `v3/scripts/run_v3_state_induction.py` の実装を統合
- `v3/primary/run_rq2_spatiotemporal_maps.py`: `v3/scripts/run_v3_spatiotemporal_maps.py` の実装を統合
- `v3/primary/run_rq3_path_mediation.py`: 修正済み `run_v3_path_mediation.py` の実装を統合
- `v3/primary/run_confirmatory_replication.py`: `v3/scripts/run_v3_confirmatory_replication.py` の実装を統合

### Step 5: 全 README の修正
- root `README.md`, `v1/README.md`, `v2/README.md`, `v3/README.md`, `behavioral/README.md` を更新。

### Step 6: 旧 results のアーカイブと初期化
- `scripts/archive_and_clean_results.py` により `archive/results_pre_rerun_20260918/` にアーカイブし、各 `results/` を空にする。

### Step 7: ゴミファイルと周辺ファイルの整理
- `.DS_Store`, `._*` 削除
- `scratch/` 削除
- `get_stats.py`, `get_md_tables.py`, `get_arousal_stats.py` を `tools/reporting/` へ移動
- `fairshare_gpu/` を `docs/infrastructure/` へ移動
- `docs/` 内の古い draft を `docs/archive/` へ移動

---

## 3. 検証計画
- `python -m pytest -q` を実行し、100% pass を確認
- 各 Primary スクリプトの `--dry-run` 実行による整合性確認
- ディレクトリ構造とファイルのクリーン度確認
