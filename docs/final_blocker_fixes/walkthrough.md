# 最終 Blocker 修正およびパイプライン検証結果 (Walkthrough)

ユーザーレビューで指摘された **5大 BLOCKER** および統計・再現性項目について、指示された順序で厳密に修正・検証を実施しました。

---

## 修正内容のまとめ

| 優先度 | 項目 | 修正対象ファイル | 修正内容 |
|---|---|---|---|
| **BLOCKER 1** | V1 Phase C `yaml` 未インポート | `v1/primary/run_phase_c.py` | `import yaml` を追加し NameError を解消 |
| **BLOCKER 2** | AIPsy 集計 CSV 名不整合 & サイレントスキップ | `behavioral/analysis/summarize_behavioral_aipsy.py` | `behavioral_aipsy_*_4split.csv` を canonical 対象とし、0件時は `FileNotFoundError` を送出。モデル名抽出プレフィックス・サフィックス除去を修正 |
| **BLOCKER 3** | AIPsy dry-run の偽の green | `behavioral/primary/run_behavioral_aipsy.py` | `pd.read_csv(stim_path).copy()` ベースで `split`, `pair_id`, `triplet_id`, `emotion` を完全保持。感情方向に応じたモック数値を付与 |
| **BLOCKER 4** | Sequence-Likelihood の論文定義不一致 | `src/affective_empathy_eval/likelihood.py`<br>`behavioral/primary/run_behavioral_aipsy.py`<br>`v1/primary/run_phase_c.py`<br>`v1/primary/phase_c/run_e6_specialization.py`<br>`configs/v1_experiments.yaml`<br>`configs/v2_experiments.yaml`<br>`configs/v3_experiments.yaml` | `normalize_length: bool = True` をデフォルトとし、全 Primary 呼び出しを length-normalized log-likelihood $s_y(x)=\frac{1}{\|y\|}\sum_t\log p(y_t\|x,y_{<t})$ に統一。各 YAML 設定に `sequence_likelihood.normalize_length: true` を明示 |
| **BLOCKER 5** | 古いキャッシュの再利用防止 | `src/affective_empathy_eval/manifests.py` | `DEFAULT_CODE_VERSION` を `"2.2.0"` に更新。`is_manifest_matching` のデフォルト引数で `expected_code_version=DEFAULT_CODE_VERSION` を検証。manifest に `sequence_likelihood_normalization: "token_mean"` を保存・照合 |
| **要修正 6** | `compute_d_z()` ゼロ分散処理 | `src/affective_empathy_eval/statistics.py` | 差分の標準偏差 `s_delta < 1e-9` または $N < 2$ の場合、`0.0` ではなく `np.nan` を返却するよう数理的修正 |
| **要修正 7** | FDR family の明文化 | `behavioral/README.md` | Sensitivity, Dose-response, Specificity, Coupling の検定 Family（全8モデル × 条件）を明文化 |
| **要修正 8** | AIPsy direction map の固定・記録 | `src/affective_empathy_eval/affect_directions.py`<br>`behavioral/primary/run_behavioral_aipsy.py` | `AIPSY_DIRECTION_VERSION = "1.0.0"` と sha256 ハッシュを定義し、実行 manifest に保存 |
| **要修正 10** | 再現性・環境メタデータ記録 | `pyproject.toml`<br>`scripts/run_production_*.sh` | `requires-python = ">=3.12,<3.13"` に修正。`which python`, `python --version`, `git rev-parse HEAD`, `sha256sum uv.lock` のログ保存を追加 |

---

## 検証結果

### 1. 単体テスト (`pytest -q`)
- **結果**: **115 passed, 1 deselected, 7 warnings in 18.87s**
- **追加テスト**: `test_summarize_behavioral_aipsy_artifacts_exist` を新設し、集計スクリプトによって以下の 4 大成果物実ファイルが正常に生成されることを検証済：
  - `behavioral_aipsy_sensitivity_rq1.csv`
  - `behavioral_aipsy_dose_response_rq2.csv`
  - `behavioral_aipsy_specificity_rq3.csv`
  - `behavioral_aipsy_coupling_rq4.csv`

### 2. 全 Stage Dry-run 実行確認
全 4 ステージを `.venv` 環境下で順次 dry-run 実行し、すべて Return Code 0 で完走しました。

| Stage | コマンド | 判定 | 成果物確認 |
|---|---|---|---|
| **Behavioral** | `python -m affective_empathy_eval.run --stage behavioral --dry-run` | **PASS (RC=0)** | Sensitivity, Dose-Response, Specificity, Coupling の 4 CSV が実生成 |
| **V1** | `python -m affective_empathy_eval.run --stage v1 --dry-run` | **PASS (RC=0)** | 全8モデル Phase A → Phase B → Phase C → E6 → Summarize (`phase_c_cross_model_summary.csv`) 完走 |
| **V2** | `python -m affective_empathy_eval.run --stage v2 --dry-run` | **PASS (RC=0)** | 全4ファミリー RQ1/RQ2 → RQ3 → RQ4 → Confirmatory Analysis 完走 |
| **V3** | `python -m affective_empathy_eval.run --stage v3 --dry-run` | **PASS (RC=0)** | RQ1 (GO判定) → RQ2 (4-Maps) → RQ3 (Mediation) → Confirmatory Replication 完走 |

---

## 本番実行に向けた推奨手順

本番実行を開始する際は、以下の安全退避スクリプトで古い dry-run キャッシュ・一時ファイルを退避してから実行してください：

```bash
# 1. 既存の dry-run / 旧結果を archive/results_<timestamp>/ 配下へ安全退避
bash scripts/clear_stage_results.sh

# 2. 本番実行の開始 (例: 全パイプライン一括実行)
bash scripts/run_production_all.sh cuda:0
```
