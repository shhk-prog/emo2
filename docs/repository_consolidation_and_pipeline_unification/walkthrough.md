# Walkthrough: リポジトリ正本一本化・Primary Pipeline 確立・構成整理

## 1. 概要
本改修では、リポジトリ全体の正本一本化、Primary / Exploratory / Legacy の物理的完全分離、論文ストーリー（Behavioral → V1 → V2 → V3）とディレクトリ構造の完全同期、仕様（相対深度 $d = l / (L-1)$、Prompt-End Normalized）の統一を実施しました。

---

## 2. 実施した主要改修内容

### 2.1 共通コードの一本化 (Root `src/affective_empathy_eval` への完全集約)
- **重複パッケージ問題の解消**:
  - `src/affective_empathy_eval/` に全モジュール（`likelihood.py`, `geometry.py`, `interventions.py`, `intervention.py`, `statistics.py`, `probing.py`, `data.py`, `controls.py`, `splits.py`, `metrics.py`, `schemas.py`, `manifests.py`, `models/`, `prompts.py`）を集約。
  - `v3/src/ot_utils.py` を `src/affective_empathy_eval/optimal_transport.py`、`v3/src/diagnostics.py` を `src/affective_empathy_eval/diagnostics.py` として正式モジュール化。
  - `src/affective_empathy_eval/__init__.py` で全モジュール・主要関数を一括エクスポート。
  - `v1/src/affective_empathy_eval/__init__.py` を root パッケージへの自動リダイレクトおよび `DeprecationWarning` 仕様に変更し、実行場所依存のバージョン不整合を完全排除。

### 2.2 ステージごとの Primary / Exploratory / Legacy 物理的分離

#### ① Behavioral (`behavioral/`) の新設・独立
- `v1` 内にあった行動実験を独立した第0ステージとして切り出し。
- **Primary スクリプト**:
  - `behavioral/primary/run_behavioral_emobank.py`: EmoBank 3-Way VAD 729候補 Sequence-Likelihood 評価。
  - `behavioral/primary/run_behavioral_aipsy.py`: AIPsy-Affect 4-Split 729候補 Sequence-Likelihood 評価。
- **Analysis スクリプト**:
  - `behavioral/analysis/summarize_behavioral_emobank.py`: EmoBank 集計（人間アノテーション相関、MAE、4大指標）。
  - `behavioral/analysis/summarize_behavioral_aipsy.py`: AIPsy 集計（Cohen's $d_z$、Benjamini-Hochberg FDR補正）。
- **README**:
  - `behavioral/README.md`: 論文導線となる4大行動評価指標（1. Human Grounding, 2. Sensitivity, 3. Dose-Response & Specificity, 4. Reader-Self Coupling）を明記。

#### ② V1 (`v1/`) の整理
- **Primary スクリプト**:
  - `v1/primary/run_phase_a.py`: E1 (Shared Decodability) & E2 (Shared Geometry)
  - `v1/primary/run_phase_b.py`: Phase B Semantic vs. Lexical Controls Audit
  - `v1/primary/run_phase_c.py`: E3 (Causal Map) & E4 (Interchangeability)
  - `v1/primary/phase_c/`:
    - `run_e3_causal_map.py`: E3 全層因果マッピング
    - `select_e4_sites.py`: Discovery スプリットに基づく E4 候補層選定
    - `run_e4_interchangeability.py`: E4 Confirmation スプリット特異性検証
    - `run_e6_specialization.py`: E6 標的消去・LMM交互作用検定
    - `summarize_phase_c.py`: Phase C モデル間統合集計
- **論文ストーリー統一**:
  - `v1/README.md` を更新。未実装の LLM Judge 自由生成評価（旧計画E5）を主表から除外し、5大実験体系（E1, E2, E3, E4, E6）および Phase B（Semantic Controls）に整理。
  - Prompt-End Normalized 介入規則を明記。

#### ③ V2 (`v2/`) の整理
- **Primary スクリプト**:
  - `v2/primary/run_rq1_rq2_cross_decoding.py`: V2-RQ1 & RQ2 クロスデコード・表現幾何
  - `v2/primary/run_rq3_causal_map.py`: V2-RQ3 因果回路再配置・ピーク解離
  - `v2/primary/run_rq4_recovery_patching.py`: V2-RQ4 分布復元パッチング
  - `v2/primary/run_confirmatory_analysis.py`: 確証的仮説検証（LMM, FDR補正）
- **ドキュメント**:
  - `v2/primary/README.md` を作成し、RQ1〜RQ4 の明確な研究体系と実行方法を明記。

#### ④ V3 (`v3/`) の整理
- **Primary スクリプト**:
  - `v3/primary/run_rq1_state_induction.py`: V3-RQ1 状態誘発と部分空間幾何
  - `v3/primary/run_rq2_spatiotemporal_maps.py`: V3-RQ2 時空間4-Map構築とピーク解離
  - `v3/primary/run_rq3_path_mediation.py`: V3-RQ3 因果媒介解析
  - `v3/primary/run_confirmatory_replication.py`: 確証的追試・反証実験
- **ドキュメント・アーカイブ**:
  - `v3/docs/legacy/README.md`: 過去の論文ドラフト（`paper_*.md` 等）をアーカイブ・隔離。
  - `v3/primary/README.md`: 3大 RQ + Confirmatory の研究体系を明記。

---

### 2.3 仕様・数学的定義の全ステージ統一
- **相対計算深度の完全統一**:
  - 全スクリプトにおける相対深度計算式を $$d = \frac{l}{L - 1} \quad (\text{0-based layer } l \in [0, L-1])$$ に置換・統一。旧式の $(l + 1) / L$ を全廃。
- **Prompt-End Normalized の厳格化**:
  - `add_special_tokens=False` かつ `prompt_end = len(prompt_ids) - 1` に統一。
- **メタデータ・再現性追跡**:
  - 全 Primary スクリプトに `create_run_manifest` を組み込み、Git commit ハッシュ、設定パラメータ、実行時刻を自動保存。

---

### 2.4 設定・成果物管理の整理
- **`.gitignore` の厳格化**:
  - OS 生成ゴミファイル（`.DS_Store`, `._*`）、実験キャッシュ（`**/results/cache/`, `**/results/logs/`）、巨大バイナリ（`*.npz`, `*.pt`）、シークレット（`.env`）を除外。
- **`pyproject.toml` の正本化**:
  - `statsmodels>=0.14.0`, `tqdm>=4.66.0`, `POT>=0.9.0` を追加し、依存関係および pytest 設定の単一正本として固定。
- **テストの集約**:
  - `tests/test_v3_causal_extensions.py` を新設。root `tests/` から `pytest` を実行するだけで、V1〜V3 の全共通・拡張機能テストが一括検証可能。

### 2.5 旧実験結果のアーカイブ退避と results のクリーン初期化
- **安全な旧データ退避パイプライン**:
  - `scripts/archive_and_clean_results.py` を作成。
  - `v2/results/prompt_hashes.json` を `v2/configs/prompt_hashes.json` へ安全に保護コピー。
  - 全ステージ（`behavioral`, `v1`, `v2`, `v3`）の既存 `results/` の内容を `archive/results_pre_rerun_20260918/` および `archive/results_pre_rerun_20260918.tar.gz` へ一括退避移動。
  - 全ステージの `results/` を再生成し、`.gitkeep` のみ配置してクリーン初期化（空の状態）。
  - `.gitignore` に `archive/` を追加し、Git 管理外に分離。

---

## 3. ターミナルでの再現・検証手順

ユーザーのローカル環境（仮想環境有効化状態）で、以下のコマンドを実行してリポジトリの整合性を検証できます。

### 3.0 旧結果のアーカイブ退避とクリーン初期化（必須推奨）
```bash
python scripts/archive_and_clean_results.py
```
これにより、以下の状態が自動構築されます：
- `archive/results_pre_rerun_20260918/`（旧結果原本）
- `archive/results_pre_rerun_20260918.tar.gz`（NAS・バックアップ用圧縮ファイル）
- `behavioral/results/`（空・.gitkeep のみ）
- `v1/results/`（空・.gitkeep のみ）
- `v2/results/`（空・.gitkeep のみ）
- `v3/results/`（空・.gitkeep のみ）


### 3.1 共通パッケージ・テストの検証
```bash
# 仮想環境の確認
which python
python --version

# 全ユニットテストの一括実行
pytest tests/ -v
```

### 3.2 各ステージ Primary スクリプトの Dry-run 検証

#### ① Behavioral
```bash
# EmoBank 3-Way 評価 (dry-run: 2サンプル)
python behavioral/primary/run_behavioral_emobank.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --is_instruct \
    --tag test_run \
    --limit 2 \
    --out-dir behavioral/results/test_run

# AIPsy 4-Split 評価 (dry-run: 2サンプル)
python behavioral/primary/run_behavioral_aipsy.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --is-instruct \
    --tag test_run \
    --limit 2 \
    --out-dir behavioral/results/test_run
```

#### ② V1 Primary
```bash
# Phase A (dry-run)
python v1/primary/run_phase_a.py \
    --model-id Qwen/Qwen2.5-0.5B-Instruct \
    --model-prefix test_qwen \
    --limit 2 \
    --out-dir v1/results/derived/test_phase_a

# Phase B (dry-run)
python v1/primary/run_phase_b.py \
    --model-id Qwen/Qwen2.5-0.5B-Instruct \
    --model-prefix test_qwen \
    --layer 12 \
    --out-dir v1/results/derived/test_phase_b

# Phase C (dry-run)
python v1/primary/run_phase_c.py \
    --model-id Qwen/Qwen2.5-0.5B-Instruct \
    --model-prefix test_qwen \
    --limit 2 \
    --alphas 0.0 1.0 \
    --layers 12 \
    --out-dir v1/results/derived/test_phase_c
```

#### ③ V2 Primary
```bash
# Confirmatory 分析 (dry-run)
python v2/primary/run_confirmatory_analysis.py \
    --config v2/configs/v2_experiments.yaml \
    --models-config v2/configs/models.yaml
```

#### ④ V3 Primary
```bash
# Confirmatory 追試・反証実験 (dry-run)
python v3/primary/run_confirmatory_replication.py \
    --config v3/configs/v3_experiments.yaml \
    --models-config v3/configs/models.yaml \
    --dry-run
```
