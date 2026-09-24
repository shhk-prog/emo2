# 変更内容の確認 (Walkthrough): 3B/7B 帯スケーリング・アブレーション実験環境の整備

## 1. 実施概要
ユーザーからの要望「ablationとして同様の4ファミリーの3b,7bあたりもやりたい」および事後確認（Gemma除外、Qwen・Llama・OLMoの3ファミリーで実施）に基づき、3B帯（`scale_3b`）および 7B帯（`scale_7b`）のスケーリング・アブレーション実験コホートを定義し、実験環境へ統合しました。

## 2. 変更・追加された設定と実装

### 2.1 モデル定義設定ファイル (`configs/models.yaml`)
- `scale_3b` コホートを追加:
  - **Qwen 2.5 3B**: `Qwen/Qwen2.5-3B` (Base), `Qwen/Qwen2.5-3B-Instruct` (Instruct)
  - **Llama 3.2 3B**: `meta-llama/Llama-3.2-3B` (Base), `meta-llama/Llama-3.2-3B-Instruct` (Instruct)
  - ※OLMo 2 には 3B 版が非存在のため 2 ファミリー構成
- `scale_7b` コホートを追加:
  - **Qwen 2.5 7B**: `Qwen/Qwen2.5-7B` (Base), `Qwen/Qwen2.5-7B-Instruct` (Instruct)
  - **Llama 3.1 8B**: `meta-llama/Llama-3.1-8B` (Base), `meta-llama/Llama-3.1-8B-Instruct` (Instruct)
  - **OLMo 2 7B**: `allenai/OLMo-2-1124-7B` (Base), `allenai/OLMo-2-1124-7B-Instruct` (Instruct)

### 2.2 モデルレジストリ (`src/affective_empathy_eval/models/registry.py`)
- `KNOWN_MODEL_DIMS` に 3B/7B モデルのレイヤー数および隠れ層次元を正確に登録:
  - `Qwen/Qwen2.5-3B` / `-Instruct`: (36 layers, 2048 dim)
  - `Qwen/Qwen2.5-7B` / `-Instruct`: (28 layers, 3584 dim)
  - `meta-llama/Llama-3.2-3B` / `-Instruct`: (28 layers, 3072 dim)
  - `meta-llama/Llama-3.1-8B` / `-Instruct`: (32 layers, 4096 dim)
  - `allenai/OLMo-2-1124-7B` / `-Instruct`: (32 layers, 4096 dim)

### 2.3 出力ディレクトリの自動分離実装 (`src/affective_empathy_eval/io.py`)
- `resolve_output_dirs(config, model_set, stage, is_dry_run)` を新設:
  - `primary_small`: `v2/results/raw/`, `v2/results/derived/`（主結果）
  - `scale_3b`: `results/ablation/scale_3b/raw/`, `results/ablation/scale_3b/derived/`
  - `scale_7b`: `results/ablation/scale_7b/raw/`, `results/ablation/scale_7b/derived/`
  - `scale_validation`: `results/ablation/scale_validation/raw/`, `results/ablation/scale_validation/derived/`
- `run_rq1_rq2_cross_decoding.py`, `run_rq3_causal_map.py`, `run_rq4_recovery_patching.py`, `run_confirmatory_analysis.py` に適用し、Primary成果物との混同・上書きを物理的に防止。
- `run.py` の `run_v2` から `confirmatory_analysis` への `--model-set` 引数伝播を追加。

### 2.4 ログディレクトリの自動分離実装 (`src/affective_empathy_eval/io.py`)
- `resolve_log_dir(model_set, stage, is_dry_run)` を新設:
  - `primary_small`: `results/logs/` (本番 Primary 実行ログ)
  - `scale_3b`: `results/ablation/scale_3b/logs/`
  - `scale_7b`: `results/ablation/scale_7b/logs/`
  - `scale_validation`: `results/ablation/scale_validation/logs/`
- `src/affective_empathy_eval/run.py` の `main()` でファイルロガー（`FileHandler`）を自動設定し、実行ログを各コホートの `logs/run_{stage}_{timestamp}.log` に自動保存。
- 本番アブレーション実行用 bash スクリプト [scripts/run_production_scale_ablation.sh](file:///mnt/nas/home/hiromi/src/emo2/scripts/run_production_scale_ablation.sh) を新設し、実行ログを `results/ablation/${MODEL_SET}/logs/production_v2_${TIMESTAMP}.log` に tee して追跡可能化。

### 2.5 単体テストと動作検証
- `tests/test_model_registry_and_adapters.py`:
  - `test_load_scale_3b()`, `test_load_scale_7b()` を新設（14件パス）。
- `tests/test_io_modular.py`:
  - `test_resolve_output_dirs()`, `test_resolve_log_dir()` を新設（全18件パス）。
- 実機検証:
  - `bash scripts/run_production_scale_ablation.sh scale_3b cpu --dry-run` を実行し、成果物が `results/ablation/scale_3b/raw/` および `derived/` に、ログが `results/ablation/scale_3b/logs/` にそれぞれ完全分離して保存されることを実機確認。

## 3. 実験実行方法

本リポジトリの統合 CLI (`affective_empathy_eval.run`) を用いて、`--model-set` を指定することで簡単にアブレーション実験を実行できます。

### 3.1 Dry-run（動作検証）
```bash
# 3B 帯 Dry-run
python -m affective_empathy_eval.run --stage v2 --model-set scale_3b --dry-run

# 7B 帯 Dry-run
python -m affective_empathy_eval.run --stage v2 --model-set scale_7b --dry-run
```

### 3.2 単一ファミリーごとの実行（GPU）
```bash
# 3B 帯 Qwen のみ実行
python -m affective_empathy_eval.run --stage v2 --model-set scale_3b --family qwen --device cuda:0

# 7B 帯 OLMo のみ実行
python -m affective_empathy_eval.run --stage v2 --model-set scale_7b --family olmo --device cuda:0
```

### 3.3 コホート全体のバッチ実行（GPU）

**推奨: 本番用 bash スクリプトによる自動ログ付き実行**
```bash
# 3B 帯全体 (ログ: results/ablation/scale_3b/logs/ に自動保存)
bash scripts/run_production_scale_ablation.sh scale_3b cuda:0

# 7B 帯全体 (ログ: results/ablation/scale_7b/logs/ に自動保存)
bash scripts/run_production_scale_ablation.sh scale_7b cuda:0
```

または Python 統合 CLI での直接実行:
```bash
python -m affective_empathy_eval.run --stage v2 --model-set scale_3b --device cuda:0
python -m affective_empathy_eval.run --stage v2 --model-set scale_7b --device cuda:0
```

> **注意 (AGENTS.md 規則 1.7)**:
> 実際の GPU を用いた推論・アブレーション実験の実行は、リソース状況や優先度に合わせて事前に確認の上で進めてください。

