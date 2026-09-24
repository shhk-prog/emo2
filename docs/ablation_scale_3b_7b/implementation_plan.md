# 実装計画: 4ファミリー 3B/7B 帯スケーリング・アブレーション実験の整備

## 1. 背景と動機
本プロジェクトの Primary 分析では、計算効率と統制の観点から 1〜1.5B 帯の 4 ファミリー（Qwen 2.5, Llama 3.2, Gemma 3, OLMo 2）を主対象としてきた。また、単独の外部スケール検証として Mistral 7B（`scale_validation`）が用意されている。
今回、モデル規模（パラメータ数）に対する現象の普遍性と頑健性を多角的に検証するアブレーションとして、同様の 4 ファミリーについて 3B 帯および 7B 帯のモデルコホートを実験環境に追加する。

## 2. 確定モデルコホート仕様

ユーザーの指定により、Gemma を除外した **Qwen**, **Llama**, **OLMo** を対象とする。
OLMo 2 には 3B モデルが提供されていないため、3B 帯は 2 ファミリー、7B 帯は 3 ファミリーとする。

### 2.1 scale_3b (3B 帯コホート)
- **Qwen 2.5 3B**:
  - Base: `Qwen/Qwen2.5-3B`
  - Instruct: `Qwen/Qwen2.5-3B-Instruct`
  - レイヤー数: 36, 隠れ層次元: 2048
  - アダプター: `qwen` (`LlamaFamilyAdapter`)
- **Llama 3.2 3B**:
  - Base: `meta-llama/Llama-3.2-3B`
  - Instruct: `meta-llama/Llama-3.2-3B-Instruct`
  - レイヤー数: 28, 隠れ層次元: 3072
  - アダプター: `llama` (`LlamaFamilyAdapter`)

### 2.2 scale_7b (7B 帯コホート)
- **Qwen 2.5 7B**:
  - Base: `Qwen/Qwen2.5-7B`
  - Instruct: `Qwen/Qwen2.5-7B-Instruct`
  - レイヤー数: 28, 隠れ層次元: 3584
  - アダプター: `qwen` (`LlamaFamilyAdapter`)
- **Llama 3.1 8B**:
  - Base: `meta-llama/Llama-3.1-8B`
  - Instruct: `meta-llama/Llama-3.1-8B-Instruct`
  - レイヤー数: 32, 隠れ層次元: 4096
  - アダプター: `llama` (`LlamaFamilyAdapter`)
- **OLMo 2 7B**:
  - Base: `allenai/OLMo-2-1124-7B`
  - Instruct: `allenai/OLMo-2-1124-7B-Instruct`
  - レイヤー数: 32, 隠れ層次元: 4096
  - アダプター: `olmo2` (`Olmo2Adapter`)

## 3. 設定およびコードの改修内容

### 3.1 `configs/models.yaml` への `model_sets` 追加
既存の `primary_small` (1-1.5B), `scale_validation` (Mistral 7B) に加え、以下を追加：
- `scale_3b`: 3B 帯コホート（Qwen 3B, Llama 3.2 3B, Gemma 3 4B 等）
- `scale_7b`: 7B 帯コホート（Qwen 7B, Llama 3.1 8B, Gemma 9B/12B, OLMo 2 7B）
※ または単一の `scale_ablation` 等として階層管理するか、スケール別に分けるのが運用上明確。

### 3.2 `src/affective_empathy_eval/models/registry.py` の更新
- `KNOWN_MODEL_DIMS` に 3B/7B 帯各モデルのレイヤー数・隠れ層次元（(num_layers, hidden_dim)）を追加し、オフライン環境・HF接続不可時でも確実に動作できるようにする。
  - `Qwen/Qwen2.5-3B`: (36, 2048)
  - `Qwen/Qwen2.5-7B`: (28, 3584) [既登録]
  - `meta-llama/Llama-3.2-3B`: (28, 3072) [既登録]
  - `meta-llama/Llama-3.1-8B`: (32, 4096)
  - `allenai/OLMo-2-1124-7B`: (32, 4096)
  - `google/gemma-3-4b-pt`: (36, 2560) 等
  - `google/gemma-2-9b`: (42, 3584) 等

### 3.3 結果出力先と分離性
- `AGENTS.md` 規則 1.3条「実験結果を上書きしない」および「Primary 4 family と混ぜない」に従い、
  - 出力先は `results/ablation/scale_3b/` および `results/ablation/scale_7b/` または `v2/results/ablation/...` とし、Primary 成果物（`v2/results/raw/`）と物理的に分離する。

### 3.4 検証方針
- テストコード `tests/test_model_registry_and_adapters.py` で新モデルセットのパース・検証が通ることを確認。
- `--dry-run` オプションでパイプライン（V2 Primary の RQ1〜RQ4）が正常にビルド・初期化されることを確認。
- GPU実行はユーザーの合意を得た後に段階的に実施。
