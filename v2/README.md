# V2 Stage: Mechanistic Reorganization (Post-training Analysis: Base vs. Instruct)

本ディレクトリ (`v2/`) は、事前学習モデル（Base）から指示チューニングモデル（Instruct）への事後学習（Post-training）を通じて、大規模言語モデル（LLM）内部の情動表現幾何（Representational Geometry）と因果回路（Causal Circuits）がどのように再編されるかを解明するための実験スイートです。

4大モデルファミリー（Qwen 2.5 1.5B, Llama 3.2 1B, Gemma 3 1B, OLMo 2 1B: Primary 1–1.5B コホート）および外部スケール検証（Mistral 7B v0.3）× 2水準（Base, Instruct）を対象とし、他者感情認識（Reader）と自己報告（Self）の共有性および介入可能性を包括的に比較・検証します。

---

## 1. 4大リサーチクエスチョン (RQ1〜RQ4)

一本化した現在の V2 実験パイプラインは、以下の 4 つの核心的リサーチクエスチョンを中心に設計されています：

1. **V2-RQ1: Post-training による表現幾何の変容**
   - 事後学習によって、情動情報（Valence / Arousal）のデコード可能層や内部幾何構造（Procrustes 歪み、RSA相関）はどう再編されるか？
2. **V2-RQ2: Reader–Self 共有性の幾何学的再編**
   - 他者感情認識（Reader）と自己報告（Self）の表現空間は共有されているか？ 事後学習によってその共有度（Held-out Cross-decoding, $\Delta \text{Sharing}$）はどう変化するか？
3. **V2-RQ3: 因果回路の再配置とピーク解離**
   - 表現のデコードピーク（Decodability Peak）と、出力変位を引き起こす因果ピーク（Causal Peak）の間に層解離（Dissociation）が存在するか？
4. **V2-RQ4: 因果的復元（Distribution Recovery Patching）**
   - 中立文脈において感情活性化を注入することで、自然な情動刺激提示時と同等の自己報告分布を真に復元できるか？（Wasserstein / EMD 距離に基づく回復率）

---

## 2. Primary 実験スイート (`v2/primary/`)

本リポジトリの正式な正本実装は `v2/primary/` 配下に集約されています：

- **`v2/primary/run_rq1_rq2_cross_decoding.py`**:
  - V2-RQ1 & RQ2: 表現幾何・交差デコード解析（Held-out Cross-decoding, RSA, Procrustes Alignment, $\Delta \text{Sharing}$）
- **`v2/primary/run_rq3_causal_map.py`**:
  - V2-RQ3: 因果回路再配置・ピーク解離解析（Interchangeability, Reader $\rightarrow$ Self パッチング）
- **`v2/primary/run_rq4_recovery_patching.py`**:
  - V2-RQ4: 分布復元パッチング（EMD / Wasserstein 距離による分布回復度評価）
- **`v2/primary/run_confirmatory_analysis.py`**:
  - 確証的仮説検証（線形混合効果モデル LMM, FDR 多重比較補正）

---

## 3. 実行方法 (Quick Start)

### 3.1 統合ランナーによる一括実行（推奨）
```bash
# Primary 1-1.5B コホートの全 RQ1〜RQ4 を一括実行
python -m affective_empathy_eval.run --stage v2 --model-set primary_small --device cuda

# Supplementary 7B 外部スケール検証 (Mistral 7B)
python -m affective_empathy_eval.run --stage v2 --model-set scale_validation --device cuda
```

### 3.2 個別スクリプト実行（特定ファミリーのみ）
```bash
python v2/primary/run_rq1_rq2_cross_decoding.py \
    --config configs/v2_experiments.yaml \
    --models-config configs/models.yaml \
    --family qwen \
    --device cuda
```

### 3.2 RQ3: 因果マッピングとピーク解離
```bash
python v2/primary/run_rq3_causal_map.py \
    --config configs/v2_experiments.yaml \
    --models-config configs/models.yaml \
    --family Qwen \
    --device cuda
```

### 3.3 RQ4: 因果的復元パッチング
```bash
python v2/primary/run_rq4_recovery_patching.py \
    --config configs/v2_experiments.yaml \
    --models-config configs/models.yaml \
    --family Qwen \
    --device cuda
```

### 3.4 Dry-run（軽量シミュレーション検証）
```bash
python v2/primary/run_rq1_rq2_cross_decoding.py --dry-run
python v2/primary/run_rq3_causal_map.py --dry-run
python v2/primary/run_rq4_recovery_patching.py --dry-run
```

---

## 4. 出力成果物構造 (`v2/results/`)

- **`v2/results/raw/`**:
  - `v2_geometry_{family}.json`: 各モデルファミリーの幾何・クロスデコード生データ
  - `v2_causal_{family}.json`: 因果マッピング生データ
  - `v2_recovery_{family}.json`: 分布復元パッチング生データ
- **`v2/results/derived/`**:
  - `v2_cross_family_summary.json`: 4ファミリー統合の Bootstrap 95% CI および Paired 比較サマリー
  - `v2_lmm_confirmatory.json`: 確証的 LMM 解析結果

---

## 5. Legacy / Exploratory Experiments (`v2/scripts/`)

以前の探索的実験（コンポーネント別アブレーション、経路分析、旧プロトタイプスクリプト）は `v2/scripts/` に保持されています。
これらは後方互換性および追加の感度分析用途であり、主論文の検証には上記の Primary スイートを使用してください。
