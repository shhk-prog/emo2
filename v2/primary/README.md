# V2 Primary Pipeline (Mechanistic Interpretability: Base vs. Instruct)

本ディレクトリ (`v2/primary/`) は、事前学習モデル（Base）から指示チューニング（Instruct）への事後学習（Post-training）を通じて、LLMの情動表現幾何（Representational Geometry）と因果回路（Causal Circuits）がどのように再編されるかを解明するための正式な Primary 実験スイートです。

---

## 1. 論文の主筋：4大リサーチクエスチョン (RQ1〜RQ4)

本研究は、4大モデルファミリー（Qwen 2.5, Llama 3.2, Gemma 2, Mistral）× 2水準（Base, Instruct）を対象に、以下の4つの核心的問いを厳密に検証します：

1. **V2-RQ1: Post-training による表現幾何の変容**
   - 指示チューニングによって、情動情報（Valence / Arousal）のデコード可能層や表現空間の幾何構造（固有次元・ノルム分布）はどう変化するか？
2. **V2-RQ2: Reader–Self 共有性の幾何学的再編**
   - 他者感情認識（Reader）と自己報告（Self）の表現幾何は共有されているか？（Held-out Cross-decoding, RSA, Procrustes Alignment）
3. **V2-RQ3: 因果回路の再配置とピーク解離**
   - 表現のデコードピーク（Decodability Peak）と、出力変位を引き起こす因果ピーク（Causal Peak）の間に層解離（Dissociation）が存在するか？
4. **V2-RQ4: 因果的復元（Recovery Patching）**
   - 中立文脈において感情活性化を介入注入することで、情動刺激提示時と同等の自己報告分布を真に復元できるか？（Wasserstein / OT 距離による回復率）

---

## 2. スクリプト構成

- **`run_rq1_rq2_cross_decoding.py`**: V2-RQ1 & RQ2: 表現幾何・交差デコード解析
- **`run_rq3_causal_map.py`**: V2-RQ3: 因果回路再配置・ピーク解離解析
- **`run_rq4_recovery_patching.py`**: V2-RQ4: 分布復元パッチング
- **`run_confirmatory_analysis.py`**: 確証的仮説検証（LMM, FDR補正）

---

## 3. 実行方法

### 3.1 RQ1 & RQ2: 幾何構造とクロスデコーディング
```bash
python v2/primary/run_rq1_rq2_cross_decoding.py \
    --config configs/v2_experiments.yaml \
    --models-config configs/models.yaml \
    --family Qwen \
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

### 3.4 Confirmatory 統合統計解析
```bash
python v2/primary/run_confirmatory_analysis.py \
    --config configs/v2_experiments.yaml \
    --models-config configs/models.yaml
```

---

## 4. 評価指標と標準化

- **相対計算深度の統一**:
  - $0 \le l < L$ に対し、相対深度を $d = \frac{l}{L - 1}$（0-based）として統一。
- **2D Joint Optimal Transport (OT)**:
  - $9 \times 9$ グリッド上の結合分布間 Wasserstein 距離（EMD）に基づき、真の回復率（True Recovery Ratio）を算出。
- **Prompt-End Normalized**:
  - トークン境界のずれを防ぐため、プロンプト末尾トークン位置を標準化。
