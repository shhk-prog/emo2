# V1 Phase A: E1 (Decodability) & E2 (Geometry) 全モデル総合評価レポート

本レポートは、EmoBank テストセット（$N=1,006$）および AIPsy-Affect（$N=2,196$）を活用し、
全8モデル（4ファミリー × Base/Instruct）の**全層（全隠れ層）にわたる感情情報の存在と幾何構造**を体系的に分析した結果です。

---

## 1. リサーチクエスチョン（RQ）と E1 $\rightarrow$ E2 の論理的階段構造

行動実験において他者認識（Reader）と自己報告（Self）の間に強い連動（$r \approx 0.80 \sim 0.97$）が観測されたことを受け、以下の2段階で内部表現を解明します：

```text
【E1: Shared Decodability (存在)】
  問い: Readerタスク時だけでなく、Selfタスク時にも、感情情報が内部表現から線形に読み出し可能か？
  データ: EmoBank人間VADへのRidge回帰 (R²) ＋ AIPsy刺激分類へのLogistic回帰 (ROC-AUC)
        │
        ▼ (情報が存在することを確認した上で)
【E2: Shared Geometry (形式)】
  問い: Reader と Self は、その感情情報を「同じ座標系（Shared）」で持っているのか、
        それとも「異なる座標系だが線形回転可能な形式（Alignable）」なのか？
  手法: Direct Cross-Decoding ＋ RSA (相関距離RDM) ＋ Orthogonal Procrustes Alignment
```

---

## Part 1: モデルファミリー別 Base vs. Instruct 統合対比表

### 表 1: E1 (Decodability) ピーク層と復元精度（EmoBank Valence）

| モデルファミリー | アライメント | 総層数 | Reader ピーク層 ($l^*_R$) | Reader $R^2$ | Self ピーク層 ($l^*_S$) | Self $R^2$ | ピーク層間乖離 $|l^*_R - l^*_S|$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen 2.5 1.5B** | Base | 28層 | Layer 25 | **0.371** | Layer 23 | **0.375** | **2 層** |
| **Qwen 2.5 1.5B** | Instruct | 28層 | Layer 14 | **0.234** | Layer 14 | **0.288** | **0 層** |
| **Llama 3.2 1B** | Base | 16層 | Layer 14 | **0.327** | Layer 13 | **0.275** | **1 層** |
| **Llama 3.2 1B** | Instruct | 16層 | Layer 10 | **0.326** | Layer 10 | **0.390** | **0 層** |
| **Gemma 2 2B** | Base | 26層 | Layer 26 | **-0.006** | Layer 26 | **0.007** | **0 層** |
| **Gemma 2 2B** | Instruct | 26層 | Layer 26 | **-0.032** | Layer 26 | **-0.027** | **0 層** |
| **Mistral 7B** | Base | 32層 | Layer 32 | **0.172** | Layer 32 | **0.197** | **0 層** |
| **Mistral 7B** | Instruct | 32層 | Layer 21 | **0.487** | Layer 18 | **0.480** | **3 層** |

### 表 2: E2 (Geometry) 幾何構造パターンと転移性能

| モデルファミリー | アライメント | ピークRSA $\rho$ (層) | Direct Cross-Decoding $R^2$ | Aligned (Procrustes) $R^2$ | 判定された幾何分類 | 科学的帰結 |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Qwen 2.5 1.5B** | Base | **0.990** (L2) | 0.505 | **0.942** | `Shared Geometry` | **座標回転による幾何整列可能** |
| **Qwen 2.5 1.5B** | Instruct | **0.946** (L11) | 0.000 | **0.848** | `Alignable Geometry` | **座標回転による幾何整列可能** |
| **Llama 3.2 1B** | Base | **0.981** (L2) | 0.384 | **0.961** | `Shared Geometry` | **座標回転による幾何整列可能** |
| **Llama 3.2 1B** | Instruct | **0.987** (L2) | 0.000 | **0.908** | `Alignable Geometry` | **座標回転による幾何整列可能** |
| **Gemma 2 2B** | Base | **0.554** (L13) | 0.000 | **0.491** | `Alignable Geometry` | **座標回転による幾何整列可能** |
| **Gemma 2 2B** | Instruct | **0.909** (L17) | 0.000 | **0.720** | `Alignable Geometry` | **座標回転による幾何整列可能** |
| **Mistral 7B** | Base | **0.995** (L2) | 0.669 | **0.975** | `Shared Geometry` | **座標回転による幾何整列可能** |
| **Mistral 7B** | Instruct | **0.988** (L1) | 0.000 | **0.954** | `Alignable Geometry` | **座標回転による幾何整列可能** |

---

## Part 2: 各モデル別 詳細分析（全8モデル個別の層別プロファイル）

### ■ モデル: `Qwen/Qwen2.5-1.5B` (Base)
- **プレフィックス**: `qwen2.5_1.5b_base`
- **総層数**: 28 layers

#### 【主要層における E1 & E2 測定値】
| Layer | Reader $R^2$ (Valence) | Self $R^2$ (Valence) | Direct Cross $R^2$ | Aligned $R^2$ | RSA $\rho$ | 幾何判定 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| Layer  0 | -0.001 | -0.001 | 0.000 | -0.000 | 0.000 | `Directly non-transferable / poorly alignable representation` |
| Layer  4 | 0.046 | 0.053 | 0.583 | 0.939 | 0.984 | `Shared Geometry` |
| Layer  8 | 0.050 | 0.019 | 0.000 | 0.905 | 0.973 | `Alignable Geometry` |
| Layer 12 | 0.060 | 0.077 | 0.000 | 0.852 | 0.957 | `Alignable Geometry` |
| Layer 16 | 0.147 | 0.191 | 0.000 | 0.745 | 0.905 | `Alignable Geometry` |
| Layer 20 | 0.322 | 0.307 | 0.000 | 0.734 | 0.857 | `Alignable Geometry` |
| Layer 24 | 0.359 | 0.372 | 0.000 | 0.826 | 0.819 | `Alignable Geometry` |
| Layer 28 | 0.293 | 0.322 | 0.206 | 0.916 | 0.828 | `Alignable Geometry` |


---

### ■ モデル: `Qwen/Qwen2.5-1.5B-Instruct` (Instruct)
- **プレフィックス**: `qwen2.5_1.5b_instruct`
- **総層数**: 28 layers

#### 【主要層における E1 & E2 測定値】
| Layer | Reader $R^2$ (Valence) | Self $R^2$ (Valence) | Direct Cross $R^2$ | Aligned $R^2$ | RSA $\rho$ | 幾何判定 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| Layer  0 | -0.001 | -0.001 | 0.000 | -0.000 | 0.000 | `Directly non-transferable / poorly alignable representation` |
| Layer  4 | 0.092 | 0.095 | 0.000 | 0.885 | 0.866 | `Alignable Geometry` |
| Layer  8 | 0.084 | 0.102 | 0.000 | 0.832 | 0.930 | `Alignable Geometry` |
| Layer 12 | 0.116 | 0.150 | 0.000 | 0.845 | 0.935 | `Alignable Geometry` |
| Layer 16 | 0.203 | 0.268 | 0.000 | 0.761 | 0.939 | `Alignable Geometry` |
| Layer 20 | 0.150 | 0.162 | 0.000 | 0.676 | 0.893 | `Alignable Geometry` |
| Layer 24 | 0.161 | 0.174 | 0.000 | 0.648 | 0.840 | `Alignable Geometry` |
| Layer 28 | 0.144 | 0.168 | 0.000 | 0.686 | 0.840 | `Alignable Geometry` |


---

### ■ モデル: `meta-llama/Llama-3.2-1B` (Base)
- **プレフィックス**: `llama3.2_1b_base`
- **総層数**: 16 layers

#### 【主要層における E1 & E2 測定値】
| Layer | Reader $R^2$ (Valence) | Self $R^2$ (Valence) | Direct Cross $R^2$ | Aligned $R^2$ | RSA $\rho$ | 幾何判定 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| Layer  0 | -0.001 | -0.001 | 0.000 | -0.000 | 0.000 | `Directly non-transferable / poorly alignable representation` |
| Layer  2 | 0.117 | 0.118 | 0.384 | 0.961 | 0.981 | `Shared Geometry` |
| Layer  4 | 0.106 | 0.112 | 0.235 | 0.946 | 0.971 | `Alignable Geometry` |
| Layer  6 | 0.125 | 0.148 | 0.000 | 0.894 | 0.967 | `Alignable Geometry` |
| Layer  8 | 0.173 | 0.140 | 0.000 | 0.868 | 0.939 | `Alignable Geometry` |
| Layer 10 | 0.228 | 0.224 | 0.000 | 0.827 | 0.902 | `Alignable Geometry` |
| Layer 12 | 0.294 | 0.256 | 0.000 | 0.832 | 0.870 | `Alignable Geometry` |
| Layer 14 | 0.327 | 0.275 | 0.096 | 0.904 | 0.870 | `Alignable Geometry` |
| Layer 16 | 0.229 | 0.220 | 0.678 | 0.942 | 0.898 | `Shared Geometry` |


---

### ■ モデル: `meta-llama/Llama-3.2-1B-Instruct` (Instruct)
- **プレフィックス**: `llama3.2_1b_instruct`
- **総層数**: 16 layers

#### 【主要層における E1 & E2 測定値】
| Layer | Reader $R^2$ (Valence) | Self $R^2$ (Valence) | Direct Cross $R^2$ | Aligned $R^2$ | RSA $\rho$ | 幾何判定 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| Layer  0 | -0.001 | -0.001 | 0.000 | -0.000 | 0.000 | `Directly non-transferable / poorly alignable representation` |
| Layer  2 | 0.120 | 0.120 | 0.000 | 0.908 | 0.987 | `Alignable Geometry` |
| Layer  4 | 0.197 | 0.210 | 0.169 | 0.936 | 0.975 | `Alignable Geometry` |
| Layer  6 | 0.272 | 0.310 | 0.000 | 0.826 | 0.955 | `Alignable Geometry` |
| Layer  8 | 0.306 | 0.324 | 0.000 | 0.744 | 0.920 | `Alignable Geometry` |
| Layer 10 | 0.326 | 0.390 | 0.000 | 0.615 | 0.835 | `Alignable Geometry` |
| Layer 12 | 0.277 | 0.303 | 0.000 | 0.515 | 0.711 | `Alignable Geometry` |
| Layer 14 | 0.273 | 0.332 | 0.000 | 0.354 | 0.689 | `Alignable Geometry` |
| Layer 16 | 0.264 | 0.323 | 0.000 | 0.209 | 0.662 | `Alignable Geometry` |


---

### ■ モデル: `google/gemma-2-2b` (Base)
- **プレフィックス**: `gemma2_2b_base`
- **総層数**: 26 layers

#### 【主要層における E1 & E2 測定値】
| Layer | Reader $R^2$ (Valence) | Self $R^2$ (Valence) | Direct Cross $R^2$ | Aligned $R^2$ | RSA $\rho$ | 幾何判定 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| Layer  0 | -0.052 | -0.071 | 0.239 | 0.266 | 0.430 | `Directly non-transferable / poorly alignable representation` |
| Layer  3 | -0.039 | -0.031 | 0.000 | 0.493 | 0.506 | `Alignable Geometry` |
| Layer  6 | -0.043 | -0.017 | 0.050 | 0.502 | 0.551 | `Alignable Geometry` |
| Layer  9 | -0.032 | -0.020 | 0.000 | 0.497 | 0.506 | `Alignable Geometry` |
| Layer 12 | -0.026 | -0.008 | 0.000 | 0.493 | 0.534 | `Alignable Geometry` |
| Layer 15 | -0.031 | -0.016 | 0.000 | 0.489 | 0.524 | `Alignable Geometry` |
| Layer 18 | -0.019 | -0.003 | 0.000 | 0.486 | 0.514 | `Alignable Geometry` |
| Layer 21 | -0.026 | -0.005 | 0.000 | 0.486 | 0.511 | `Alignable Geometry` |
| Layer 24 | -0.013 | 0.002 | 0.000 | 0.484 | 0.514 | `Alignable Geometry` |
| Layer 26 | -0.006 | 0.007 | 0.156 | 0.477 | 0.523 | `Alignable Geometry` |


---

### ■ モデル: `google/gemma-2-2b-it` (Instruct)
- **プレフィックス**: `gemma2_2b_instruct`
- **総層数**: 26 layers

#### 【主要層における E1 & E2 測定値】
| Layer | Reader $R^2$ (Valence) | Self $R^2$ (Valence) | Direct Cross $R^2$ | Aligned $R^2$ | RSA $\rho$ | 幾何判定 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| Layer  0 | -0.047 | -0.050 | 0.084 | 0.123 | 0.853 | `Alignable Geometry` |
| Layer  3 | -0.038 | -0.046 | 0.000 | 0.779 | 0.815 | `Alignable Geometry` |
| Layer  6 | -0.049 | -0.052 | 0.000 | 0.806 | 0.855 | `Alignable Geometry` |
| Layer  9 | -0.040 | -0.046 | 0.000 | 0.773 | 0.836 | `Alignable Geometry` |
| Layer 12 | -0.044 | -0.055 | 0.000 | 0.758 | 0.882 | `Alignable Geometry` |
| Layer 15 | -0.049 | -0.055 | 0.000 | 0.743 | 0.891 | `Alignable Geometry` |
| Layer 18 | -0.051 | -0.051 | 0.000 | 0.714 | 0.893 | `Alignable Geometry` |
| Layer 21 | -0.050 | -0.056 | 0.000 | 0.721 | 0.867 | `Alignable Geometry` |
| Layer 24 | -0.058 | -0.056 | 0.000 | 0.724 | 0.867 | `Alignable Geometry` |
| Layer 26 | -0.032 | -0.027 | 0.000 | 0.724 | 0.864 | `Alignable Geometry` |


---

### ■ モデル: `mistralai/Mistral-7B-v0.1` (Base)
- **プレフィックス**: `mistral7b_v0.1_base`
- **総層数**: 32 layers

#### 【主要層における E1 & E2 測定値】
| Layer | Reader $R^2$ (Valence) | Self $R^2$ (Valence) | Direct Cross $R^2$ | Aligned $R^2$ | RSA $\rho$ | 幾何判定 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| Layer  0 | -0.001 | -0.001 | 0.000 | -0.000 | 0.000 | `Directly non-transferable / poorly alignable representation` |
| Layer  4 | 0.148 | 0.151 | 0.765 | 0.970 | 0.988 | `Shared Geometry` |
| Layer  8 | 0.128 | 0.135 | 0.401 | 0.926 | 0.968 | `Shared Geometry` |
| Layer 12 | 0.105 | 0.127 | 0.000 | 0.799 | 0.896 | `Alignable Geometry` |
| Layer 16 | 0.106 | 0.091 | 0.000 | 0.663 | 0.839 | `Alignable Geometry` |
| Layer 20 | 0.089 | 0.064 | 0.000 | 0.540 | 0.799 | `Alignable Geometry` |
| Layer 24 | 0.075 | 0.058 | 0.000 | 0.649 | 0.777 | `Alignable Geometry` |
| Layer 28 | 0.080 | 0.085 | 0.034 | 0.771 | 0.772 | `Alignable Geometry` |
| Layer 32 | 0.172 | 0.197 | 0.314 | 0.838 | 0.823 | `Shared Geometry` |


---

### ■ モデル: `mistralai/Mistral-7B-Instruct-v0.1` (Instruct)
- **プレフィックス**: `mistral7b_v0.1_instruct`
- **総層数**: 32 layers

#### 【主要層における E1 & E2 測定値】
| Layer | Reader $R^2$ (Valence) | Self $R^2$ (Valence) | Direct Cross $R^2$ | Aligned $R^2$ | RSA $\rho$ | 幾何判定 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| Layer  0 | -0.001 | -0.001 | 0.000 | -0.000 | 0.000 | `Directly non-transferable / poorly alignable representation` |
| Layer  4 | 0.180 | 0.203 | 0.524 | 0.974 | 0.969 | `Shared Geometry` |
| Layer  8 | 0.330 | 0.350 | 0.449 | 0.920 | 0.948 | `Shared Geometry` |
| Layer 12 | 0.354 | 0.410 | 0.310 | 0.837 | 0.956 | `Shared Geometry` |
| Layer 16 | 0.459 | 0.474 | 0.365 | 0.868 | 0.950 | `Shared Geometry` |
| Layer 20 | 0.485 | 0.477 | 0.363 | 0.839 | 0.882 | `Shared Geometry` |
| Layer 24 | 0.477 | 0.474 | 0.216 | 0.851 | 0.852 | `Alignable Geometry` |
| Layer 28 | 0.453 | 0.470 | 0.461 | 0.860 | 0.842 | `Shared Geometry` |
| Layer 32 | 0.462 | 0.471 | 0.476 | 0.867 | 0.846 | `Shared Geometry` |


---
