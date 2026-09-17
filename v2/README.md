# LLM情動反応性評価実験 v2 (Mechanistic Interpretability: Base vs. Instruct)

本リポジトリ (`v2`) は、v1で観測された「事後学習（Post-training / Instruction Tuning）に伴う情動自己報告の抑制・中立化（Neutralization）」のメカニズムを、深層学習モデルの内部表現レベル（機械論的解釈可能性: Mechanistic Interpretability）から解明するための第2フェーズ実験環境です。

4大モデルファミリー（**Qwen 2.5 1.5B**, **Llama 3.2 1B**, **Gemma 2 2B**, **Mistral 7B**）の各 Base / Instruct モデル（計8モデル）を対象とし、感情情報が内部から「消去」されたのか、あるいは「表現幾何の変化」や「因果的出力マッピングの再編」によって自己報告が変化したのかを多角的な介入実験と分布間距離解析（Wasserstein距離, 2D EMD, JSD）により検証します。

---

## 1. 核心的リサーチクエスチョン (Core Research Questions)

事後学習モデルにおいて情動自己報告が中立化する現象に対し、以下の4つのリサーチクエスチョン（RQ1〜RQ4）を体系的に検証します：

```text
【V2-RQ1 & RQ2: 表現幾何の変化と整列可能性 (Geometry & Alignment)】
  問い: Baseモデルに存在した感情情報は、Instructモデルで消去されたのか (Erasure)？
        それとも表現空間が回転・歪曲されただけで保持されているのか (Transformation)？
  手法: Held-out 線形プローブ (R²), 直交 Procrustes 整列, 表現類似度分析 (RSA)

【V2-RQ3: 因果回路の再配置とピーク解離 (Causal Map & Peak Dissociation)】
  問い: 出力生成を因果的に駆動する活性化サイト（層）は、事後�| **Mistral** | `mistralai/Mistral-7B-v0.1` | `mistralai/Mistral-7B-Instruct-v0.2` | 32層 | Layer 16 / $d=4096$ |
```

プロンプト形式の交絡を統制するため、InstructモデルのChat Templateを用いる `native` 条件と、特殊トークンを除外した `matched_plain` 条件の双方で検証が可能です（`configs/v2_experiments.yaml` で制御）。

---

## 3. ディレクトリ構成とスクリプトの役割

v2のスクリプト群は、4モデルファミリー横断の中核パイプラインと、詳細なコンポーネント・回路単位の検証スクリプト（Phase 1〜9）で構成されています。

```text
v2/
├── README.md                          # 本ドキュメント
├── REPRODUCIBILITY.md                 # 再現性ガイド
├── src/
│   └── likelihood.py                  # 81候補 Sequence-Likelihood 尤度計算モジュール
├── scripts/
│   ├── # --- 4モデルファミリー横断 中核パイプライン ---
│   ├── run_v2_2x2_cross_decoding.py   # V2-RQ1/2: Base ↔ Instruct 幾何・交差デコード解析
│   ├── run_v2_2x2_causal_map.py       # V2-RQ3: 因果回路再配置・ピーク解離解析
│   ├── run_v2_recovery_patching.py    # V2-RQ4: 分布回復パッチング (EMD_VA)
│   │
│   ├── # --- データセット準備 ---
│   ├── prepare_aipsy_strict.py        # 厳密マッチングトリプレット (Strict Subset) 抽出
│   ├── prepare_aipsy.py               # AIPsy-Affect 基本前処理
│   │
│   ├── # --- 予備・確証プロービング解析 (Phase 1 / 1.5 / 2 / 3) ---
│   ├── run_probing_preliminary.py     # 予備的表現プロービング
│   ├── run_confirmatory_analysis.py   # 確証的デコーディング分析
│   ├── run_cross_decoding.py          # 詳細交差デコーディング
│   ├── run_strict_cross_decoding.py   # 厳密データセット上での交差デコーディング
│   ├── run_rsa_and_controlled_coupling.py # RSAおよび統制結合解析
│   │
│   ├── # --- コンポーネント単位の因果・回路パッチング (Phase 4 / 5 / 6 / 7) ---
│   ├── run_patching_screening.py      # Attention / MLP / Residual のスクリーニング
│   ├── run_strict_patching_screening.py # 厳密データセット上でのコンポーネントスクリーニング
│   ├── run_circuit_patching.py        # 詳細回路パッチング
│   ├── run_path_patching.py           # パスパッチングによる情報伝播経路同定
│   ├── run_strict_path_patching.py    # 厳密データセット上でのパスパッチング
│   ├── run_synergy_patching.py        # 複数コンポーネント同時介入 (相乗効果検証)
│   ├── run_output_gating_test.py      # Late-Residual バイパスによる出力ゲーティング検証
│   ├── run_unembedding_norm_swap.py   # 最終Residual / RMSNorm / lm_head の8条件スワップ
│   ├── run_unembedding_swap.py        # lm_head 単体スワップ
│   │
│   ├── # --- 厳密検証・高度な介入 (Phase 8 / 9) ---
│   ├── run_strict_causal_scrubbing.py # Causal Scrubbing による厳密回路検証
│   ├── run_sae_patching.py            # Sparse Autoencoder (SAE) 特徴量パッチング
│   ├── run_introspective_accessibility_test.py # 内省的アクセス可能性検証
│   │
│   ├── # --- ステアリング・感度解析 ---
│   ├── run_steering_and_likelihood.py # 対照的方向ベクトルによる表現ステアリング
│   ├── run_lambda_dose_response.py    # パッチング強度の連続スイープ (用量反応性)
│   ├── run_temperature_scaling.py     # 温度パラメータ τ スケーリング
│   ├── run_mixed_effects_coupling.py  # 混合効果モデルによる結合勾配 (Coupling Slope) 評価
│   │
│   └── # --- 集計・可視化 ---
│       ├── dump_all_metrics.py        # 全メトリクスの一括ダンプ
│       ├── dump_tables.py             # 論文掲載用テーブルの Markdown/CSV 出力
│       ├── aggregate_strict_experiments.py # 厳密実験結果の統合
│       ├── calculate_bootstrap_ci.py  # ブートストラップ 95% 信頼区間計算
│       └── plot_*.py                  # 各種可視化スクリプト
└── results/
    ├── raw/                           # 生データ: v2_geometry_{family}.json, v2_causal_map_{family}.json, v2_recovery_{family}.json, aipsy/
    └── derived/                       # 集計結果: v2_*_summary.json, phase1_preliminary/ 〜 phase9_advanced_patching/
```

---

## 4. 評価指標と出力フォーマット

v2では、単純な期待値変位だけでなく、2次元VA同時分布の形状変化を捉える幾何・情報幾何的指標を用います。

### 4.1 幾何・デコーディング指標 (Geometry & Decoding)
- **Held-out $R^2$**: 独立したテストセット（70% train / 30% test、`pair_id` による Group Split）に対する線形プローブの予測決定係数（Valence および Arousal の双軸評価）。
- **Direct Cross-Decoding $R^2$**: 一方のモデル（例: Base）で学習したプローブを、座標変換を行わずに他方（Instruct）に直接適用した決定係数。
- **Orthogonal Procrustes $R^2$**: 直交行列 $Q$（回転・鏡映）により表現空間を整列させた後の決定係数（Alignable Geometry の判定）。
- **RSA Correlation ($\rho_{\mathrm{RSA}}$)**: 刺激間相関距離行列（RDM）の Spearman 順位相関。表現空間のトポロジー的一致度。

### 4.2 因果・分布間距離指標 (Causal & Distribution Distance)
- **Sequence-Likelihood Protocol**: 81通りの JSON 候補列に対する正規のシーケンス対数尤度から算出される 81 マス同時確率分布 $p(v, a)$ および期待値 $E[V], E[A]$。
- **$WD_V$ / $WD_A$ (1D Wasserstein Distance)**: Valence / Arousal 周辺分布間の1次元最適輸送距離。
- **$EMD_{VA}$ (2D Earth Mover's Distance)**: 2次元VA平面上（81候補）の同時確率分布間における最適輸送コスト。Base分布への回復率（Recovery Rate）評価に使用：
  $$\text{Recovery Rate} = \frac{EMD(\text{Instruct}, \text{Base}) - EMD(\text{Patched}, \text{Base})}{EMD(\text{Instruct}, \text{Base})} \times 100\%$$
- **JSD (Jensen-Shannon Divergence)**: 分布間の対称化カルバック・ライブラー情報量（$0 \le JSD \le \ln 2$）。
- **因果変位 $C(l)$ & Peak Dissociation**: 介入（Zero Ablation等）時の期待値変化 $C(l)$、各タスク（Reader / Self）における最大因果層（$l^*$ または相対深度 $d^*$）、タスク間のピーク層乖離 $|l^*_R - l^*_S|$（$\Delta d^*$）、および因果重心乖離 $\Delta \bar{d}$。

---

## 5. 実行手順 (How to Run)

### 5.1 環境構築
プロジェクト全体の規約に従い、プロジェクトルートの `.venv` 仮想環境を使用します。

```bash
# プロジェクトルートに移動
cd /mnt/nas/home/hiromi/src/emo

# 仮想環境の有効化
source .venv/bin/activate

# 共通ライブラリのインストール確認
pip install -e .
```

### 5.2 4モデルファミリー横断 中核パイプラインの実行
設定ファイル `configs/v2_experiments.yaml` に基づき、主要RQを順次実行します。

```bash
# 1. V2-RQ1 & RQ2: 表現幾何と交差デコーディング解析
python v2/scripts/run_v2_2x2_cross_decoding.py --device cuda

# 2. V2-RQ3: 因果回路の再配置とピーク解離解析
python v2/scripts/run_v2_2x2_causal_map.py --device cuda

# 3. V2-RQ4: Base activation パッチングによる Instruct 分布回復実験
python v2/scripts/run_v2_recovery_patching.py --device cuda
```

> [!NOTE]
> 特定のモデルファミリー（例: Qwen）のみを対象とする場合は `--family Qwen`、動作確認用の軽量実行には `--dry-run` または `--max-samples 50` を付与します。

### 5.3 詳細コンポーネント解析の実行（Qwen等を対象とした個別検証）
```bash
# 厳密データセットの準備 (AIPsy-Affect matched triplets)
python v2/scripts/prepare_aipsy_strict.py

# コンポーネントパッチングスクリーニング (Attention / MLP / Residual)
python v2/scripts/run_strict_patching_screening.py

# 複数コンポーネント同時介入 (相乗効果検証)
python v2/scripts/run_synergy_patching.py

# 出力層 8-Condition スワップ解析 (Residual, RMSNorm, lm_head)
python v2/scripts/run_unembedding_norm_swap.py

# Late-Residual バイパスによる出力ゲーティング検証
python v2/scripts/run_output_gating_test.py

# Causal Scrubbing による厳密因果検証
python v2/scripts/run_strict_causal_scrubbing.py
```

### 5.4 結果の集計とプロット生成
```bash
# 全メトリクスの集計
python v2/scripts/dump_all_metrics.py

# 論文用テーブルの生成
python v2/scripts/dump_tables.py

# 幾何・パッチング結果の可視化
python v2/scripts/plot_strict_cross_decoding.py
python v2/scripts/plot_patching_and_gating.py
```     └── plot_*.py                  # 各種可視化スクリプト
└── results/
    ├── raw/                           # 尤度スコアリング・活性化抽出ログ
    └── derived/                       # 幾何メトリクス、EMD、パッチング効果等の集計表
```

---

## 4. 評価指標と出力フォーマット

v2では、単純な期待値変位だけでなく、2次元VA同時分布の形状変化を捉える幾何・情報幾何的指標を用います。

### 4.1 幾何・デコーディング指標 (Geometry & Decoding)
- **Held-out $R^2$**: 独立したテストセット（70% train / 30% test）に対する線形プローブの予測決定係数。
- **Direct Cross-Decoding $R^2$**: 一方のモデル（例: Base）で学習したプローブを、座標変換を行わずに他方（Instruct）に直接適用した性能。
- **Orthogonal Procrustes $R^2$**: 直交行列 $Q$（回転・鏡映）により表現空間を整列させた後の予測性能（Alignable Geometry の判定）。
- **RSA Correlation ($\rho_{\mathrm{RSA}}$)**: 刺激間相関距離行列（RDM）の Spearman 順位相関。表現空間のトポロジー的一致度。

### 4.2 因果・分布間距離指標 (Causal & Distribution Distance)
- **$WD_V$ (1D Wasserstein Distance)**: Valence 周辺分布間の1次元最適輸送距離。
- **$EMD_{VA}$ (2D Earth Mover's Distance)**: 2次元VA平面上（81候補）の同時確率分布間における最適輸送コスト。Base分布への回復率（Recovery Rate）評価に使用：
  $$\text{Recovery Rate} = \frac{EMD(\text{Instruct}, \text{Base}) - EMD(\text{Patched}, \text{Base})}{EMD(\text{Instruct}, \text{Base})}$$
- **JSD (Jensen-Shannon Divergence)**: 分布間の対称化カルバック・ライブラー情報量（$0 \le JSD \le \ln 2$）。
- **Peak Depth & Dissociation**: 各タスク（Reader / Self）における因果的影響の最大層（$l^*$）と、タスク間の層乖離 $|l^*_R - l^*_S|$。

---

## 5. 実行手順 (How to Run)

### 5.1 環境構築
プロジェクト全体の規約に従い、プロジェクトルートの `.venv` 仮想環境を使用します。

```bash
# プロジェクトルートに移動
cd /mnt/nas/home/hiromi/src/emo

# 仮想環境の有効化
source .venv/bin/activate

# 共通ライブラリのインストール確認
pip install -e .
```

### 5.2 4モデルファミリー横断 中核パイプラインの実行
設定ファイル `configs/v2_experiments.yaml` に基づき、主要RQを順次実行します。

```bash
# 1. V2-RQ1 & RQ2: 表現幾何と交差デコーディング解析
python v2/scripts/run_v2_2x2_cross_decoding.py --device cuda

# 2. V2-RQ3: 因果回路の再配置とピーク解離解析
python v2/scripts/run_v2_2x2_causal_map.py --device cuda

# 3. V2-RQ4: Base activation パッチングによる Instruct 分布回復実験
python v2/scripts/run_v2_recovery_patching.py --device cuda
```

> [!NOTE]
> 特定のモデルファミリー（例: Qwen）のみを対象とする場合は `--family Qwen`、動作確認用の軽量実行には `--dry-run` または `--max-samples 50` を付与します。

### 5.3 詳細コンポーネント解析の実行（Qwen等を対象とした個別検証）
```bash
# 厳密データセットの準備
python v2/scripts/prepare_aipsy_strict.py

# コンポーネントパッチングスクリーニング (Attention / MLP / Residual)
python v2/scripts/run_strict_patching_screening.py

# 複数コンポーネント同時介入 (相乗効果検証)
python v2/scripts/run_synergy_patching.py

# 出力層 8-Condition スワップ解析 (Residual, RMSNorm, lm_head)
python v2/scripts/run_unembedding_norm_swap.py

# Late-Residual バイパスによる出力ゲーティング検証
python v2/scripts/run_output_gating_test.py
```

### 5.4 結果の集計とプロット生成
```bash
# 全メトリクスの集計
python v2/scripts/dump_all_metrics.py

# 論文用テーブルの生成
python v2/scripts/dump_tables.py

# 幾何・パッチング結果の可視化
python v2/scripts/plot_strict_cross_decoding.py
python v2/scripts/plot_patching_and_gating.py
```
