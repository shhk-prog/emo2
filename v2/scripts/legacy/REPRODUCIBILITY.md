# 論文再現性検証ガイド: "When Affective Self-Reports Do Not Trace Internal Representations"

本文書は、本論文における実験および分析を外部の研究者が完全に再現するための手順を詳細に記載したガイドです。

## 1. 環境構築 (Environment Setup)

### 動作要件
- **OS**: Linux (Ubuntu 20.04/22.04 等を推奨)
- **Python**: 3.10以上
- **GPU**: NVIDIA GPU (最低 8GB VRAM が必要。16GB以上を推奨。Qwen2.5-1.5B のロードと中間表現のキャッシュにメモリを消費します)
- **CUDA**: 11.8 または 12.x

### インストール手順
Pythonのパッケージ管理および仮想環境構築には `uv` の使用を強く推奨します（`venv` または `conda` も利用可能です）。

```bash
# 仮想環境の作成と有効化
uv venv .venv
source .venv/bin/activate

# 依存パッケージのインストール (開発用パッケージを含む)
uv pip install -e ".[dev]"
```

## 2. データセットの準備 (Dataset Preparation)

本実験では、Valence（感情価）を制御した最小対データセット（AIPSY minimal-pair dataset）を使用します。

- **原データ (Read-only)**: `data/raw/` に配置されています。このデータは直接変更せず、読み取り専用の原本として扱います。
- **実験用データ (Processed Stimuli)**: 前処理されたデータは `v2/data/processed/aipsy_annotated/test_strict.csv` に出力されます。

**データの前処理の実行:**
原データから実験用の厳密なデータセット（test_strict.csv）を生成するには、以下のコマンドを実行します。
```bash
python v2/scripts/prepare_aipsy_strict.py
```
> **期待される結果**: `v2/data/processed/aipsy_annotated/test_strict.csv` が生成され、後続の実験の準備が完了します。

## 3. 実験の実行 (Running the Experiments)

実験は `Qwen/Qwen2.5-1.5B` (Baseモデル) および `Qwen/Qwen2.5-1.5B-Instruct` (Instructモデル) を対象に行われます。実行時に初回のみ Hugging Face Hub からモデルがダウンロードされます。

### 3.1 事前行動評価とプローブ学習 (Preliminary Behavioral Evaluation & Probe Training)
モデルが感情価をどのように表現しているか、そして線形プローブを用いた感情価の分類精度を評価します。
```bash
python v2/scripts/run_probing_preliminary.py
```
> **出力先**: 精度やプローブの重み等の結果が `v2/results/` 下に保存されます。

### 3.2 表現の変換・交差デコーディング (Representation Transformation)
BaseモデルとInstructモデルの中間表現を相互にデコードし、両者の表現空間がどのように共有・分離されているかを分析します。
```bash
python v2/scripts/run_strict_cross_decoding.py
```
> **出力先**: `v2/results/derived/` 下に交差デコーディングの精度や距離メトリクスが出力されます。

### 3.3 Path Patching と Substitution (置換) テスト
本論文の核心となる、因果的なメカニズムを特定するための介入実験群です。

**① 活性化パッチングによるスクリーニング (Activation Patching Screening)**
どの層・コンポーネントが自己報告に影響を与えているかを広くスクリーニングします。
```bash
python v2/scripts/run_strict_patching_screening.py
```
> **出力**: Table 3の元となるコンポーネントごとの影響度（$\Delta E[V]$ や $\Delta WD_V$）が計算されます。

**② 後期残差への Substitution テスト (Late-Residual Substitution Test - Table 4)**
特定のコンポーネントから最終層への直接的な寄与が、Instructモデルの分布をBaseモデルの分布へと回復させるか（十分性）をテストします。
```bash
python v2/scripts/run_strict_causal_scrubbing.py
```
> **出力**: `v2/results/derived/phase8_causal_scrubbing/strict_causal_scrubbing_results.csv` が出力されます。

**③ 出力段（Unembedding/RMSNorm）のパラメータスワップ (Swap Analysis - Table 5)**
出力層（`lm_head`）および最終レイヤー正規化（`RMSNorm`）の重みをBaseモデルとInstructモデル間で入れ替えた際の影響を分析します。
```bash
python v2/scripts/run_unembedding_norm_swap.py
```
> **出力**: 2次元の分布間距離（EMDやJSD）を含む結果が記録されます。

## 4. プロンプトテンプレートとハッシュ値の検証 (Prompt Templates and Hashes)

本研究の介入実験（事前評価を除く）では、トークナイゼーションやフォーマットの違いによる交絡（Confounding）を完全に排除するため、**一言一句違わない厳密なプロンプト（Strictly Identical Prompt）**を使用しています。

具体的には、Instructモデルのチャットテンプレートに基づく文字列を生成し、JSONのキー（`{ "valence": `）の直前までの同一文字列を入力として用いています。

外部検証の際、使用したプロンプトが元の実験と完全に一致しているかを確認するために、以下のスクリプトで SHA-256 ハッシュ値を計算できます。
```bash
python v2/scripts/generate_reproducibility_artifacts.py
```
> **期待されるハッシュ値**: `645e97c5ee5afde7d322f77684905f71edb9df7af721e553f566686acc8eaf0b`
> **出力先**: `v2/results/prompt_hashes.json` にハッシュ値とプロンプトの完全なテキストが保存されます。

## 5. 統計分析と信頼区間の計算 (Statistical Analyses & Bootstrap CIs)

Table 4などで示されている95%信頼区間（CI）や、Matched BaseとRandom Source間の差分は、ペア単位のクラスターブートストラップ（Cluster Bootstrap、再サンプリング10,000回）により計算されています。

計算を実行するには以下のコマンドを使用します。
```bash
python v2/scripts/calculate_bootstrap_ci.py
```
> **期待される結果**: コンソール上に各コンポーネントの信頼区間が出力され、Markdown形式でそのまま論文（Table 4）に貼り付けられる行データが出力されます。
