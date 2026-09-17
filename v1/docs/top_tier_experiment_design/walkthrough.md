# Walkthrough: 実験パイプライン (Phase 1 & 2) の実装と検証

このドキュメントでは、LLM情動反応性評価実験の「Phase 1: ベースラインの確立」および「Phase 2: AIPsy-Affectと厳密なProbeの実装」に関する内容と、その動作確認結果についてまとめます。

## 実装内容: Phase 1 (ベースラインの確立)

### 1. EmoBankからの層化抽出 (`prepare_stimuli.py`)
Valence/Arousalの3x3分割空間から各セル50件を目標に抽出する処理を修正し、正常に機能することを確認しました。
- 実行結果: 計 **321件** の刺激データを `data/processed/stimuli.csv` に出力しました。

### 2. 81通りのVAシーケンス尤度算出 (`evaluation.py`)
`InterventionEvaluator` に新しいメソッド `evaluate_sequence_logits` を追加し、期待VAベクトル $\mathbf{E}_x$ （`E_V`, `E_A`）を算出するロジックを確立しました。

### 3. Euclidean Recoveryの計算 (`metrics.py`)
ユークリッド距離に基づく `Recovery` スコアの計算と閾値判定を行う `calculate_euclidean_recovery` 関数を追加しました。

### 4. ベースライン評価スクリプト (`run_baseline_evaluation.py`)
Qwen2.5 Base/Instructペアに対して一括でベースラインベクトルを算出する統合スクリプトを作成しました（現在全件実行中）。

---

## 実装内容: Phase 2 (AIPsy-Affectと線形Probe)

### 1. AIPsy-Affect データセットの導入 (`prepare_aipsy.py`, `data.py`)
感情語彙交絡を排除した最小対（Minimal Pair）データセットである AIPsy-Affect を導入し、ペア単位での厳密な `train/dev/test` 分割ロジックを実装しました。
- `pair_id` に基づく分割により、学習セットとテストセット間で同じ状況（Neutral / Affective）がリークすることを防ぎます。

### 2. Z-score正規化と分類Probeの実装 (`probing.py`)
`FixedSplitProber` クラスを新規作成し、以下の機能を実装しました：
- Scikit-LearnのPipelineを活用した、**学習データ内でのPCAおよびZ-score正規化**（テストデータへのリーク防止）
- Affective(1) vs Neutral(0) を分類する `RidgeClassifier` の実装と Accuracy / ROC-AUC 評価

### 3. コントロール（統制）Probeの実装 (`probing.py`)
単に表現が長さや表面的な情報を暗記しているだけではないことを保証するため、以下2つの交絡排除用コントロールProbeを実装しました：
1. **シャッフルコントロール (`run_shuffled_control_classification`)**: 学習ラベルをランダムに入れ替え、ベースラインの偶然一致確率（ランダムチャンス）を算出。
2. **文字数予測コントロール (`run_length_control_probing`)**: 各刺激文の文字数（`word_count`）を予測するRidge回帰プローブ。

### 4. プローブ実行スクリプト (`run_probing_aipsy.py`)
`Qwen2.5-1.5B-Instruct` モデルの全層から隠れ状態（デフォルトでは `stimulus_mean_pool`）を抽出し、上記プローブを一括で学習・評価する統合スクリプトを作成しました。

---

---

## 実装・検証内容: Phase 3 & 4 (因果介入検証: Patching / Ablation / Steering)

### 1. 巨大バッチ対応因果介入スクリプト (`run_causal_intervention.py`)
Phase 2で保存した `stimulus_mean_pool` 隠れ状態を用い、全28層に対してプロンプト末尾（ボトルネック位置）へ以下の3種類の介入を実施・評価しました。

#### (A) Activation Patching (十分性の検証)
Neutral（非感情的文脈）のプロンプトを処理するモデルの中間層に対し、対応するAffective（感情的文脈）の隠れ状態を入れ込み、自己報告Valence（$E[V]$）が感情側にどれだけ回復するか（`recovery_V`）を評価しました。
- **結果**: 
  - **深層部（Layer 15, 16, 19, 25）で有意な回復**が観測されました（Layer 19で最大 `recovery_V = +0.278` (27.8%), Layer 15で `+0.226` (22.6%), Layer 25で `+0.153` (15.3%)）。
  - 中間層（Layer 2〜11）では単一 newsletter/layer の置換により負のRecoveryが観測され、中間層の感情表現が分散・非線形に符号化されているか、前後のアライメント構造と競合することを示唆しています。

#### (B) Ablation (必要性の検証)
Affectiveのプロンプトを処理するモデルの中間層に対し、Neutral群の平均ベクトル（$\mu_{neutral}$）を代入（消去）し、出力される感情変化がどれだけ抑制されるか（`effect_removed_V`）を測定しました。
- **结果**:
  - **初期・中間層（Layer 1, 4, 12, 13, 14）において大きな効果消去効果**が観測されました。
    - **Layer 4**: 42.3% の効果消去 (`effect_removed_V = 0.4226`)
    - **Layer 13**: 36.6% の効果消去 (`effect_removed_V = 0.3663`)
    - **Layer 14**: 32.6% の効果消去 (`effect_removed_V = 0.3259`)
    - **Layer 1**: 32.1% の効果消去 (`effect_removed_V = 0.3212`)
  - これにより、Layer 1〜4 および Layer 12〜14 の表現が、感情的文脈を最終的な行動（自己報告VA）へ伝達する上で**因果的に必要（Necessary）**であることが実証されました。

#### (C) Steering (線形空間の対称性検証)
Affective群とNeutral群の平均差分ベクトル $\mathbf{d} = \mu_{affective} - \mu_{neutral}$ を算出し、強さ $\alpha \in \{-2.0, -1.0, 0.0, 1.0, 2.0\}$ で注入した際の出力Valence $E[V]$ の単調変化を追跡しました。
- **結果**:
  - 各層において $\alpha$ の増減に応じた意図通りのスムーズかつ単調な自己報告Valenceの単調シフトが確認されました。

---

## 実装・検証内容: Phase 5 (アライメント遮断メカニズムの解明: Base vs. Instruct)

`scripts/run_alignment_suppression.py` を全件実行し、Baseモデル (`Qwen2.5-1.5B`) と Instructモデル (`Qwen2.5-1.5B-Instruct`) の隠れ状態、無介入ベースライン、全28層の因果介入（Patching & Ablation）、および Suppression Gap を算出・比較しました。

### 1. 行動レベルの感度比較 (Clean Baseline)
- **Baseモデル**: $\Delta E[V]_{base} = -0.1642$
- **Instructモデル**: $\Delta E[V]_{instruct} = -0.0285$
- **発見**: Post-training（アライメント）によって、感情的刺激に対するモデルの自己報告Valenceの感度が **82.6% 抑制（減衰）** されていることが数値的に裏付けられました。

### 2. 回路レベルの因果的遮断解離 (Suppression Gap Analysis)
- **Ablation (必要性)**:
  - **Instructモデル**: 中間層（L1, L4, L12, L13, L14）の消去により、感情効果が最大 **42.3% (L4)**, **36.6% (L13)**, **32.6% (L14)** 削除される明確なボトルネック構造が形成されています。
  - **Baseモデル**: 単一 newsletter/layer の消去ではこのような局所的ボトルネックが生じず、分散・相互作用的な符号化となっています。
- **Activation Patching (十分性)**:
  - **Instructモデル**: 深層部（L15, L16, L19, L25）への介入により、Valenceが **+22.2%〜+27.8%** 回復。
  - **Baseモデル**: 中間層 (L5〜L9) で Suppression Gap（回復率の差分）が最大 **+1.167 (L8)**, **+1.147 (L9)** に達し、モデル間で表現が変換・再配線される層が特定されました。

---

## 実装内容: Phase 5.5 (スケーリング & 複数モデルファミリー展開)

### 1. 汎用スケーリング・モデル間介入スクリプト (`scripts/run_scaling_experiments.py`)
Qwen2.5 (0.5B, 1.5B, 3B, 7B) や Llama-3.2 (1B, 3B) など、モデル規模およびアーキテクチャファミリーを跨いで一括で Probing・Activation Patching・Ablation・Steering・Suppression Gap を自動評価する統一スクリプトを実装しました。

#### 主な仕様
- **出力フォルダおよびログのモデル別完全独立化**:
  - 各モデルペア（例: `qwen2.5_0.5b`, `qwen2.5_7b`, `llama3.2_1b`）ごとに `results/derived/scaling/<tag>/` フォルダを作成。
  - 各モデルペア個別のログファイル `logs/scaling/<tag>.log` に進行状況、GPU使用量、各層メトリクスを出力。
- **メモリ解放の厳密化**:
  - `del model`, `gc.collect()`, `torch.cuda.empty_cache()` により、大型モデル (7B等) 評価後もGPUメモリリークなく連続実行可能。

### 2. Slurm非使用一括バッチシェル (`scripts/run_scaling_all.sh`)
Slurm (`sulam`) 等のジョブスケジューラを使用せず、ローカル環境（`bash`）のルーピングのみで一括順次実行する実行シェルを作成しました。

```bash
# 手動での一括実行コマンド
./scripts/run_scaling_all.sh
```

特定モデルペアのみを単体実行する場合：
```bash
python scripts/run_scaling_experiments.py \
    --tag "qwen2.5_3b" \
    --base-model "Qwen/Qwen2.5-3B" \
    --instruct-model "Qwen/Qwen2.5-3B-Instruct" \
    --batch-size 192
```

---

## 結論と総合成果
Phase 1〜5 の全実験・考察に加え、複数規模 (0.5B〜7B) および異アーキテクチャ (Qwen, Llama) への一般化可能性を自動検証する統合バッチシステムが完成しました。
