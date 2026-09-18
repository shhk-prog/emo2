# Implementation Plan: AIPsy-Affect 4-split（強度・複雑性・感情カテゴリ統制）評価システム

## 1. 背景と目的
従来の「Affective vs. Neutral」の二値比較から発展させ、AIPsy-Affect が持つ4つのsplit（clinical, moderate, neutral, complex_neutral）をフル活用して、以下の3要因を完全に分離・統制して評価する：
1. **感情刺激の有無 (Sensitivity)**: clinical vs. neutral
2. **感情強度 (Dose-Response)**: neutral → moderate → clinical
3. **文章複雑性 (Specificity / Complexity Control)**: complex_neutral vs. neutral, clinical vs. complex_neutral
4. **8感情カテゴリ特異性 (Discrete Emotion Profiles)**: grief, terror, rage, loathing, ecstasy, admiration, amazement, vigilance
5. **認識と自己報告の解離 (Recognition–Self-report Coupling)**: Reader-Response Prediction vs. Self-Report

## 2. 4大リサーチクエスチョン (RQ) の数式定義

### RQ1: Sensitivity（感情感受性）
- 対象: 192組の matched minimal pairs (clinical vs. neutral)
- 指標:
  $$\Delta V_{\text{affect}} = E[V]_{\text{clinical}} - E[V]_{\text{neutral}}$$
  $$\Delta A_{\text{affect}} = E[A]_{\text{clinical}} - E[A]_{\text{neutral}}$$
- 統計的検定: 対応のあるt検定 / Wilcoxon符号付順位検定、効果量 Cohen's $d$

### RQ2: Dose-Response（感情強度の段階性）
- 対象: 48組の matched triplets ($\text{neutral} \rightarrow \text{moderate} \rightarrow \text{clinical}$)
- 指標:
  - 段階的差分: $\Delta_{\text{mod-neu}} = E[\cdot]_{\text{moderate}} - E[\cdot]_{\text{neutral}}$, $\Delta_{\text{peak-mod}} = E[\cdot]_{\text{clinical}} - E[\cdot]_{\text{moderate}}$
  - 単調性レート (Monotonicity Rate): 感情の極性に応じた単調変化（例: negative感情で $E[V]_{\text{neu}} > E[V]_{\text{mod}} > E[V]_{\text{peak}}$）を満たす刺激組の割合
  - 線形トレンド傾き (Linear Trend Slope) および順位相関 Spearman $\rho$

### RQ3: Specificity（複雑性統制による感情特異性）
- 対象: 48件の complex_neutral 文（文長・複雑性はclinicalと同等、感情価はneutral）
- 指標:
  - 複雑性効果: $\Delta V_{\text{complexity}} = E[V]_{\text{complex\_neutral}} - E[V]_{\text{neutral}}$
  - 純感情効果: $\Delta V_{\text{controlled}} = E[V]_{\text{clinical}} - E[V]_{\text{complex\_neutral}}$
  - 特異性比率: $|\Delta V_{\text{clinical-neutral}}| / (|\Delta V_{\text{complexity}}| + \epsilon)$

### RQ4: Recognition–Self-Report Coupling（認識と自己報告の連動性）
- 各条件において Reader-Response Prediction ($R$) と Self-Report ($S$) を独立測定
- 指標:
  - 結合相関: $\text{Corr}(\Delta \text{VA}_R, \Delta \text{VA}_S)$
  - 解離検定: $\Delta \text{VA}_R$ は有意に動くが $\Delta \text{VA}_S$ は中立固定（Collapse）される度合いのモデル間比較

## 3. 実装計画

### Step 1: データセット前処理スクリプトの作成
- ファイル: `v1/scripts/prepare_aipsy_4splits.py`
- 出力: `v1/data/processed/aipsy_4split_all.csv` (全480件)
- `triplet_id`（48組の neutral-moderate-clinical）および `pair_id`（192組の clinical-neutral）を付与。

### Step 2: 推論スクリプトの作成
- ファイル: `v1/scripts/run_aipsy_4split_evaluation.py`
- 729候補のバッチ尤度計算により、各文に対して Reader Prediction ($R$) と Self-Report ($S$) の $E[V], E[A], E[D]$ および最尤 Greedy 出力を算出。
- 出力: `v1/results/aipsy_4split_eval/{model_tag}_aipsy_4split.csv`

### Step 3: 詳細集計・分析スクリプトの作成
- ファイル: `v1/scripts/summarize_aipsy_4split.py`
- RQ1〜RQ4の各表、8感情カテゴリ別プロファイル表、全モデル横断サマリー表をMarkdownおよびCSVで出力。

### Step 4: 一括バッチ実行スクリプトの作成
- ファイル: `v1/scripts/run_all_aipsy_4split.sh`
- 全8モデル（Base/Instruct × 4ファミリー）の一括実行。

## 4. 検証手順
- データセット生成の整合性確認（480件、ペアおよびトリプレットの結合検証）
- 1モデルでのドライラン検証
- 集計スクリプトの出力確認（RQ1〜RQ4のテーブルフォーマット検証）
