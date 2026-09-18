# 実装計画: v2におけるRSA（表現類似度分析）詳細結果の体系的提示

## 1. 概要
v2で実施された「三空間RSA（Representational Similarity Analysis）」について、その目的、数理的定義、全層（Layer 0〜27）の実測値、中間層における特異的傾向、および統制回帰分析との統合的解釈をまとめる。

## 2. 検討項目と構成

### 2.1 実験設計と三空間の定義
- **Human Geometry ($D^H$)**: 人間注釈（辞書ベース・刺激評価）のValence-Arousal空間における刺激間ユークリッド距離行列。
- **Internal Geometry ($D_l^I$)**: モデル内部隠れ状態（Layer $l$）から射影された情動部分空間における刺激間ユークリッド距離行列。
- **Self-Report Geometry ($D^S$)**: モデルの行動出力である自己報告期待値 $(E_v, E_a)$ 空間における刺激間ユークリッド距離行列。
- **3つのRSA指標**:
  - $\mathrm{RSA}_{H,I,l} = \mathrm{corr}(\mathrm{vec}(D^H), \mathrm{vec}(D_l^I))$: 人間感情と内部幾何の一致度
  - $\mathrm{RSA}_{H,S} = \mathrm{corr}(\mathrm{vec}(D^H), \mathrm{vec}(D^S))$: 人間感情と自己報告幾何の一致度（全層共通定数）
  - $\mathrm{RSA}_{I,S,l} = \mathrm{corr}(\mathrm{vec}(D_l^I), \mathrm{vec}(D^S))$: 内部幾何と自己報告幾何の一致度（Coupling幾何）

### 2.2 実測結果（全28層のテーブルと統計サマリー）
- `v2/results/derived/phase1.5_confirmatory/rsa_analysis.csv` の全層データを整理。
- $\mathrm{RSA}_{H,S} = 0.124$（一定・微小な相関）。
- $\mathrm{RSA}_{I,S,l}$：中間層（Layer 10で $-0.176$、Layer 18で $-0.197$ など）で顕著な**負の相関**を示し、幾何構造のねじれ・解離が発生。
- $\mathrm{RSA}_{H,I,l}$：多くの層で $-0.13 \sim +0.10$ の狭い範囲に留まり、人間評価空間とInstruct内部幾何が単純な等長写像になっていない。

### 2.3 統制回帰分析（Controlled Regression）との統合解釈
- $E_v \sim z_V + V_H + \text{word\_count}$
  - $z_V$（内部表現）の寄与: $\beta = -0.0113, p = 0.676$（非有意・切断）
  - $V_H$（表層テキストVAD）の寄与: $\beta = +0.2933, p = 0.003$（有意）
- 結論：内部空間に表現が存在しても、それが自己報告には反映されず、表層特徴が表面的な中立・微小応答を駆動している「System Dissociation」の確証。
