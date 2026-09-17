# 修正内容・解説の確認 (Walkthrough): v2 活性化ステアリング（Steering）の詳細結果

## 1. 実施概要
ユーザーからの「v2 実験③ Steeringの詳しい結果を教えて」という要望に対し、v2で実施されたActivation Steering実験（`v2/scripts/run_steering_and_likelihood.py`, `plot_steering_results.py`）の実測データ（`v2/results/derived/phase2_steering/`）および生成されたプロットを詳細に分析・整理しました。

---

## 2. 実験設計と介入仕様

- **目的**: 内部の感情方向（Valence Contrastive Direction $d_V$）に直接介入した際、自己報告期待値 $E[V_{self}]$ がどれだけ感度よく応答するか（Steering Slope: $\frac{\partial E[V]}{\partial \alpha}$）をBaseとInstructで比較し、Readout Suppression (H3) を検証する。
- **対象モデル**:
  - `Qwen/Qwen2.5-1.5B` (Base)
  - `Qwen/Qwen2.5-1.5B-Instruct` (Instruct)
- **介入層**: Layer 20, Layer 24, Layer 27（後期残差接続）
- **介入強度**: $\alpha \in \{-3.0, -1.5, 0.0, +1.5, +3.0\}$
- **数式**: $h_{l, p} \leftarrow h_{l, p} + \alpha \cdot \sigma_l \cdot d_V$
  - ここで $\sigma_l$ は各層の活性化標準偏差、$d_V$ は単位対立ベクトル。
  - プロンプト最終トークン以降の全生成ポジションにフックを継続適用。

---

## 3. 実測結果とプロット分析

### (1) Layer 20 における Steering Slope
- **Baseモデル**: $\alpha = -3.0 \to +3.0$ において、$E_V \approx 5.22 \to 5.19$ と極めて平坦。
- **Instructモデル**: $\alpha = -3.0 \to +3.0$ において、$E_V \approx 5.52 \to 5.54$ と平均値はほぼ不変。ただし、95%信頼区間が 4.65〜6.40 と極めて広く、分散が増大。

### (2) Layer 24 における Steering Slope
- **Baseモデル**: $\alpha = -3.0 \to +3.0$ において、$E_V \approx 5.20 \to 5.23$ と微小な変化。
- **Instructモデル**: $\alpha = -3.0 \to +3.0$ において、$E_V \approx 5.57 \to 5.51$ と微弱な負の傾きを示すが、誤差範囲内で有意な単調制御はみられない。

### (3) Layer 27 における Steering Slope
- **Baseモデル**: $\alpha = -3.0 \to +3.0$ において、$E_V \approx 5.14 \to 5.28$ とわずかに正の傾斜。
- **Instructモデル**: $\alpha = -3.0 \to -1.5$ までは $E_V \approx 5.51$ だが、$\alpha = 0.0$ で $E_V \approx 6.28$ へ跳ね上がり、正の過大介入（$\alpha = 1.5, 3.0$）では出力の崩壊・フォーマット破綻（NaN / 出力異常）が発生。

---

## 4. 科学的解釈とNegative Resultの意味

1. **「Decodability $\neq$ Steerability（解読可能性と操作可能性の解離）」**:
   - 線形プローブによって内部表現から感情情報が高精度に復元可能（Decodable）であっても、その単一の線形軸に沿って活性化を加算するだけでは、自己報告を安定して単調に制御（Steerable）することはできませんでした。
2. **Readout Suppressionの単一線形モデルの限界**:
   - 単純な「1本の読み出しパイプラインが絞られている（一様減衰）」という仮説（H3）では説明がつかず、複数層・複数コンポーネントが非線形に干渉し合って中立出力を形成している「分散的再写像（Distributed Remapping, H4）」を強く支持する結果となりました。
