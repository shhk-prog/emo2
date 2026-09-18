# 実装計画: 査読クリティークへの対応策（論理防衛＋追加実験プロトコル）

## 概要
査読者による「Weak Reject / Borderline (4-5/10)」の評価と12項目の指摘を全面的に受け止め、本論文が真にAcceptを獲得するための「論理的防御」と「最小追加実験セット（Exp A〜D）」を策定します。

## 査読指摘の核心と対応方針

### 1. 【最優先/致命的】Within-model positive causal controlの確立 (Exp A)
- **問題**: Instruct内部ですら同一介入（中間MLP最終トークンパッチ）で報告が動かないなら、クロスモデルパッチの0%は「結合変化」ではなく「介入サイトの因果的不十分性（causal insufficiency）」にすぎない。
- **対応**: 
  1. 同一Instruct内で `Peak -> Neutral` パッチを実施。
  2. MLP最終トークンだけでなく、**Residual stream** や **プロンプト内全トークン（all prompt tokens）**、あるいは **複数層同時residualパッチ** を試し、`Recovery_within >> 0` となる介入サイト/プロトコルを突き止める。
  3. その正の対照（positive control）が成立するサイトにおいて、初めてクロスモデル（Raw Base vs Aligned Base）の比較を行う。

### 2. 【重大】Ridge AlignmentのFull-state再構築度評価 (Exp B)
- **問題**: $d=1536, N=124$ で推定されたRidgeは、低次元のValenceプローブ値しか復元しておらず、下流計算に必要な1500次元以上の情報が脱落している可能性が高い。
- **対応**: 
  - $R^2_{\text{activation}} = 1 - \frac{\|\hat{\mathbf{h}}_I - \mathbf{h}_I\|^2}{\|\mathbf{h}_I - \bar{\mathbf{h}}_I\|^2}$ （全次元平均）を算出。
  - CKA (Centered Kernel Alignment) / PWCCA / Pair retrieval accuracy の導入。

### 3. 【重大】Mahalanobis距離・Cosineの経験的参照分布 (Exp C)
- **問題**: $D_M \approx 6$ は $\chi^2_{1536}$ （理論典型値 $\sqrt{1536}\approx 39$）から見て絶対値単独では意味をなさない。Cosine 0.995 も異方性（anisotropy）の可能性がある。
- **対応**: 
  - Held-out natural Instruct 活性化自身の $D_M$ 分布（median, 5-95% tile）との比較（percentile rank）。
  - Natural vs Aligned の Two-sample classifier (Logistic Regression / Linear SVM) による判別可能性（AUC $\approx 0.5$ か否か）。
  - Cosineの対照群（Matched vs Unmatched same/diff valence vs Natural Instruct-Instruct）。

### 4. 【重要】第2のモデルファミリーでの検証 (Exp D)
- **問題**: Qwen2.5-1.5B 1ペアのみでは「post-training」の一般的性質と言えない。
- **対応**: Llama-3-8B / Llama-3.2-1B / Gemma-2-2B 等のBase/Instructペアで、C2（尤度測定＋プロービング＋アライメント＋パッチング）の最小パイプラインを再現。

### 5. 即時テキスト改訂（Paper Draftの緊急防御）
- **C1の格下げ**: 主貢献ではなく、Martorell等の先行知見に準拠した「妥当な測定セットアップ（Measurement setup）」として位置づけ。
- **C4の切り離し**: L16のrandom controlがvalenceより強い（Generic disruption）問題を認め、本文主結果から除外してAppendixへ落とす、または完全に削除。
- **Singh et al. (2026) への準拠**: 「first-person report」は出力フォーマット（grammatical/task format）を指し、内省（introspection）や主観的体験の証拠ではないことを明記。
- **EMD Recoveryの分母問題**: 分母分布の提示と、Raw EMD shift, $\Delta E[V]$, JSD等との一貫性を明記。

## 実行フェーズ
- **フェーズ1（即時）**: 論文草稿（`v3/docs/paper.md`）の論理防御改訂（C1格下げ、C4のAppendix移動/除外、Singh et al. 注記、限界と前提の厳密化）
- **フェーズ2（実験準備・実行）**: Exp A (Within-model positive control: residual stream & token positions) のスクリプト作成と検証
- **フェーズ3（実験実行）**: Exp B (Full-state reconstruction) & Exp C (Empirical OOD) の解析スクリプト作成と実行
- **フェーズ4（拡張）**: Exp D (Second model family) の実行
