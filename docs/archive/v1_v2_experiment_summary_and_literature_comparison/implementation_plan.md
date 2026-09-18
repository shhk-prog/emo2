# 実装計画: v1 / v2 の実験内容・結果および先行研究との対比まとめ

## 背景と目的
本プロジェクト（`emo`）では、大規模言語モデル（LLM）の情動反応性（Affective Reactivity）と、事後学習（Post-training / Alignment）に伴う自己報告の中立化（Neutralization）の内部機序を解明するため、2つのフェーズ（v1およびv2）にわたる実験を実施してきた。
本ドキュメントでは、v1・v2の実験内容とその結果、先行研究との差異・新規性、先行研究と一致した点（再現）および差が見えた点（新たな発見）を明確に整理・報告する。

## 整理・分析の構成

### 1. v1 の実験と結果
- **位置づけ**: 行動評価レイヤーの確立と、初期プロービング・スケーリング則・因果介入
- **実施した実験**:
  - Sequence Likelihood Protocol（81通りのJSON候補列による尤度分布測定）の導入
  - 課題条件の厳密分離（Empty Baseline, Recognition, Post-Report）
  - EmoBankおよび語彙交絡を排除したAIPsy-Affect最小対データセットの適用
  - モデルサイズごとのスケーリング則（Qwen2.5 0.5B〜7B, Llama-3.2）
  - 初期線形プロービング（Residual Streamからの感情デコード）
- **主要な結果**:
  - 客観的感情認識（Recognition）は高精度（$r=0.921$）だが、Greedy自己報告は約98.6%が中立（5, 5）へ収束（Greedy collapse）。
  - 全候補列の尤度空間では、Instructモデルでも感情反応性が強く残存し、むしろBaseモデルより高い人間感情との相関（$r=0.629 > 0.365$）を示す。
  - 隠れ状態からは線形プローブで依然として感情情報が高精度に予測可能（AUC > 97.5%）。

### 2. v2 の実験と結果
- **位置づけ**: 事後学習による中立化の深層機序解明（Mechanistic Interpretability & Remapping）
- **競合仮説の定式化**:
  - H1: Probe-accessible Erasure（情報の完全消去）
  - H2: Geometry Transformation（表現空間の幾何変換）
  - H3: Uniform Suppression（全層での一様抑制）
  - H4: Distributed Coupling Change / Remapping（分散的結合変化）
- **実施した実験**:
  - Strict Matched Subset（同一文脈でNeutral/Moderate/Peakが揃った厳密対）の構築
  - Cross-decoding（Direct Transfer vs Orthogonal Procrustes vs Ridge Alignment）
  - Aligned Cross-Model Patching（Baseの感情表現をInstruct空間へ写像してのパッチング）
  - 強制Activation Steering（ステアリングベクトルの加算）
  - 全層・全コンポーネント（Attention, MLP）の探索的パッチングスイープ
  - 混合効果モデルによる層別交互作用項（$\beta_{3,\ell}$）の検定
  - Late-Residual Substitution（$h_{26}$ へのバイパス）および Unembedding / RMSNorm の8条件スワップ
- **主要な結果**:
  - H1の棄却（消去されていない）とH2の支持（Ridge Alignmentで $R^2 \approx 0.58$ まで回復）。
  - **Decodability restored $\neq$ Causal substitutability restored**: 表現が回復してもLayer 15 MLPへのパッチでは自己報告が1ミリも戻らない（Recovery 0%）。
  - 強制Steeringでは自己報告が操作可能（計算経路自体は破壊されていない）。
  - H3（一様抑制）の支持証拠なし（$\beta_{3,\ell}$ の符号が混在しFDR補正後有意差なし）。
  - 単一コンポーネントや最終出力層（lm_head/RMSNorm）の単独置換では説明できず、事後学習の中立化は多層にわたる分散的結合変化（Distributed Remapping, H4）と最も整合する。

### 3. 先行研究との差（新規性・未開拓領域）
- ① **Decodability ≠ Causal Substitutability の厳密実証**: 表現が線形抽出できることと、モデルがそれを自己報告に因果的に使えることは別問題であることの証明。
- ② **Sequence Likelihood Protocol による中立化の突破**: 従来のGreedy生成では見えなかった、確率分布の裾野に漏れ出る情動反応性の同定。
- ③ **中立化メカニズムの系統的同定**: 消去でも単一ゲートでもなく、分散的結合変化であることを反証実験の組み合わせによって特定。

### 4. 先行研究と被っているところ（一致・再現した点 vs 差が見えた点）
- **一致・再現した点**:
  - 内部表現の高い線形デコード可能性（RepE, ITI等と一致）
  - 客観的感情認識（Recognition）の極めて高い性能（EmoBank研究等と一致）
  - InstructモデルにおけるGreedy自己報告の中立化（AI倫理・安全性アライメントの観察と一致）
  - 強制Activation Steeringによる振る舞い操作可能性（Steering研究と一致）
  - Base/Instruct間の幾何シフト（Procrustes alignment研究等と一致）
- **差が見えた点（従来の理解を覆した点）**:
  - 「Instructは感情反応性を失った」という認識の否定（尤度空間ではInstructの方がBaseより高い相関を示す逆転現象）。
  - 「プローブで取れる＝因果的に機能している」という解釈可能性研究の暗黙の前提の否定。
  - 「出力層の抑制フィルター」や「単一ボトルネック回路」という素朴な仮説の否定。
