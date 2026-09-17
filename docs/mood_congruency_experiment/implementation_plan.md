# 実装計画: LLMにおける「気分一致バイアス（Mood Congruency Bias）」の因果実証実験

## 1. 実験の背景と目的
人間心理学における**気分一致効果（Mood Congruency Effect）** とは、「自分が快（ポジティブ）な気分の時は他者の表情や文章も好意的に解釈し、不快（ネガティブ）な気分の時は敵対的・悲観的に解釈しやすくなる」という認知的偏向である。

本実験では、LLMの隠れ状態（Residual stream）に **Activation Steering** を施して「擬似的な気分状態（Induced Mood: Positive / Negative）」を誘導した上で、**他者の感情を推定する客観認識タスク（Recognition）** を行わせる。
もし誘導された気分によって他者の感情評価値（推定Valence）に統計的に有意な偏向（バイアス）が生じるならば、**「モデルの内部感情表現と他者認識回路が内部特徴量を因果的に共有している」** ことが直接証明される。

---

## 2. 実験設計（Experimental Design）

### 2.1 実験条件と介入
- **モデル**: `Qwen/Qwen2.5-1.5B-Instruct`（および対照として `Qwen/Qwen2.5-1.5B` Base）
- **介入層 (Steering Layers)**: Layer 14, 16, 20（表現プロービングで高精度だった中間層）
- **介入方向 (Steering Directions)**:
  1. `valence`（ターゲット条件: ポジティブ $\leftrightarrow$ ネガティブ）
  2. `random`（陰性統制条件: ノルムを揃えたランダムベクトル）
  3. `arousal`（直交性統制条件: 活性度を操作してもValence評価に影響しないか確認）
- **介入強度 ($\alpha$)**: $\alpha \in \{-3.0, -1.5, 0.0, +1.5, +3.0\}$（学習データの標準偏差 $\sigma$ 単位）

### 2.2 刺激データセット（Stimuli）
- **EmoBank 評価セット**:
  - **曖昧・中立刺激（Ambiguous/Neutral subset）**: 人間アノテーションのValenceが $2.5 \sim 3.5$（1〜5スケール）の刺激。人間でも解釈が割れる刺激において、バイアスが最も鋭敏に現れる。
  - **明確刺激（Clear Positive / Negative subset）**: Valence $< 2.0$ または $> 4.0$ の刺激。床効果・天井効果および全体的なオフセットの確認。
- **AIPsy-Affect Strict Subset**: 語彙交絡が統制された文脈刺激（Neutral, Moderate, Peak）。

### 2.3 プロンプト設計（Recognition vs Reactivity の厳密分離）
- **他者感情認識（Recognition Prompt）**:
  ```text
  Read the following text and estimate the emotional state of the writer/speaker.

  Text: {text}

  Respond strictly in JSON format with 'valence' and 'arousal' keys (1-9).
  ```
  ※ 一人称の自己報告（"report your affective state"）ではなく、**三人称の他者推定** を指示することで、中立化ガードレールを回避し、純粋な認識回路を駆動する。

### 2.4 測定指標（Endpoints）
- **Sequence Likelihood Protocol**: 81通り（$V, A \in \{1..9\}$）の候補JSONの同時対数尤度から期待値 $E[V_{\text{rec}} \mid x, \alpha]$ を算出。
- **気分一致バイアス量**:
  $$\Delta V_{\text{rec}}(\alpha) = E[V_{\text{rec}} \mid x, \alpha] - E[V_{\text{rec}} \mid x, \alpha=0]$$
- **Greedy JSON出力**: 離散出力レベルでもバイアスが観測されるかを記録。

### 2.5 統計モデリングと仮説検証
線形混合効果モデル（Linear Mixed-Effects Model）を用いて、刺激個体差をランダム効果として制御しつつ気分効果を推定する：

$$E[V_{\text{rec}}]_{i,\alpha} = \beta_0 + \beta_1 \cdot V^{\text{human}}_i + \beta_{\text{mood}} \cdot \alpha + u_{\text{item}(i)} + \epsilon_{i,\alpha}$$

- **主仮説 (H_mood)**: $\beta_{\text{mood}} > 0$（有意水準 $p < 0.001$）。
- **統制仮説**: Random方向およびArousal方向では $\beta_{\text{mood}} \approx 0$。

---

## 3. 実装計画とファイル構成

### [NEW] スクリプトの新規作成
#### `v3/scripts/run_mood_congruency_experiment.py`
- 既存の `v2/scripts/run_steering_and_likelihood.py` と `v3/scripts/run_dual_outcome_behavior.py` を統合・洗練。
- 認識プロンプト（Recognition）に対するステアリング介入と81候補尤度測定をバッチ処理。
- 介入ログ、尤度分布、$E[V_{\text{rec}}]$、$E[A_{\text{rec}}]$、Greedy出力をJSONLで保存。

#### `v3/scripts/analyze_mood_congruency.py`
- 統計解析（混合効果モデルのフィッティング、$\beta_{\text{mood}}$ の推定、95%信頼区間、p値の算出）。
- 刺激の感情強度（曖昧 vs 明確）ごとのバイアス量プロット生成。

---

## 4. 実行とリソース（GPU利用方針）
- **事前の安全確認**:
  - `AGENTS.md` のルール（1.7 GPUを勝手に用いない、5.5 コストとパイロット）に従い、まずは **10サンプル程度のDry-run / パイロット実行** を手元で確認した上で、本番実行を行います。
  - スクリプトの作成・検証まではGPUを占有せず、準備が整った段階で実行手順をご案内します。

## 5. ユーザーへの確認事項 (User Review Required)
1. **刺激セットの優先順位**:
   - 主分析に **EmoBankの曖昧刺激群（人間評価 $V \approx 3.0$ 付近）** を中心に据える方針でよろしいでしょうか？（心理学の実験パラダイムと完全に一致します）
2. **比較対象**:
   - Instructモデル単体での検証に加え、Baseモデルとの比較も含めるか、まずはInstructモデルに集中するか。
