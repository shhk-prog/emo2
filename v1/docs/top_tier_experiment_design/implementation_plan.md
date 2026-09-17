# LLM情動反応性評価・追加実験設計書 (Top-Tier Conference向け)

## 1. 目的と背景
本計画は、「単一層・単一位置のActivation Patchingでは自己報告VAが回復しなかった」というネガティブ結果を起点とし、分散表現・非媒介・アライメント遮断などの競合仮説を識別するmechanistic interpretability研究への転換を目的とします。

「LLMが感情を経験するか」という検証不能な問いは避け、**「感情的文脈の内部表現が、自己報告や共感的応答といった出力行動へどのように媒介され、Base-to-Instruct post-trainingによってどの経路が保持・変容・遮断（post-training-dependent causal dissociation）されるか」**という計算論的・操作可能な対象を検証します。

### 1.1 仮説識別表
以下の予測パターンの差異から、内部表現の存在と機能的役割を切り分けます。アライメント遮断を「感情自体の不在」ではなく「因果経路の再配線」として捉える点が本研究の核心です。

| 仮説 | Probe $R^2$ | 介入効果 (Patching / Steering) | Base–Instruct差 | 判定のための追加条件 |
| :--- | :--- | :--- | :--- | :--- |
| **表現なし** | 低い | 低い | 小さい | ランダムprobeと有意差なし、または本研究で定義する刺激情動表現は線形probeで検出可能な水準では確認されない |
| **分散表現** | 広い層で高い | 単一部品で低く、複数部品で増加 | 条件依存 | 上位部品の集合介入で効果増大 |
| **表現はあるが非媒介**| 高い | 低い | 小さい／混在 | Ablationでも出力不変 |
| **機能的媒介** | 高い | 高い | 条件依存 | Patching・Ablation・Steeringが整合 |
| **アライメント遮断** | Base/Instructとも高い | Baseで高くInstructで低い | 大きい | Instructの**抑制候補ablation後に、感情文脈に整合的な出力方向への感度が部分的に回復** |

---

## 2. 実験設定とデータセットの役割分担

### 2.1 データセットの役割と優先順位
1. **AIPsy-Affect (主実験・最優先)**: 語彙交絡への最強の対処。感情語彙を排した192の最小対ヴィネット（感情 vs 中立）を提供。本研究では直接VAに写像せず、「感情的状況 vs 中立状況」のカテゴリカルな内部表現の差分とその因果効果の検証を主眼とします。
2. **EmoBank (主ベンチマーク)**: 連続VA評価とRussell円環空間の対応関係を検証する、本研究の連続空間理論の中心。
3. **crowd-enVENT (外部妥当性)**: Appraisal理論との対応検証。提供されるtrain/dev/test分割を利用し、Probeの訓練・ハイパーパラメータ探索と最終検証を分離します。
4. **EmpatheticDialogues (外部妥当性・最終段階)**: 自由応答への影響検証。独立LLM評価だけでなく、別ファミリー・盲検化・介入条件を隠蔽した人間評価を併用します。
5. **XANEW (補助的感度分析)**: 単語レベルでの信号駆動度合いを定量化。

### 2.2 モデルの選定と正規化
対応する Base / Instruct ペアで比較を行いますが、比較にあたっては以下の正規化を必須とします。
- **Primary**: Qwen2.5-7B (または 1.5B) Base / Instruct
- **Secondary (再現性)**: Llama-3.1-8B Base / Instruct
- **正規化**: Hidden stateを各モデル内でZ-score標準化してProbeを学習。Steering方向 $\alpha$ は残差ストリームの標準偏差に対する倍率として $\ell_2$ 正規化し、Recoveryはモデル内の元のSource-Target効果量で正規化します。

---

## 3. 分析手法と実装基盤（因果識別のパイプライン）

### 3.1 期待VA指標と距離ベースのRecovery（シーケンス確率への移行）
VAの81通り（$V \in [1,9], A \in [1,9]$）のJSON文字列全体 `{"valence": V, "arousal": A}` に対する条件付きシーケンス確率を使用します。

$$ p(v,a\mid x) = \frac{ \exp\left( \log P\left(\operatorname{format}(v,a)\mid x\right) \right) }{ \sum_{v'=1}^{9}\sum_{a'=1}^{9} \exp\left( \log P\left(\operatorname{format}(v',a')\mid x\right) \right) } $$

これに基づく期待VAベクトル $\mathbf{E}_x=(E[V\mid x], E[A\mid x])$ は以下となります：
$$ E[V\mid x] = \sum_{v=1}^{9}\sum_{a=1}^{9} v \cdot p(v,a\mid x) $$
$$ E[A\mid x] = \sum_{v=1}^{9}\sum_{a=1}^{9} a \cdot p(v,a\mid x) $$

主指標としてのRecoveryは、二次元空間でのユークリッド距離に基づく以下を採用し、Logit Difference (LD) は候補探索のための感度指標として扱います。
$$ \mathrm{Recovery}_{\mathrm{VA}} = 1- \frac{ \left\| \mathbf{E}_{\mathrm{patch}} - \mathbf{E}_{\mathrm{source}} \right\|_2 }{ \left\| \mathbf{E}_{\mathrm{target}} - \mathbf{E}_{\mathrm{source}} \right\|_2 } $$

### 3.2 表現から因果への段階的検証
1. **情報の存在**: 全層Probeによる予測（交絡排除済）。
2. **候補探索**: Attribution Patching (局所的線形近似での走査)。
3. **十分性**: Activation Patching (候補部位への情報移植)。
4. **必要性**: Ablation (候補部位の機能無効化)。
5. **用量反応**: Steering (連続的応答の単調性と対称性)。

---

## 4. 実行順序 (Phases)

### Phase 1: ベースラインの確立 (小さく厳密に)
- EmoBankから層化抽出（各セル32〜50件、計288〜450件）。
- 期待VA $\mathbf{E}_x$ とRecoveryを算出し、分母が閾値未満のペアを除外する規則を定めます。

### Phase 2: 主実験 - AIPsy-Affect (最小対) と 厳密なProbe
- AIPsy-Affectを用い「感情的状況 vs 中立状況」の差分を確保。
- **Probeの交絡排除**:
  - 最小対ペアID単位で train/dev/test を分割し、情報リークを防止。
  - 感情ラベルをシャッフルしたランダムベースラインとの比較。
  - 文字数・トークン数・話題ドメインを予測する非感情Probeとの層プロファイル比較。

### Phase 3 & 4: 因果検証 (Patching・Ablation・Steering)
- Attribution Patchingで候補選定後、通常Patching・Ablationを実施。
- **Steering**: 平均差分ベクトル $\mathbf{d}^{(l)}$ を用い、$\alpha \in \{-2, -1, -0.5, 0, 0.5, 1, 2\}$ で操作。
- $\alpha$ の正負対称性の評価、および「形式的一貫性・一般的な指示追従・安全関連の非標的指標」の低下がないかを確認。

### Phase 5: アライメント遮断（自己非感情化出力）の因果的証拠
「alignment suppression hypothesisを支持する因果的証拠」として、Instructモデルの出力抑制プロセスを以下のように解明します。
1. **自己非感情化出力に寄与する候補部品の特定**:
   - Instructモデルに感情刺激を与え、「I do not have feelings」や中立値JSON等の自己非感情化文字列の対数尤度を定義。
   - Attribution Patchingを用いて、その対数尤度を増大させているhead/MLPを順位付けし、抑制候補部品として特定する。
2. **Ablationによる回復の検証**:
   - 上位候補部品をAblateした際に、自己非感情化のlogitが低下し、かつ**感情文脈に整合的な出力方向への感度が部分的に回復**するかを検証。
   - ランダム部品や別のペルソナ部品とのAblation比較を行う。

---

## 6. Verification Plan (追加の統制条件)

以下の統制実験により、交絡を徹底的に排除します。
1. **JSONフォーマット対照**: 「JSON出力回路」の誤認を防ぐため、選択肢形式（Valence: [1]...[9] 等）や自然言語形式（low/high等）でも同様の表現が再現するか確認。
2. **刺激ペアの方向反転**: affective $\rightarrow$ neutral と neutral $\rightarrow$ affective でパッチングを実施し、効果の非対称性を確認。
3. **探索の独立化**: devセットでパラメータ（層・正則化係数・介入候補）を選び、testセットで最終検証を行う。
4. **位置の比較**: 最終トークンだけでなく、刺激トークン位置・全トークン平均の表現を比較する。
5. **話題バイアス (Topic Bias) 統制**: AIPsy-Affectの領域跨ぎ評価や、EmoBank/crowd-enVENTでの話題層化による統制。
6. **効果量と信頼区間**: 刺激ペアの再標本化によるbootstrap 95% CIと、介入位置・seed変更による感度分析。
