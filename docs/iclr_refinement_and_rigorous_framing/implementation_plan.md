# 実装計画: ICLR 2027 基準に基づく paper.md の全面改訂

## 1. 改訂の基本理念
本研究のコアメッセージを以下の統一命題に据える：
> **"Post-training does not necessarily eliminate information that is linearly predictive of affective context. Instead, it can alter how such information is converted into a constrained first-person reporting distribution; linear cross-model recoverability alone is insufficient evidence of causal interchangeability."**

過度な誇張（「世界初」「完全再現」「確定」）や擬人化（「モデル自身の感情」「気分一致」）を徹底的に排除し、厳密な操作的定義と因果的制約に基づき論理を再構築する。

---

## 2. セクションごとの改訂設計

### タイトル案
* **採用タイトル**:
  `Post-Training Alters the Coupling Between Affect-Relevant Representations and First-Person Reports in Language Models`

### Abstract（5文構造）
1. **問い**: Post-trainingに伴う一人称の制約付き自己報告中立化は、情報の消去か、readout結合の変化か。
2. **方法**: Base/Instructペアモデル、制約付きsequence-likelihood protocol、厳密グループ分割アライメント、活性化パッチング／ステアリング。
3. **発見1 (N1)**: Greedy崩壊にもかかわらず、尤度由来の期待値は感情ラベルと共変動し、Baseより感度が高い場合がある。
4. **発見2 (N2)**: 線形アライメントがcross-model予測を回復しても、単一部位のaligned patchingは一人称自己報告分布を回復しない（Decodability restored $\neq$ Causal substitutability restored）。
5. **発見3+含意 (N3, N4)**: 局所的出力層のみの機序より分散的readout変化と整合し、valence方向ステアリングは曖昧な三人称感情認識を系統的に変位させる（valence-congruent recognition shift）。主観的感情を意味しない。

### Section 1: Introduction
- 構成概念の厳密な定義（Reader-rated text affect / Character attribution / First-person report likelihood）。
- 科学的健全性チェック（Consistency checks: S1〜S5）と新規知見（Empirical findings: N1〜N4）の明確な関係付け。
- 4つの明確なリサーチクエスチョン（RQ1〜RQ4）と限定的コントリビューション。

### Section 2: Problem Formulation & Competing Accounts
- 4つの競合仮説（H1〜H4）の対照表（表1）。
- N3の慎重な位置づけ：H1および単純な局所説明を排除し、分散的変化と整合的であることを示すが、因果経路の全集合を一意同定したとは主張しない。

### Section 3: Experimental Setup
- **データ**: AIPsy-Affect Strict Subset（n=10 pair, 30 sample）を統制されたケーススタディとして明確に位置づけ、作成規則と対照統制を明記。
- **3分割**: Train / Alignment-dev / Test の完全非重複。
- **因果介入の要因計画**: Source (Base-peak, Base-neutral) × Target (Instruct-peak, Instruct-neutral) の対照構造。
- **Sequence Likelihood Protocol の数理**: テンプレート依存性、tokenization、長さ正規化の根拠、$\tau=1.0$ の妥当性。
- **Primary Endpoint**: Normalized 2D EMD Recovery の定義と除外基準（$\epsilon=0.01$）。

### Section 4: Results
- **4.1 Behavioral Neutralization & Likelihood Sensitivity (N1)**: EmoBank reader valenceとの共変動、従属相関差の検定、スケーリング解析。
- **4.2 Retained Linear Information (H1 rejection)**: Within-model probeの性能と情報保持。
- **4.3 Geometry Transformation (H2)**: Procrustes vs Ridge Alignment の限界と意義。
- **4.4 Decodability Does Not Guarantee Causal Substitutability (N2)**: Pre-specified Layer 15 MLPパッチでの0.0%回復（実数値・CI）、Within-model positive control、診断的解釈。
- **4.5 Mixed-Effects Test of Uniform Suppression (H3)**: 階層モデルの完全な数式定義、FDR補正、独立標本単位の明示。
- **4.6 Evidence Consistent with Distributed Coupling Changes (N3)**: 単一部位・Late-residual・出力層スワップの限界。
- **4.7 Valence-Congruent Recognition Shift (N4)**: $\Delta V_{\mathrm{rec}} \sim \alpha \times \mathrm{ambiguity} \times \mathrm{direction\ type}$ の交互作用モデル、曖昧刺激における特異性。

### Section 5: Related Work & Conceptual Positioning
- 表6の刷新：先行知見との整合性（Consistency Checks）と本研究の差分（Novel Empirical Findings）の正確な対応。
- 引用の厳密化（不適切な過剰帰属の修正）。

### Section 6: Discussion, Evidence Map & Limitations
- Evidence Map の論理構造化（何が棄却され、何が整合的証拠に留まるか）。
- 限界：小標本（n=10 pair）、OOD活性化、単一モデルファミリーの限界。

### Section 7: Conclusion

### Appendix & Ethics Statement
- **Appendix D の置換**: EU AI Act の法的断定を削除し、国際査読標準の **Ethics and Broader Impact Statement** および **Reproducibility Statement** に置き換え。
- **匿名化の徹底**: 著者同定情報、絶対パス、Gitハッシュの排除。
