# 実装計画: paper.md の詳細化と体系的改訂

## 1. 改訂の基本方針
既存の `v3/docs/paper.md` の学術的トーン（厳密な測定科学、擬人観の排除）を完全に維持しつつ、v1、v2、v3の全容を統合し、先行研究との対比表・論述、およびAIガバナンス（EU AI Act等）への適合性補遺を正式に追加する。

## 2. セクションごとの改訂項目

### セクション1: 導入 (Introduction) & リサーチクエスチョン
- 背景として「認知的感情認識」と「機能的情動反応性」の解離を明記。
- 先行研究の限界（プロービングの暗黙の前提、Greedy評価の限界）を明確化。
- RQ1〜RQ4の明示（RQ4として下流行動・他者認識への波及効果・気分一致効果を追加）。

### セクション2: 課題と競合仮説 (Problem Formulation and Competing Accounts)
- H1 (Erasure), H2 (Transformation), H3 (Uniform Suppression), H4 (Distributed Remapping) の対照表。
- 「内部表現の幾何学的変化」と「内部表現から報告・行動への結合変化（Coupling change）」の概念的区別を明確化。

### セクション3: 実験設定 (Experimental Setup)
- **v1由来**: Sequence Likelihood Protocol の数理的導出（81候補列の対数尤度正規化、2D EMD Recovery）、EmoBank（外部妥当性）、スケーリング解析（Qwen2.5 0.5B〜7B, Llama-3.2）。
- **v2由来**: AIPsy-Affect Strict Subset（語彙交絡の排除、10トリプレット）、Cross-decoding、コンポーネントパッチング、8条件重みスワップ。
- **v3由来**: 厳密3分割プロトコル（Train/Alignment-dev/Test）、Dual-Outcome測定枠組み（$E[V]$ vs $B(x)$）、気分一致バイアス（Mood Congruency）因果検証パイプライン（認識プロンプト、曖昧刺激選定、ステアリング強度・対照条件）。

### セクション4: 結果 (Results)
- 4.1 行動的中立化とスケーリング則（Greedy collapse vs Sequence likelihoodでの逆転現象、モデル規模の影響）
- 4.2 内部表現の残存（AUC > 97.5%、H1棄却）
- 4.3 表現幾何の変換（Ridge Alignmentによる回復、H2支持）
- 4.4 内部感度と自己報告感度の逆転（Base vs Instructでの記述的乖離）
- 4.5 **Decodability $\neq$ Causal Substitutability**（事前指定L15パッチでの0%回復、探索スイープ、強制ステアリング）
- 4.6 混合効果モデルによる一様抑制（H3）の検証（有意な一様抑制なし）
- 4.7 局所化の否定と分散的結合変化（単一コンポーネントの不完全性、Late-Residual無効、出力層スワップでのResidual支配）
- 4.8 **自己報告の枠を超えた因果効果（Dual-Outcome & Mood Congruency）**:
  - 自己報告中立化の裏での下流共感対話選択 $B(x)$ への影響
  - 内部感情ステアリングによる他者感情認識のバイアス実証（曖昧刺激における特異的発現）

### セクション5: 関連研究および先行研究との対比 (Related Work & Novelty)
- プロービング研究、行動評価、アライメント機序、ステアリング研究との詳細対比表と論述。
- 「再現された点」と「従来の理解を覆した点」の体系的整理。

### セクション6: 考察・制限・結論 (Discussion, Limitations, Conclusion)
- 分散的再マッピング仮説の総合的証拠マップ（Evidence Map）。
- 方法論的・理論的インプリケーション。

### セクション7: Appendix（補遺）
- Appendix A: Sequence Likelihood 実装および指標定義
- Appendix B: データセット設計（AIPsy-Affect Strict Subset）
- Appendix C: プローブ・アライメント・介入詳細
- **Appendix D: AIガバナンスおよび法規制適合性（EU AI Act等への非該当性論証）**:
  - 科学研究開発免除（Article 2(6), 2(8)）
  - 禁止される感情認識システム（Article 5(1)(f)）への非該当性（自然人ではなくLLM自身が対象）
  - ハイリスクAI（Annex III）への非該当性
  - 汎用AI（GPAI: Articles 51-55）に関する位置づけ
  - 個人情報の非保持と倫理的指針
