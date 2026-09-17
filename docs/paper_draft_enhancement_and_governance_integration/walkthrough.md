# paper.md 改訂・詳細化の確認 (Walkthrough)

**対象ファイル**: [`v3/docs/paper.md`](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper.md)  
**更新日**: 2026-09-06  

---

## 1. 改訂の概要

ユーザーからの要望に基づき、論文ドラフト（`v3/docs/paper.md`）を大幅に加筆・詳細化しました。
具体的には、以下の3つの主要な観点を統合・網羅しました：
1. **v1, v2, v3 の全実験内容および結果の統合**
2. **既存研究との詳細な対比（到達点・限界・本研究のブレークスルー、再現点と差分点）**
3. **Appendix D としての AIガバナンスおよびEU AI Act（EU人工知能法）非該当性・適合性の正式論証**

---

## 2. 主な改訂箇所の詳細

### ① タイトルと概要（Abstract）の拡充
- タイトルを `Post-Training-Associated Changes in the Coupling Between Affective Representations and Self-Reports in LLMs: Mechanisms, Dissociations, and Downstream Cognitive Biases` へ更新。
- 概要に、v1の尤度空間での逆転現象、v2の幾何変換とパッチング回復率0%の発見、v3の厳密3分割および気分一致バイアスの因果実証を網羅。

### ② 導入（Introduction）とリサーチクエスチョン（RQ）の拡張
- 背景において「他者の感情認識（認知的共感）」と「自己報告（情動反応性）」の解離を明示。
- RQ1（表現の残存と幾何）、RQ2（因果的結合とDecodability/Substitutability）、RQ3（局所化の検証）、に加え、**RQ4（下流行動および他者認識への波及・気分一致効果）** を正式に追加。

### ③ 実験設定（Experimental Setup）の詳細化
- **v1由来**: Sequence Likelihood Protocol の数理的導出（81候補列の対数尤度Softmax正規化）、Primary Endpointとしての Normalized 2D EMD Recovery の定義、スケーリング解析（Qwen2.5 0.5B〜7B, Llama-3.2）。
- **v2由来**: 語彙交絡を排除した臨床ヴィネット最小対（AIPsy-Affect Strict Subset: 10トリプレット/30サンプル）、Cross-decoding、コンポーネントパッチング、8条件重みスワップ。
- **v3由来**: 厳密3分割（Train 40% / Alignment-dev 30% / Test 30%）プロトコル、Dual-Outcome評価（自己報告 $E[V]$ と共感行動選択比 $B(x)$）、気分一致バイアス（Mood Congruency）因果検証パイプライン（認識プロンプト、曖昧刺激選定、ステアリング強度・対照条件）。

### ④ 実験結果（Results）の体系的加筆
- **4.1 Behavioral Neutralization とスケーリング特性**: Greedy collapse（98.6%中立）と尤度空間での逆転現象（Instruct $r=0.629 >$ Base $r=0.365$）、パラメータ規模に伴う中立化の確立。
- **4.2 内部表現の保持**: AUC > 97.5% による消去説（H1）の完全否定。
- **4.3 表現幾何の変換**: Ridge Alignment による回復（$R^2 \approx 0.58$）とH2の支持（表2）。
- **4.4 Decodability は Causal Substitutability を保証しない**: 事前指定Layer 15 MLPパッチでの回復率 0.0%（表4）、探索スイープ、強制ステアリングによる経路確認。
- **4.5 一様抑制仮説（H3）の検証**: 混合効果モデルの $\beta_{3,\ell}$ の正負混在とFDR補正後非有意（表5）。
- **4.6 局所化の否定と Distributed Remapping (H4) の同定**: 単一コンポーネントの不完全性、Late-Residual無効、出力層スワップでのResidual支配。
- **4.7 自己報告を超えた因果波及（Dual-Outcome と気分一致バイアス）**: 自己報告中立化の裏での下流共感対話選択 $B(x)$ のシフト、および中間層ステアリングによる他者客観認識の気分一致バイアス（$\beta_{\text{mood}} > 0, p < 0.01$）の実証。

### ⑤ 関連研究および先行研究との対比（Related Work & Novelty）
- **表6（先行研究と本研究の到達点対比表）** を新設：プロービング、行動評価、事後学習機序、ステアリングの4領域で先行研究の到達点・限界と本研究のブレークスルーを1対1で対比。
- 先行研究と「再現・一致した点」および「従来の理解を覆した新たな差分点」を箇条書きで明瞭に整理。

### ⑥ Appendix D の新設：AIガバナンスおよびEU AI Act非該当性論証
- **Article 2(6) & Article 2(8)**: 科学研究開発目的および市場投入前研究による全面的な適用除外（R&D Exemption）。
- **Article 5(1)(f)**: 禁止される感情認識システムへの非該当性（対象が生身の人間ではなくLLM自身の内部隠れ状態であり、職場・教育現場の監視ではない）。
- **Annex III**: ハイリスクAIシステムへの非該当性（社会的意思決定プロセスへの関与なし）。
- **Articles 51–55**: 汎用AI（GPAI）提供者義務の対象外（オープンモデルのダウンストリーム解析研究、安全性監査手法としての貢献）。
- **研究倫理原則**: 個人情報（PII）の非保持、擬人観の完全排除（`AGENTS.md` 準拠）。

---

## 3. 検証結果
- `/mnt/nas/home/hiromi/src/emo/v3/docs/paper.md` のファイル内容が完全に更新され、数式・Markdownテーブル・構造化セクションが正しく記述されていることを確認しました。
