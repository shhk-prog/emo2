# Walkthrough: 既存研究（すでにあるところ）と本研究の新規性（ないところ）の完全対比

本ドキュメントは、LLMの感情表現・感情認識・自己報告・事後学習（Post-training）・回路解析に関する先行研究の到達点（すでにあるところ）と、本研究（`v3/docs/paper.md`）が初めて明らかにした新規な貢献（ないところ）を体系的に整理したものである。

---

## 1. エグゼクティブ・サマリー（一目でわかる対比表）

| 領域・テーマ | 既存研究で「すでにあるところ」 (Established) | 既存研究に「ないところ」＝本研究の新規性 (Gap & Novelty) |
|---|---|---|
| **内部表現 (Representation)** | ・内部状態から感情（Valence, Arousal, 離散感情）を高精度に線形プローブで抽出できる。<br>*(e.g., Zou et al. 2023, Tigges et al. 2023)* | ・事後学習（Post-training）を経ても感情情報が**消去（Erasure）されず幾何変換（Transformation）を受けて残存**していることの厳密証明。 |
| **行動的現象 (Behavior)** | ・Instructモデルに「あなたの感情は？」と聞くと「AIなので感情はない」や中立（5, 5）へ収束する（Greedy collapse）現象の観察。<br>*(e.g., EmoBank自己報告研究)* | ・Greedy生成では中立に完全に潰れていても、**全候補列の尤度空間（Sequence Likelihood Protocol）では人間感情との相関が極めて強く残存している**ことの実証。 |
| **因果介入 (Causal Intervention)** | ・感情ベクトルを強制加算（Activation Steering）すると、生成文のトーンや共感性が変わる。<br>*(e.g., Representation Engineering, Steering Vectors)* | ・**「プローブで情報が取れること（Decodability）」と「自己報告に因果的影響を与えること（Causal Substitutability）」の乖離**の実験的実証（プローブが回復しても自己報告は戻らない）。 |
| **事後学習の機序 (Mechanisms of Post-training)** | ・BaseモデルとChat/Instructモデルの間で表現空間の幾何がシフトすることの記述的報告。<br>*(e.g., Procrustes alignment)* | ・自己報告の中立化が「情報消去」「幾何変換」「一様抑制」「分散的結合変化」のいずれであるかを検証し、**層依存的・分散的な結合の組み替え（Distributed Remapping）** と最も整合することを特定。 |
| **回路の局所化 (Circuit Localization)** | ・特定ヘッドや単一MLPが特定機能（事実想起など）を担うという回路特定。<br>*(e.g., ROME, Meng et al. 2022)* | ・感情自己報告の中立化は**単一コンポーネント（L15 MLP等）や最終出力層（lm_head / RMSNorm）の単独置換では説明できず**、多層に分散していることの実証。 |

---

## 2. 既存研究で「すでにあるところ」（先行研究の到達点）

### ① 内部表現の抽出（Representation Extraction / Probing）
- **内容**: トランスフォーマーの残差ストリームやMLP層に線形プローブ（Ridge回帰、ロジスティック回帰）を当てると、テキストのValence（快・不快）、Arousal（興奮度）、あるいは特定感情（怒り、喜び、悲しみ等）が極めて高精度にデコードできる。
- **代表例**: Representation Engineering (RepE, Zou et al., 2023), ITI (Li et al., 2023), Tigges et al. (2023)。

### ② 他者の感情認識タスクの高性能（Affective Recognition）
- **内容**: 「この文章に書かれている感情は何か？」と客観的な分類・スコアリングを求めると、最新のLLMは人間評価（EmoBank, GoEmotionsなど）と高い一致率を示す（認知的共感能力）。

### ③ 活性化操作による振る舞いの改変（Activation Steering）
- **内容**: 感情方向の活性化ベクトルを抽出して推論時に足し込む（Activation Addition / Steering）と、モデルの出力テキストが攻撃的になったり、過剰に親切・共感的になったりする。

### ④ Instructモデルの自己報告における中立化（Greedy Neutral Collapse）
- **内容**: Instruct/Chatモデルに対して「あなた自身の感情はどうですか？」と一人称で尋ねると、安全対策やAIアシスタントの規範（Sycophancy / Harmlessness）により、ほとんどのプロンプトで「中立」または「感情はありません」と答えることが経験的に知られていた。

---

## 3. 既存研究に「ないところ」（本研究が初めて明らかにしたこと）

### ① 「プローブで取れること」と「自己報告に使えること」の決定的な乖離
- **既存の前提**: 多くの解釈可能性研究は「プローブで精度良く取れた表現（Decodability）＝モデルがその情報を使って判断している（Causal use）」と暗黙に仮定しがちであった。
- **本研究の新規性**:
  - Baseモデルの感情表現をInstructモデルの幾何に線形写像（Ridge Alignment）して移送（Aligned Cross-Model Patching）すると、**表現上は $R^2 \approx 0.58$ で復元されているにもかかわらず、自己報告（$E[V]$）は1ミリもBase型に戻らなかった（回復率0%）**。
  - すなわち、**「Decodability restored $\neq$ Causal substitutability restored」** を厳密に実証した。

### ② Greedy生成の裏に潜む「尤度空間（Sequence Likelihood）」の情動感度
- **既存の盲点**: 先行研究は貪欲法（Greedy decoding）で生成された単一のテキスト（`{"valence": 5, "arousal": 5}`）だけを見て「Instructモデルは感情反応性を完全に失った」と結論づけていた。
- **本研究の新規性**:
  - 全候補列（81通り）の対数尤度を測定する **Sequence-Likelihood Protocol** を開発。
  - Greedyでは98.6%が中立に潰れているInstructモデルでも、**尤度空間全体では人間感情との相関が $r=0.629$ と極めて高く残存している（Baseの $r=0.365$ よりも高い）** ことを発見した。

### ③ 事後学習（Post-training）による感情中立化のメカニズム解明
- **既存の空白**: なぜ事後学習で中立化するのか、その内部機序（情報が消えたのか、出口でブロックされているのか、回路が壊されたのか）は未解明だった。
- **本研究の新規性**:
  - 4つの競合仮説（H1: 消去、H2: 幾何変換、H3: 全層一様抑制、H4: 分散的結合変化）を対照実験でテスト。
  - **H1（消去）を明確に棄却**（Instruct内部からAUC>0.97で復元可能）。
  - **H3（一様抑制）を支持しない**（混合効果モデルで層ごとの交互作用項に一様な負のシフトが見られない）。
  - 単一コンポーネントパッチングや最終出力層（lm_head / RMSNorm）のスワップ実験により、中立化は特定の単一箇所の抑制ではなく、**層依存的かつ分散的な representation-to-report coupling の変化（Distributed Remapping）** であると結論づけた。

---

## 4. 論文執筆・主張における推奨フレーズ

論文の Introduction や Related Work で新規性をアピールする際は、以下のように整理すると査読者に極めてクリアに伝わります：

> **"While prior work has established that affect-relevant representations exist in LLM hidden states (probing) and can artificially steer free-form generation (activation steering), it remains unknown how post-training alters the *natural causal coupling* between these representations and constrained self-reports.  
> We show that post-training does not erase affective representations nor simply filter them at the final readout layer; rather, it induces a distributed, layer-dependent remapping between internal representations and self-report distributions."**
