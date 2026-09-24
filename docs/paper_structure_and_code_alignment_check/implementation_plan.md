# 論文構成・コード整合性・重複排除 実装計画書 (Implementation Plan)

## 1. 概要と目的
本ドキュメントは、`/mnt/nas/home/hiromi/src/emo2/iclr2027/iclr2027_conference2.tex` について、
1. **リポジトリ内のコード・実験結果（Behavioral, V1, V2, V3, configs, results）との厳密な整合性確認**
2. **ユーザー指定の19セクション順序への適合性確認と未記載セクション（結果・考察・接続・全体の考察）の再構成方針**
3. **セクション間・ステージ間の重複の洗い出しと排除方針**
を体系的に整理し、論文ファイルの完全な再構成に向けた計画を策定するものである。

---

## 2. ユーザー指定の19セクション構成と現状の対応状況

ユーザー指定の順序：
1. 背景
2. 関連研究
3. 実験全体の概要
4. Behavioral概要
5. Behavioral結果
6. Behavioral考察
7. Behavioralからv1への接続
8. v1概要
9. v1結果
10. v1考察
11. v1からv2への接続
12. v2概要
13. v2結果
14. v2考察
15. v2からv3への接続
16. v3概要
17. v3結果
18. v3考察
19. 全体の考察

### 現状の適合状況分析
| # | 指定セクション名 | 現状のファイル内ステータス | 現状の行番号等 | 課題と対応方針 |
|---|---|---|---|---|
| 1 | 背景 | 執筆済み | L475--575 | コード・設計と整合。微修正のみ。 |
| 2 | 関連研究 | 執筆済み | L576--724 | 最新の先行研究を網羅。コード・設計と整合。 |
| 3 | 実験全体の概要 | 執筆済み | L725--1930 | モデル、候補空間（729 VAD / 81 VA）、Sequence-Likelihood、抽出位置など極めて詳細。各ステージ概要との重複を整理する。 |
| 4 | Behavioral概要 | 執筆済み | L1931--2954 | プロトコル・RQ1〜4・解析手法が詳細。末尾の「Behavioral Stageの境界」は考察・接続へ移管。 |
| 5 | Behavioral結果 | **未執筆 (TODO)** | L2958 (コメント) | `results/derived/paper_summary/tables/table_b*.csv` の確定済み数値を基に執筆。Table B1〜B5, Table B Summary。 |
| 6 | Behavioral考察 | **未執筆 (TODO)** | L2964 (コメント) | 出力共変動（$\Delta R$ vs $\Delta S$ の有意な結合）の意味と、内部表現・因果の同一性は主張できない境界を整理。 |
| 7 | Behavioralからv1への接続 | **未執筆 (TODO)** | L2970 (コメント) | 行動連動の背後にある「内部表現の共有度と因果経路の同一性」を検証する動機付けを明示。 |
| 8 | v1概要 | 執筆済み | L2977--4487 | Phase A/B/C、E1〜E6の証拠階層が詳細。L2972の ```` ```latex ```` 記法バグを削除。 |
| 9 | v1結果 | **未執筆 (TODO)** | L4490 (コメント) | `results/derived/paper_summary/tables/table_v1_*.csv` を基に執筆。Table V1-1〜V1-6。 |
| 10 | v1考察 | **未執筆 (TODO)** | L4496 (コメント) | 見出し `\subsection{V1から直接主張できる範囲}` (L4498) に本文を執筆。同一モデル内の表現・因果共有と部分特殊化の解釈。 |
| 11 | v1からv2への接続 | **未執筆 (TODO)** | L4503 (コメント) | 同一モデル内の共有から、事後学習（Post-training: Base vs Instruct）による幾何・因果配置の再編検証への接続。 |
| 12 | v2概要 | 執筆済み | L4510--6124 | H1a, H1b, H2, H3, H4のプロトコル・LMM・OT EMDが詳細。 |
| 13 | v2結果 | **未執筆 (TODO)** | L6127 (コメント) | `results/derived/paper_summary/tables/table_v2_*.csv` を基に執筆。Table V2-1〜V2-4, Table V2 Confirmatory。 |
| 14 | v2考察 | **未執筆 (TODO)** | L6133 (コメント) | 見出し `\subsection{V2から直接主張できる範囲}` (L6135) に本文を執筆。表現の消去ではなく深層への移行（Relocation）と回復可能性。 |
| 15 | v2からv3への接続 | **未執筆 (TODO)** | L6140 (コメント) | 静的な表現再編から、Instructモデル生成過程のどの時空間で自己報告へ因果的利用が起きるか（時空間因果利用）への接続。 |
| 16 | v3概要 | 執筆済み | L6147--8058 | Gate、Discovery 4-Maps、Mediated attenuation、Cross-model replicationが詳細。 |
| 17 | v3結果 | **未執筆 (TODO)** | L8062 (コメント) | `results/derived/paper_summary/tables/table_v3_*.csv` を基に執筆。Gate（NO_GO/停止）、Discovery 4-Maps、Confirmatory。 |
| 18 | v3考察 | **未執筆 (TODO)** | L8068 (コメント) | Gate NO_GO の厳格な科学的意味（State Inductionの閾値未達）、Discovery 4-Mapの局所的因果構造の解釈。 |
| 19 | 全体の考察 | **未執筆 (TODO)** | L8074 (コメント) | 4つの証拠レベル（Covariation $\to$ Sharing $\to$ Reorganization $\to$ Causal Utilization）の統合的考察、限界、倫理・人間AI相互作用への含意。 |

---

## 3. コードおよび実験成果物との整合性確認結果

### 3.1 モデルコホート（完全一致）
- `configs/models.yaml` の正本定義：
  - Primary (1--1.5B): Qwen 2.5 1.5B, Llama 3.2 1.23B, Gemma 3 1B, OLMo 2 1B (Base & Instruct, 計8モデル)
  - V3: Instruct 4モデル（Qwen 1.5B-Instruct が Discovery、Llama, Gemma, OLMo が Confirmatory）
  - Supplementary: Mistral 7B v0.3, 3B/7B scale ablation
- 論文中の Table 1 (`\label{tab:primary_models}`) および本文記述は上記と完全に合致している。

### 3.2 測定空間およびプロトコル規約（完全一致）
- Behavioral & V1: 729-state VAD candidate space (compact JSON)
- V2 & V3: 81-state VA candidate space (compact JSON)
- 「729 VAD と 81 VA の絶対値を直接比較しない」規約が論文本文（L1662--1678等）に明記され、コード側の仕様と完全一致。

### 3.3 内部表現および因果介入プロトコル（完全一致）
- 抽出位置: prompt-end residual stream
- 相対深度: $l / (L - 1)$
- V1 Phase A/B/C/E6: Ridge regression, Procrustes, RSA, Nonfallback N保持, Causal interchangeability, Task $\times$ SiteType interaction
- V2 H1a/H1b/H2/H3/H4: Procrustes/RSA, Peak shift, Sharing shift, LMM causal relocation, 2D OT EMD distribution recovery
- V3 Gate/RQ2/RQ3/RQ4: State Induction Gate (0.05/0.10 raw threshold), 4-Maps (`pre_V` as primary), Mediated attenuation, Cross-model replication
- すべてコード（`v1/`, `v2/`, `v3/`）の実装と完全一致。

### 3.4 確定済み結果データ（Result Artifacts）の存在確認
- `results/derived/paper_summary/tables/` に全25個のCSVテーブルが存在。
- `results/derived/paper_summary/results_summary.md` に結果章の数値サマリーが確定済み。
- これらを直接引用して結果章を執筆可能。

---

## 4. 重複の洗い出しと排除方針

### 重複1: 「実験全体の概要」と「各ステージ概要」における共通基盤の重複
- **現状**:
  - `実験全体の概要`（§3）で、モデル一覧、プロンプト例、Sequence-Likelihood計算式、729 VAD / 81 VA 候補空間、relative depth、内部表現抽出位置が網羅されている。
  - しかし、`Behavioral概要`、`V1概要`、`V2概要`、`V3概要` のそれぞれで、全く同一の数式（$E[V] = \sum \dots$）、プロンプトの全文、モデルの表、729/81 の候補空間の定義が再三再四繰り返されている。
- **排除方針**:
  - §3「実験全体の概要」を全ステージ共通の「土台（Foundations）」として位置づけ、共通の定義・数式・モデル・候補空間は§3に一元化する。
  - 各ステージ概要（§4 Behavioral概要, §8 v1概要, §12 v2概要, §16 v3概要）では、「前節§3で定義した共通プロトコルに基づき、本ステージでは…」として参照し、各ステージ固有の仮説、特有の実験条件（介入プロトコル、統制条件、固有の統計検定モデル）のみを記述する。

### 重複2: 「考察」「接続」「次ステージ概要冒頭」における論理の三重重複
- **現状**:
  - 各ステージの境界・限界に関する議論が、概要末尾（例: L2914「Behavioral Stageの境界」）、考察、接続、そして次ステージ概要冒頭（例: L2980「目的とBehavioral Stageからの位置づけ」）で同じ論理（「出力相関だけでは内部表現の同一性は言えない」「同一モデル内の共有だけでは事後学習の影響は言えない」「静的な表現変化だけでは生成時の因果利用は言えない」）が繰り返し語られている。
- **排除方針**:
  - **「概要」**: 「何をどう測定するか」（実験プロトコル、解析指標）に特化。前ステージの議論を長々と繰り返さない。
  - **「結果」**: 「客観的測定結果」（数値、有意性、表、図の事実）に特化。解釈を混ぜない。
  - **「考察」**: 「その結果が示すこと、およびそのステージ固有の境界・限界（何を主張でき、何を主張できないか）」を論じる。
  - **「接続」**: 「その限界から必然的に導かれる次の研究課題・問いの引き渡し（Bridge）」を2〜3段落で明確に提示する。
  - **「次ステージ概要」**: 冒頭の重複した振り返りを最小限にし、直ちに固有の実験設計の説明に入る。

### 重複3: ICLR テンプレートのボイラープレート（L70--470）
- **現状**:
  - 提出手順、LaTeXマージン、サンプル著者、サンプル表・数式など 400 行以上のテンプレート本文が先頭に残存。
- **排除方針**:
  - 論文本文（背景以降）の整合性を保ちつつ、テンプレートの指示文を削除または整理する。

### 重複4: Markdown コードブロック記法の混入（L2972）
- **現状**:
  - L2972 に ```` ```latex ```` が記述されており、LaTeX のコンパイルエラーになる。
- **排除方針**:
  - 直ちに削除する。

---

## 5. 次のステップ
1. ユーザーへ、確認結果（コード整合性、構成順序適合状況、重複箇所）を詳細報告し、本実装計画の合意を得る。
2. 承認後、`iclr2027/iclr2027_conference2.tex` を指定の19セクション構成に従って再編成し、結果・考察・接続・全体の考察を確定済みデータに基づいて記載する。
3. 変更結果の確認（Walkthrough）を作成し、学術的・形式的不変性を検証する。

---

## 6. 重複修正（Deduplication）詳細作業計画【ユーザー指示により着手】

ユーザーからの「重複のみを修正して」という明確な指示に基づき、本文・数値の勝手な捏造や無関係な改変を行わず、以下の「重複A」および「重複B」の具体的箇所を対象として重複の排除・簡潔化・参照一元化を実行する。

### 対象1: Behavioral概要における重複排除
1. **プロンプト全文・テンプレートの重複排除 (Line 2029--2119)**:
   - §3（Line 1305--1390）に記載済みの Reader, Self, Writer の英文 instruction、共通 JSON instruction、Chat template、Base plain prompt の全文再掲を削除。
   - §3への明確な参照記法（`§\ref{app:prompts}`）へ置き換え、Behavioral Stage固有の条件指定のみを記述。
2. **数式の重複排除 (Line 1974--2008)**:
   - Sequence-Likelihood による $E[V], E[A], E[D]$ 算出数式（§3 Line 1504--1587 と重複）の冗長な展開を整理し、共通算出式を参照。
3. **境界・考察記述の整理 (Line 2914--2954)**:
   - 概要末尾の「Behavioral Stageの境界」は、次セクションの考察・接続と論理重複するため、概要としての分析プロトコル記述に絞る。
4. **Markdown記法の削除 (Line 2972)**:
   - 誤って混入した ```` ```latex ```` を完全削除。

### 対象2: V1概要における重複排除
1. **目的の振り返り重複の最小化 (Line 2980--3004)**:
   - Behavioral Stage の境界論理の重複再述を省き、直ちに V1 固有の目的（同一モデル内の内部表現幾何・因果共有と特殊化）へ直結。
2. **共通測定の重複排除 (Line 3097--3144)**:
   - Reader/Self のプロンプト英文、相対深度数式 $d_l = \frac{l}{L-1}$、729 VAD 候補空間の再掲を省き、§3 の定義を参照。
3. **空見出しの整理 (Line 4498--4505)**:
   - 本文のない `\subsection{V1から直接主張できる範囲}` を整理。

### 対象3: V2概要における重複排除
1. **Prompt-format control の重複排除 (Line 4656--4700)**:
   - §3（Line 1458--1503）に定義済みの Base Plain, Matched-Plain, Native-Chat のフォーマット説明を整理し、V2 固有の比較役割（Primary: matched-plain、Secondary: native-chat）に集約。
2. **目的の重複最小化 (Line 4513--4536)**:
   - 前ステージの振り返りを簡潔にし、V2 固有の Base--Instruct 幾何・共有・因果再編（H1--H4）へ直結。
3. **空見出しの整理 (Line 6135--6142)**:
   - 本文のない `\subsection{V2から直接主張できる範囲}` を整理。

### 対象4: V3概要における重複排除
1. **測定空間・相対深度の再掲排除 (Line 6150--6350)**:
   - 81-state VA candidate space や相対深度正規化の再定義を整理し、§3 の定義を参照。
   - V3 固有の State Induction Gate や時空間 4-Maps、局所方向推定 ($d_{l,t}$) に集中。
2. **目的の重複最小化 (Line 6150--6165)**:
   - 前ステージの振り返りを最小化し、時空間因果利用の核心へ直結。

---

## 7. 論文結果挿入用テーブルまとめ生成コード群の体系的実装計画

ユーザーの要望：
- `v1/scripts/legacy/summarize_*.py` のように、論文の結果章にそのまま挿入できる結果まとめを作成するコードをそれぞれ作成。
- 段階ごとに別の結果ファイルを作成（無理に1つの表にまとめず、分かりやすく分割）：
  1. **`scripts/summarize_behavioral_emobank.py`**:
     - EmoBank における Writer, Reader, Self の VAD 対応をファミリーごとに明示。
     - Base vs Instruct の直接対比表。
     - 内部認知結合度 $\text{Corr}(R, S)$ の表。
  2. **`scripts/summarize_behavioral_aipsy.py`**:
     - AIPsy-Affect における RQ1 (Sensitivity: $d_z, \Delta$), RQ2 (Dose-Response: slope), RQ3 (Specificity), RQ4 (Reader-Self Coupling: $r_\Delta$)。
  3. **`scripts/summarize_v1_internal_sharing.py`**:
     - E1〜E6 の各分析に対応する独立テーブル：
       - E1: Decodability（ファミリーごとの Reader vs Self ピーク深度・相関 $r$・決定係数 $R^2$）
       - E2: Geometry（Procrustes disparity / RSA 類似度）
       - E3: Causal Map（Reader vs Self 因果プロファイルの Spearman 順位相関、ピーク層、方向コサイン）
       - E4: Interchangeability（Reader $\to$ Self 転移率・特異性）
       - E6: Specialization（Task $\times$ SiteType 交互作用）
  4. **`scripts/summarize_v2_reorganization.py`**:
     - ファミリーごとの Base と Instruct の対応：
       - H1a: 幾何学的歪み（Procrustes 歪み、RSA、Matched-Plain vs Native）
       - H1b: デコードピーク深度のシフト（$\Delta d^*$）
       - H2: 表現共有度の変化（$\Delta\text{Sharing}$）
       - H3: 因果再配置・LMM（深層移行、Canonical metric $C_{net,rand}$）
       - H4: 分布回復率（Base 活性化注入による 2D OT EMD 回復）
  5. **`scripts/summarize_v3_causal_utilization.py`**:
     - Gate 判定表（State Induction Gate: NO_GO 閾値未達）
     - RQ2: 時空間 4-Maps（Discovery Qwen の各時空間ステージでの因果影響）
     - RQ3/RQ4: 独立検証・モデル間再現性表
  6. **`scripts/generate_paper_results_tables.py`**:
     - 全ステージの一括オーケストレーター。
- 出力形式：
  - `iclr2027/tables/` に `.tex` ファイル（`booktabs`, `caption`, `label` 付き完全な table 環境）を出力し、論文本文から `\input{tables/...}` または直接貼付け可能な構造とする。
  - ターミナルおよびプレビュー用の Markdown レポート（`.md`）も同時出力。

