# タスク: AIPsy-Affect 4-split（強度・複雑性・感情カテゴリ統制）評価システムの構築

- [x] **AIPsy-Affect 4-split データの準備とペアリング構造の抽出** <!-- id: 0 -->
  - [x] 4つのsplit（clinical, moderate, neutral, complex_neutral）のarrow/parquetデータの統合読み込み
  - [x] minimal pair / matched control の紐付け（clinical ↔ neutral, moderate ↔ neutral, complex_neutral ↔ neutral）
  - [x] 8感情カテゴリ（grief, terror, rage, loathing, ecstasy, admiration, amazement, vigilance）の属性抽出
  - [x] 統合評価用データセット `v1/data/processed/aipsy_4split_all.csv` の生成スクリプト作成
- [x] **推論スクリプト（Reader Prediction & Self-Report）の実装** <!-- id: 1 -->
  - [x] 729候補の尤度ベース $E[V], E[A], E[D]$ 評価
  - [x] 3タスク（Reader Prediction / Self-Report / Writer Estimation）の測定
  - [x] 結果の出力ディレクトリ `v1/results/aipsy_4split_eval/` への保存
- [x] **4大リサーチクエスチョン（RQ1〜RQ4）に対応する詳細集計スクリプトの実装** <!-- id: 2 -->
  - [x] RQ1 (Sensitivity): clinical vs. neutral（基本情動反応性）
  - [x] RQ2 (Dose-response): neutral → moderate → clinical（線形トレンド、単調性レート、段階的変位）
  - [x] RQ3 (Specificity): complex_neutral vs. clinical（文章複雑性の統制効果検証）
  - [x] RQ4 (Coupling): Recognition vs. Self-Report（他者予測と自己報告の結合度）
  - [x] 8感情カテゴリ別のVA変位プロファイル分析
  - [x] 全モデル横断（Base / Instruct × 4ファミリー）比較表の生成
- [x] **一括実行バッチスクリプト（`v1/scripts/run_all_aipsy_4split.sh`）の実装** <!-- id: 3 -->
- [x] **ユーザーへの実行コマンド案内および分析ドキュメント整備** <!-- id: 4 -->
- [x] **人間スケール（1〜5尺度、中立3.0）への標準化および集計スクリプト改修** <!-- id: 5 -->
  - [x] $E[Y]_{1..5} = (E[Y]_{1..9} + 1.0) / 2.0$ による線形標準化
  - [x] 感情相殺を防ぐ3大感情クラスタ（Negative, Positive, Alert）の導入
  - [x] Reader（他者認識）と Self（自己報告）の完全分離対比表（表1-A/B〜表5-A/B）生成
- [x] **AIPsy-Affect 4-Split 5大リサーチクエスチョン総合考察の執筆** <!-- id: 6 -->
- [x] **`v3/docs/- [x] paper4.md の背景・序論部分の統合と学術的定式化完了 <!-- id: 16 -->
- [ ] V1 (Self vs. Other) 6大実験の実装と検証 <!-- id: 22 -->
  - [ ] Phase A: E1 (Shared Decodability) & E2 (Reader-Self Geometry & Alignment) <!-- id: 23 -->
  - [ ] Phase B: E5 (Lexical Confound Audit & 4大Semantic Controls) <!-- id: 24 -->
  - [ ] Phase C: E3 (Causal Map: Magnitude + 2D Direction Vector) <!-- id: 25 -->
  - [ ] Phase C: E4 (Matched Difference Patching with α-sweep & 4 Controls) <!-- id: 26 -->
  - [ ] Phase C: E6 (Targeted Ablation & Mixed-effects Interaction Test) <!-- id: 27 -->
  - [x] 実験の位置づけ・学術的背景の明記
  - [x] 5大リサーチクエスチョン（RQ1〜RQ5）の定義
  - [x] 8モデル要因配置デザイン・データセット構造（480件）の整理
  - [x] プロンプト仕様（Base/Instruct）および連続対数尤度期待値の数理的定式化
- [x] **`v3/docs/paper4.md` EmoBank 実験概要セクションの体系的・学術的完成** <!-- id: 8 -->
  - [x] V1（妥当性検証）の位置づけと Buechel & Hahn (2017) の背景
  - [x] VAD 3次元の心理学的定義テーブルと Writer vs. Reader 視点の区別（具体例付き）
  - [x] 3タスク（Writer / Reader / Self）＋ 1内部結合度の対比表とアスキー概念フロー図
  - [x] 5つの主要RQ、評価対象8モデル選定理由、3大評価指標（Alignment, Calibration, Collapse）の定式化
- [x] **`v3/docs/paper4.md` 関連研究（Related Work）セクションの体系的・学術的完成** <!-- id: 9 -->
  - [x] 哲学的基盤と認知・情動の分離（Haladjian & Montemayor; Elyoseph; Ayers）
  - [x] 感情表現の幾何構造と制御（Di Palma; Reichman; Wu; Li ITI; Zhang TruthX）
  - [x] 内省能力と自己報告の解離・モード崩壊（Burns CCS; Turpin; Binder; Lindsey; Martorell）
  - [x] 内省への批判（Singh et al. COLM 2026）と本研究の3大貢献・学術的空白の明確化
- [x] **`v3/docs/paper4.md` 序論・研究背景（Introduction）セクションの体系的・学術的完成** <!-- id: 10 -->
  - [x] 心を支える対話型生成AIの普及と過剰依存リスク（社会的背景・人間中心のAI）
  - [x] Bornsteinの対人関係依存モデル（健全な依存 / 過剰な依存 / 分離）のAI関係性への拡張
  - [x] Ivey et al.のカウンセリング面接技法に基づく「感情への同調（情動的共感・迎合性）」の危険性と心理的ガードレール
  - [x] 根本的問い（LLMの情動的共感メカニズム）とV1（妥当性）→V2（特異性と事後学習）→V3（内部動態と因果利用）の全体ロードマップ
  - [x] 参考文献 [1]〜[20] の正式学術フォーマット化




