# Walkthrough: AIPsy-Affect 4-Split（強度・複雑性・感情カテゴリ統制）評価システムの実装

## 概要
AIPsy-Affect の持つ4つのsplit（`clinical`, `moderate`, `neutral`, `complex_neutral`）をフルに活用し、従来の単純な「感情あり vs なし」の二値比較を超えて、以下の **3大要因（感情強度・文章複雑性・感情カテゴリ）を完全に分離・統制して評価できる体系的な実験・集計パイプライン** を設計・実装しました。

---

## 1. 4大リサーチクエスチョン (RQ) と分析構造

| リサーチクエスチョン | 比較条件 | 測定している学術的問い |
|:---|:---|:---|
| **RQ1: Sensitivity（感情感受性）** | `clinical vs. neutral` (N=192 pairs) | 強い感情刺激に対して、モデルの認識および自己報告が有意に変位するか（$\Delta V, \Delta A$, Cohen's $d$） |
| **RQ2: Dose-Response（用量反応性）** | `neutral → moderate → clinical` (N=48 triplets) | 刺激の感情強度（None → Mod → Peak）に応じて、反応量が段階的・単調に変化しているか（単調性成立率 %, Linear Slope） |
| **RQ3: Specificity（複雑性統制）** | `complex_neutral vs. neutral & clinical` (N=48 controls) | 反応変位が単なる文長や構文複雑性によるものではなく、純粋な感情操作に特異的なものか（感情特異性比率 ASR） |
| **RQ4: Recognition–Self Coupling** | $\Delta \text{VA}_{\text{Reader}} \leftrightarrow \Delta \text{VA}_{\text{Self}}$ | 人間の感情反応を的確に予測・認識した変化量が、モデル自身の自己報告にどれだけ連動・反映されるか（変化量相関, 振幅比率） |

さらに、
- **8感情カテゴリ特異性**: `grief`, `terror`, `rage`, `loathing`, `ecstasy`, `admiration`, `amazement`, `vigilance` ごとの V/A 変位プロファイル
- **モデル横断（Base vs. Instruct × 4ファミリー）**: Qwen 2.5 (1.5B), Llama 3.2 (1B), Mistral (7B), Gemma 2 (2B) の比較

---

## 2. 実装したスクリプト一覧

1. **データ準備スクリプト**: [`v1/scripts/prepare_aipsy_4splits.py`](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/prepare_aipsy_4splits.py)
   - ローカル arrow キャッシュから 4 split（全480件）を抽出・統合。
   - 192組の Clinical ↔ Neutral ペア（`pair_id`）、および 48組の Neutral → Moderate → Clinical トリプレット（`triplet_id`）を自動マッピング。
   - 出力: `v1/data/processed/aipsy_4split_all.csv` (検証済み: 48組のトリプレット完全一致)
2. **推論実行スクリプト**: [`v1/scripts/run_aipsy_4split_evaluation.py`](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/run_aipsy_4split_evaluation.py)
   - 729候補のバッチ対数尤度による期待値 $E[V], E[A], E[D]$ および最尤 Greedy 出力を算出。
   - Reader-Response Prediction ($R$)、Self-Report ($S$)、Writer-State Estimation ($W$) を独立測定。
   - 出力: `v1/results/aipsy_4split_eval/{model_tag}_aipsy_4split.csv`
3. **詳細集計・分析スクリプト**: [`v1/scripts/summarize_aipsy_4split.py`](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/summarize_aipsy_4split.py)
   - RQ1 (Sensitivity), RQ2 (Dose-Response), RQ3 (Specificity), RQ4 (Coupling) の4大表を出力。
   - 8感情カテゴリ別の詳細変位プロファイル表を出力。
   - 全モデル横断の比較サマリー表および CSV を自動生成。
4. **一括バッチスクリプト**: [`v1/scripts/run_all_aipsy_4split.sh`](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/run_all_aipsy_4split.sh)
   - 全8モデルの一括実行および自動レポート生成。

---

## 3. 手元ターミナルでの実行方法

手元環境（`h200-01` ターミナル、仮想環境 `.venv`）で以下のコマンドを実行できます。

### (A) 個別モデルの実行（例: Qwen 2.5 1.5B Instruct）
```bash
source .venv/bin/activate

python v1/scripts/run_aipsy_4split_evaluation.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --tag qwen2.5_1.5b_instruct \
  --is-instruct \
  --stimuli-path v1/data/processed/aipsy_4split_all.csv \
  --out-dir v1/results/aipsy_4split_eval
```

### (B) 全8モデルの一括実行
```bash
source .venv/bin/activate
bash v1/scripts/run_all_aipsy_4split.sh
```

### (C) 集計レポートの生成（推論結果のCSVがある場合、一瞬で完了します）
```bash
source .venv/bin/activate
python v1/scripts/summarize_aipsy_4split.py --results-dir v1/results/aipsy_4split_eval
```
- 生成されるレポート: `v1/results/aipsy_4split_eval/aipsy_4split_detailed_report.md`
- 生成される集計CSV: `v1/results/aipsy_4split_eval/aipsy_4split_summary.csv`

---

## 4. 今回の改修と `v3/docs/paper4.md` への実験概要の反映

### (1) 人間スケール（1〜5尺度、中立3.0）への標準化
- EmoBankとの整合性を図るため、生期待値（1〜9尺度）を $E[Y]_{1..5} = \frac{E[Y]_{1..9} + 1.0}{2.0}$ により 1〜5 尺度に標準化。
- 快（$V+$）と不快（$V-$）の相殺を防ぐ **3大感情クラスタ（Negative 4感情、Positive 2感情、Alert 2感情）** を導入。

### (2) `v3/docs/paper4.md` 実験概要セクションの体系的完成
`v3/docs/paper4.md` の冒頭に、学術論文の Experimental Setup / Methodology として完全に機能する以下の構造を正式に書き込みました：
1. **実験の位置づけと学術的背景**: EmoBank（妥当性）と AIPsy-Affect（統制・特異性）の役割分担、文章複雑性による困惑交絡の排除。
2. **5大リサーチクエスチョン（RQ1〜RQ5）**: ①感度、②用量反応、③特異性、④他者-自己連動、⑤8感情プロファイル。
3. **実験要因デザイン（Factorial Design）**: 4ファミリー（Qwen 1.5B, Mistral 7B, Llama 1B, Gemma 2B）× 2水準（Base vs. Instruct）× 3強度 × 8感情 × 2視点（Reader vs. Self）の全8モデル。
4. **データセット構造（480データ）**: 4 split（clinical 192, neutral 192, moderate 48, complex_neutral 48）の設計とメタデータ定義。
5. **プロンプト設計とタスク指示**: Reader / Self の正確なプロンプト本文、Instruct（チャットテンプレート）と Base（継続文脈）の提示仕様。
6. **測定・推定メカニズムと標準化**: 729候補の完全結合対数尤度による連続重心期待値算出（パース失敗率0%・再現性100%）、1〜5尺度変換、3大感情クラスタの定義。

### (3) `v3/docs/paper4.md` 関連研究（Related Work）セクションの体系的完成
入力いただいたメモ・議論のテキストを、学術論文の「第2章：関連研究」として以下の4階層に完全に体系化・清書し、ファイルの最冒頭に配置しました：
1. **感情・共感の哲学的基盤とLLMにおける認知・情動の分離**:
   - 現象的意識と情報処理の分離（Haladjian & Montemayor, 2016）
   - 認知共感 vs. 情動共感の理論的境界（Montemayor et al., 2021/2022）
   - LLMの共感的振る舞い（Elyoseph 2023, Ayers 2023 JAMA）と情動的実態の乖離
2. **LLM内部における感情表現の幾何構造と表現制御 (Representation & Steering)**:
   - 感情表現の存在と中間層局在（Di Palma ACL 2025; Zhang 2025）
   - 低次元マニフォールドと人間整合成（Reichman ICLR 2026; Wu 2026）
   - 活性化操舵（Activation Steering: Li NeurIPS 2023 ITI; Zhang ACL 2024 TruthX）
3. **LLMの内省能力（Introspection）と自己報告の解離 (Dissociation)**:
   - 潜在知識と表層出力の不一致（Burns 2023 CCS; Turpin 2023）
   - 自己知識と人工概念の内省（Binder ICLR 2025; Lindsey 2025/2026）
   - 心理状態の数値自己報告とモード崩壊（Martorell 2026）
   - 内省批判と入力セマンティクス交絡（Singh, Linzen & Ravfogel, COLM 2026）
4. **既存研究の学術的空白と本研究の位置づけ**:
   - 既存5大系統（Latent Knowledge, Emotion Probing, Emotion Geometry, Quantitative Introspection, Introspection Critique）との対比マトリクス表
   - 本研究の3大貢献（他者認識-自己報告の完全分離測定、Complex Neutralによる複雑性交絡の排除、Post-trainingによる覚醒度増幅とペルソナ発現の解明）の明示

### (4) `v3/docs/paper4.md` 序論・研究背景（Introduction）セクションの洗練と学術的定式化 (Refinement Complete)
「社会課題 → 共感の分類 → 既存研究 → 計算論的ギャップ → 本研究の4比較軸 → V1〜V3」が一直線につながる論理構造に整理・改稿を完了しました：
1. **1.1 心を支える対話型生成AIの普及と共感応答の重要性**:
   - 汎用対話型生成AIやAIコンパニオンの「心の支え」としての利用拡大（Perez [2], Awarefy [3], Zao-Sanders [1]）。
   - 単なるハルシネーションを超えた、人間の認知・情動・自律性に影響を与える「関係性レベルのリスク（感情的過剰依存・自律性の喪失・心理的危機）」（Weidinger [4], Fang et al. [5], CBS News [6]）。
   - Bornstein（1998, 2002）の関係依存モデル（健全な依存、過剰な依存、分離）をAI関係性に拡張した定義表（表 1）。
   - 人間中心の理念：「主導権は人間に、AIは相棒に（Human-in-the-lead, AI as an assistant）」。
2. **1.2 認知的共感と情動的共感**:
   - 認知的共感（他者理解）と情動的共感（情動共鳴・同調）の心理学的区別。
   - Yu et al. [21]（*Computers in Human Behavior: Artificial Humans* 2025）のIRI/BES評価の限界（回答テキストのブラックボックスな測定であり内部処理は未解明）。
   - 「共感的な出力を生成できること」と「感情刺激が内部でどのように処理され出力に利用されるか」の厳密な峻別。
3. **1.3 心理的ガードレールと共感応答の制御**:
   - カウンセリング面接技法（Ivey et al. [18]）における認知的共感（理解と自律性）vs. 情動的共感・同調（巻き込みと過剰依存）。
   - LLM特有の迎合性（Sycophancy, Sharma et al. [19]）の危険性と、心理的ガードレールの設計課題（正確な理解を維持しながら過度な情動的同調を抑制）。
4. **1.4 既存研究：LLMは感情をどのように処理しているのか**:
   - 出力レベルの行動評価：Huang et al. [24] **EmotionBench** (NeurIPS 2024)。
   - 内部表現の幾何解析：Tak et al. [22] (Findings of ACL 2025)、Reichman et al. [23] (ICLR 2026)。
5. **1.5 研究ギャップ：認識された感情情報はどのように出力へ利用されるのか**:
   - 4つの観点：
     - ① **ReaderとSelfの区別**: Selfを情動的共感そのものではなく「感情刺激に対するモデル自身のaffective reactivityを測定する操作的指標」として慎重に定義。
     - ② **入力・内部表現・出力の分離**: Decodability（線形プローブでの読み出し）と Causal Leverage（実際の出力生成への利用）の峻別。
     - ③ **BaseとInstructの比較**: 事後学習による感情処理能力の変化様式を解明。
     - ④ **モデルファミリー間比較**: アーキテクチャ横断の普遍性とモデル依存性の同定。
6. **1.6 本研究の目的**:
   - 処理過程：$\boxed{\text{感情刺激} \rightarrow \text{感情認識} \rightarrow \text{内部感情表現} \rightarrow \text{因果的利用} \rightarrow \text{自己報告・応答}}$
   - 比較軸：$\boxed{\text{Reader vs. Self} \times \text{Base vs. Instruct} \times \text{Input vs. Internal vs. Output} \times \text{Model Family}}$
7. **1.7 研究ロードマップ**:
   - V1（EmoBank妥当性・他者認識と自己報告の分離）、V2（AIPsy-Affect因果的特異性と事後学習・モデル差）、V3（内部動態と因果介入、Decodability vs. Causal Leverage）。
8. **1.8 本研究の位置付けと将来的展開**:
   - 心を支えるAIにおける共感を内部メカニズムから理解し、制御可能にする心理的ガードレールの基礎研究。
9. **参考文献 [1]〜[24]**:
   - 既存の [1]〜[20] を完全維持した上で、[21] Yu et al. (2025), [22] Tak et al. (Findings of ACL 2025), [23] Reichman et al. (ICLR 2026), [24] Huang et al. (NeurIPS 2024 EmotionBench) を正確に整備。



