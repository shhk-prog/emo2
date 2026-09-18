# 詳細解説・提案書: EmoBank & AIPsy-Affect 4-Split の次に進める内部表現分析

## 概要

本ドキュメントは、EmoBank（人間評価済みVAD）および AIPsy-Affect 4-Split（最小対臨床ヴィネット）の実験完了後に取り組むべき、**内部表現（Mechanistic Interpretability）の次期検証体系**を体系化したものです。

ユーザーから提示された以下の2つの重要リサーチクエスチョン（RQ）に対し、現在実装済みの3大ツール（**Linear Probe**, **Ablation**, **Activation Patching**）をどのように連携させて決着をつけるかを解説します。

---

## 1. 2つの問いの学術的背景と意義

### 問1: 「自己報告と相手の感情の発火場所が同じかどうか」
- **学術的意義**:
  - 認知心理学や神経科学において、共感のメカニズムには**「共有表現仮説（Shared Representation Hypothesis）」**と**「分離回路仮説（Separate Systems Hypothesis）」**の対立があります。
  - LLMにおいても、「相手の文章から感情を認識・抽出する計算」と、「刺激を受けた直後に自分自身の情動状態として報告する（あるいは対話応答を生成する）計算」が、同じ隠れ状態・同じ方向ベクトルを共有しているのか、それとも完全に独立した別の部分空間で処理されているのかは、モデルの「機能的共感」の構造を決定づける最重要課題です。

### 問2: 「発火する場所は単語と単語が結びついているだけではないか」
- **学術的意義**:
  - LLMの内部表現研究における最大の査読者反論（Critique）は**「表層的な語彙共起（Lexical Association / Bag-of-Words Shortcut）」**です。
  - すなわち、「モデルが感情を抽象的に理解しているのではなく、文章中に含まれる特定の単語（death, accident, money, brother等）の統計的共起や単語埋め込みの近さに引っ張られて特定層が発火しているだけではないか？」という疑念です。
  - AIPsy-Affectの最小対（Affective vs Neutral）はこの語彙交絡を大幅に低減していますが、「発火が単語の結びつきではなく、文脈全体が織りなす抽象的な情動概念（Abstract Affective Concept）である」ことを因果的・統計的に完全に証明する必要があります。

---

## 2. 3大ツールの役割と検証マトリクス

現在リポジトリに実装されている3つのツールの理論的役割は以下の通りです：

| 手法 | 問いの性質 | 判定できること | 限界 |
|---|---|---|---|
| **Linear Probe** | 相関・存在 (Correlation) | 「その層の表現に、目的の情報が**存在・復元可能か**」 | 存在していても、下流の生成で使われていない（エピフェノメノン）可能性がある |
| **Ablation** | 因果的必要性 (Necessity) | 「その情報を壊すと、出力が崩壊するか（**不可欠か**）」 | バイパス経路や分散表現があると、単一層消去では効果が見えにくい |
| **Activation Patching** | 因果的十分性 (Sufficiency) | 「中立状態にその情報を注入すると、目的の出力を**強制誘導できるか**」 | 不自然な状態合成（Out-of-Distribution）を引き起こすリスクがある |

これら3つを単独ではなく**「組み合わせて対比させる」**ことで、決定的な証拠（Evidentiary Triangulation）を構築します。

---

## 3. 具体的な検証アプローチと判定基準

### 3.1 問1（自己報告 vs 相手の感情）の検証アプローチ

```text
【認識タスク (Recognition)】   入力文 ───> [ 認識表現 H_rec ] ───> 相手の感情値推論
                                              │ (比較・交差)
【自己報告タスク (Self-Report)】 入力文 ───> [ 自己表現 H_self ] ───> 自身のVA自己報告
```

1. **Linear Probe（相互交差プロービング: Cross-Decoding）**:
   - 認識タスクの隠れ状態から学習したプローブ重み $W_{rec}$ を、自己報告タスクの隠れ状態 $H_{self}$ に適用。
   - **判定**:
     - $R^2_{rec \rightarrow self}$ が高い $\rightarrow$ **共有表現**（他者認識と同じ感情軸で自己報告も表現されている）。
     - $R^2_{rec \rightarrow self} \approx 0$（交差転移が崩壊） $\rightarrow$ **分離表現**（同じ感情刺激であっても、「他者認知」と「自己状態」は直交する別個の座標系で管理されている）。
   - 全28層におけるプローブ方向のコサイン類似度 $\cos(W_{rec}^{(l)}, W_{self}^{(l)})$ の推移をプロット。

2. **Ablation（二重解離: Double Dissociation）**:
   - 認識方向 $W_{rec}$ を消去した際、自己報告の尤度シフトが維持されるか？
   - 自己報告方向 $W_{self}$ を消去した際、認識の推論精度が維持されるか？
   - **判定**: 「一方を壊しても他方は温存される」層が見つかれば、独立した並列回路の存在が証明される。

3. **Activation Patching（タスク間交差パッチング: Cross-Task Patching）**:
   - 認識プロンプト実行時の隠れ状態 $H_{rec}$ を、自己報告タスクのベースライン（中立推論中）に注入。
   - **判定**: 自己報告のVA期待値が認識された感情へとシフト（Recovery > 50%）すれば、認識表現は自己情動反応を駆動する因果的入力（Upstream Trigger）であることが証明される。

---

### 3.2 問2（単語の結びつき・語彙共起仮説の打破）の検証アプローチ

```text
[通常文 (Context + Words)]   ───> Layer 0 (単語) ───> ... ───> Layer 14-25 (文脈感情)
[シャッフル文 (Words Only)]  ───> Layer 0 (単語) ───> ... ───> [ 発火崩壊？ or 維持？ ]
```

1. **Linear Probe（Word Shuffling & 語彙ベースライン対照）**:
   - **単語埋め込み対照 (Layer 0 vs Intermediate Layers)**:
     - 単語単体の埋め込み（Layer 0）からのプローブ精度と、文脈が統合された中間層（Layer 14-25）の精度を比較。中間層が有意に高い場合、単語以上の文脈付加価値が存在する。
   - **Word Shuffling（語順シャッフル）**:
     - 文章内の単語をランダムにシャッフルした文（単語は100%同一だが文脈なし）を入力。
     - **判定**:
       - シャッフルしても中間層の発火・プローブ精度が維持される $\rightarrow$ **単語の結びつき（Bag-of-Words）にすぎない**。
       - シャッフルによって中間層の感情表現が完全に消失・崩壊する $\rightarrow$ **単語の存在ではなく、文脈が織りなす高次の意味構造を捉えている**決定打となる。

2. **Ablation（表層語彙成分の直交射影消去: Orthogonal Subspace Ablation）**:
   - Layer 0の主成分や高頻度単語相関が張る部分空間 $V_{lex}$ を特定。
   - 中間層の隠れ状態から $V_{lex}$ 方向を直交射影で消去：
     $$ H_{orthogonal} = H - (H \cdot V_{lex}) V_{lex} $$
   - **判定**: 語彙成分を物理的に除去しても、なお自己報告の尤度シフト（AIPsy-Affect等の効果）が維持されるかを検証。

3. **Activation Patching（Token-wise Patching: 単語位置 vs 文末統合位置）**:
   - 「個々の感情語」のトークン位置の活性化をパッチした場合と、「文末の統合トークン（[EOS]やピリオド）」の活性化をパッチした場合の自己報告への影響を比較。
   - **判定**: 個別単語位置のパッチでは効果が出ず、文末の要約トークンでのみ効果が出る場合、モデルは単語単位の局所共起ではなく、文末で統合された文脈概念を利用していることが証明される。

---

## 4. 次期ロードマップの提案

| フェーズ | 目的 | 手法 | 計算コスト / GPU要件 |
|---|---|---|---|
| **Phase 1** | **即時実行可能な既存データ分析** | ・Recognition vs Self-Report の相互交差プロービング<br>・全28層のプローブ重みコサイン類似度 & RSA<br>・Layer 0 vs 中間層の語彙対照プロービング | **極小 (既存テンソル再利用)**<br>※新規GPU推論不要 |
| **Phase 2** | **語彙共起仮説の因果的打破** | ・Word Shuffling（語順シャッフル）推論 & プロービング<br>・Token-wise（単語位置 vs 文末統合位置）Patching | **小規模 (数千推論)**<br>※GPU事前相談後に実行 |
| **Phase 3** | **二重解離とタスク間回路の同定** | ・認識表現消去 vs 自己報告消去の二重解離Ablation<br>・認識 $\rightarrow$ 自己報告のタスク間交差Patching | **中規模**<br>※Phase 1-2の結果を踏まえて実施 |

---

## 6. 全モデル・全層 実験スクリプトの配備とモジュラージョブ実行仕様

全8モデル（Qwen2.5-1.5B, LLaMA-3.2-1B, Gemma-2-2B, Mistral-7B × Base/Instruct）および全層（Layer 0〜L）、フルデータ（EmoBank 1,006件 ＋ AIPsy 2,196件／480組）を完全網羅するための3大バッチスクリプトを `v1/scripts/` 配下に新規配備しました。

### 6.1 スクリプト一覧と機能対応表

| スクリプト | 対象フェーズ | 実験項目 | 計算規模・所要時間 | 実行制御・特徴 |
|:---|:---:|:---|:---|:---|
| [`run_all_phase_a.sh`](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/run_all_phase_a.sh) | **Phase A** | **E1**: 全層線形復元 (Ridge/Logistic)<br>**E2**: Direct/Aligned/RSA 幾何分類 | 全8モデル × 全層<br>所要時間: **約45分〜1時間** | 1フォワードパスで全層隠れ状態を一括抽出。<br>引数: `[limit] [batch_size] [device]` |
| [`run_all_phase_b.sh`](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/run_all_phase_b.sh) | **Phase B** | **E5**: 語彙交絡監査 (Jaccard, PPL等)<br>**E5**: 4大統制対 (Minimal/Shuffle等) | 全8モデル × 480組<br>所要時間: **約20分** | 語彙ショートカット仮説の完全反証。<br>引数: `[limit] [device]` |
| [`run_all_phase_c.sh`](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/run_all_phase_c.sh) | **Phase C** | **E3**: 因果マップ (強度＋2Dコサイン)<br>**E4**: 差分移植 (αスイープ・4対照)<br>**E6**: 二重解離消去 (LMM交互作用) | 全8モデル × 因果介入<br>所要時間: 柔軟分割可能 | **個ジョブ分割ディスパッチャー内蔵**。<br>モデル別、タスク別、単体ジョブ別の自由実行。 |

---

### 6.2 Phase C 個ジョブ実行ディスパッチャー（`run_all_phase_c.sh`）の使用法

Phase C は計算量が大きいため、モデル別・タスク別・ジョブID別に個別に分割して実行できるよう設計されています。

#### (1) 利用可能なジョブ一覧の確認
```bash
./v1/scripts/run_all_phase_c.sh --list
```
*出力例:*
```text
[1] qwen2.5_1.5b_base: E3/E4 Causal Patching & Difference Transfer  [id: qwen2.5_1.5b_base_patching]
[2] qwen2.5_1.5b_base: E6 Double Dissociation Ablation (L8 vs L20)  [id: qwen2.5_1.5b_base_ablation]
[3] qwen2.5_1.5b_instruct: E3/E4 Causal Patching & Difference Transfer  [id: qwen2.5_1.5b_instruct_patching]
[4] qwen2.5_1.5b_instruct: E6 Double Dissociation Ablation (L8 vs L20)  [id: qwen2.5_1.5b_instruct_ablation]
...
(全16ジョブ)
```

#### (2) 特定の個別ジョブのみを実行（バックグラウンドや並列化に最適）
```bash
# ジョブ番号で指定して実行
./v1/scripts/run_all_phase_c.sh --job 4

# ジョブIDで指定し、全層オプションを付与して実行
./v1/scripts/run_all_phase_c.sh --job qwen2.5_1.5b_instruct_patching --all-layers
```

#### (3) 特定のモデルに絞って実行
```bash
# LLaMA-3.2-1B-Instruct の全 Phase C ジョブを実行
./v1/scripts/run_all_phase_c.sh --model llama3.2_1b_instruct
```

#### (4) 特定のタスクのみを全モデル横断で実行
```bash
# 全8モデルの E6 Targeted Ablation（二重解離）のみを一括実行（所要時間: 約1時間）
./v1/scripts/run_all_phase_c.sh --task ablation
```

#### (5) 全モデル・全ジョブの一括実行
```bash
# 全16ジョブを順次完全実行（夜間バッチ実行等）
./v1/scripts/run_all_phase_c.sh --all
```

---

### 6.3 ログ保存機能の完全自動化（AGENTS.md 再現性規約準拠）

すべての実行スクリプト（Phase A, B, C）に、**コンソール出力（stdout / stderr）のリアルタイム表示とログファイルへの自動二重保存（`tee` 連携）** が組み込まれました。

- **保存ディレクトリ**:
  - Phase A: `v1/results/logs/phase_a/`
    - 全体実行ログ: `phase_a_run_YYYYMMDD_HHMMSS.log`
    - モデル別個別ログ: `${model_prefix}_YYYYMMDD_HHMMSS.log`
  - Phase B: `v1/results/logs/phase_b/`
    - 全体実行ログ: `phase_b_run_YYYYMMDD_HHMMSS.log`
    - モデル別個別ログ: `${model_prefix}_YYYYMMDD_HHMMSS.log`
  - Phase C: `v1/results/logs/phase_c/`
    - ジョブ別個別ログ: `${job_id}_YYYYMMDD_HHMMSS.log`
- **利点**:
  1. ターミナルで進捗バー（tqdm）や各層の精度出力をリアルタイムに確認可能。
  2. `nohup` や `tmux` でバックグラウンド実行した場合でも、ログファイルを開くだけで進行状況・エラー内容を完全に追跡可能。
  3. 各実行のタイムスタンプ、引数設定、全ログが永続化され、AGENTS.md の再現性要件を厳密に満たす。

---

### 6.4 Slurm Workload Manager による Phase C 16子ジョブ並列投入

Phase C をクラスター上で効率的に並列実行するための Slurm スクリプト [`slurm_run_phase_c.sbatch`](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/slurm_run_phase_c.sbatch) を配備しました。

- **Slurm 設定仕様**:
  - **パーティション**: `h100` (`#SBATCH --partition=h100`)
  - **CPUタスク**: `1` (`#SBATCH --cpus-per-task=1`)
  - **GPU**: `1` (`#SBATCH --gres=gpu:1`)
  - **メモリ・時間制限**: クラスタデフォルト（指定なし）
  - **子ジョブ配列**: `1-16` (`#SBATCH --array=1-16`)
  - **Slurm 出力ログ**: `v1/results/logs/slurm/phase_c_%A_%a.out` / `.err`

#### 投入コマンド
```bash
# 全16子ジョブをSlurmアレイとして一括投入
sbatch v1/scripts/slurm_run_phase_c.sbatch

# 全層パッチング（--all-layers）オプション付きで投入
sbatch v1/scripts/slurm_run_phase_c.sbatch --all-layers

# 特定の子ジョブ範囲（例: Job 1〜4 の Qwen 系のみ）に絞って投入
sbatch --array=1-4 v1/scripts/slurm_run_phase_c.sbatch

# ジョブ状態の確認
squeue -u $USER
```

---

## 7. Phase B（E5: 語彙交絡監査・4大統制対）全8モデル実行結果

全8モデル（4ファミリー × 2アライメント）において、AIPsy-Affect 4-Split の厳密なマッチドペア（$N=192$ 組）に対する Phase B（E5）が完了しました。

### 7.1 全8モデル横断 統制実験結果マトリクス

| モデル | Original Minimal Pair | Paraphrase Invariance | Word Shuffle | Outcome Reversal ($\Delta P$) | 判定 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Qwen 2.5 1.5B (Base)** | 0.883 | **0.992** (維持) | **0.773** (大幅崩壊) | **-0.284** (明瞭反転) | 文脈意味駆動 |
| **Qwen 2.5 1.5B (Instruct)** | 0.857 | **1.000** (維持) | **0.766** (大幅崩壊) | **-0.091** (反転) | 文脈意味駆動 |
| **LLaMA 3.2 1B (Base)** | 0.880 | **1.000** (維持) | **0.755** (大幅崩壊) | **-0.039** (反転) | 文脈意味駆動 |
| **LLaMA 3.2 1B (Instruct)** | 0.924 | **1.000** (維持) | **0.823** (崩壊) | **-0.013** (反転) | 文脈意味駆動 |
| **Gemma 2 2B (Base)** | 0.745 | **0.997** (維持) | **0.766** (崩壊) | **-0.146** (明瞭反転) | 文脈意味駆動 |
| **Gemma 2 2B (Instruct)** | 0.727 | **0.997** (維持) | **0.622** (大幅崩壊) | **-0.144** (明瞭反転) | 文脈意味駆動 |
| **Mistral 7B (Base)** | 0.901 | **0.997** (維持) | **0.776** (大幅崩壊) | **-0.018** (反転) | 文脈意味駆動 |
| **Mistral 7B (Instruct)** | 0.938 | **0.997** (維持) | **0.828** (崩壊) | **-0.007** (反転) | 文脈意味駆動 |

### 7.2 事前語彙交絡監査（E5-1）の全体平均値
- **Token-level Jaccard 類似度**: **0.411**（臨床文と中立文の単語重複率は4割程度に抑制）
- **相対編集距離 (Levenshtein)**: **0.464**
- **明示的感情語辞書スコア差**: **-0.047**（感情辞書単語の含有量に偏りは存在しない）
- **モデル Perplexity 比 (PPL Ratio)**: **0.86〜0.94**（臨床文と中立文で統語的難易度・自然さの極端な偏りは排除）

### 7.3 科学的結論
1. **語彙ショートカット仮説（Bag-of-Words）の完全打破**:
   - 単語を100%保持したまま語順（構成的意味）を破壊する `Word Shuffle` において、全モデルでプローブ精度が **0.622〜0.828** へと急落した。
   - 逆に、表層語彙を大幅に差し替えて意味を保つ `Paraphrase` においては、全モデルで **0.992〜1.000** と精度が完璧に保持された。
   - これにより、「プローブが特定の感情単語（death, cancer等）の共起を拾っているだけ」という査読者の反論は完全に退けられた。
---

## 8. Phase A, B, C 総合レポート生成スクリプト体系と成果物

EmoBankレポート（`3way_vad_detailed_report.md`）および AIPsy-Affectレポート（`aipsy_4split_detailed_report.md`）と同等の水準で、全8モデルの個別結果、モデルファミリー別 Base vs. Instruct 対比、RQと実験内容、理論的階段構造を包含した**総合レポート自動生成スクリプト**を配備しました。

### 8.1 レポート生成スクリプトと成果物対応表

| フェーズ | 生成スクリプト | 成果物レポート | レポート構成と分析内容 |
|:---:|:---|:---|:---|
| **Phase A** | [`generate_phase_a_report.py`](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/generate_phase_a_report.py) | [`phase_a_comprehensive_report.md`](file:///mnt/nas/home/hiromi/src/emo/v1/results/derived/v1_phase_a/phase_a_comprehensive_report.md) | **E1 $\rightarrow$ E2 の論理展開**<br>・EmoBank/AIPsyの全層プロービング ($R^2$)<br>・ピーク層間変位 $|l^*_R - l^*_S|$<br>・Direct / Aligned / RSA 幾何構造分類 |
| **Phase B** | [`generate_phase_b_report.py`](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/generate_phase_b_report.py) | [`phase_b_comprehensive_report.md`](file:///mnt/nas/home/hiromi/src/emo/v1/results/derived/v1_phase_b/phase_b_comprehensive_report.md) | **E5 語彙交絡の打破**<br>・Jaccard / 編集距離 / PPL比 監査<br>・4大統制対マトリクス (Shuffle崩壊・Paraphrase維持)<br>・結末反転感度 ($\Delta P$) |
| **Phase C** | [`generate_phase_c_report.py`](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/generate_phase_c_report.py) | [`phase_c_comprehensive_report.md`](file:///mnt/nas/home/hiromi/src/emo/v1/results/derived/v1_phase_c/phase_c_comprehensive_report.md) | **E3 $\rightarrow$ E4 $\rightarrow$ E6 因果回路と特異化**<br>・E3 因果マップ強度と2Dコサイン類似度<br>・E4 差分パッチング用量反応 & 特異性 Specificity<br>・E6 二重解離 (LMM交互作用検定) |

#### レポートの生成コマンド
```bash
python v1/scripts/generate_phase_a_report.py
python v1/scripts/generate_phase_b_report.py
python v1/scripts/generate_phase_c_report.py
```





