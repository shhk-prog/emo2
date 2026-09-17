# Implementation Plan: V1 and Behavioral Stage Statistical & Methodological Refinements

本計画は、ユーザーから提示された「Behavioral Stage」および「V1 (Phase A, B, C)」に関する計29項目の学術的・統計的・方法論的改善要求をコードベースに完全反映するための実装詳細です。

---

## ユーザー確認事項 (User Review Required)

> [!IMPORTANT]
> - **原データ・既存ログの保護**: `data/raw/` および既存の実験結果ログ（`v1/results/raw/`）は一切上書きせず、必要に応じて派生データや修正版スクリプトの出力として新規保存します。
> - **優先度に基づく改修順序**: 
>   1. Behavioral: Joint Sequence Likelihood、LMM Dose-response、Reader–Self coupling CI、用語・命名の中立化
>   2. V1: E1-B Group Split、E2 held-out化・操作的分類注記、Phase C tokenization統一・完全撹乱・Discovery/Confirmation分離、E5/E6表現緩和

---

## 提案される変更 (Proposed Changes)

### 1. Behavioral Stage

#### [MODIFY] [run_3way_vad_evaluation.py](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/run_3way_vad_evaluation.py) & [run_aipsy_4split_evaluation.py](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/run_aipsy_4split_evaluation.py)
- **Joint Sequence Likelihood**:
  - `prompt_ids + cand_ids` の連結方式を廃止し、`prompt + candidate` を joint tokenize して candidate 部分のトークンのみ teacher-forced scoring する方式に変更。
  - `encode_prompt_canonical` (`add_special_tokens=False`) による統一された tokenization を適用。

#### [MODIFY] [summarize_3way_vad.py](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/summarize_3way_vad.py)
- **命名の中立化**:
  - `collapse_rates`, `pct_dim_5` などの命名・表示を `exact_neutral_argmax_rate` / `exact_neutral_555_rate` 等の客観的観測量に改名（旧「中立化」仮説の排除）。
- **用語の厳密化**:
  - Self ↔ Human Reader を「accuracy」と呼ばず「human-affect correspondence (alignment)」に変更。
  - 3種類の関係性（① Reader ↔ Human Reader: 認識モデルの対人間一致、② Self ↔ Human Reader: 自己報告の対人間一致、③ Reader ↔ Self: モデル内部の認知結合度）を明確に区別して解説。
- **結合度の統計強化**:
  - Reader–Self coupling (`internal_rs`) の Pearson $r$ に対し、1,000回の pair bootstrap による 95% 信頼区間を追加。
  - 外れ値依存性を検証するため、Spearman $\rho$ を併記。

#### [MODIFY] [summarize_aipsy_4split.py](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/summarize_aipsy_4split.py)
- **RQ1 (Sensitivity)**:
  - 効果量を単なる $\Delta / \text{std}$ から、対応のある効果量（paired effect size） $d_z = \frac{\bar\Delta}{s_\Delta}$（`ddof=1`）に変更。
  - Reader/Self × V/A × cluster 等の多数の検定に対して FDR 補正（Benjamini-Hochberg）を適用し、Primary contrast を事前固定。
- **RQ2 (Dose-response)**:
  - 単なる monotonicity rate に依存せず、混合効果モデル $Y \sim \text{Intensity} + (1|\text{pair})$ の傾き（固定効果係数）、標準誤差、p値を Primary 解析として追加。
- **RQ3 (Specificity / Complexity Control)**:
  - 不安定な単純比率 $|\Delta_{\rm affect}|/|\Delta_{\rm complexity}|$ を Primary から外し、$|\Delta_{\rm affect}| > |\Delta_{\rm complexity}|$ の contrast（$\Delta_{\rm contrast} = |\Delta_{\rm affect}| - |\Delta_{\rm complexity}|$）と pair bootstrap 95% CI による評価を導入。
- **RQ4 (Reader-Self Coupling)**:
  - Pearson $r$ に pair bootstrap 95% CI を追加。Spearman $\rho$ も追加。cluster 別相関にも CI を付与。
  - `self_collapse_555_pct` などの命名を廃止し、中立的命名（`exact_neutral_argmax_rate`）へ変更。
- **Primary 構成の整理**:
  - レポート構成を「1. Human grounding / 2. Sensitivity / 3. Dose-response & specificity / 4. Reader–Self coupling」の 4 部構成に再整理。

---

### 2. V1 (Phase A, B, C)

#### [MODIFY] [run_v1_phase_a_probe.py](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/run_v1_phase_a_probe.py) & [generate_phase_a_report.py](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/generate_phase_a_report.py)
- **E1-B Group Split**:
  - AIPsy affective-vs-neutral 分類において、`StratifiedKFold` を廃止し、`StratifiedGroupKFold(n_splits=5)` に `groups=df["pair_id"]` を渡すことで同一ペアの漏洩（pair leakage）を完全に防止。
- **E2 Held-out Cross-Decoding**:
  - 全データでの fit/evaluate を廃止し、pair 単位の train/test split（GroupKFold または 50/50 held-out split）を導入。
  - `StandardScaler`, `Ridge`, `Procrustes` の回転行列 $Q$ を train split のみで fit し、test split に適用して held-out transfer を評価。
  - RSA も held-out 刺激上で評価。
- **操作的分類 (Operational Classification) の明記**:
  - Shared Geometry / Alignable Geometry の閾値分類を理論的事実ではなく「操作的分類」と明記。
  - Gemma などの低 decodability モデルにおいて、幾何構造の類似を安易に「affect-specific geometry の共有」と解釈しないよう注意書きを追加。

#### [MODIFY] [run_v1_phase_b_semantic_audit.py](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/run_v1_phase_b_semantic_audit.py) & [generate_phase_b_report.py](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/generate_phase_b_report.py)
- **レポート表現の是正**:
  - 「Semantic Validity Confirmed」等の強い表現を「Evidence Consistent with Contextual Affective Processing」等に緩和。
  - Word Shuffle について「collapse」と書かず「performance decreased」と記述。
  - Outcome Reversal で実コードで保証していない「lexical overlap $\ge 80\%$」の記述を削除。
  - Paraphrase を「語彙を全面変更したparaphrase」と呼ばず「rule-based surface-form perturbation」と正確に記述。

#### [MODIFY] [run_v1_phase_c_causal_patching.py](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/run_v1_phase_c_causal_patching.py) & [generate_phase_c_report.py](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/generate_phase_c_report.py)
- **Tokenization & Sequence Likelihood 統一**:
  - `prompt + candidate` の joint tokenization へ改修。
  - special token の扱いを含め canonical tokenization を patch 位置計算と likelihood 計算で共通化。
- **E3 拡張**:
  - 現在の層平均ベクトルのコサイン類似度 $\cos(\bar{C}_R, \bar{C}_S)$ に加え、pairごとの $\text{mean}\{\cos(C_{R,i}, C_{S,i})\}$ を計算・記録。
- **E3/E4 Pair-level Long-form CSV 保存**:
  - 層平均サマリーだけでなく、全 pair の介入前後値およびシフトを保持する long-form CSV（`e3_causal_map_pair_level.csv`, `e4_interchangeability_pair_level.csv`）を保存。
- **E4 Random Control 完全撹乱 (Derangement)**:
  - `np.random.permutation` の固定点（自己一致）を許さず、全 pair について `perm[i] != i` となる完全撹乱アルゴリズムを実装。
  - Matched vs Random の差に対する paired test および bootstrap 95% CI を追加。
- **Pilot / Confirmatory の区分**:
  - `--limit=20` は pilot 用と明記し、全ペア実行オプションを整理。

#### [MODIFY] [run_v1_phase_c_targeted_ablation.py](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/run_v1_phase_c_targeted_ablation.py)
- **Discovery / Confirmation 分割**:
  - データを Discovery split（サイト探索・確認用）と Confirmation split（独立な仮説検定用）に 50/50 で分割。Confirmation split 上で LMM 交互作用検定を実行。
- **介入条件の拡充**:
  - 単なる Zero Ablation だけでなく、neutral mean replacement（中立文の平均活性化ベクトルへの置換）および matched neutral activation（当該ペアの中立文活性化への置換）を対照介入として追加。
- **Double Dissociation 判定の厳格化**:
  - interaction $p<.05$ だけでなく、cross-over パターン（Reader-site で Reader 影響 > Self 影響、かつ Self-site で Self 影響 > Reader 影響）の成立を必須条件とし、結論は「task-specific causal specialization / partial dissociation」と表現。
- **Fallback t-test の対応付け**:
  - `pair_id` で明示的に pivot して差分を計算し、配列順依存を完全に排除。

---

## 検証計画 (Verification Plan)

### 自動テスト / ユニットテスト
- 修正した関数（$d_z$、完全撹乱アルゴリズム、LMM フィッティング、Bootstrap CI、Joint Likelihood 等）の単体テストスクリプトを作成・実行。
```bash
/mnt/nas/home/hiromi/src/emo/.venv/bin/python -m pytest tests/test_v1_refinements.py -v
```

### スクリプトの整合性・ドライラン確認
- `summarize_3way_vad.py` および `summarize_aipsy_4split.py` を既存結果に対して実行し、エラーなくレポート・CSV が生成されることを確認。
```bash
/mnt/nas/home/hiromi/src/emo/.venv/bin/python v1/scripts/summarize_3way_vad.py --help
/mnt/nas/home/hiromi/src/emo/.venv/bin/python v1/scripts/summarize_aipsy_4split.py --help
/mnt/nas/home/hiromi/src/emo/.venv/bin/python v1/scripts/run_v1_phase_a_probe.py --help
/mnt/nas/home/hiromi/src/emo/.venv/bin/python v1/scripts/run_v1_phase_c_causal_patching.py --help
/mnt/nas/home/hiromi/src/emo/.venv/bin/python v1/scripts/run_v1_phase_c_targeted_ablation.py --help
```
