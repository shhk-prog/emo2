# Task: V1 and Behavioral Stage Statistical and Methodological Refinements

## 概要
ユーザーから提示された「Behavioral Stage」および「V1 (Phase A, B, C)」に関する計29項目の学術的・統計的・方法論的改善指示を反映し、測定妥当性・統計的厳密性・再現性を最高水準に引き上げる。

## タスクリスト

### 1. Behavioral Stage の改修
- [ ] **1.1 Joint Sequence Likelihood & Tokenization条件統一**
  - [ ] `v1/scripts/run_3way_vad_evaluation.py` の `compute_likelihoods_batched` を `prompt + candidate` の joint tokenization に変更
  - [ ] `v1/scripts/run_aipsy_4split_evaluation.py` の `compute_likelihoods_batched` を joint tokenization に変更
  - [ ] `v1/scripts/run_cross_family_recognition.py` の likelihood 計算も確認・統一
  - [ ] hidden extraction / likelihood / anchor 計算で `encode_prompt_canonical` (`add_special_tokens=False`) による tokenization 条件を統一
- [ ] **1.2 統計解析の厳密化 (AIPsy 4-Split)**
  - [ ] `v1/scripts/summarize_aipsy_4split.py` の RQ1 effect size を paired effect size $d_z = \frac{\bar\Delta}{s_\Delta}$ (`ddof=1`) に変更
  - [ ] Reader/Self × V/A × cluster 等の多数の検定に対して FDR 補正（Benjamini-Hochberg）を追加、Primary contrast を事前固定
  - [ ] RQ2 Dose-response に $Y \sim \text{Intensity} + (1|\text{pair})$ の混合効果モデル（LMM）の傾き・検定結果を Primary 解析として追加
  - [ ] RQ3 Complexity control の単純比率 $|\Delta_{\rm affect}|/|\Delta_{\rm complexity}|$ を Primary から除外し、$|\Delta_{\rm affect}| > |\Delta_{\rm complexity}|$ の contrast + bootstrap CI で評価
  - [ ] RQ4 Reader–Self coupling の Pearson $r$ に pair bootstrap 95% CI を追加
  - [ ] RQ4 に Spearman $\rho$ を追加（外れ値依存性の確認）
  - [ ] cluster 別相関にも bootstrap CI を付与
  - [ ] Primary 結果を「Human grounding / Sensitivity / Dose-response & specificity / Reader–Self coupling」の 4 部構成に再整理
- [ ] **1.3 用語・命名の厳密化と中立化 (EmoBank 3-Way VAD)**
  - [ ] `v1/scripts/summarize_3way_vad.py` で Self ↔ Human Reader を「accuracy」と呼ばず「human-affect correspondence (alignment)」に変更
  - [ ] Reader ↔ Human Reader、Self ↔ Human Reader、Reader ↔ Self の3種類の意味を明確に分離して記述
  - [ ] `self_collapse_555_pct` や `neutralization rate` 等の旧仮説命名を廃止し、`exact_neutral_argmax_rate` 等の中立的観測量に改名
  - [ ] Reader–Self coupling の Pearson $r$ に pair bootstrap 95% CI および Spearman $\rho$ を追加

---

### 2. V1 (Phase A, B, C) の改修
- [ ] **2.1 Phase A: Probing & Geometry**
  - [ ] `v1/scripts/run_v1_phase_a_probe.py`: E1-B の AIPsy affective-vs-neutral 分類を `StratifiedKFold` から `StratifiedGroupKFold(groups=pair_id)` に変更し pair leakage を防止
  - [ ] `v1/scripts/run_v1_phase_a_probe.py`: E2 Direct Cross-Decoding を全データ fit から pair 単位 train/test split の held-out 評価に変更（Scaler / Ridge / Procrustes は train のみで fit し test に適用）
  - [ ] `v1/scripts/run_v1_phase_a_probe.py`: E2 RSA を held-out 刺激上で評価
  - [ ] `v1/scripts/run_v1_phase_a_probe.py` & `generate_phase_a_report.py`: Shared/Alignable Geometry の閾値分類を「操作的分類 (operational classification)」と明記し、Gemma などの低 decodability モデルでの解釈上の注意を注記
- [ ] **2.2 Phase B: Semantic Audit & Report**
  - [ ] `v1/scripts/run_v1_phase_b_semantic_audit.py` & `generate_phase_b_report.py`:
    - [ ] 「Semantic Validity Confirmed」等の強い表現を「Evidence Consistent with Compositional Affective Processing」等へ緩和
    - [ ] Word Shuffle について「collapse」と書かず「performance decreased」と記述
    - [ ] Outcome Reversal の「lexical overlap ≥80%」の記述を削除
    - [ ] Paraphrase を「語彙を全面変更したparaphrase」と呼ばず「rule-based surface-form perturbation」へ修正
- [ ] **2.3 Phase C: Causal Patching & Targeted Ablation**
  - [ ] `v1/scripts/run_v1_phase_c_causal_patching.py`:
    - [ ] Sequence Likelihood を `prompt + candidate` の joint tokenization へ変更
    - [ ] patch 位置計算と likelihood 計算で canonical tokenization を共通化
    - [ ] E3 で $\cos(\text{mean vectors})$ に加え、pair ごとの $\text{mean}\{\cos(C_{R,i}, C_{S,i})\}$ を保存
    - [ ] E3/E4 で層平均だけでなく pair-level long-form CSV を保存
    - [ ] E4 Random control で自己一致（固定点）を除外する完全撹乱（derangement）アルゴリズムを導入
    - [ ] E4 の Matched vs Random 差について bootstrap CI および paired test を追加
    - [ ] `--limit` の default=20 を pilot 用と明記し、全ペア実行オプションを整理
  - [ ] `v1/scripts/run_v1_phase_c_targeted_ablation.py`:
    - [ ] Reader-site / Self-site 選択を Discovery split、統計検定を独立 Confirmation split に分離
    - [ ] Zero Ablation に加え、neutral mean replacement / matched neutral activation 等の対照介入を追加
    - [ ] Double Dissociation 判定を interaction $p<.05$ のみで行わず、cross-over パターン（Reader-site で Reader > Self かつ Self-site で Self > Reader）を確認する
    - [ ] 結論表現を「task-specific causal specialization / partial dissociation」に是正
    - [ ] fallback t-test で `pair_id` で明示的に pivot して対応付け
- [ ] **2.4 全体検証 & レポート**
  - [ ] 改修後のスクリプトの構文チェック・単体テスト
  - [ ] `walkthrough.md` の作成と完了報告
