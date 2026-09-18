# Walkthrough: V1 および Behavioral Stage の学術的・統計的・方法論的改善完了報告

本ドキュメントは、Behavioral Stage（3-Way VAD, AIPsy 4-Split）および V1 実験パイプライン（Phase A, Phase B, Phase C）における計29項目の学術的・統計的改善の完了内容をまとめたものです。

---

## 1. 修正の全体概要

| ステージ | 対象ファイル | 主要な改善点 |
|:---|:---|:---|
| **共通統計基盤** | `src/affective_empathy_eval/statistics.py`<br>`v1/src/affective_empathy_eval/statistics.py` | ・`compute_paired_cohen_dz` ($d_z = \bar\Delta / s_\Delta, \text{ddof}=1$)<br>・`compute_bivariate_bootstrap_ci` (ペアブートストラップ 95% CI)<br>・`generate_derangement` (固定点なし完全撹乱順列) |
| **Behavioral** | `v1/scripts/run_3way_vad_evaluation.py`<br>`v1/scripts/run_aipsy_4split_evaluation.py` | ・`prompt + candidate` の **Joint Tokenization** による teacher-forced scoring（サブワード境界アーティファクト解消） |
| **Behavioral** | `v1/scripts/summarize_3way_vad.py` | ・用語中立化（`collapse` 廃止 $\to$ `exact_neutral_rates`）<br>・Self ↔ Human Reader を「accuracy」から「human-affect correspondence (alignment)」へ是正<br>・Reader–Self coupling に bootstrap 95% CI & Spearman $\rho$ 追加<br>・3者の明確な概念分離解説を追加 |
| **Behavioral** | `v1/scripts/summarize_aipsy_4split.py` | ・RQ1: paired effect size $d_z$ & Benjamini-Hochberg FDR 補正<br>・RQ2: $Y \sim \text{Intensity} + (1\|\text{pair})$ の LMM 傾き・p値を Primary 解析に追加<br>・RQ3: 単純比率 ASR を Primary から外し、差分 contrast ＋ 95% CI を主軸化<br>・RQ4: coupling に bootstrap CI & Spearman $\rho$ 追加<br>・Primary 結果を4本柱（Grounding / Sensitivity / Specificity / Coupling）に体系化 |
| **V1 Phase A** | `v1/scripts/run_v1_phase_a_probe.py`<br>`v1/scripts/generate_phase_a_report.py` | ・E1-B 分類に `StratifiedGroupKFold(groups=pair_id)` 適用（pair-level 漏洩防止）<br>・E2 Direct Cross-Decoding, RSA, Procrustes を held-out split 評価に改修<br>・幾何分類を「操作的分類 (Operational Classification)」と明記し、低 decodability モデル（Gemma等）での解釈注意注記を追加 |
| **V1 Phase B** | `v1/scripts/run_v1_phase_b_semantic_audit.py`<br>`v1/scripts/generate_phase_b_report.py` | ・表現緩和（Semantic Validity Confirmed $\to$ Indicative of affective sensitivity）<br>・`collapse` を `decreased` へ是正<br>・`lexical overlap >= 80%` を削除<br>・Paraphrase を rule-based surface perturbation へ修正 |
| **V1 Phase C** | `v1/scripts/run_v1_phase_c_causal_patching.py`<br>`v1/scripts/generate_phase_c_report.py` | ・Joint Tokenization への移行<br>・`generate_derangement` による Random control の自己一致完全排除<br>・E3: $\cos(\text{mean vectors})$ に加え各 pair の $\text{mean}\{\cos(C_{R,i}, C_{S,i})\}$ を記録<br>・E3/E4: 全 pair の介入値を保持する long-form CSV 出力<br>・E4: Matched vs Random 差に対する paired test ($d_z$) & bootstrap 95% CI 追加<br>・`--full` オプション整備 |
| **V1 Phase C** | `v1/scripts/run_v1_phase_c_targeted_ablation.py` | ・Joint Tokenization への移行<br>・Discovery / Confirmation 独立 50/50 分割オプション追加<br>・Zero Ablation に加え matched neutral / neutral mean replacement を追加<br>・Double Dissociation 判定に厳格な cross-over 条件を必須化<br>・結論表現を「task-specific causal specialization / partial dissociation」に是正<br>・`df.pivot` による安全なペア対応付け |

---

## 2. 統計的・方法論的改善の詳細

### 2.1 Joint Tokenization による Sequence Likelihood の健全化
- **問題点**: 従来の `prompt_ids + cand_ids` の単純トークン列連結では、BPE/SentencePiece 境界において「プロンプト末尾と候補先頭が1つのトークンにマージされる」ケースで不自然な境界分割（sub-word boundary artifact）が発生し、尤度計算に歪みが生じる恐れがありました。
- **改善**:
  ```python
  prompt_encoded = tokenizer.encode(prompt, add_special_tokens=False)
  prompt_len = len(prompt_encoded)
  joint_encoded_all = [tokenizer.encode(prompt + c, add_special_tokens=False) for c in candidates]
  # candidate トークン位置のみを正確にスライスして teacher-forced log_softmax を gather
  sub_logits = logits[local_i, prompt_len - 1 : prompt_len - 1 + c_len, :]
  ```
  すべての Behavioral スクリプトおよび Phase C 介入スクリプトでこの統一プロトコルを実装しました。

### 2.2 多重比較補正 (FDR) & Paired Effect Size ($d_z$)
- **改善**:
  各刺激ペアの差分 $\Delta = x_i - y_i$ に対し、独立サンプルの Cohen's $d$ ではなく、相関を考慮した適切な対応のある効果量
  $$d_z = \frac{\bar\Delta}{s_\Delta}, \quad (\text{ddof}=1)$$
  を計算。さらに、多数の検定（Reader/Self × V/A × 各クラスタ）に対して Benjamini-Hochberg 法による FDR 補正（$q$-value）を適用し、偽発見率を厳格に制御しました。

### 2.3 混合効果モデル (LMM) による Dose-Response の Primary 評価
- **改善**:
  臨床強度の単調増加率（monotonicity rate）だけでなく、刺激ペアのランダム切片を取り入れた線形混合効果モデル
  $$Y \sim \text{Intensity} + (1 \mid \text{pair})$$
  の固定効果傾き $\beta_{\text{intensity}}$ および $p$ 値を Primary 解析として算出し、用量依存的反応性を統計的に検証しました。

### 2.4 完全撹乱順列 (Derangement) による厳格なシャッフル統制
- **改善**:
  E4 の Random control において、単純な `np.random.permutation` では $O(1/e) \approx 36.8\%$ の確率で少なくとも1つのペアで $j = i$（自己一致、すなわち matched と同じ刺激）が混入してしまいます。
  `generate_derangement` により固定点（$\forall i, \pi(i) \neq i$）を数学的に排除し、真の刺激間シャッフル統制を実現しました。

### 2.5 二重解離 (Double Dissociation) の厳格な判定基準
- **改善**:
  単に交互作用検定 $p < 0.05$ のみで二重解離を主張するのではなく、以下の **Cross-over（交差）条件** を必須としました：
  1. Reader-Site において：$\text{Impact}_{\text{Reader}} > \text{Impact}_{\text{Self}}$
  2. Self-Site において：$\text{Impact}_{\text{Self}} > \text{Impact}_{\text{Reader}}$
  交差条件を満たさない場合は「Single Dissociation / Asymmetric Specialization」と分類し、過度なモジュール化の主張を排して「task-specific causal specialization / partial dissociation」と中立的に報告します。

---

## 3. テスト検証結果

新たに作成した `tests/test_v1_refinements.py` を含む全テストスイートを実行し、全件合格を確認しました。

```bash
.venv/bin/python -m pytest tests/
```

**実行結果**:
```text
====================== test session starts ======================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0
rootdir: /mnt/nas/home/hiromi/src/emo
plugins: anyio-4.15.1
collected 29 items                                            

tests/test_geometry.py .....                            [ 17%]
tests/test_interventions.py ......                      [ 37%]
tests/test_likelihood.py ........                       [ 65%]
tests/test_models_hooks.py .                            [ 68%]
tests/test_statistics.py .....                          [ 86%]
tests/test_v1_refinements.py ....                       [100%]

================ 29 passed, 1 warning in 2.18s ================
```

### 検証項目一覧
1. `test_paired_cohen_dz`: $d_z = \bar\Delta / s_\Delta$ (`ddof=1`) の正確性
2. `test_generate_derangement`: 自己一致（固定点）が皆無であることの検証
3. `test_bivariate_bootstrap_ci`: ペアブートストラップによる 95% CI の計算精度
4. `test_e6_pivoted_rm_anova_equivalence`: データフレームの行順序に依存しない `pivot` 対応付けの堅牢性
5. 既存の `test_geometry`, `test_interventions`, `test_likelihood`, `test_models_hooks`, `test_statistics` の後方互換性

---

## 4. ドキュメント保存規約の遵守

- `docs/v1_and_behavioral_refinements/` 配下に以下のファイルが保存・整備されています：
  - `task.md`: 29項目の改善要件と実装チェックリスト
  - `implementation_plan.md`: 詳細な実装方針・設計ドキュメント
  - `walkthrough.md`: 本検証・完了報告ドキュメント
- 原データ（`data/raw/`）および既存の raw results ログは上書きせず保持されています。
