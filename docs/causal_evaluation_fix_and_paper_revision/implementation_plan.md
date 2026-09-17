# 実装計画: 因果評価スクリプトの修正および論文原稿の改訂

## 1. 概要と背景

本件では、コードと既存結果の突合により判明した以下の重大な実装上・記述上の問題を修正し、論文原稿を査読に耐えうる堅牢な内容へ改訂します。

1. **EMD計算における重大バグの解消**:  
   `compute_likelihoods_for_candidates()` の第2返り値は確率ではなく `candidate_lengths`（トークン長）です。`run_causal_localization_sweep.py` および `run_within_model_positive_control.py` がこれを確率分布として扱い `reshape(9, 9)` してEMDを計算していたため、同一のトークン長行列同士の距離比較となり、全28層で recovery = 0.00% が出力されていました。この結果は無効であり、コードを正しく Softmax 確率分布 `p` を渡すように修正します。
2. **プローブターゲット記述の整合**:  
   全層スイープコード（`run_causal_localization_sweep.py`）で予測しているのは Valence ではなく `intensity == 'peak'`（Peak-vs-Neutral intensity indicator, peak=1, neutral=0）です。したがって、Layer 15 $R^2=0.548$ は「Valence $R^2$」ではなく「Peak-vs-Neutral intensity indicator の線形回帰 $R^2$」として記述を厳格に修正します。
3. **尤度プロトコルの記述と実装の整合**:  
   実装は candidate token log-probability の単純和（raw sequence log-likelihood）です。`v2/src/likelihood.py` を拡張して raw likelihood と length-normalized likelihood の双方を算出・選択可能とし、論文内でも両者の区別と取り扱いを明記します。
4. **論文原稿（`v3/docs/paper.md`）の改訂**:  
   「全28層0%」を確定した結果として前面に押し出さず、ユーザーより提示された「Decodability Without Causal Sufficiency: A Case Study of Affect-Relevant Representations in a Paired Base/Instruct Language Model」の骨子に基づき、Decodability・高次元アライメント（Ridge $\alpha$ スイープ、CKA 0.829、retrieval 86.6%）・Center Collapse（$D_M=39.63 \rightarrow 8.30$）を中心に据え、因果介入については「EMD修正後に再計算・検証すべき要求事項（Required Validation）」として整理した安全かつ説得力のある論構成へ全面的に改訂します。

---

## 2. ユーザー確認事項 (User Review Required)

> [!IMPORTANT]
> **GPUを用いた実験の再実行について**:  
> 本リポジトリのルール（`AGENTS.md` 1.7「GPUを勝手に用いない」）に従い、修正コードによる全28層の再実行（GPUが必要なスクリプト）は、本計画の承認後およびユーザーからの明示的な指示があってから実施します。本ターンではコードの修正・構文検証・ドライラン準備と、論文原稿の改訂を完了させます。

---

## 3. 変更内容の詳細

### A. コード修正

#### 1. [MODIFY] [likelihood.py](file:///mnt/nas/home/hiromi/src/emo/v2/src/likelihood.py)
- `compute_likelihoods_for_candidates(..., normalize_length=False)` に引数を追加し、
  - `normalize_length=False`: `sum(cand_log_probs)` (raw sequence log-likelihood)
  - `normalize_length=True`: `sum(cand_log_probs) / len(cand_log_probs)` (length-normalized log-likelihood)
  を選択可能にする。
- 返り値のタプル `(likelihoods, candidate_lengths)` のドキュメント（docstring）を明確化し、第2返り値が確率ではなくトークン長であることを強調。

#### 2. [MODIFY] [run_causal_localization_sweep.py](file:///mnt/nas/home/hiromi/src/emo/v3/scripts/run_causal_localization_sweep.py)
- `evaluate_layer_causal_effect()` 内の EMD 計算バグを修正:
  ```python
  # 修正前
  l_peak, p_peak = compute_likelihoods_for_candidates(...)
  ev_peak, _, _, _, _ = compute_expected_va(l_peak, va_pairs)
  ...
  p_source_mat = np.array(p_peak).reshape(9, 9)

  # 修正後
  l_peak, _ = compute_likelihoods_for_candidates(...)
  ev_peak, _, _, _, p_peak = compute_expected_va(l_peak, va_pairs)
  ...
  p_source_mat = np.array(p_peak).reshape(9, 9)
  ```
- Neutral および Patch の呼び出しについても同様に `compute_expected_va` から返される確率分布 `p_neut`, `p_patch` を使用。
- コメントおよび出力表示において、プローブが「Peak-vs-Neutral intensity indicator (peak=1, neutral=0)」であることを明記。
- 長さ正規化フラグ `--normalize-length` オプションを追加（デフォルトは False）。

#### 3. [MODIFY] [run_within_model_positive_control.py](file:///mnt/nas/home/hiromi/src/emo/v3/scripts/run_within_model_positive_control.py)
- 同様に EMD 計算部分で `compute_expected_va` から返される確率分布 `p_peak, p_neut, p_patch` を使用するように修正。

---

### B. 論文原稿改訂

#### 4. [MODIFY] [paper.md](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper.md)
ユーザー提示の構成案を忠実に反映し、以下の章立てと内容で全面的に書き直します。

- **Title**: Decodability Without Causal Sufficiency: A Case Study of Affect-Relevant Representations in a Paired Base/Instruct Language Model
- **Abstract**:
  - representation-use distinctionのケーススタディとしての位置づけ
  - 4つの観点：(1)表層的中立化と尤度分布の刺激感応性、(2)中間層からの線形decodability（Peak-vs-Neutral indicator）、(3)Base→Instructの予測アライメント（CKA 0.83, Top-1 retrieval 86.6%）、(4)Mahalanobis解析によるCenter Collapse（$D_M=39.6 \rightarrow 8.3$）
- **§1. Introduction**:
  - Decodability $\not\Rightarrow$ Behavioral Use
  - Motivation: 一人称VA報告の中立化に対する4つの競合説明
  - Research Questions (RQ1–RQ5)
- **§2. Construct Definition**:
  - Reader-rated text affect / Third-person affect recognition / Constrained first-person report distribution の3区分
  - first-person の文法・課題的性格（主観的感情・内省の完全な排除）
  - affect-relevant representation の定義
- **§3. Experimental Setup**:
  - Qwen2.5-1.5B Base/Instruct
  - AIPsy-Affect Strict Expanded (192 pair groups, 422 samples) & 3-way split (Train / Alignment-dev / Held-out test)
- **§4. Constrained Report Measurement**:
  - 81候補尤度プロトコルの定式化
  - raw sequence log-likelihood と length-normalized score の明示的な区別
- **§5. Results**:
  - 5.1 Greedy Neutralization Does Not Imply Distributional Invariance
  - 5.2 Affect-Related Conditions Are Linearly Decodable (Peak-vs-Neutral indicator $R^2=0.548$)
  - 5.3 Base and Instruct Representations Are Predictively Alignable (CKA 0.829, Retrieval 86.6%)
  - 5.4 Predictive Alignment Does Not Guarantee Distributional Typicality (Center Collapse, $\alpha$-sweep)
- **§6. Causal Intervention: Required Validation**:
  - 尤度由来の確率分布に基づく EMD / Recovery の定式化
  - 今後再検証すべきコンポーネント（MLP, residual, attention, prompt-final, response tokens）
  - sufficiency と necessity の両面評価要件
- **§7. Discussion**:
  - 7.1 Decodability Is a Representational Claim, Not a Mechanistic Claim
  - 7.2 Predictive Alignment Is Not Functional Equivalence
  - 7.3 What This Study Does Not Claim
- **§8. Limitations**
- **§9. Conclusion**

---

## 4. 検証計画

### 自動検証
1. **構文チェック & インポートテスト**:
   - `python -m py_compile v2/src/likelihood.py`
   - `python -m py_compile v3/scripts/run_causal_localization_sweep.py`
   - `python -m py_compile v3/scripts/run_within_model_positive_control.py`
2. **単体機能検証（CPUドライラン）**:
   - `likelihood.py` における `compute_likelihoods_for_candidates` の `normalize_length` フラグの動作確認
   - `compute_expected_va` から確率分布 `probs` が正しく返り、合計が 1.0 に正規化され、9×9 に reshape されて EMD 関数に渡せることのスクラッチ検証

### レビュー確認
- 論文原稿（`paper.md`）の整合性確認（ユーザー提示の英文・和文構造、数値データ表の完全一致）。
