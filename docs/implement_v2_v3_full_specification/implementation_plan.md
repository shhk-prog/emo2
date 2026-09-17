# 実装計画: V2・V3 再定義および複数モデルファミリー対応 全仕様実装

## 目的
ユーザーから提示された「V2・V3 再定義および複数モデルファミリー対応 実装計画書（最終仕様固定版）」に基づき、前回の監査で特定された未実装・重大バグ（RQ3/RQ4のSequence-Likelihood未実装、RQ4実パッチング未実装、RQ1/RQ2のArousal・GroupSplit・matched_plain未対応等）をすべて解消し、完全な仕様として本番実装を行う。

---

## 修正・実装コンポーネント詳細

### 1. 共通基盤ライブラリ (`src/affective_empathy_eval/`)
#### [MODIFY] `src/affective_empathy_eval/likelihood.py`
- **真の Sequence-Likelihood 計算の実装**:
  - `compute_sequence_likelihoods_for_candidates(model, tokenizer, prompt, candidates, device, normalize_length=False)` を追加。
  - プロンプトに続く81候補 JSON 列（`{"valence": v, "arousal": a}`）の各トークンの条件付き対数尤度の和 $\sum_t \log P(token_t \mid prompt, cand_{<t})$ を計算。
  - バッチ化による高速計算に対応しつつ、厳密な対数確率分布を出力。
- **期待値算出のシームレスな統合**:
  - 対数尤度配列から Softmax 分布を経て $E[V], E[A]$ を算出する関数を完備。

#### [MODIFY] `src/affective_empathy_eval/interventions.py`
- **Centered Projection Removal の実装**:
  - `apply_centered_projection_removal_1d(h, mu_neu, d)`:
    $$h' = h - [(h - \mu_{\text{neu}})^\top d] d$$
  - `apply_centered_projection_removal_subspace(h, mu_neu, P)`:
    $$h' = h - P (h - \mu_{\text{neu}}) \quad (P = Q Q^\top)$$
- 既存の `extract_conditional_directions` (重回帰), `compute_orthonormal_subspace` (QR分解), `compute_causal_leverage` との整合確認。

---

### 2. V2-RQ3 因果マップ (`v2/scripts/run_v2_2x2_causal_map.py`)
#### [MODIFY] `v2/scripts/run_v2_2x2_causal_map.py`
- **先頭81ロジット切り出しバグの排除**:
  - `logits_clean[:81]` を完全に廃止。
  - `compute_sequence_likelihoods_for_candidates` を呼び出し、Clean run および Patched run の 81 候補シーケンス対数尤度から期待値 $E[V], E[A]$ を正確に算出。
- **因果変位の厳密計算**:
  - $C_V(l) = |E[V]_{\text{patched}} - E[V]_{\text{clean}}|$, $C_A(l) = |E[A]_{\text{patched}} - E[A]_{\text{clean}}|$
  - 符号付き変位 $C_V^{\text{dir}}, C_A^{\text{dir}}$ も記録。
- **重み非負化重心と解離指標の出力**:
  - $\bar{d}_C = \frac{\sum_l d_l \max(C_l, 0)}{\sum_l \max(C_l, 0)}$ とピーク深度 $d_C^* = \arg\max_d C(d)$ の算出。
  - Decodability との解離 $\Delta d^* = d_C^* - d_D^*, \Delta \bar{d} = \bar{d}_C - \bar{d}_D$ の記録。

---

### 3. V2-RQ4 分布回復パッチング (`v2/scripts/run_v2_recovery_patching.py`)
#### [MODIFY] `v2/scripts/run_v2_recovery_patching.py`
- **先頭81ロジット切り出しバグの排除**:
  - `out.logits[0, -1, :81]` を完全に廃止し、正規の 81 状態 Sequence Likelihood による同時確率分布 $P(V, A)$ を算出。
- **Base $\rightarrow$ Instruct 実パッチングの実装**:
  - `ActivationHookManager` を用い、Base モデルの各層 $l$ の残差ストリーム活性化（`POST_MLP_RESID`）を抽出し、Instruct モデルの層 $l$ に注入する本物の介入処理を実装。
- **分布間距離と回復率の正確な算出**:
  - 81状態同時分布間の 2D EMD ($EMD_{VA}$)、周辺 $W_1(V), W_1(A)$、JSD の算出。
  - $\text{Recovery}(l) = \frac{EMD_0 - EMD_l}{EMD_0}$ の正確な計算。

---

### 4. V2-RQ1 & RQ2 幾何・交差デコード (`v2/scripts/run_v2_2x2_cross_decoding.py`)
#### [MODIFY] `v2/scripts/run_v2_2x2_cross_decoding.py`
- **`pair_id` に基づく Group Split の実装**:
  - データセットに `pair_id` 列が存在する場合は、`GroupKFold` または pair 単位のハッシュ/シャッフルで train (70%) / test (30%) に分割し、同一ヴィネットの漏洩を防止。
- **Arousal (`reader_A`) の完全評価**:
  - Valence ($V$) と Arousal ($A$) の双方について、Held-out 線形プローブ $R^2$、Cross-decoding $R^2$ を算出。
- **`matched_plain` フォーマット条件の追加**:
  - Instruct モデルに対する `native` (chat template) と `matched_plain` (plain text completion) の両方を評価可能にする。
- **差の差（Difference-in-Differences）と Bootstrap CI の統合**:
  - $\Delta\Delta_{\text{post}} = (R^2_{IR \to IS} - R^2_{BR \to BS}) - (R^2_{IS \to IS} - R^2_{BS \to BS})$ 等の定義を明記し、`compute_bootstrap_ci` を接続。

---

### 5. 単体テストの拡充と検証 (`tests/`)
#### [MODIFY] `tests/test_likelihood.py`
- 81候補 Sequence-Likelihood 計算関数の正確性、確率正規化、期待値算出のテストを追加。
#### [MODIFY] `tests/test_interventions.py`
- Centered Projection Removal（1D および 2D サブスペース）の直交性・ノルム減少テストを追加。

---

## 検証手順
1. `PYTHONPATH=src .venv/bin/pytest tests/` を実行し、既存テストおよび新規テストの全パス（100% Green）を確認。
2. `python v2/scripts/run_v2_2x2_cross_decoding.py --dry-run` を実行し、Valence/Arousal/GroupSplit の動作を確認。
3. `python v2/scripts/run_v2_2x2_causal_map.py --dry-run` を実行し、Sequence-Likelihood/重心非負化の動作を確認。
4. `python v2/scripts/run_v2_recovery_patching.py --dry-run` を実行し、回復率計算の動作を確認。
5. `walkthrough.md` に全変更内容と検証結果を記録。
