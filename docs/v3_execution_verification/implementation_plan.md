# 実装計画: V3 期待値計算の修正・全件監査およびRQ1完全再実行 (改訂版)

## 概要
直近の V3 パイプライン実行で生じた `NO_GO` 判定は、科学的な negative result ではなく、尤度 API の戻り値受取と期待値計算における「二重 Softmax 適用」による期待値の一様平滑化バグ（効果量の微小化・消失）が原因でした。
本計画では、Gate 閾値や実験条件を一切変更せず、**期待値計算関数の分離・仕様明確化**、**リポジトリ全体（behavioral, v1, v2, v3, src, tests）のコールサイト監査・修正**、**4 つの単体テスト（回帰テスト含む）追加**、**手計算 Softmax との厳格な一致検証（Sanity Check）**、**キャッシュ・マニフェスト退避**、および **RQ1 からの完全再実行** を実施します。

---

## 1. リポジトリ全体のコールサイト監査 (behavioral, v1, v2, v3, src, tests)
V3 のプライマリコードだけでなく、リポジトリ全体で以下のパターンを監査・統一します。

```bash
grep -R "compute_expected_va(" behavioral v1 v2 v3 src tests --include="*.py"
grep -R "compute_sequence_likelihoods_for_candidates" behavioral v1 v2 v3 src tests --include="*.py"
```

各コールサイトについて：
```python
log_likelihoods, probs = compute_sequence_likelihoods_for_candidates(...)
```
で受け取り、
- 未正規化対数尤度を使う場合は必ず `compute_expected_va(log_likelihoods, candidates)`
- 既に正規化された確率を使う場合は必ず `compute_expected_va_from_probs(probs, candidates)`
のいずれかに厳格に統一します。

特に最優先対象：
- `v3/primary/run_rq1_state_induction.py`
- `v3/primary/run_rq2_spatiotemporal_maps.py`
- `v3/primary/run_rq3_path_mediation.py`
- `v3/primary/run_confirmatory_replication.py`

---

## 2. `likelihood.py` の修正と新設
自動判定方式（`0<=x<=1 and sum~1`）は採用せず、入力の型と意味を API レベルで厳格に分離します。

1. **`compute_expected_va(log_scores, candidates=None)`**:
   - 引数名を `log_scores` に統一し、未正規化の対数尤度（log-likelihood / log-scores）専用であることを docstring に明記。
   - `np.exp(log_scores - np.max(log_scores))` により Softmax を 1 回適用して期待値 $(E[V], E[A])$ を算出。
2. **`compute_expected_va_from_probs(probs, candidates)`** の新設:
   - 既に Softmax 済みの確率分布を受け取る関数。
   - `np.isclose(probs.sum(), 1.0, atol=1e-6)` による確率和 1.0 の厳格バリデーション（満たさない場合は `ValueError`）。
   - `vals = np.asarray([[c["valence"], c["arousal"]] for c in candidates], dtype=np.float64)`
   - `expected = probs @ vals` による直接の内積で $(E[V], E[A])$ を算出。
3. `affective_empathy_eval/__init__.py` で `compute_expected_va_from_probs` をエクスポート。

---

## 3. 単体テストの追加 (`tests/test_likelihood.py`)
以下の 4 つのテスト + API 戻り値仕様テストを追加します。

1. **Test 1 (一様分布)**:
   - 81 候補が完全一様（`log_scores = np.zeros(81)`）なら $E[V]=5.0, E[A]=5.0$ となること。
2. **Test 2 (極端集中)**:
   - $V=9, A=9$ に集中（`log_scores[idx_9_9] = 0.0`, 他は `-100.0`）している場合、$E[V] > 8.9, E[A] > 8.9$ となること。
3. **Test 3 (log score 経由と probability 経由の数値一致)**:
   - 最重要テスト。`probs = softmax(log_scores)` のとき、`compute_expected_va(log_scores, candidates)` と `compute_expected_va_from_probs(probs, candidates)` の出力が `np.allclose` で完全に一致すること。
4. **Test 4 (二重 Softmax 検出の回帰テスト)**:
   - 関数名: `def test_regression_double_softmax_distorts_expected_va():`
   - 偏りのある確率分布 `probs` に対して、誤って `compute_expected_va(probs, candidates)` を呼んだ場合、真の期待値 `compute_expected_va_from_probs(probs, candidates)` と大きく乖離することを回帰テストとして明示（「これは正しい API 利用ではない」というコメントを付与）。
5. **Test 5 (API 返り値アンパック整合性)**:
   - `compute_sequence_likelihoods_for_candidates` の返り値が `(log_likelihoods, probs)` のタプルであり、それぞれの形状とプロパティを検証。

---

## 4. Cache / Manifest Versioning and Archival
過去のバグ結果およびキャッシュが再利用されないよう、確実に退避・無効化します。

1. **結果ディレクトリの退避**:
   ```bash
   mkdir -p archive/results_v3_double_softmax_bug_20260920
   cp -a v3/results/raw archive/results_v3_double_softmax_bug_20260920/
   cp -a v3/results/derived archive/results_v3_double_softmax_bug_20260920/
   rm -f v3/results/raw/v3_rq1_results.json
   rm -f v3/results/derived/v3_gate_decision.json
   ```
2. **Cache / Manifest 再利用の防止**:
   - パイプラインが旧結果をキャッシュ再利用しない状態（新規実行として扱われる状態）を担保。

---

## 5. 少数サンプル Sanity Check (Qwen 8〜16 サンプル)
Qwen 2.5 1.5B Instruct を用いて、8〜16 サンプルの Sanity Check を実行します。

### 合否判定基準
- **合否基準**: 手計算 Softmax 期待値との完全一致
  \[
  E_{\text{implementation}} = \sum_i \operatorname{softmax}(\ell)_i \cdot VA_i
  \]
  が `np.allclose(ev, manual_ev)` かつ `np.allclose(ea, manual_ea)` で一致すること。
- ※「0.1 以上動く」や「5.0000x にならない」は参考値（モデルの真の出力が中立なら 5 付近になることもあり得るため合否基準にはしない）。
- 各サンプルについて以下のテーブルを出力・記録：
  `sample_id`, `clean_aff_ev`, `clean_neu_ev`, `clean_aff_ea`, `clean_neu_ea`, `natural_shift_v`, `natural_shift_a`, `manual_ev`, `manual_ea`

---

## 6. Gate 閾値の維持と本番完全再実行

- `configs/v3_experiments.yaml` の Gate 閾値（`min_specificity_diff: 0.05`, `min_necessity_attenuation: 0.05`, `max_topic_tvd: 0.15` 等）は**一切変更せず維持**。
- `bash scripts/run_production_v3.sh cuda:0` を実行し、RQ1 から完全再実行。
- 修正後の Gate 判定結果をそのまま科学的結果として受け入れる：
  - **GO の場合**: RQ2 → RQ3 → Confirmatory Replication へ進む。
  - **NO_GO の場合**: 「内部表現はデコード可能だが、事前定義された因果的十分性ゲート（causal sufficiency criterion）を満たさなかった」という正当な科学的 negative result（$\text{representation} \neq \text{causal utilization}$）として停止を尊重。

---

## 7. 修正後の実行順序
1. `pytest -q tests/test_likelihood.py`
2. `pytest -q tests/test_v3*`
3. `pytest -q -m "not slow"`
4. 8〜16 件の Qwen sanity run & 手計算 Softmax との一致確認
5. 旧 V3 結果/cache を archive へ退避
6. `bash scripts/run_production_v3.sh cuda:0` を RQ1 から完全再実行
7. 結果検証と `docs/v3_execution_verification/walkthrough.md` の作成
