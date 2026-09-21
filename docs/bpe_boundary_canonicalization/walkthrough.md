# Walkthrough: BPE Boundary Canonicalization

**日付:** 2026-09-21  
**対象バグ:** `production_v2_20260921_120717.log` の `run_rq3_causal_map.py` 実行時 `ValueError`

---

## 問題の要約

Qwen2.5-1.5B の BPE トークナイザーが `"Response: "` 末尾の空白と候補 `{` を
マージしてトークンが変化し、`require_strict_prefix=True` の検証が失敗してパイプラインが停止していた。

```
ValueError: Strict prefix property violated across boundary for candidate:
{"valence":1,"arousa...
Prompt ids (98): [2582, 25, 220], Full ids prefix: [2582, 25, 5212]
```

---

## 採用した修正方針

**Boundary Canonicalization**：`likelihood.py` の内部のみで、
prompt 末尾の trailing ASCII space を candidate 先頭へ移動する。

- `require_strict_prefix=True` を維持
- `build_prompt()` は変更しない
- 連結文字列（モデルへの入力）は assertion で不変性を保証
- token-level continuation boundary を明示的に再定義（「影響なし」とは書かない）

---

## 変更ファイル

| ファイル | 変更内容 |
|---|---|
| [`src/affective_empathy_eval/likelihood.py`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/likelihood.py) | `canonicalize_prompt_candidate_boundary` 関数を追加; `compute_sequence_likelihoods_for_candidates` のループ内で適用 |
| [`src/affective_empathy_eval/__init__.py`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/__init__.py) | `canonicalize_prompt_candidate_boundary` を import・`__all__` に追加 |
| [`tests/test_likelihood.py`](file:///mnt/nas/home/hiromi/src/emo2/tests/test_likelihood.py) | 4テスト追加 |
| [`docs/decision_log.md`](file:///mnt/nas/home/hiromi/src/emo2/docs/decision_log.md) | 2026-09-21 エントリを追加 |

---

## 追加テスト

| テスト名 | 種別 | 内容 |
|---|---|---|
| `test_canonicalize_prompt_candidate_boundary_text_invariance` | モック・高速 | 連結文字列が一文字も変わらないことを検証 |
| `test_canonicalize_no_op_when_no_trailing_space` | モック・高速 | 末尾スペースなし・改行のケースで no-op を確認 |
| `test_canonicalize_enables_strict_prefix_with_bpe_merge` | モック・高速 | Qwen-style BPE mock で strict-prefix 成立を確認 |
| `test_canonicalize_all_candidates_strict_prefix_real_tokenizer` | `@pytest.mark.slow` | 実 tokenizer × 810候補（VA81 + VAD729）全数で strict prefix failure = 0 を確認 |

---

## テスト結果

```
tests/test_likelihood.py -q -m "not slow"
21 passed, 1 deselected in 2.03s
```

新規3テストを含む全モックテスト通過。ruff エラーは既存コードのみ（今回追加分は0件）。

---

## 重要な注意事項

> [!WARNING]
> この変更は token-level continuation boundary の **再定義**である。
> 旧実装（クラッシュ）との数値的一致は保証しない。
> **Behavioral/V1/V2/V3 全 Stage を fresh run すること。**
> RQ1/RQ2 は正常完了済みだが、本定義を統一適用するため再実行が必要。

---

## 次のステップ

1. `python -m pytest tests/test_likelihood.py -q -m slow` で実 tokenizer 統合テストを実行
2. 全 Stage（Behavioral / V1 / V2 / V3）を fresh run
3. RQ3 から再実行：  
   `python v2/primary/run_rq3_causal_map.py --models-config configs/models.yaml --model-set primary_small --device cuda:0`
