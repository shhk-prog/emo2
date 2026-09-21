# BPEマージによるプレフィックス検証失敗の修正（第3版）

## 背景・問題

`results/logs/production_v2_20260921_120717.log` の `run_rq3_causal_map.py` 実行時に以下のエラーが発生し、パイプラインが停止した。

```
ValueError: Strict prefix property violated across boundary for candidate:
{"valence":1,"arousa...
Prompt ids (98): [2582, 25, 220], Full ids prefix: [2582, 25, 5212]
```

## 根本原因

`build_prompt` の `plain` フォーマットはプロンプト末尾を `"Response: "`（末尾スペース）で終わらせる。

Qwen2.5-1.5Bの BPE トークナイザーは：
- `"Response: "` 単独 → 末尾トークン `220`（空白）
- `"Response: {"` → スペース + `{` が **BPEマージ** → トークン `5212`（異なる）

これにより `prepare_joint_sequence_with_boundary` の厳密プレフィックス検証が失敗する。

## 採用しない方針と理由

| 方針 | 却下理由 |
|---|---|
| `require_strict_prefix=False` | length-normalized likelihood の分母が曖昧になる。論文用 metric として不適切 |
| `build_prompt()` の変更 | Behavioral/V1/V2/V3 の既存プロンプトテキストが変わり、過去結果との比較条件が崩れる |

## 修正方針：Boundary Canonicalization（境界正規化）

`likelihood.py` 内に `canonicalize_prompt_candidate_boundary()` を追加し、
`compute_sequence_likelihoods_for_candidates` の内部でのみ適用する。

### 動作

```text
元の入力:
  prompt    = "...Response: "
  candidate = '{"valence":1,"arousal":1}'

scoring 時（内部のみ）:
  scoring_prompt    = "...Response:"
  scoring_candidate = ' {"valence":1,"arousal":1}'

連結結果: どちらも完全に同一
  "...Response: {"valence":1,"arousal":1}"
```

### Token-level metric 境界の再定義について

> [!IMPORTANT]
> 「モデルへの入力テキスト」および「連結後の完全入力列」は変化しない。
> ただし、**token-level continuation boundary は変化する**。
>
> 元の実装では `" {"` が prompt 末尾トークンとして隠れる形でマージされていたが、
> canonicalize 後は `" {"` が candidate 側の先頭トークンとして length normalization
> の分子・分母に明示的に含まれる。
>
> これは「影響なし」ではなく、**token-level continuation boundary を明示的に再定義する変更**である。
> 旧実装（BPEマージでクラッシュする実装）との数値的一致は保証しない。
> 変更後の定義を Behavioral/V1/V2/V3 の全 Stage・全モデルで統一して用いる。

---

## 変更内容

### [MODIFY] [`likelihood.py`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/likelihood.py)

#### 1. 追加する関数 `canonicalize_prompt_candidate_boundary`

`prepare_joint_sequence_with_boundary` の直前（line 302）に追加。

```python
def canonicalize_prompt_candidate_boundary(
    prompt: str,
    candidate: str,
) -> tuple[str, str]:
    """
    プロンプト末尾の trailing ASCII space を候補先頭へ移動し、
    BPEマージによるプレフィックス検証失敗を防ぐ境界正規化を行う。

    連結後の完全文字列は保存される（scoring_prompt + scoring_candidate == prompt + candidate）。
    build_prompt() の戻り値やモデルへの入力テキストは変更しない。

    token-level continuation boundary は再定義される：
    canonicalize 後は trailing whitespace を含む先頭トークン（例: ' {'）が
    candidate 側の最初のトークンとして length normalization の対象に含まれる。
    旧実装（マージによりクラッシュ）との数値的一致は保証しない。
    本定義を全 Stage・全モデルで統一して用いる。

    例:
        prompt    = "...Response: "
        candidate = '{"valence":1,"arousal":1}'
        → scoring_prompt    = "...Response:"
        → scoring_candidate = ' {"valence":1,"arousal":1}'
    """
    n = len(prompt) - len(prompt.rstrip(" "))
    if n == 0:
        return prompt, candidate
    suffix = prompt[-n:]
    assert prompt[:-n] + suffix + candidate == prompt + candidate
    return prompt[:-n], suffix + candidate
```

#### 2. `compute_sequence_likelihoods_for_candidates` の修正

lines 449–458 のループ内、`prepare_joint_sequence_with_boundary` 呼び出し前に境界正規化を適用。

```diff
     for c in cand_strings:
+        scoring_prompt, scoring_c = canonicalize_prompt_candidate_boundary(prompt, c)
+        assert scoring_prompt + scoring_c == prompt + c, (
+            "Boundary canonicalization violated text invariance"
+        )
         full_ids, c_start = prepare_joint_sequence_with_boundary(
-            prompt=prompt,
-            candidate=c,
+            prompt=scoring_prompt,
+            candidate=scoring_c,
             tokenizer=tokenizer,
             delimiter=delimiter,
             require_strict_prefix=True,
         )
```

`require_strict_prefix=True` は**維持**。

---

### [MODIFY] [`test_likelihood.py`](file:///mnt/nas/home/hiromi/src/emo2/tests/test_likelihood.py)

既存テストは変更しない。以下の4テストを末尾に追加する。

#### テスト1: 文字列不変性（モック不要・高速）

```python
def test_canonicalize_prompt_candidate_boundary_text_invariance():
    from affective_empathy_eval.likelihood import canonicalize_prompt_candidate_boundary

    cases = [
        ("Response: ",  '{"valence":1,"arousal":1}'),
        ("Response:  ", '{"valence":1,"arousal":1}'),  # 複数スペース
        ("Response:\n", '{"valence":1,"arousal":1}'),  # 末尾が改行: 変化なし
        ("Response:",   '{"valence":1,"arousal":1}'),  # スペースなし: 変化なし
    ]
    for prompt, candidate in cases:
        p2, c2 = canonicalize_prompt_candidate_boundary(prompt, candidate)
        assert p2 + c2 == prompt + candidate

    p2, c2 = canonicalize_prompt_candidate_boundary("Response: ", '{"valence":1,"arousal":1}')
    assert p2 == "Response:"
    assert c2 == ' {"valence":1,"arousal":1}'
```

#### テスト2: no-op確認（末尾スペースなし・改行）

```python
def test_canonicalize_no_op_when_no_trailing_space():
    from affective_empathy_eval.likelihood import canonicalize_prompt_candidate_boundary

    p2, c2 = canonicalize_prompt_candidate_boundary("Response:", '{"valence":5,"arousal":5}')
    assert p2 == "Response:"
    assert c2 == '{"valence":5,"arousal":5}'

    # 改行は strip しない
    p3, c3 = canonicalize_prompt_candidate_boundary("Response:\n", '{"valence":5,"arousal":5}')
    assert p3 == "Response:\n"
    assert c3 == '{"valence":5,"arousal":5}'
```

#### テスト3: モック BPE tokenizer で strict-prefix 成立確認

```python
def test_canonicalize_enables_strict_prefix_with_bpe_merge():
    """
    ' {' がマージされる Qwen-style tokenizer モックで、
    canonicalize 後に require_strict_prefix=True が通ることを検証。
    """
    import pytest
    from affective_empathy_eval.likelihood import (
        canonicalize_prompt_candidate_boundary,
        prepare_joint_sequence_with_boundary,
    )

    class QwenStyleTokenizer:
        pad_token_id = 0
        eos_token_id = 1
        _MERGES = {" {": 5212}

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            tokens = []
            i = 0
            while i < len(text):
                if i + 2 <= len(text) and text[i:i+2] in self._MERGES:
                    tokens.append(self._MERGES[text[i:i+2]])
                    i += 2
                else:
                    tokens.append(ord(text[i]))
                    i += 1
            return tokens

    tok = QwenStyleTokenizer()
    prompt = "Response: "
    candidate = '{"valence":1,"arousal":1}'

    # canonicalize なし → strict prefix 違反
    with pytest.raises(ValueError, match="Strict prefix property violated"):
        prepare_joint_sequence_with_boundary(
            prompt=prompt, candidate=candidate, tokenizer=tok, require_strict_prefix=True
        )

    # canonicalize 後 → strict prefix 成立
    p2, c2 = canonicalize_prompt_candidate_boundary(prompt, candidate)
    assert p2 + c2 == prompt + candidate   # 文字列不変
    full_ids, c_start = prepare_joint_sequence_with_boundary(
        prompt=p2, candidate=c2, tokenizer=tok, require_strict_prefix=True
    )
    prompt_ids = tok.encode(p2)
    assert full_ids[:len(prompt_ids)] == prompt_ids
    assert c_start == len(prompt_ids)
```

#### テスト4: 実 tokenizer × 全候補 Integration Test（`@pytest.mark.slow`）

```python
@pytest.mark.slow
def test_canonicalize_all_candidates_strict_prefix_real_tokenizer():
    """
    Qwen2.5-1.5B 実 tokenizer を使い、81 VA 候補 + 729 VAD 候補すべてで
    canonicalize 後に require_strict_prefix=True が通ることを検証する。

    - strict prefix failure = 0
    - 全候補で連結文字列が完全一致
    - canonicalization 後も p + c == original_prompt + original_candidate

    モデルウェイトは不要（tokenizer のみ）。
    """
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer
    from affective_empathy_eval.likelihood import (
        build_va_candidates,
        build_vad_candidates,
        canonicalize_prompt_candidate_boundary,
        prepare_joint_sequence_with_boundary,
    )

    model_id = "Qwen/Qwen2.5-1.5B"
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        pytest.skip(f"Tokenizer not available: {e}")

    prompt = "Text: Hello world.\n\nRate the emotional state of the reader.\nRespond in JSON:\n{\"valence\": <int 1-9>, \"arousal\": <int 1-9>}\n\nResponse: "

    all_candidates = (
        [c["json_str"] for c in build_va_candidates()] +
        [c["json_str"] for c in build_vad_candidates()]
    )
    assert len(all_candidates) == 81 + 729

    failures = []
    for candidate in all_candidates:
        p2, c2 = canonicalize_prompt_candidate_boundary(prompt, candidate)
        # 連結文字列不変
        assert p2 + c2 == prompt + candidate, (
            f"Text invariance violated for candidate: {candidate[:30]}"
        )
        try:
            full_ids, c_start = prepare_joint_sequence_with_boundary(
                prompt=p2,
                candidate=c2,
                tokenizer=tok,
                require_strict_prefix=True,
            )
        except ValueError as e:
            failures.append(f"{candidate[:30]}: {e}")

    assert failures == [], (
        f"strict prefix failed for {len(failures)} candidates:\n" + "\n".join(failures[:5])
    )
```

---

### [MODIFY] [`docs/decision_log.md`](file:///mnt/nas/home/hiromi/src/emo2/docs/decision_log.md)

既存行の下に以下を追加する。

| 日付 | 変更者 | 対象 | 変更前 | 変更後 | 理由 |
|---|---|---|---|---|---|
| 2026-09-21 | agent | token-level continuation boundary 定義 | 未定義（Qwen2.5でBPEマージによりクラッシュ） | `canonicalize_prompt_candidate_boundary` を導入し、prompt 末尾 trailing whitespace を candidate 先頭へ移動した上で strict prefix を検証。連結文字列は不変。token-level boundary は再定義（`" {"` 等が candidate 側 token として length normalization に含まれる）。旧実装との数値的一致は保証しない。全 Stage・全モデルで統一適用。 | Qwen2.5 BPEマージによる `ValueError` でパイプライン停止。require_strict_prefix=True を維持しつつ length-normalized metric の分母を明示的に定義するため。 |

---

## 影響範囲と再実行方針

> [!WARNING]
> この変更は token-level continuation boundary の **再定義**である。
> Behavioral/V1/V2/V3 すべてにわたって `compute_sequence_likelihoods_for_candidates` を使う実験が対象となる。
> **既存のキャッシュ・raw results を継承せず、全 Stage を fresh run すること。**

| 項目 | 評価 |
|---|---|
| `build_prompt()` | **変更なし** |
| モデルへの連結入力テキスト | **変更なし**（`p2 + c2 == prompt + candidate` をassertionで保証） |
| `require_strict_prefix=True` | **維持** |
| token-level likelihood の分母 | **再定義あり**（trailing space が candidate 側トークンに含まれる） |
| 過去結果との数値的一致 | **保証しない** |
| 再実行対象 | Behavioral / V1 / V2 / V3 全 Stage |

---

## 検証計画

### 自動テスト（モックのみ、CI で必須）
```bash
source .venv/bin/activate
python -m pytest tests/test_likelihood.py -q
```

### 統合テスト（実 tokenizer、slow マーク）
```bash
python -m pytest tests/test_likelihood.py -q -m slow
```

### lint / format
```bash
python -m ruff check src/affective_empathy_eval/likelihood.py tests/test_likelihood.py
python -m ruff format --check src/affective_empathy_eval/likelihood.py tests/test_likelihood.py
```
