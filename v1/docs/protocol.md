# V1 プロトコル補遺（層選択）

## Phase B: 相対深度の事前固定

Primary 解析では、Phase B のプローブ層を Phase A の decodability peak から選ばない。

- 相対深度を $d = 0.5$ に事前固定する。
- 層番号は $l = \operatorname{round}(d(L-1))$（0-based）でモデルごとに決める。
- これは「中間深度を a priori に固定する」設計であり、結果を見て層を動かさない。
- CLI で `--layer` を明示したときだけ上書きする。その場合は感度分析として記録する。

## Phase B: 統制の性格

`prepare_v1_phase_b_controls.py` の paraphrase / outcome reversal は **rule-based controlled perturbation** である。semantic perturbation とは書かない。

Outcome Reversal で規則置換が当たらない場合の fallback

```text
Fortunately, everything was completely resolved without any harm.
```

は語彙追加が大きい。Outcome Reversal の強い証拠として過大解釈しない。
