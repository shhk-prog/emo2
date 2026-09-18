# V1 プロトコル補遺（層選択）

## Phase B: 相対深度の事前固定

Primary 解析では、Phase B のプローブ層を Phase A の decodability peak から選ばない。

- 相対深度を $d = 0.5$ に事前固定する。
- 層番号は $l = \operatorname{round}(d(L-1))$（0-based）でモデルごとに決める。
- これは「中間深度を a priori に固定する」設計であり、結果を見て層を動かさない。
- CLI で `--layer` を明示したときだけ上書きする。その場合は感度分析として記録する。
