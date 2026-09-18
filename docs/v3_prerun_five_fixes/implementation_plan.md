# 実装計画: 正式再実行前の V3 5点修正

## 1. データセット

`configs/v3_experiments.yaml` の既定を

`v1/data/processed/aipsy_4split_all.csv`

にする。EmoBank 3-way は pair / matched-neutral を持たないため V3 Primary に使わない。

`load_v3_matched_pair_table()` が clinical–neutral の 192 pair を wide 形式にする。

| 列 | 意味 |
|---|---|
| `text` | 情動側（clinical） |
| `neutral_text` | 同一 `pair_id` の中立側 |
| `pair_id` | 対 ID |
| `condition` | `affective` |

人工文 `"This is a neutral and ordinary statement."` は使わない。中立が取れなければ `ValueError`。
`neutral_expected_*` は CSV に無くてよい。各行の `neutral_text` を通し、モデル自己報告を baseline にする。

人間 `reader_V` / `reader_A` が無い場合、方向推定は train のモデル自己報告、β の共変量は `domain` を使う。5.0 や乱数で埋めない。

## 2. RQ2 patch 位置

活性化抽出・介入とも `prepare_joint_sequence_with_boundary` で joint を作り、

`t = cand_start + stage_offset`

を使う。`min(t, prompt_len-1)` は削除。範囲外はエラー。
尤度計算の forward も同じ絶対位置で hook する。

## 3. V/A 介入分離

- `d_V` 注入 → `gamma_V`, `C_V`
- `d_A` 注入 → `gamma_A`, `C_A`

RQ1 の用量反応と Confirmatory Sufficiency も Valence sweep と Arousal sweep を分ける。
ゲートの `dose_pass_a` は $d_A$ 注入への応答であり、$d_V$ 注入時の Arousal 変化ではない。
β は `beta_V`（符号付き）と `abs_beta_V` を両方保存する。

## 4. Go/No-Go

`run.py` の V3 は RQ1 のあと `v3/results/derived/v3_gate_decision.json` を読む。

- `GO` → RQ2 / RQ3 / Confirmatory
- `NO_GO` → 終了コード 2
- `--force-after-no-go` のときだけ継続

## 5. 候補空間

コードは変えない。役割を固定する。

- Behavioral / V1 Primary: 729 VAD（Dominance を含む 3 軸測定）
- V2 / V3 Primary: 81 VA
- 729 空間の E[V], E[A] と 81 空間のそれを同一尺度として比較しない

## 6. 論文

`docs/v3_prerun_five_fixes/paper_outline.md` に現行設計の構成を書く。
旧 tex の結果・旧 4 family・「劇的増幅」は再実行前に使わない。
