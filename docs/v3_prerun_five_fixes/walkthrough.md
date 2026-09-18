# Walkthrough: 正式再実行前の V3 5点修正

## 1. データセット

`configs/v3_experiments.yaml` の既定を AIPsy 4-split にした。`load_v3_matched_pair_table` が clinical–neutral 192 pair を wide 化する。人工中立文は削除し、中立が無い場合は `ValueError`。necessity baseline は matched-neutral 文のモデル自己報告。

EmoBank 3-way を渡すとテストで拒否される。

## 2. RQ2 patch 位置

`resolve_joint_stage_index` は `cand_start + stage_offset` だけを返す。`prompt_end` への `min` は無い。範囲外はエラー。尤度計算は `generation_patch` で同じ絶対位置に hook する。

## 3. V/A 分離と β

RQ1 / RQ2 / Confirmatory Sufficiency は $d_V$ 注入と $d_A$ 注入を分け、対応する軸の変位だけを読む。`beta_*` は符号付き、`abs_beta_*` を併記。

## 4. Go/No-Go

`run.py` は RQ1 のあと `v3_gate_decision.json` を読む。完全一致の `GO` だけ継続。`NO_GO` および軸片方の GO は終了コード 2。`--force-after-no-go` のみ継続。`run_production_v3.sh` も同じフラグを渡せる。

## 5. 候補空間

コードは変更していない。decision_log と Root / V3 README で、Behavioral/V1 = 729 VAD、V2/V3 = 81 VA、直接比較しないと固定した。

## 論文

`paper_outline.md` に現行 README 準拠の構成を書いた。旧 tex の結果は使っていない。

## 検証

この macOS checkout の `.venv` は Linux 向けのため、`PYTHONPATH=src` と `/opt/anaconda3/bin/python` で確認した。

- `compileall`: 成功
- `pytest -q tests`: 42 passed, 1 skipped
- `pytest -q tests v1/tests v3/tests`: 57 passed, 2 skipped
- `bash -n scripts/run_production_v3.sh`: 成功
- 編集ファイルの UTF-8: 成功

GPU 本番は起動していない。
