# 再実行前7点修正 — 実装計画

## 目的

レビューで指摘された7点を直し、正式再実行に進める状態にする。
実験の主仮説・指標定義の「中身」は変えない。危険なのは、逆順実行・楽観的時間・静かな別モデル実行・正本以外のモデルID・旧コホート表記・件数の手書きである。

## 方針

1. **実行順**: 統合 CLI (`affective_empathy_eval.run`) の `--stage all` を `run_production_all.sh` と同じ `Behavioral → V1 → V2 → V3` に固定する。定数 `PRODUCTION_STAGE_ORDER` を正本にする。
2. **所要時間**: 固定レンジ（30分〜1時間 等）を削除する。実測は Qwen 1 family の benchmark 後に
   \(T_{\mathrm{total}} \approx \sum_{\mathrm{family}} T_{\mathrm{family}}\) で更新する。
3. **V1 モデル選択**: `--model-id` の Qwen default を廃止する。`--model-id` または `--family` が無い単独実行は例外とする。統合 runner は従来どおり明示的に `--model-id` を渡す。
4. **V3 fallback**: registry に family が無い場合は Qwen 文字列へ落とさず `KeyError` を送出する。`target_family` も registry 解決に限定する。
5. **モデル正本**: `configs/scale_validation.yaml` から `base` / `instruct` を削除し、`model_set: scale_validation` のみ残す。ID は `configs/models.yaml` のみ。
6. **文書**: V2 Primary の Gemma 2 / Mistral 表記を Qwen / Llama / Gemma 3 / OLMo 2 に更新する。Path Mediation は mediated attenuation。Topic control は Topic 課題への非特異的摂動が小さいことを確認する control。
7. **件数と results**: 件数は CSV 実ロード数をログする。再実行前に stage 配下の results を `.gitkeep` 以外削除する。

## 変更しないもの

- `data/raw/` および刺激抽出規則
- 主指標の定義式そのもの（ADA、相対深度式、matched-neutral baseline）
- legacy 論文ドラフトの歴史的記述（現行導線の README / primary docstring のみ更新）
- 本番フル実験の実行（本作業ではテストと構文チェックまで）

## 検証

- `.venv` で `pytest -q`、`ruff check`（対象ファイル）
- Primary スクリプトの `py_compile`
- `--stage all` の順序定数テスト、V1 単独実行の必須化テスト、V3 未知 family の `KeyError` テスト
