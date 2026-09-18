# 再実行前7点修正 — タスクリスト

本番フル再実行の直前レビューで残った修正点を、実験仕様を変えずに潰す。

## 最優先（レビュー指定の7点）

- [x] 1. `--stage all` の実行順を Behavioral → V1 → V2 → V3 に統一
- [x] 2. 本番所要時間の固定目安を削除し、未計測として記録
- [x] 3. V1 Primary の Qwen default `--model-id` を廃止（`--model-id` または `--family` 必須）
- [x] 4. V3 の Qwen silent fallback を `KeyError` 化
- [x] 5. `configs/scale_validation.yaml` のモデルID重複を削除し、`configs/models.yaml` へ一本化
- [x] 6. V2 README / docstring の Gemma 2 / Mistral 表記を Primary コホートへ更新
- [x] 7. `behavioral/results/`, `v1/results/`, `v2/results/`, `v3/results/` を `.gitkeep` 以外クリア

## 関連整理

- [x] Path Mediation 用語を mediated attenuation に統一（現行 docs / README）
- [x] V3 Topic control を「同尺度の差」ではなく非特異的摂動の確認統制として再位置づけ
- [x] V1 Phase B は Primary で相対深度 $d=0.5$ を事前固定することを明記
- [x] V2 RQ1/RQ2/RQ4 の重複 import を整理
- [x] production scripts の件数手書きをやめ、実ロード数をログ出力
- [x] 本番は stage 分割実行を推奨する旨を現行ドキュメントへ反映
