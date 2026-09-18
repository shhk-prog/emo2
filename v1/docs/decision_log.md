# Decision Log

| 日付 | 変更者 | 変更内容 | 理由 |
|---|---|---|---|
| 2026-09-18 | agent | 統合 CLI `--stage all` を Behavioral → V1 → V2 → V3 に固定 | `run_production_all.sh` と逆順だったため |
| 2026-09-18 | agent | 本番所要時間の固定目安を削除（未計測） | GPU 1枚・4 family 直列では楽観的すぎるため。Qwen 1 family の実測後に更新する |
| 2026-09-18 | agent | V1 Primary の Qwen default `--model-id` を廃止 | 単独実行時に静かに Qwen が走るのを防ぐ。`--model-id` または `--family` 必須 |
| 2026-09-18 | agent | V3 未知 family を `KeyError` 化 | Qwen 文字列への silent fallback を禁止 |
| 2026-09-18 | agent | `scale_validation.yaml` からモデルIDを削除 | `configs/models.yaml` を唯一の正本にする |
| 2026-09-18 | agent | V3 Topic control を非特異的摂動の確認統制として再位置づけ | Self VA shift と Topic TVD は同一構成概念ではない |
| 2026-09-18 | agent | Phase B 層選択を $d=0.5$ 事前固定と明記 | Phase A peak を使わない a priori 設計 |
| 2026-09-18 | agent | stage results を `.gitkeep` 以外クリア | 正式再実行前の混在を防ぐ |
