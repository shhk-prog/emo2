# Stage README 詳細化 — 実装計画

## 目的

Behavioral / V1 / V2 / V3 の README を、現行の Primary 実装・モデル正本・用語に合わせて詳細化する。
実験コードは変えない。文書だけを更新する。

## 方針

- 正本は `*/primary/` と統合 CLI。legacy スクリプトは補助として明示する。
- モデルは `configs/models.yaml` の `primary_small`（Qwen 2.5 / Llama 3.2 / Gemma 3 / OLMo 2）。Mistral 7B は scale validation。
- 件数は手書き固定値にしない。実 CSV をロードした件数をログする、と書く。
- 所要時間の固定目安は書かない。
- 「モデルが感情を経験する」等の断定は避ける。
- V3 の主指標名は mediated attenuation。NDE/NIE は使わない。
- Topic control は非特異的摂動の確認統制であり、Self VA shift と同尺度の効果量ではない。
