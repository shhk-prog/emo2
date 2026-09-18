# Stage README 詳細化 — Walkthrough

Behavioral / V1 / V2 / V3 の README を、現行 Primary 実装に合わせて書き直した。コードは変更していない。

## 共通で直した点

- 正本を `*/primary/` と統合 CLI に揃えた
- コホートを Qwen 2.5 / Llama 3.2 / Gemma 3 / OLMo 2 にした
- 件数・所要時間の固定値を書かない
- 「感情を経験する」表現を避けた
- V3 は mediated attenuation。Topic control は非特異的統制

## ファイル

- [`behavioral/README.md`](../../behavioral/README.md)
- [`v1/README.md`](../../v1/README.md)
- [`v2/README.md`](../../v2/README.md)
- [`v3/README.md`](../../v3/README.md)
- 実行面の短い案内: `v2/primary/README.md`, `v3/primary/README.md`
