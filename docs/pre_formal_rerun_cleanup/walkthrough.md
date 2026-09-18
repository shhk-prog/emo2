# Walkthrough: 正式再実行前の results クリアと再現性整理

## 必須2点

1. **results を完全クリアした。**  
   `behavioral/results`, `v1/results`, `v2/results`, `v3/results` は `.gitkeep`（および `raw/` `derived/` のプレースホルダ）以外を削除した。Git 追跡されていた V1 derived CSV/JSON と V2 geometry JSON も `git rm` 済み。旧 run は checkout しても戻らない。
2. **`.gitignore` で混入を止めた。**  
   `**/results/raw/**` と `**/results/derived/**` を無視し、`.gitkeep` だけ例外にした。試し書きした CSV は ignore されることを確認した。

## 確認結果（再実行ブロッカーではないもの）

- **V2 RQ4 Recovery** は README の代数と実装が一致する。  
  \(P_{\mathrm{target}}=\) Base、\(P_{\mathrm{clean}}=\) Instruct 未介入、\(P_{\mathrm{patch}}=\) Base 活性化注入後、\(W_1=\) `emd_va`。中立文脈への感情注入ではない。`compute_emd_recovery_ratio` と unit test で符号・分母・完全回復 / 無変化 / 悪化を固定した。
- **V3 4-Map** はコードが D, β, γ, C（各 V/A）を出している。README を D/C だけから 4 枚の説明に詳細化した。名前は 4-Map のまま。
- **V1 Phase B** は rule-based controlled perturbation。README / protocol / 生成スクリプトの docstring を合わせ、reversal fallback 文の過大解釈を避けた。
- **V2 dry-run** の `reader_V` 乱数 fallback を `np.linspace(1, 9, n)` の固定 fixture に変更。本番経路は未変更。
- **Primary ランナー** の `transformers` import を `ImportError` 時にフォールバック。`--dry-run` は transformers 未導入でも起動できる。
- **V3 mock 出力** に `dry_run=true` を残す。RQ1 は既存、RQ2/RQ3/Confirmatory に追加。
- **本番実行** は stage 別スクリプトを推奨。順序は Behavioral → V1、問題なければ V2 → V3。

## 検証

この macOS checkout の `.venv` は Linux の `python3.12` への symlink のため使えない。`PYTHONPATH=src` と `/opt/anaconda3/bin/python` で実行した。

- pytest: **50 passed / 1 skipped**
- Primary コード: `compileall` 成功
- `scripts/run_production_*.sh` と `clear_stage_results.sh`: `bash -n` 成功
- 対象ファイル: UTF-8 問題なし
- ruff: この環境に未導入のため未実行

## 残しているもの

- `docs/` の current/archive 再配置は公開時の整理であり、実行ブロッカーではないので未実施。
- GPU 本番は未実行。results は空なので、この状態から Behavioral → V1 を始めてよい。

## コミット

結果削除と gitignore は index に載っている。コミットは指示があれば行う。
