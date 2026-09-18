# タスク: 正式再実行前の results クリアと再現性整理

## 目的

正式な本番再実行の前に、旧 run の途中結果が ZIP / Git に混ざる事故を止める。あわせて、再実行前に確認した表現・再現性の改善を入れる。

## 必須（再実行前に完了）

- [x] `behavioral/results`, `v1/results`, `v2/results`, `v3/results` を `.gitkeep` 以外完全クリア
- [x] Git 追跡中の raw/derived 成果物を `git rm` し、旧 CSV/JSON が checkout で戻らないようにする
- [x] `.gitignore` で `**/results/raw/**` と `**/results/derived/**` を無視し、`.gitkeep` だけ残す

## 再実行前に入れておく改善

- [x] V2 RQ4 の Recovery 式を関数化し、README と同じ符号・分母・target 定義を unit test で固定
- [x] V2 RQ1/RQ2 dry-run の `reader_V` 乱数 fallback を固定 fixture に変更
- [x] V1 Phase B を rule-based controlled perturbation として README / protocol に明記
- [x] V3 RQ2 の 4-Map を D, β, γ, C の4枚として README に詳細化
- [x] V3 dry-run mock 出力に `dry_run=true` を残す
- [x] Primary ランナーの transformers import を遅延（`--dry-run` がモデル依存なしで起動できるようにする）
- [x] 本番は stage 別スクリプト（Behavioral → V1、問題なければ V2 → V3）を推奨として明記

## 今回やらないこと

- GPU 本番実行
- 実験仕様・刺激セット・主指標の変更
- `docs/` 全体の current/archive 再配置（公開時の整理であり実行ブロッカーではない）
