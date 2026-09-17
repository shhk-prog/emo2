# スケーリング実験結果の集約・分析計画

スケーリング実験（`./scripts/run_scaling_all.sh`）の実行が完了し、`logs/scaling/` および `results/derived/scaling/` に結果が出力されていることを確認しました。
この結果を整理・可視化し、報告書としてまとめるための実装計画を提案します。

## プロジェクトルールへの準拠
プロジェクトルール（`AGENTS.md`）の「回答言語およびプロジェクトドキュメント保存ルール」に従い、今回の分析に関連するタスクリスト、実装計画、修正内容の確認（Walkthrough）は、`docs/scaling_results_analysis/` ディレクトリに保存します。

## Proposed Changes

### 1. データ集計スクリプトの作成
#### [NEW] [scripts/summarize_scaling.py](file:///mnt/nas/home/hiromi/src/emo/scripts/summarize_scaling.py)
- `logs/scaling/*.log` を読み込み、各モデルの `Base Clean Shift`, `Instruct Clean Shift`, `Behavioral Suppression Ratio` などを抽出する。
- 各モデルのディレクトリ（`results/derived/scaling/*`）から `suppression_gap.csv` を読み込み、レイヤーごとの結果を集約する。
- 集計結果を `results/derived/scaling_summary.csv` として出力する。

### 2. 可視化スクリプトの作成
#### [NEW] [scripts/plot_scaling.py](file:///mnt/nas/home/hiromi/src/emo/scripts/plot_scaling.py)
- `results/derived/scaling_summary.csv` を元に、以下の図表を `results/figures/scaling/` に作成する。
  1. モデルサイズ（パラメータ数）ごとの Clean Shift の変化（Base vs Instructの比較）。
  2. モデルサイズごとの Behavioral Suppression Ratio の推移。
  3. 各モデルにおける、レイヤーごとの Suppression Gap の変化。

### 3. ドキュメント化とレポートの作成
#### [NEW] [docs/scaling_results_analysis/scaling_report.md](file:///mnt/nas/home/hiromi/src/emo/docs/scaling_results_analysis/scaling_report.md)
- 生成された図表と集計されたCSVを元に、実験結果の傾向と考察をまとめたレポートを作成する。
- Qwen2.5 および Llama-3.2 ファミリーにおける、アライメント（Instruct化）とモデルサイズが自己報告VA（情動反応）に与える影響についてまとめる。

#### [NEW] [docs/scaling_results_analysis/task.md](file:///mnt/nas/home/hiromi/src/emo/docs/scaling_results_analysis/task.md)
#### [NEW] [docs/scaling_results_analysis/walkthrough.md](file:///mnt/nas/home/hiromi/src/emo/docs/scaling_results_analysis/walkthrough.md)
#### [NEW] [docs/scaling_results_analysis/implementation_plan.md](file:///mnt/nas/home/hiromi/src/emo/docs/scaling_results_analysis/implementation_plan.md)
- 本計画を含む管理用ドキュメントをローカルのファイルシステム上に保存する。

## Verification Plan

### Automated Tests
- 新規作成した Python スクリプト（集計・描画用）をエラーなく実行できることを確認します。

### Manual Verification
- `results/figures/scaling/` に生成された画像を開き、正しくプロットされているか確認します。
- `docs/scaling_results_analysis/scaling_report.md` の内容に矛盾がないか、結果の抽出が正確に行われているかを確認します。

## Open Questions

> [!NOTE]
> スケーリング則のプロットにおいて、モデルパラメータ数（0.5B, 1.5B, 3B, 7Bなど）をX軸の対数スケールとしてプロットする予定ですが、これで問題ないでしょうか？また、描画したい追加のメトリクスやグラフの希望があればお知らせください。
