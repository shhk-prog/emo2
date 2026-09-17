# 実験タスクリスト (Top-Tier Conference向け追加実験)

## Phase 1: ベースラインの確立 (小さく厳密に)
- `[x]` EmoBankデータセットからの層化抽出スクリプト作成（V/Aそれぞれ低中高から各32〜50件）
- `[x]` 81通りのVA JSONフォーマットに対する条件付きシーケンス確率 $p(v,a\mid x)$ の算出ロジック実装（Qwen2.5 Base/Instruct用）
- `[x]` 期待VAベクトル $\mathbf{E}_x$ および正規化Recoveryの計算ロジック実装
- `[x]` ペア除外規則（分母が閾値未満のペアを除外）の実装とベースライン評価の実行

## Phase 2: 主実験 - AIPsy-Affect (最小対) と 厳密なProbe
- `[x]` AIPsy-Affectデータセットのロード・前処理と、最小対ペアID単位での train/dev/test 分割実装
- `[x]` 全層Hidden state抽出およびZ-score正規化ロジックの実装
- `[x]` Ridge回帰による線形Probeの実装と学習
- `[x]` 交絡排除用コントロールProbeの実装（感情ラベルシャッフル、文字数/トークン数予測）

## Phase 3 & 4: 因果検証 (Patching・Ablation・Steering)
- `[x]` `nnsight` などを利用したAttribution Patchingによる候補部位（head/MLP）の高速走査・順位付け (※巨大バッチでの直接 Activation Patching に変更して実装完了)
- `[x]` 抽出した候補部位に対するActivation Patching (十分性) の実装
- `[x]` 抽出した候補部位に対するAblation (必要性) の実装
- `[x]` 平均差分ベクトル $\mathbf{d}^{(l)}$ を用いたSteeringの実装と、$\alpha \in \{-2, \dots, 2\}$ での対称性評価

## Phase 5: アライメント遮断の因果的証拠
- `[x]` Baseモデル (`Qwen2.5-1.5B`) vs Instructモデル (`Qwen2.5-1.5B-Instruct`) の対照的な隠れ状態抽出スクリプトの実装 (`run_alignment_suppression.py`)
- `[x]` 両モデルに対する全層 Activation Patching / Ablation による感情伝達特性の統一比較ロジックの実装
- `[x]` Post-training (アライメント) による因果的遮断ギャップ (Suppression Gap = Base - Instruct) の定量評価メトリクスの実装
- `[x]` 実験の実行と Suppression Gap の確認（BaseとInstructにおける層別の因果的伝達率の解離評価）

## Phase 5.5: スケーリング & 複数モデルファミリー展開 (Slurm非使用)
- `[x]` スケーリング (0.5B, 1.5B, 3B, 7B) & 異ファミリー (Qwen2.5, Llama-3.2) 対応スクリプトの実装 (`run_scaling_experiments.py`)
- `[x]` モデル規模・ファミリーごとの独立出力フォルダ (`results/derived/scaling/<tag>/`) および個別のログ分離 (`logs/scaling/<tag>.log`) の実装
- `[x]` Slurm（sulam）を使用しない一括バッチ実行シェルの実装 (`scripts/run_scaling_all.sh`)

## Phase 6: 分析とレポーティング
- `[ ]` 実験結果の集計と可視化（仮説識別表との照合）
- `[ ]` `walkthrough.md` への実験結果の反映と最終レポート作成
