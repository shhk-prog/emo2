# [作業完了報告] LLM情動反応性評価・本実験（Main Experiment）コードの実装

本実験（Main Experiment: 機能的内部表現の検証）に関するパイプラインコードの実装およびテストが完了しました。

## 1. 実装された機能

### コア幾何指標モジュール (`src/affective_empathy_eval/metrics.py`)
- **Anchor Direction Alignment (ADA)**: 刺激VAと反応VAの角度的一致を計算。ゼロベクトル時の除外処理を実装。
- **Reactivity Magnitude ($R$) & Distance ($D$)**: 反応強度および人間のアンカー値からの距離計算。
- **Stimulus-to-response Gain ($G$)**: $R / \|h\|$ のゲイン計算。
- **Over-empathy Index**: 反応強度と刺激強度の差分による過剰共感の定量化。
- **RSA (Representational Similarity Analysis)**: 距離行列に基づく空間構造の相関（Spearman / Pearson）を算出するロジック。

### 内部表現抽出モジュール (`src/affective_empathy_eval/extraction.py`)
- Hugging Face / PyTorch の Transformer モデルに対応し、forward hook を用いて指定した層の hidden states / residual stream を抽出する `PyTorchRepresentationExtractor` を実装。
- 刺激終端トークン, 回答開始直前トークン, 刺激列 mean-pooling などの位置切り出し処理。
- ローカル検証・テスト用のモックテンソルを生成する `MockRepresentationExtractor` を実装。

### 内部表現解析モジュール (`src/affective_empathy_eval/probing.py`)
- 抽出された表現からの目標値（Stimulus-VA, Recognition-VA, Reception-VA, Shift-VA）予測のための `LayerProber` (Ridge回帰 / 交差検証) を実装。
- シャッフル統制条件（Permutation Test）用のベースライン予測モジュール `run_shuffled_baseline_probing` を実装。

### 因果的介入モジュール (`src/affective_empathy_eval/intervention.py`)
- **Activation Patching**: ソース刺激とターゲット刺激のテンソル置換・補間 (`ActivationPatcher`)。
- **Ablation**: 表現の Zero / Mean / Neutral 化 (`RepresentationAblator`)。
- **Steering Vector**: 感情差分に基づく方向ベクトルの算出とアディティブ注入 (`SteeringController`)。

### 統制条件モジュール (`src/affective_empathy_eval/controls.py`)
- 感情語辞書を用いた Lexical-emotion と Contextual-emotion の分類フィルタ、中性刺激フィルタの実装。
- 置換検定用の Label-shuffled データセット生成関数。

### 実験実行スクリプト (`scripts/`)
- **`run_main_experiment.py`**: Llama, Qwen 等の公開重みモデルを用いた4条件推論および表現抽出パイプライン。
- **`run_probing.py`**: 抽出されたテンソル(`.npz`)を読み込み、プロービングおよび RSA 解析を実行するパイプライン。
- **`run_causal_intervention.py`**: 介入実験（Patching, Ablation, Steering）の実行パイプライン。
- **`run_analysis.py` (更新)**: $G$, Over-empathy Index, `has_explicit_emotion` の統合とデータセット結合に対応。

---

## 2. プロービングと因果的介入の厳密化
- プロービング (`probing.py`):
  - 予測精度の漏洩（リーク）を防ぐため `GroupKFold` を導入し、交差検証の学習時のみPCAと標準化を適用するように修正。
  - 指標として $R^2$, RMSE, MAE, Pearson, Spearman を統合し、シャッフルテストを標準機能として追加。
- 因果的介入 (`intervention.py`):
  - 実証の厳密性を高めるため、Activation Patching の効果を測る Recovery Score の計算を追加。
  - 主分析のためのニュートラル・アブレーション（`RepresentationAblator`）と、任意のスカラー強度による因果介入（`SteeringController`）を実装。
- 評価システム (`evaluation.py`):
  - 実験フェーズB向けの直接的評価（次トークン確率の抽出）と自由応答評価（生成的評価）を追加。

## 3. テストと動作検証
- 修正・追加された各モジュールに対し、`pytest` によるユニットテストを実施し、すべて（15件）のテストがパスすることを確認。
- コマンドラインスクリプト群 (`scripts/run_main_experiment.py`等) のドライランモードを統合し、`ManifestManager` や `GroupKFold` などの新仕様に追従。現在エンドツーエンドでの動作確認を進めています。
- **Dry-run 検証**: 
  - `python scripts/run_main_experiment.py --mode dry-run` を実行し、全刺激・全条件での JSONL ログおよび `.npz` ファイル出力が完了。
  - `python scripts/run_probing.py --dry-run` にて、RSA行列・層別 $R^2$ などの解析サマリ (csv) が正常に生成。
  - `python scripts/run_causal_intervention.py --dry-run` にて、介入サマリが生成。

---

## 3. 次のステップ

1. **実機実行**: 十分な GPU メモリ (VRAM) を持つ環境にて、`configs/main_experiment.yaml` の `--mode hf` を使用し、小規模な Llama-3.2-1B または Qwen2.5-1.5B モデルのパイプライン実行をお試しください。
2. **プロービングの可視化**: 抽出結果の $R^2$ を層ごとにプロットする可視化スクリプト等への拡張が可能です。
3. **論文執筆への反映**: 今回実装された指標や構成は `docs/paper_draft.md` に記載された実験プロトコルと完全に整合しています。
