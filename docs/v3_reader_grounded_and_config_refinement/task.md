# タスクリスト: V3 Reader-Grounded 情動方向への刷新および設定・サンプル数精緻化

## 1. 課題と背景
- **V3 情動方向の再定義（最優先）**:
  - 現状の V3（RQ1〜RQ4 / Confirmatory）では、AIPsy に人間の連続 VA ラベルがないため、モデル自身の Self-report（$y_{\text{train}}^S$）をターゲットに方向 $d_V, d_A$ を学習していた。
  - これは「自己報告に関連づけられた内部状態 $\rightarrow$ 自己報告」という循環構造を一部含み、「独立に情動的と同定された内部状態を自己報告が参照している」という主張に対して弱点となっていた。
  - train split（および各刺激）においてモデル自身の「Reader Prediction（他者感情認識タスク）」を出力させ、その予測値 $y_V^R, y_A^R$ をターゲットとして情動方向 $d_V^R, d_A^R$ を学習する「Reader-Grounded Affect Direction」を Primary に刷新。
  - これにより、全体の論理構造が：
    Behavioral coupling $\rightarrow$ V1 shared representation $\rightarrow$ V2 post-training reorganization $\rightarrow$ V3 Reader-grounded affect state $\rightarrow$ Self-report
    と美しく一本化された。
  - 従来の Self-derived 方向は Secondary analysis（補助分析）として結果辞書に保持。
- **設定ファイルの明確化**:
  - `configs/v3_experiments.yaml` の `neutral_text_column: "neutral_text"` が loader 生成列であることをコメントで明記。
- **RQ2 Discovery 因果介入サンプル数の増量**:
  - 既定の 5 件から 15 件（10〜20件の推奨範囲）へ増量し、設定ファイルから調整可能にした。

## 2. タスク進捗
- [x] 実装計画の作成・提示 (`implementation_plan.md`)
- [x] `configs/v3_experiments.yaml` の更新
  - [x] `neutral_text_column` の loader 生成フィールドに関する注記明記
  - [x] `spatiotemporal.n_causal_samples: 15` の追加
- [x] `v3/primary/run_rq1_state_induction.py` の改修
  - [x] train split における Reader Prediction ($y_V^R, y_A^R$) の取得
  - [x] Primary 方向を Reader-grounded ($d_V^R, d_A^R$) に刷新
  - [x] Secondary 分析として Self-derived 方向およびコサインアライメント（$d^R \text{ vs } d^S$）を保持
- [x] `v3/primary/run_rq2_spatiotemporal_maps.py` の改修
  - [x] Reader Prediction をデコードターゲット・介入方向の Primary に設定
  - [x] 因果介入サンプル数を 5 件から設定値（既定 15 件）へ拡張
  - [x] Secondary 分析として Self-derived D-map 等を保持
- [x] `v3/primary/run_rq3_path_mediation.py` の改修
  - [x] Discovery 段階での情動ターゲットを Reader Prediction に設定
  - [x] Reader-grounded な刺激情動状態 $d_{\text{stim}}^R \rightarrow M \rightarrow Y$ の因果媒介パスを検証
- [x] `v3/primary/run_confirmatory_replication.py` の改修
  - [x] 各モデルの Reader Prediction に基づく layer-specific $d_V^{R,(l)}, d_A^{R,(l)}$ を推定して Self-report への介入を実施
  - [x] シミュレーション側および実モデル実行側のキー整合性維持
- [x] 単体テスト・回帰テストの実行
  - [x] `pytest tests/test_production_entrypoints.py` (2 passed)
  - [x] `pytest -q` (60 passed)
  - [x] 各スクリプトの `--dry-run` 検証完了
- [x] ドキュメント（`task.md`, `implementation_plan.md`, `walkthrough.md`）の保存
