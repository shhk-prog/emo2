# READMEコード整合・詳細化 (README Synchronization) — 確認ドキュメント (Walkthrough)

直近の本番実行前修正（全25項目）を正確に反映し、リポジトリ内の全 README（ルート `README.md`, `v3/README.md`, `v1/README.md`, `v2/README.md`, `behavioral/README.md`）の更新・詳細化を完了しました。

---

## 主な更新内容とコード整合

### 1. `v3/README.md`
- **$\beta(l,t)$ の定義と目的変数**:
  刺激共変量を統制した内部 Reader 予測スコア $\rightarrow$ **Self-report ($y_{\text{self}}$)** の偏回帰係数であること、および pair-bootstrap 95% CI を併記することを明記。
- **意味段階（Teacher-forced candidate sequence）**:
  `response_start` を最初の候補トークン生成直前（`cand_start - 1` = `prompt_end`）とし自己回帰的因果関係に整合。`response_end` は因果効果が原理上消失する **Negative control** として保持することを明記。
- **$\gamma$ スロープの定義**:
  $\gamma = \Delta \text{Report} / \Delta \alpha$（「1 SD 正規化介入用量あたりの自己報告変化率」）として数式および説明を統一。
- **RQ1 特異性統制**:
  `num_random_controls: 5` に連動し、各軸 $K=5$ 本のランダム方向および直交方向コントロールを生成・評価し平均効果との差分を記録することを記載。
- **RQ3 媒介除去統制**:
  2D affect subspace removal に対し、**matched-rank random 2D subspace removal コントロール ($Q_{\text{rand}}$)** を追加し、任意の 2 次元破壊による非特異的変位減少と情動特異的減衰を分離することを記載。
- **Confirmatory Replication**:
  架空値フォールバックの完全排除と、全仮説（H1〜H4）の判定基準を **CI lower bound > preregistered threshold** に統一したことを明記。

### 2. `v1/README.md`
- **Phase C `--alphas`**:
  CLI default ではなく `configs/v1_experiments.yaml` の `[0.0, 0.5, 1.0, 2.0]` を source of truth として優先適用することを記載。
- **E4 Interchangeability**:
  単一の derangement ではなく、**20 回の固定シード derangements ($K=20$)** による random donor distribution との比較（matched − mean(random)、標準偏差、permutation p、bootstrap CI）を明記。
- **科学的主張の境界**:
  $\Delta h$ は感情差に関連した隠れ状態変位（**affect-manipulation-associated hidden-state difference**）であり、コンテキストから完全に遊離した純粋情動コードとは主張しないこと、E6 の zero ablation は **task-specific causal site sensitivity (whole residual zeroing)** であることを注記。
- **共通プロトコル**:
  `checkpoint_manifest.json` による中断再開の厳密照合、left/right padding 双対応の `valid_pos[-1]`、`max_length=1024` silent truncation ガード、Phase A `evaluated_mask` を記載。

### 3. `v2/README.md`
- **RQ3 特異性コントロール**:
  同一ノルムを持つランダム方向統制 ($d_{\text{rand}}$) および直交方向統制 ($d_{\perp}$) を追加し、主たる因果指標を **Net Causal Effect ($C - C_{\text{rand}}, C - C_{\perp}$)** として深層摂動感受性の交絡を排除することを明記。
- **RQ4 分布復元の概念的峻別**:
  Matched-Plain Raw（direct interchangeability）と Procrustes Aligned（coordinate-remapping-adjusted recovery）を対比し、Raw で回復しないのは情報消失ではなく事後学習に伴う表現座標系の変化（*off-manifold* 化）によるものである解釈境界を明記。
- **マニフェスト照合**:
  キャッシュ検証マニフェストに `v2_config` 全体を含め、YAML 設定変更時のキャッシュ無効化を保証することを記載。

### 4. `behavioral/README.md`
- **Dry-run 成果物の完全隔離**:
  `--dry-run` 時の出力先が `.../dry_run` に隔離され、本番成果物を汚染しないことを明記。
- **チェックポイント再開の厳密性**:
  `checkpoint_metadata.json` が存在し全メタデータが完全一致する場合のみ resume を許可し、メタデータ欠損時は破棄・退避する設計を記載。
- **Specificity 解釈境界**:
  Complex Neutral（48件）と Clinical（192件）の比較における語彙・文長交絡と感度分析の位置付けを明記。

### 5. ルート `README.md`
- 各 Stage のサマリ記述に最新の統制条件（20-derangements, random/perp net causal effect, random 2D subspace removal, CI lower bound 判定, truncation ガード）を反映。
- **§4.7 Cross-Stage Methodological Matrix** の Primary Metric、Token Position、Statistical Test を最新の実装仕様に完全同期。

---

## 検証結果
- **単体テスト**: **114 passed, 1 deselected, 5 warnings in 16.23s**
- 全テストスイートが正常にパスすることを確認済み。
