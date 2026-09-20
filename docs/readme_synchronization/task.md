# Task: READMEコード整合・詳細化 (README Synchronization)

## 概要
直近で実施された25項目の必須修正を反映し、ルートおよび各ステージ（Behavioral, V1, V2, V3）の README を現行コードと完全に一致させ、測定量・統制条件・解釈境界を詳細に記述する。

## タスクリスト
- [x] 1. `v3/README.md` の更新
  - [x] $\beta(l,t)$ の回帰目的変数を「Reader-affect score $\rightarrow$ Self-report ($y_{\text{self}}$)」として明記
  - [x] `response_start = prompt_end`（`cand_start - 1`）自己回帰位置、`response_end` negative control
  - [x] $\gamma$ スロープの「1 SD 正規化用量あたりの report 変位量」定義と変数名
  - [x] RQ1: `num_random_controls: 5` のランダム・直交方向コントロール
  - [x] RQ3: matched-rank random 2D subspace removal コントロール ($Q_{\text{rand}}$)
  - [x] Confirmatory: H1〜H4 pair-bootstrap CI および CI 下限判定、架空値排除
- [x] 2. `v1/README.md` の更新
  - [x] Phase C `--alphas` の YAML 優先 (`[0.0, 0.5, 1.0, 2.0]`)
  - [x] E4 Interchangeability: $K=20$ 回固定シード derangements と random donor distribution
  - [x] 科学的主張の境界（$\Delta h$ は情動操作関連変位、E6 は site sensitivity）
  - [x] `checkpoint_manifest.json` による厳密再開と activation cache メタデータ拡充
  - [x] left padding 対応、truncation ガード、Phase A `evaluated_mask`
- [x] 3. `v2/README.md` の更新
  - [x] RQ3: 同 norm ランダム/直交コントロール ($C_{\text{affect}} - C_{\text{random}}, C_{\text{affect}} - C_{\perp}$)
  - [x] RQ4: plain-raw vs Procrustes aligned の off-manifold 峻別
  - [x] manifest への `v2_config` 全体含浸
- [x] 4. `behavioral/README.md` の更新
  - [x] `--dry-run` 時の出力隔離とチェックポイント metadata 厳密一致要件
- [x] 5. ルート `README.md` の更新
  - [x] 全体概要・Cross-Stage Methodological Matrix のコントロール・メトリクス最新化
- [x] 6. 変更内容確認ドキュメント (`walkthrough.md`) の作成
