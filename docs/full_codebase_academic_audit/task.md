# Task: 全コードベース学術監査 (Full Codebase Academic Audit)

## 概要
コードベース全体（Behavioral, V1, V2, V3, 共通src）を網羅的に精査し、トップ国際会議／学術論文として測定妥当性・統計処理・因果主張・再現性に問題がないかを確認・評価する。

## タスク進捗
- [x] 1. 共通モジュール (`src/affective_empathy_eval/`) の精査
  - [x] 尤度計算・期待値算出 (`likelihood.py`)
  - [x] 介入フック・用量反応スロープ (`interventions.py`, `intervention.py`)
  - [x] 幾何・分解能・距離指標 (`geometry.py`)
  - [x] 統計検定・ブートストラップ・LMM (`statistics.py`)
  - [x] マニフェスト・キャッシュ検証 (`manifests.py`)
- [x] 2. Behavioral パイプラインの精査
  - [x] AIPsy / EmoBank 刺激提示と独立セッション性 (`run_behavioral_*.py`)
  - [x] 分析・集計・FDR 補正・LMM モデル (`summarize_behavioral_*.py`)
- [x] 3. V1 (Mechanistic Localization & Interchangeability) の精査
  - [x] Phase A (線形プロービング・評価マスク)
  - [x] Phase B (感情特異性・回帰分析)
  - [x] Phase C (E3 shared map, E4 20-derangements interchangeability, E6 site sensitivity)
- [x] 4. V2 (Post-training Representational Reorganization) の精査
  - [x] RQ1/RQ2 (Cross-decoding, Procrustes, held-out test 分離)
  - [x] RQ3 (Causal map, random/perp 同 norm コントロール)
  - [x] RQ4 (Causal recovery, plain vs aligned off-manifold 解釈)
- [x] 5. V3 (Spatiotemporal Mapping & Causal Path Utilization) の精査
  - [x] RQ1 (State induction, K=5 random/perp コントロール)
  - [x] RQ2 (時空間マップ, $\beta$ の Self-report 目的変数回帰, prompt_end 位置)
  - [x] RQ3 (Path mediation, random 2D subspace removal コントロール)
  - [x] Confirmatory replication (H1〜H4 CI 下限判定, 架空値フォールバック排除)
- [x] 6. 総合監査レポート作成 (`walkthrough.md`)
