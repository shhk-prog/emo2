# コードベース全精査・学術論文適格性監査 (Implementation Plan)

## 目的
本リポジトリ内の全コードベース（Behavioral, V1, V2, V3, 共通ライブラリ `src/affective_empathy_eval`）を対象に、トップ国際会議／学術誌（ACL, EMNLP, NeurIPS 等）投稿水準での測定妥当性、因果推論の厳密性、統計処理の妥当性、再現性、およびクレームの整合性を網羅的に精査・監査する。

---

## 監査の観点

1. **研究上の測定原則と構成概念妥当性 (Construct Validity)**
   - 測定対象の限定: 「モデルが感情を経験する」等の過剰解釈がなく、行動的代理指標・自己報告変位として正しく定義・命名されているか。
   - Recognition（他者感情推定）と Reactivity（自己報告変位）の混同がないか。
   - 独立セッションの担保（Baseline, Recognition, Post のコンテキスト非汚染）。
   - スケーリング基準の統一（1–9 raw expected scale が Primary であること）。

2. **因果推論と統制条件の妥当性 (Internal Validity & Causal Rigor)**
   - V1: $\Delta h$ の解釈（affect-associated vs pure code）、E4 の random donor 統制、E6 の site sensitivity 解釈。
   - V2: Cross-decoding held-out 分離、Procrustes アライメント、RQ3 介入時の同 norm random/orthogonal 統制、RQ4 off-manifold 解釈。
   - V3: $\beta(l,t)$ の自己報告目的変数回帰、自己回帰 token 位置（`response_start = prompt_end`）、$\gamma$ スロープの正規化用量定義、RQ3 random 2D subspace 統制、Confirmatory H1〜H4 の CI 下限判定。

3. **統計的検定と多重比較補正 (Statistical Conclusion Validity)**
   - 多重比較補正（BH-FDR, Family-wise error rate）の適用単位と事前定義の整合性。
   - 混合効果モデル（LMM）の収束診断・singular fit ハンドリング。
   - ブートストラップ信頼区間（ペアブートストラップ）の実装と除外・ゼロ除算ガード。

4. **データ完全性・再現性・アーティファクト管理 (Reproducibility & Traceability)**
   - Discovery / Confirmation スプリットの完全分離（ペア単位）。
   - シード固定、モデル revision pinning、マニフェスト照合。
   - dry-run による本番データ汚染防止、チェックポイント再開時の安全性。

---

## 監査計画とステップ

- [ ] Step 1: 監査計画書 (`task.md`, `implementation_plan.md`) の作成
- [ ] Step 2: 共通モジュール (`src/affective_empathy_eval/`) の精査
  - `likelihood.py`, `interventions.py`, `geometry.py`, `statistics.py`, `manifests.py`
- [ ] Step 3: Behavioral 実験パイプラインの精査
  - `behavioral/primary/`, `behavioral/analysis/`
- [ ] Step 4: V1 (Mechanistic Localization & Interchangeability) の精査
  - `v1/primary/run_phase_a.py`, `run_phase_b.py`, `run_phase_c.py`
- [ ] Step 5: V2 (Post-training Representational Reorganization) の精査
  - `v2/primary/run_rq1_rq2_cross_decoding.py`, `run_rq3_causal_map.py`, `run_rq4_causal_recovery.py`
- [ ] Step 6: V3 (Spatiotemporal Mapping & Causal Path Utilization) の精査
  - `v3/primary/run_rq1_state_induction.py`, `run_rq2_spatiotemporal_maps.py`, `run_rq3_path_mediation.py`, `run_confirmatory_replication.py`
- [ ] Step 7: 総合監査報告書 (`walkthrough.md`) の作成と最終判定

---

## 検証方法
- 静的解析・コードリーディングによる論理的欠陥の抽出
- 回帰テスト・単体テスト全件通過の再確認 (`pytest -q`)
- バイトコンパイル検証 (`compileall`)
