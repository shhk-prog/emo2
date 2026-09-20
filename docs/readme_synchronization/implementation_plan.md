# READMEコード整合・詳細化計画 (Implementation Plan)

## 目的
直近で実施した本番実行前必須修正（全25項目：$\beta$ 目的変数の Self-report 化、`response_start` 自己回帰位置補正、$\gamma$ 定義統一、K=5 ランダム/直交コントロール、Confirmatory CI 下限判定、V1 E4 20-derangements、V2 RQ3 コントロール、V3 RQ3 random subspace 統制、キャッシュ/マニフェスト厳格化、left padding / truncation ガード等）に基づき、リポジトリ内の各 README ファイル（ルート、Behavioral、V1、V2、V3）を現行の実装コードと完全に同期・詳細化する。

---

## 主な更新対象ドキュメント
1. **ルート `README.md`**:
   - 4-Stage 概念、Cross-Stage Methodological Matrix の Primary メトリクス・統制条件の最新化
   - コントロール条件の統一（Random/Orthogonal directions, Random 2D subspace, 20-derangements）
2. **`v3/README.md`**:
   - $\beta(l,t)$ の Self-report 目的変数回帰（Reader internal score $\rightarrow$ Self-report $\mid$ covariates）
   - 自己回帰生成段階 `response_start = prompt_end`（`cand_start - 1`）と negative control `response_end`
   - $\gamma$ スロープの 1 SD 正規化用量定義 (`estimate_interventional_slope(dose_grid, report_shift)`)
   - RQ1: $K=5$ ランダム・直交方向コントロール
   - RQ3: matched-rank random 2D subspace removal コントロール ($Q_{\text{rand}}$)
   - Confirmatory: H1〜H4 の pair-bootstrap CI および **CI lower bound > preregistered threshold** 判定
3. **`v1/README.md`**:
   - `--alphas` の YAML 優先設定 (`configs/v1_experiments.yaml`)
   - E4 Interchangeability: $K=20$ 回固定シード derangements による random donor distribution 比較
   - 科学的主張の境界（$\Delta h$ は affect-manipulation-associated difference、E6 は task-specific causal site sensitivity）
   - `checkpoint_manifest.json` による中断再開厳格化、activation cache メタデータ拡充
   - left/right padding 対応、prompt truncation ガード、Phase A `evaluated_mask`
4. **`v2/README.md`**:
   - RQ3: 同 norm ランダム/直交コントロール ($C_{\text{affect}} - C_{\text{random}}, C_{\text{affect}} - C_{\perp}$)
   - RQ4: plain-raw vs Procrustes aligned の off-manifold 峻別
   - manifest への `v2_config` 全体含浸
5. **`behavioral/README.md`**:
   - `--dry-run` 時の `dry_run` 出力隔離
   - チェックポイント再開時の metadata 厳密一致要件
   - Specificity 解析の留意事項

---

## 検証方法
- 各 README ファイルの記載と実コード（`v1/primary/`, `v2/primary/`, `v3/primary/`, `behavioral/`, `src/`）の一致度確認
- Markdown 文法・リンク切れの検証
