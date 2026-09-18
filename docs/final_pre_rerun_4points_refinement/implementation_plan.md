# 実装計画: 全再実行直前の最終4点リファインメント

全再実行（GPU本番環境での正式実行）を完全に万全な状態とするため、ご提示いただいた4点の修正を実施します。

---

## ユーザー確認が必要な事項 (User Review Required)

> [!NOTE]
> 今回の4点はすべて局所的かつ安全な修正であり、モデルアーキテクチャやデータセット定義への破壊的変更はありません。
> 1. V3 RQ2 の実介入サンプル数メタデータを正しく実数（$\min(5, N)$）に修正し、マップ全体サンプル数（$N$）と分離します。
> 2. V3 RQ2 の内部デコード能 CV において、ペア漏洩防止（pair leakage prevention）を他ステージと統一するため、`pair_id` に基づく `GroupKFold` を採用します。
> 3. V1 E6 において、予期せぬアーキテクチャ解決失敗時に勝手に 28 層（Qwen相当）で続行するサイレントフォールバックを排除し、厳格に `raise` します。
> 4. V1 E6 の docstring および legacy 二重出力を廃止し、`Task-Specific Causal Specialization` / `e6_specialization_trials.csv` に一本化します。

---

## 提案される変更 (Proposed Changes)

### 1. V3 RQ2 スパティオテンポラル 4-Map 解析
#### [MODIFY] [run_rq2_spatiotemporal_maps.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq2_spatiotemporal_maps.py)
- **GroupKFold の導入**:
  - `eval_df` 内に `pair_id` が存在する場合、`GroupKFold(n_splits=min(3, n_groups))` を用いて交差検証スプリットを生成。存在しない場合は従来の `KFold` にフォールバック。
- **実介入サンプル数とマップサンプル数の分離・メタデータ修正**:
  - 介入計算ループの実サンプル数 `min(5, N)` を明示的に追跡。
  - 出力結果辞書および Manifest のメタデータに以下を設定：
    - `"n_map_samples": int(N)`
    - `"n_intervene_samples": int(min(5, N))`
    - `"n_causal_intervention_samples": int(min(5, N))`

---

### 2. V1 Phase C E6 サイト特異的因果解析
#### [MODIFY] [run_e6_specialization.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/phase_c/run_e6_specialization.py)
- **num_layers のサイレントフォールバック削除**:
  - `resolve_architecture_dims(args.model_id)` の例外処理で `num_layers = 28` を代入していたブロックを削除し、例外をそのまま `raise` させる。
- **Double Dissociation 表記の完全統一と Legacy 二重出力の撤去**:
  - ファイル先頭の docstring:
    `V1 Phase C Primary Script: E6 Targeted Ablation & Task-Specific Causal Specialization` へ改定。
  - dry-run ブロックおよび本番実行ブロックでの `e6_double_dissociation_trials.csv` への保存を撤去し、主出力 `e6_specialization_trials.csv` のみに一本化。

---

## 検証計画 (Verification Plan)

### 自動テスト & 静的検証
- 対象スクリプトの Python 構文チェックおよび import 検証。
- `pytest -q` で既存の単体テストがすべて通過することを確認。
- `run_rq2_spatiotemporal_maps.py --help` および `run_e6_specialization.py --help` が正常に実行できることを確認。
- `docs/final_pre_rerun_4points_refinement/` に成果物ドキュメントを配置。
