# Walkthrough: 全再実行直前の最終4点リファインメント

本ドキュメントは、全再実行（GPU本番環境での正式実行）の直前に実施された最終4点の修正および整合化結果をまとめたものです。

---

## 1. 実施した修正項目一覧

### ① V3 RQ2: 実介入サンプル数とマップサンプル数の分離・メタデータ修正
- **対象ファイル**: [`v3/primary/run_rq2_spatiotemporal_maps.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq2_spatiotemporal_maps.py)
- **背景**: 実際の介入ループでは $\min(5, N)$ サンプルで介入応答および因果変位量 $C$ を測定していましたが、出力 JSON の `"n_intervene_samples"` に全体サンプル数 $N$ が記録される不整合がありました。
- **対応内容**:
  - `run_real_spatiotemporal_maps`、`simulate_spatiotemporal_maps`、および `main()` の出力結果・Manifest メタデータを以下のように明確に分離しました：
    - `"n_map_samples": N`（活性化抽出および $D, \beta$ 算出サンプル数）
    - `"n_intervene_samples": min(5, N)`（実介入評価サンプル数）
    - `"n_causal_intervention_samples": min(5, N)`（明示的エイリアス）

### ② V3 RQ2: Decodability CV の GroupKFold 統一
- **対象ファイル**: [`v3/primary/run_rq2_spatiotemporal_maps.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq2_spatiotemporal_maps.py)
- **背景**: AIPsy データセットの同一対（`pair_id`）に由来するサンプル間の情報漏洩（pair leakage）を防ぐため、V1 および V3 Path Mediation と同じく `GroupKFold` で揃える必要がありました。
- **対応内容**:
  - `eval_df` 内に `pair_id` 列が存在する場合、`GroupKFold(n_splits=min(3, n_groups))` による交差検証スプリットを生成して held-out $R^2$ を算出する設計へ統一しました（万が一存在しない場合のみ通常の `KFold` に安全にフォールバック）。

### ③ V1 Phase C E6: `num_layers = 28` サイレントフォールバックの完全削除
- **対象ファイル**: [`v1/primary/phase_c/run_e6_specialization.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/phase_c/run_e6_specialization.py)
- **背景**: `resolve_architecture_dims(args.model_id)` が失敗した際に、旧名残で静かに 28 層（Qwen相当）へフォールバックする例外ブロックが残存していました。Model Registry が厳格化された現在、多モデル（Llama: 16層, Gemma: 26層, OLMo: 16層等）において予期せぬ挙動を招くリスクがありました。
- **対応内容**:
  - 例外を捕捉して静かに値を代入する処理を完全撤去し、エラーログを出力した上でそのまま `raise` して即座に安全停止するよう改修しました。

### ④ V1 Phase C E6: Double Dissociation 旧名称の完全統一と Legacy 二重出力の撤去
- **対象ファイル**: [`v1/primary/phase_c/run_e6_specialization.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/phase_c/run_e6_specialization.py)
- **背景**: 実装・README・CLI はすべて `Task-Specific Causal Specialization` に移行済みでしたが、スクリプト先頭の docstring および試行出力で `e6_double_dissociation_trials.csv` への二重出力が残っていました。
- **対応内容**:
  - docstring を `E6 Targeted Ablation & Task-Specific Causal Specialization` へ改定。
  - dry-run ブロックおよび本番実行ブロックにおける `e6_double_dissociation_trials.csv` への保存を削除し、主出力 `e6_specialization_trials.csv` のみに一本化しました。

---

## 2. 状態検証結果

1. **構文・整合性チェック**:
   - 変更された各スクリプトで構文エラー、未定義参照、未対応引数等が存在しないことを確認。
2. **Results 初期化状態**:
   - `results/`, `v1/results/`, `v2/results/`, `v3/results/` ともに余分なデータは一切生成されず、`.gitkeep` のみのクリーン状態を維持。
3. **再現性・運用ルール準拠**:
   - `AGENTS.md` の原データ読み取り専用・生データ非破壊・仮想環境運用・カスタム独自コード抑制の各原則に完全準拠。

---

## 3. 全再実行コマンド（GPU本番環境用）

本番再実行（Primary 4 Family: Qwen, Llama, Gemma, OLMo）は、以下の手順で GPU 環境にて実行可能です。

```bash
# 仮想環境のアクティベート
source .venv/bin/activate

# 1. 行動実験 (Behavioral: EmoBank, AIPSY, Reader/Self)
bash scripts/run_production_behavioral.sh cuda:0

# 2. V1 実験 (Phase A/B/C: E1~E6)
bash scripts/run_production_v1.sh cuda:0

# 3. V2 実験 (2x2 Cross-Decoding & Recovery Patching)
bash scripts/run_production_v2.sh cuda:0

# 4. V3 実験 (RQ1 State Induction, RQ2 Spatiotemporal Maps, RQ3 Mediation, RQ4 Steering)
bash scripts/run_production_v3.sh cuda:0

# または全4ステージ一括実行:
# bash scripts/run_production_all.sh cuda:0
```
