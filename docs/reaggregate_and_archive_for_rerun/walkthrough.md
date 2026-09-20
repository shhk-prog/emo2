# 結果退避および再集計・再実行手順 Walkthrough レポート

本ドキュメントは、コード監査指摘（21項目）の修正完了を受け、既存結果の `old_results/` への安全退避、再集計対象の集計実行、および手元で実行いただく再実行スクリプトの整備状況をまとめたものです。

---

## 1. 既存結果の `old_results/` 退避状況

再実行対象となる実験の既存結果は、すべて `old_results/archive_20260921/` 配下へ安全に移動しました。
元の結果ディレクトリには `.gitkeep` のみを残し、上書き事故を防止しています。

| 対象実験 | 移動元 | 移動先 (`old_results/archive_20260921/`) | 状態 |
|---|---|---|---|
| **V1 Phase B** (必ず再実行) | `v1/results/derived/v1_phase_b/` | `old_results/archive_20260921/v1_phase_b/` | 退避完了 (`.gitkeep` のみ維持) |
| **V3 RQ2 / RQ3 / Confirmatory** (必ず再実行) | `v3/results/raw/`<br>`v3/results/derived/` | `old_results/archive_20260921/v3/raw/`<br>`old_results/archive_20260921/v3/derived/` | 退避完了 (`.gitkeep` のみ維持) |
| **V2 RQ4** (できれば再実行) | `v2/results/raw/v2_causal_map_*`<br>`v2/results/raw/v2_recovery_*`<br>`v2/results/derived/pair_level/`<br>`v2/results/derived/v2_distribution_recovery_*` | `old_results/archive_20260921/v2_rq4/` | 退避完了 (RQ1/RQ2 の生データのみ保持) |

---

## 2. 再集計の実施状況

### 2.1 V2 RQ1 / RQ2 (クロスファミリー幾何学＆クロスデコーディング)
- **入力生データ**: `v2/results/raw/` 内の実機 4 ファミリー幾何学結果（`v2_geometry_{qwen,llama,gemma,olmo}.json`）
- **処理内容**:
  - 改訂プロトコル（Item 6）に基づき、Primary (`matched-plain`) と Secondary (`native-chat`) を分離。
  - Primary（Matched Plain）の層深度・共有度変位からクロスファミリー Bootstrap 95% CI を算出。
  - Format Confound Effect（Native Chat - Matched Plain）を分離出力。
- **出力成果物**:
  - `v2/results/derived/v2_cross_family_summary.json`
- **主要集計値**:
  - **Primary (Matched-Plain)**:
    - Base Cross-decoding Peak Depth: Mean = 0.494, 95% CI = [0.222, 0.657]
    - Instruct (Matched) Cross-decoding Peak Depth: Mean = 0.119, 95% CI = [0.000, 0.294]
    - Mean Delta Sharing: Mean = -0.557, 95% CI = [-1.283, -0.040]
    - Paired Peak Shift (Inst - Base): Mean Diff = -0.376 ($t = -3.078, p = 0.054$)
  - **Secondary (Native-Chat)**:
    - Instruct (Native) Cross-decoding Peak Depth: Mean = 0.567, 95% CI = [0.222, 0.867]
    - Mean Delta Sharing: Mean = -0.947, 95% CI = [-2.247, -0.059]

### 2.2 V1 Phase A (幾何学＆生転移スコア保持)
- **入力生データ**: `v1/results/derived/v1_phase_a/{model}/e2_emobank_geometry.csv`
- **処理スクリプト**: `scripts/reaggregate_v1_phase_a.py`
- **処理内容**:
  - `r2_cross_r_to_s` と `r2_cross_s_to_r` の平均値から生値 `direct_transfer_score_raw` を計算。
  - 負の $R^2$ を 0 にクリップせず生値を `direct_transfer_score` に反映（Item 4）。
  - クリップ値は `direct_transfer_score_clipped` として併記。

---

## 3. 手元での再実行手順（ユーザー実行用）

GPU 環境のターミナルにて、以下のコマンドを実行してください。

### 3.1 一括再実行（推奨）
全ステージ（V1-B, V3-RQ2, V3-RQ3, V3-Confirmatory, V2-RQ4）を順次実行する統合スクリプトを作成しました：

```bash
bash scripts/run_production_reruns.sh
```

### 3.2 個別実行コマンド
各ステージを個別に実行したい場合は、以下のコマンドを使用できます：

#### 1. V1 Phase B (Semantic Control Audit: Reader & Self)
```bash
# 各モデルファミリーに対して reader / self を実行
for fam in qwen llama gemma olmo; do
    .venv/bin/python v1/primary/run_phase_b.py --family $fam --task-type reader --force
    .venv/bin/python v1/primary/run_phase_b.py --family $fam --task-type self --force
done
```

#### 2. V3 RQ2 (Spatiotemporal 4-Maps & Causal Discovery)
```bash
.venv/bin/python v3/primary/run_rq2_spatiotemporal_maps.py --family qwen --force
```

#### 3. V3 RQ3 (Path Mediation & Frozen Confirmation Sites)
```bash
.venv/bin/python v3/primary/run_rq3_path_mediation.py --family qwen --force
```

#### 4. V3 Confirmatory Replication (Hold-out Families)
```bash
for fam in llama gemma olmo; do
    .venv/bin/python v3/primary/run_confirmatory_replication.py --family $fam --force
done
```

#### 5. V2 RQ4 (Recovery Patching across Families) [できれば再実行]
```bash
for fam in qwen llama gemma olmo; do
    .venv/bin/python v2/primary/run_rq4_recovery_patching.py --family $fam --force
done
```

---

## 4. 再集計スクリプトの実行コマンド

もし V1 Phase A や V2 RQ1/RQ2 の再集計を再度ローカルで走らせたい場合は、以下を実行してください：

```bash
bash scripts/reaggregate_all.sh
```
