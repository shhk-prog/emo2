# 実装計画: 再集計および再実行用結果退避・実行手順の整備

ユーザーの指示に従い、以下を実施します。
1. **再実行対象の結果を `old_results` に移動**:
   - 必ず再実行: V1 Phase B, V3 RQ2, V3 RQ3, V3 Confirmatory
   - できれば再実行: V2 RQ4
2. **再集計対象の再集計を実行**:
   - V2 RQ1/RQ2: `v2/results/raw/v2_geometry_*.json` を基に改訂ロジックで再集計
   - V1 Phase A: `v1/results/cache/` 内の既存キャッシュを基に改訂ロジックで再集計
3. **再実行スクリプトの提供**:
   - ユーザーが手元で実行するためのコマンドおよびスクリプト（`scripts/run_production_reruns.sh` 等）を整備

---

## 提案する変更と手順

### 1. 既存結果の `old_results/` 退避
- `old_results/archive_20260921/` を作成し、以下を移動：
  - `v1/results/derived/v1_phase_b/`
  - `v2/results/raw/v2_causal_map_*`, `v2/results/raw/manifest_causal_map_*`, `v2/results/derived/pair_level/`, `v2/results/raw/dry_run/v2_recovery_*`, `v2/results/derived/dry_run/v2_distribution_recovery_*` 等（V2 RQ4 関連）
  - `v3/results/raw/`, `v3/results/derived/`（V3 全体）
- 各結果ディレクトリの `.gitkeep` を保持。

### 2. 再集計の実行
- **V2 RQ1/RQ2**:
  - `v2/results/raw/` に残る実機結果（`v2_geometry_qwen.json`, `v2_geometry_llama.json`, `v2_geometry_gemma.json`, `v2_geometry_olmo.json`）を読み込み、Primary (matched-plain) と Secondary (native-chat) を分離した Bootstrap 95% CI を算出する再集計スクリプトを実行し、`v2/results/derived/v2_cross_family_summary.json` を更新。
- **V1 Phase A**:
  - `v1/results/cache/` に存在するアクティベーション（192件）を用いて、負の $R^2$ 生値保持、AIPsy プローブの NaN 化を反映した集計を実行。

### 3. 再実行スクリプトの作成
- `scripts/run_production_reruns.sh` を作成し、ユーザーが GPU 環境で順次実行できるように引数とコマンドを整理。

## 検証計画
- `old_results/archive_20260921/` への移動確認
- `v2/results/derived/v2_cross_family_summary.json` の再生成内容確認（`primary_matched_plain` が含まれ、Bootstrap CI が計算されていること）
- `v1/results/derived/v1_phase_a/` の更新確認
