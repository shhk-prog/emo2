# Walkthrough: E6タスク選択性サイト選定と全再実行直前の最終整合化

本ドキュメントは、全再実行開始直前に実施された最終リファクタリングおよび整合化作業の実施結果をまとめたものです。

---

## 1. 実施した修正項目一覧

### ① V1 E6: タスク選択性差（コントラスト）によるSite選定とNo-Go処理の実装
- **変更前**:
  - E3 Discovery結果の Reader 最大層・Self 最大層を個別単独で選択。
  - Reader site と Self site が同一層となった場合、Self site を強制的に「2番目のピーク層」に置き換えて実行（人工的な乖離の創出）。
  - E3 結果ファイルが存在しない場合、未指定時に $0.5(L-1)$ / $0.6(L-1)$ へフォールバック。
- **変更後** (`v1/primary/phase_c/run_e6_specialization.py`):
  - タスク選択性コントラストを導入:
    $$S_R(l) = C_R(l) - C_S(l)$$
    $$S_S(l) = C_S(l) - C_R(l)$$
  - 最大選択性層をそれぞれ $\arg\max_l S_R(l)$、$\arg\max_l S_S(l)$ として厳密に抽出。
  - ピーク層が同一、あるいは最大選択性が正でない（$\le 0$）場合は、人工的な第2ピークへの差し替えを一切行わず、**No-Go / Negative Result**（`status: "no_distinct_sites_identified"`）として判定・記録し、正常にスキップ（早期リターン）するよう改修。
  - E3 Discovery CSV が存在しない場合は、heuristic fallback を完全撤去し、`FileNotFoundError` を送出。

### ② "Double Dissociation" から "Task-Specific Causal Specialization" への表記完全統一
- 科学的な妥当性（二重解離の厳密な定義と不要な査読者摩擦の回避）に基づき、コード・ドキュメント全体の表記を統一:
  - `v1/primary/phase_c/run_e6_specialization.py`
  - `v1/primary/phase_c/summarize_phase_c.py`
  - `src/affective_empathy_eval/run.py`
  - `scripts/run_production_v1.sh`
  - `v1/README.md`
  - `docs/` 内の各仕様書

### ③ V3 RQ2: Discovery Role の明記
- `v3/primary/run_rq2_spatiotemporal_maps.py`:
  - 探索的解析であることを明示するため、出力 `results["analysis_role"] = "discovery"` を追加。
  - 保存される `manifest_rq2_{fam_key}.json` の `config` および `metadata` に `"analysis_role": "discovery"` を追加。
  - `results/derived/v3_spatiotemporal_summary.json` にも `"analysis_role": "discovery"` を付与。

### ④ V2 Legacy スクリプト・設定ファイルの隔離
- 単一モデル（Qwen専用時代）の旧スクリプトおよび古いハッシュ設定ファイルを `legacy/` フォルダへ移動し、Primary Multi-Family 実行環境と完全に分離:
  - `v2/scripts/run_lambda_dose_response.py` $\rightarrow$ `v2/scripts/legacy/`
  - `v2/scripts/run_module_probing.py` $\rightarrow$ `v2/scripts/legacy/`
  - `v2/scripts/run_all_models.sh` $\rightarrow$ `v2/scripts/legacy/`
  - `v2/REPRODUCIBILITY.md` $\rightarrow$ `v2/scripts/legacy/`
  - `v2/configs/prompt_hashes.json` $\rightarrow$ `v2/configs/legacy/`

---

## 2. 状態検証結果

1. **ファイル配置検証**:
   - `v2/scripts/legacy/` 内に旧スクリプト 4 ファイルが正しく配置。
   - `v2/configs/legacy/` 内に `prompt_hashes.json` が配置。
   - `v2/` 直下には最新の `README.md` のみが存在。
2. **Results 初期化状態**:
   - `results/`, `v1/results/`, `v2/results/`, `v3/results/` ともに実データは存在せず、`.gitkeep` のみが配置されたクリーン状態。
3. **再現性・運用ルール準拠**:
   - `AGENTS.md` の仮想環境ルール、生データ非破壊ルール、結果上書き禁止ルール、原データ読み取り専用ルールに完全に適合。

---

## 3. 本番実行用コマンド案内

全再実行（Primary 4 Family: Qwen, Llama, Gemma, OLMo）は、以下の手順で GPU 環境にて実行可能です。

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

# または全ステージ一括実行:
# bash scripts/run_production_all.sh cuda:0
```
