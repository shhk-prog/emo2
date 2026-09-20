# 実装計画書: 実験結果のアーカイブ退避、出力モジュール化・逐次保存・途中再開機能

## 1. 目的と背景
ユーザーからの指示に基づき、以下の3つの主要改修を実施します：
1. **既存結果の完全退避**: 現在の `behavioral/results/`, `v1/results/`, `v2/results/`, `v3/results/` の全結果を `old_results/archive_<timestamp>/` へ安全に移動し、`results/` をクリーン初期化する。
2. **段階・実験ごとのファイル名明記と分割**: 同じ段階内の複数実験（例: Behavioral の EmoBank/AIPsy、V1 Phase A の E1 Decodability / E2 Cross-decoding, RSA, Alignment、V2 の RQ1 Decodability / RQ2 Geometry 等）を出力ファイル単位で明確に分離・命名する。
3. **逐次保存・成否判定・途中再開（Resume）**: すべてのサブ実験が完了してから一括保存するのではなく、サブ実験・モデル・条件ごとに即時逐次保存する。各ファイルに `"execution_success": true/false` を付与し、途中で停止した場合も `true` のファイルはスキップして未完了部分から再開できるようにする。

---

## 2. 段階と実験の対応・出力ファイル命名規則

ユーザー指定の結論テーブルに基づき、各成果物（Raw / Derived）のファイル命名規則を以下のように統一・分離します。

| 段階 | 実験 | 結論 | 新 Raw 出力ファイル名（例） | 新 Derived 出力ファイル名（例） |
|---|---|---|---|---|
| **Behavioral** | EmoBank (Sensitivity/Coupling) | ReaderとSelfは行動的に連動する | `behavioral_emobank_<tag>_raw.json` / `.csv` | `behavioral_emobank_summary.json` / `metrics.csv` |
| **Behavioral** | AIPsy (Dose/Specificity/Coupling) | 臨床刺激で段階的・特異的に変位 | `behavioral_aipsy_<tag>_raw.json` / `.csv` | `behavioral_aipsy_summary.json` / `metrics.csv` |
| **V1 E1** | Decodability | 両task内部にaffect情報がある | `v1_e1_decodability_<tag>.json` | `v1_e1_decodability_summary.json` |
| **V1 E2** | Cross-decoding | 表現幾何が部分的に共有される (直接射影) | `v1_e2_cross_decoding_<tag>.json` | `v1_e2_cross_decoding_summary.json` |
| **V1 E2** | RSA (Representational Similarity) | 幾何構造の類似性 | `v1_e2_rsa_<tag>.json` | `v1_e2_rsa_summary.json` |
| **V1 E2** | Procrustes Alignment | 表現空間の回転整列 | `v1_e2_alignment_<tag>.json` | `v1_e2_alignment_summary.json` |
| **V1 E5** | Semantic controls (Phase B) | 語彙shortcutだけではない | `v1_e5_semantic_controls_<tag>_<task>.json` | `v1_e5_semantic_controls_summary.json` |
| **V1 E3** | Causal map (Phase C) | 一部の内部表現は出力に因果的 | `v1_e3_causal_map_<tag>.json` | `v1_e3_causal_map_summary.json` |
| **V1 E4** | Interchangeability (Phase C) | Reader/Self間で一部因果表現を共有 | `v1_e4_interchangeability_<tag>.json` | `v1_e4_interchangeability_summary.json` |
| **V1 E6** | Double dissociation | 同時にtask-specific specializationもある | `v1_e6_double_dissociation_<tag>.json` | `v1_e6_double_dissociation_summary.json` |
| **V2 RQ1** | Decodability preservation | Instructでも情報は残る | `v2_rq1_decodability_preservation_<fam>.json` | `v2_rq1_decodability_summary.json` |
| **V2 RQ2** | Geometry transformation | Base→Instructでgeometryが変わる | `v2_rq2_geometry_transformation_<fam>.json` | `v2_rq2_geometry_summary.json` |
| **V2 RQ3** | Causal relocation | causal leverageの場所も変わる | `v2_rq3_causal_relocation_<fam>.json` | `v2_rq3_causal_summary.json` |
| **V2 RQ4** | Recovery patching | 単一output suppressionだけでは説明しにくい | `v2_rq4_recovery_patching_<fam>.json` | `v2_rq4_recovery_summary.json` |
| **V3 RQ1** | Sufficiency/Necessity Gate | affect方向が本当にSelfを動かす | `v3_rq1_gate_<fam>.json` | `v3_rq1_gate_decision.json` |
| **V3 RQ2** | D/C spatiotemporal maps | 情報の存在と利用を時空間的に分離 | `v3_rq2_spatiotemporal_maps_<fam>.json` | `v3_rq2_spatiotemporal_summary.json` |
| **V3 RQ3** | Path mediation | どの計算経路を通って利用されるか | `v3_rq3_path_mediation_<fam>.json` | `v3_rq3_path_mediation_summary.json` |
| **V3 Confirmation**| Frozen-site replication | 他model familyでも同様か確認 | `v3_confirmatory_replication_<fam>.json` | `v3_confirmatory_summary.json` |

---

## 3. 逐次保存・成否判定・途中再開の実装仕様

### 3.1 標準メタデータ形式
すべての実験結果 JSON のトップレベルに以下の共通メタデータを付与します：
```json
{
  "execution_status": "success",          // "success" または "failed"
  "execution_success": true,               // true または false (明示的真偽値)
  "stage": "v1",                          // "behavioral", "v1", "v2", "v3"
  "experiment_id": "v1_e1_decodability",  // 上記の実験ID
  "model_id": "Qwen/Qwen2.5-1.5B",
  "completed_at": "2026-09-21T04:10:00Z",
  "run_id": "...",
  "config_hash": "...",
  "data": { ... }                         // 実験固有の結果データ
}
```

### 3.2 逐次保存（Incremental Save）
- **V1 Phase A**: E1 (Decodability) の計算が終わったら直ちに `v1_e1_decodability_*.json` を保存。続いて E2 Cross-decoding, RSA, Alignment が終わるごとに即時保存。
- **V2 RQ1/RQ2**: RQ1 のデコード能比較が終わったら直ちに `v2_rq1_*.json` を保存。続いて RQ2 幾何変換が終わったら直ちに `v2_rq2_*.json` を保存。
- **V3 Confirmatory**: 各モデル（Llama, Gemma, OLMo）の追試計算が完了するごとに、そのモデルの `v3_confirmatory_replication_<fam>.json` を即時保存。

### 3.3 途中再開（Resume）判定
各サブ実験の開始前に既存ファイルを検証：
```python
def should_skip_experiment(target_path: Path, manifest_path: Path, expected_config: dict, force: bool = False) -> bool:
    if force or not target_path.exists():
        return False
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # execution_success が明示的に True であること
        if not data.get("execution_success", False):
            return False
        # manifest の照合（存在する場合）
        if manifest_path.exists() and not is_manifest_matching(...):
            return False
        return True
    except Exception:
        return False
```
- 中断された（または `execution_success: false` の）不完全なファイルは自動検知されて再実行されます。
- 正常終了した実験はスキップされ、途中から即座に再開します。

---

## 4. 実行手順

1. **既存結果のアーカイブ**:
   - `scripts/archive_all_results_for_modular_run.py` を作成・実行。
   - `old_results/archive_20260921_pre_modular/` へ全ステージの results を安全退避。
   - 各ステージの `results/raw`, `results/derived`, `results/checkpoints`, `results/cache` に `.gitkeep` を配置。
2. **スクリプト改修**:
   - `behavioral/primary/run_behavioral_*.py`
   - `v1/primary/run_phase_a.py` (E1, E2 分割 & 逐次保存)
   - `v1/primary/run_phase_b.py` (E5 明記 & 逐次保存)
   - `v1/primary/run_phase_c.py` (E3, E4 分割 & 逐次保存)
   - `v1/primary/phase_c/run_e6_specialization.py` (E6 明記 & 逐次保存)
   - `v2/primary/run_rq1_rq2_cross_decoding.py` (RQ1, RQ2 分割 & 逐次保存)
   - `v2/primary/run_rq3_causal_map.py` (RQ3 明記)
   - `v2/primary/run_rq4_recovery_patching.py` (RQ4 明記)
   - `v3/primary/run_rq1_state_induction.py` (RQ1 Gate 明記)
   - `v3/primary/run_rq2_spatiotemporal_maps.py` (RQ2 明記)
   - `v3/primary/run_rq3_path_mediation.py` (RQ3 明記)
   - `v3/primary/run_confirmatory_replication.py` (Confirmatory モデルごと逐次保存)
3. **統合ランナー・テストの同期**:
   - `src/affective_empathy_eval/run.py`
   - 各集計スクリプト (`behavioral/analysis/`, `v1/primary/phase_c/summarize_phase_c.py`, `scripts/reaggregate_*.py`)
   - 単体テストのパス参照
4. **検証**:
   - `compileall`
   - `pytest`
   - Stage all dry-run（途中停止・途中再開の動作テスト含む）

---

## 5. ユーザー確認事項
- 既存のすべての結果を `old_results/archive_20260921_pre_modular/` に退避して、`results/` をクリーンにして問題ないか。
- 上記の段階・実験ごとのファイル命名規則に問題がないか。
