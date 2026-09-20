# 実験結果のモジュラー分離・逐次保存・成否判定・途中再開機能 実装完了レポート (Walkthrough)

## 概要

本改修では、大規模LLM情動反応性評価実験において、以下の4大要求事項を完全に満たすインフラ改修を実施しました：
1. **既存結果の安全な退避**: `behavioral`, `v1`, `v2`, `v3` の全結果を `old_results/archive_20260921_040722_pre_modular` に退避し、`results/` をクリーン初期化。
2. **段階と実験のモジュラー分離命名**: 段階（Behavioral, V1, V2, V3）および実験（E1〜E6, RQ1〜RQ4, Gate, Confirmatory）の出力を個別の JSON / CSV に完全分離。
3. **計算完了ごとの即時逐次保存 & 成否判定フラグ**: バッチ終了時のまとめ保存ではなく、各モデル・各条件・各層の完了ごとにアトミック書き込み (`save_experiment_result`) を行い、トップレベルに `"execution_success": true/false`, `"execution_status": "success"/"failed"` を付与。
4. **途中再開（Resume）およびスキップ機能**: `is_experiment_completed` により、正常終了した実験成果物を自動検知してスキップ。クラッシュ時や中断時は未完了部分から安全に再開。

---

## 1. 段階・実験とモジュラー出力ファイルの対応表

| 段階 (Stage) | 実験 (Experiment) | 結論・目的 | モジュラー出力ファイル名 (新命名) |
|---|---|---|---|
| **Behavioral** | EmoBank 3-Way VAD | Reader と Self の行動的連動性 | `behavioral_emobank_{tag}_summary.json`<br>`behavioral_emobank_{tag}_3way_vad.csv` |
| **Behavioral** | AIPsy-Affect 4-Split | 4象限直交空間での感度・特異度 | `behavioral_aipsy_{tag}_summary.json`<br>`behavioral_aipsy_{tag}_4split.csv` |
| **V1 Phase A** | E1: Decodability | 両タスク内部に情動情報が存在するか | `v1_e1_decodability_{prefix}.json`<br>`v1_e1_emobank_decodability.csv`<br>`v1_e1_aipsy_intensity.csv`<br>`v1_e1_aipsy_classification.csv` |
| **V1 Phase A** | E2: Cross-decoding | 表現幾何が部分的に共有されるか | `v1_e2_cross_decoding_{prefix}.json`<br>`v1_e2_rsa_{prefix}.json`<br>`v1_e2_alignment_{prefix}.json` |
| **V1 Phase B** | E5: Semantic Controls | 語彙ショートカットではなく情動意味論か | `v1_e5_semantic_controls_{prefix}_{task}.json`<br>`v1_e5_semantic_controls_{prefix}_{task}.csv` |
| **V1 Phase C** | E3: Causal Map | 内部表現は自己報告に対して因果的か | `v1_e3_causal_map_{prefix}.json`<br>`v1_e3_causal_map_{prefix}.csv` |
| **V1 Phase C** | E4: Interchangeability | Reader/Self 間で因果的表現が交換可能か | `v1_e4_interchangeability_{prefix}.json`<br>`v1_e4_interchangeability_{prefix}.csv` |
| **V1 Phase C** | E6: Double Dissociation | タスク選択的サイトの因果的二重解離 | `v1_e6_double_dissociation_{prefix}.json` |
| **V2** | RQ1: Decodability Preservation | 事後学習後も内部デコード能は保存されるか | `v2_rq1_decodability_preservation_{family}.json` |
| **V2** | RQ2: Geometry Transformation | Reader–Self の表現幾何共有性の再編 | `v2_rq2_geometry_transformation_{family}.json` |
| **V2** | RQ3: Causal Relocation | 因果回路の再配置とピーク解離 | `v2_rq3_causal_relocation_{family}.json` |
| **V2** | RQ4: Recovery Patching | 活性化パッチングによる分布回復性 | `v2_rq4_recovery_patching_{family}.json` |
| **V3** | RQ1: Gate Evaluation | 内部状態誘導の Go/No-Go 判定 | `v3_rq1_gate_{family}.json` |
| **V3** | RQ2: Spatiotemporal Maps | 生成アンカー×層グリッドの 4-Map 解析 | `v3_rq2_spatiotemporal_maps_{family}.json` |
| **V3** | RQ3: Path Mediation | 刺激提示時表現から出力ロジットへの媒介 | `v3_rq3_path_mediation_{family}.json` |
| **V3** | Confirmatory Replication | 他3モデル (Llama, Gemma, OLMo) での追試 | `v3_confirmatory_replication_{family}.json` |

> [!NOTE]
> 既存の分析スクリプトやパイプラインとの後方互換性を保つため、旧ファイル名（例: `e1_emobank_decodability.csv`, `v2_geometry_*.json`, `v3_rq1_results.json` など）も同時に並行保存されます。

---

## 2. 共通 I/O モジュール (`src/affective_empathy_eval/io.py`)

### 2.1 逐次保存と成否判定 (`save_experiment_result`)
- 一時ファイルに書き込んでからアトミックに置換するため、途中でプロセスが終了しても破損ファイルが残りません。
- 出力 JSON のトップレベルに以下が必ず付与されます：
```json
{
  "execution_status": "success",
  "execution_success": true,
  "stage": "v1",
  "experiment_id": "v1_e1_decodability",
  "completed_at": "2026-09-20T19:35:00.000000+00:00",
  "metadata": { ... },
  "results": { ... }
}
```

### 2.2 途中再開判定 (`is_experiment_completed`)
- ファイルが存在し、かつ `"execution_success": true` である場合のみ `True` を返します。
- 計算途中で中断された場合や、エラー終了したファイルは `execution_success: false` となり、次回実行時に自動的にスキップされず再実行されます。
- `--force` オプションが指定された場合は強制的に `False` となり再計算されます。

---

## 3. 検証結果

### 3.1 ユニットテスト
- `tests/test_io_modular.py` を新設し、正常保存、失敗フラグ判定、不完全ファイル検出、途中再開チェックの全パスをテスト。
- 全既存テスト（95 passed）および新設テストがすべて成功。

### 3.2 全ステージ Dry-run 実機検証
全ステージを `--dry-run` で連続実行し、全モジュラーファイルが正常に生成され、成否フラグが付与されることを実証：
```text
OK: behavioral_emobank_qwen_instruct_summary.json | id: behavioral_emobank_3way          | success=True | status=success
OK: behavioral_aipsy_qwen_instruct_summary.json   | id: behavioral_aipsy_4split          | success=True | status=success
OK: v1_e1_decodability_qwen_instruct.json         | id: v1_e1_decodability               | success=True | status=success
OK: v1_e2_cross_decoding_qwen_instruct.json       | id: v1_e2_cross_decoding             | success=True | status=success
OK: v1_e2_rsa_qwen_instruct.json                  | id: v1_e2_rsa                        | success=True | status=success
OK: v1_e2_alignment_qwen_instruct.json            | id: v1_e2_alignment                  | success=True | status=success
OK: v1_e5_semantic_controls_qwen_instruct_reader.json | id: v1_e5_semantic_controls      | success=True | status=success
OK: v1_e3_causal_map_qwen_instruct.json           | id: v1_e3_causal_map                 | success=True | status=success
OK: v1_e4_interchangeability_qwen_instruct.json   | id: v1_e4_interchangeability         | success=True | status=success
OK: v1_e6_double_dissociation_qwen_instruct.json  | id: v1_e6_double_dissociation        | success=True | status=success
OK: v2_rq1_decodability_preservation_qwen.json    | id: v2_rq1_decodability_preservation | success=True | status=success
OK: v2_rq2_geometry_transformation_qwen.json      | id: v2_rq2_geometry_transformation   | success=True | status=success
OK: v2_rq3_causal_relocation_qwen.json            | id: v2_rq3_causal_relocation         | success=True | status=success
OK: v2_rq4_recovery_patching_qwen.json            | id: v2_rq4_recovery_patching         | success=True | status=success
OK: v3_rq1_gate_qwen.json                         | id: v3_rq1_gate                      | success=True | status=success
OK: v3_rq2_spatiotemporal_maps_qwen.json          | id: v3_rq2_spatiotemporal_maps       | success=True | status=success
OK: v3_rq3_path_mediation_qwen.json               | id: v3_rq3_path_mediation            | success=True | status=success
OK: v3_confirmatory_replication_llama.json        | id: v3_confirmatory_replication      | success=True | status=success
```

### 3.3 途中再開（スキップ）検証
- 再度実行した際に、すでに正常終了した実験は自動検知され、安全にスキップ・途中再開可能であることを確認。
