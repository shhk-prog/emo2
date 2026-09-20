# コード監査指摘事項（21項目）対応完了 Walkthrough レポート

本ドキュメントは、論文「情動反応性評価（4-Stage構造）」の最新コード監査で指摘された全21項目（P1: 1〜12, P2: 13〜21）に対する修正内容、設計上の判断、および検証結果をまとめたものです。

---

## 1. 修正の概要と対応マトリクス

| No. | カテゴリ | 対象ファイル | 修正内容と改善点 |
|---|---|---|---|
| **1** | P1 (V1-A) | `v1/primary/run_phase_a.py` | キャッシュ判定で `--dataset` 引数（both/aipsy）に応じた AIPsy 3成果物（`aipsy_classification_probe_metrics.csv` 等）の存在・行数を厳格検証。 |
| **2** | P1 (V1-B) | `v1/primary/run_phase_b.py` | `StratifiedKFold` へのフォールバックと `test_pair_ids = train_pair_ids` を完全削除。`GroupKFold` のみ使用し、同一ペアが跨ぐ場合は `ValueError` を送出。`assert train_pair_ids.isdisjoint(test_pair_ids)` を追加。 |
| **3** | P1 (V1-B) | `v1/primary/prepare_v1_phase_b_controls.py`<br>`v1/primary/run_phase_b.py` | `paraphrase_fallback`, `reversal_fallback`, `paraphrase_method`, `reversal_method` 列をデータに追加。Primary 解析はフォールバックなしの厳密ペアのみ抽出し、Sensitivity として全ペア解析を併記。 |
| **4** | P1 (V1-A) | `v1/primary/run_phase_a.py` | Direct cross-decoding で負の $R^2$ を 0 にクリップせず生値 `direct_transfer_score_raw` をそのまま保存・判定に使用。 |
| **5** | P1 (V1-A) | `v1/primary/run_phase_a.py` | AIPsy 分類プローブ評価でエラー・例外時に 0.5 や 0 を代入せず `np.nan` を代入。`status`, `failure_reason`, `n_valid_folds` 列を記録。 |
| **6** | P1 (V2) | `v2/primary/run_rq1_rq2_cross_decoding.py` | `summary_metrics` を `primary_matched_plain`, `secondary_native_chat`, `format_effect` に明示分離。クロスファミリー Bootstrap CI を Primary（matched-plain）から算出。 |
| **7** | P1 (V2) | `v2/primary/run_rq4_recovery_patching.py` | サンプル不足時の `eval_indices = train_indices` フォールバックを完全撤廃し `ValueError` 送出。`assert train_pairs.isdisjoint(eval_pairs)` を追加。 |
| **8** | P1 (V2) | `v2/primary/run_rq4_recovery_patching.py` | ハードコード `0.7` を排除し、`v2_cfg.get("dataset", {}).get("train_ratio", 0.7)` から取得。 |
| **9** | P1 (V3) | `v3/primary/run_rq2_spatiotemporal_maps.py` | 事前定義アンカー（`a_priori_test_stage_v="pre_V"`, `a_priori_test_stage_a="pre_A"`）と、全 stage × 全 layer 探索によるデータドリブンピーク（`empirical_peak_stage_v/a`）の両方を算出・保存。 |
| **10** | P1 (V3) | `v3/primary/run_confirmatory_replication.py` | ハードコード `"pre_V"`, `"pre_A"` を完全撤廃。`frozen_confirmatory_sites.json` から動的に読み取った `stage_v`, `stage_a` に基づき時間的コントラストを算出。 |
| **11** | P1 (V3) | `v3/primary/run_rq3_path_mediation.py` | ハードコード `seed=42`, `ratio=0.5` を排除し、`v3_cfg` から読み込み。`assert disc_pairs.isdisjoint(conf_pairs)` を追加。`frozen_confirmatory_sites.json` に temporal stage 情報を統合出力。 |
| **12** | P1 (V3) | `v3/primary/run_rq1_state_induction.py` | ハードコード `seed=42`, `43` を排除。`base_seed = int(v3_cfg.get("seed", 42))` から統一的に派生。 |
| **13** | P2 (基盤) | `src/affective_empathy_eval/manifests.py` | AGENTS.md 5.3 準拠の一意な `generate_run_id(git_sha, config_hash)`（`YYYYMMDDTHHMMSSZ_<git>_<config>`）を実装し `RunManifest` に記録。 |
| **14** | P2 (基盤) | `configs/models.yaml`<br>`src/affective_empathy_eval/models/registry.py` | `base_revision`, `instruct_revision`（デフォルト: `"main"`）を定義し、`ModelSpec` およびモデルロード時に Hugging Face revision を固定。 |
| **15** | P2 (基盤) | `src/affective_empathy_eval/manifests.py` | `compute_prompt_hash` を実装し、プロンプトテンプレートの SHA256 ハッシュを `RunManifest` に記録・検証。 |
| **16** | P2 (基盤) | `tests/test_likelihood.py` | BPE subword merge が起きる境界条件（prompt 末尾と candidate 先頭の結合）に対する `prepare_joint_sequence_with_boundary` の整合性および `require_strict_prefix` 例外検出テストを追加。 |
| **17** | P2 (V1-C) | `v1/primary/run_phase_c.py` | `is_manifest_matching` を用いて、モデル名、VAD_729 空間、dry_run フラグ、設定ハッシュの一致を厳格に照合してからスキップするように強化。 |
| **18** | P2 (文書) | `README.md` | Candidate-space sensitivity の事前断定的な文言を改訂。同一刺激サブセットでの Pearson $r$, Spearman $\rho$, 方向一致率（direction agreement）を測定・報告する客観的記述に修正。 |
| **19** | P2 (文書) | `README.md` | Teacher-forced sequence likelihood の定義および測定プロトコル表記を確認・整合。 |
| **20** | P2 (設定) | `configs/v1_experiments.yaml`<br>`v1/primary/run_phase_b.py` | V1 Phase B の `seed: 42`, `train_ratio: 0.7`, `cv_folds: 5` を外部 YAML に外出し・連動。 |
| **21** | P2 (設定) | `configs/v1_experiments.yaml`<br>`v1/primary/run_phase_c.py`<br>`v1/primary/phase_c/run_e6_specialization.py` | V1 Phase C / E6 の `discovery_ratio: 0.5` を外部 YAML から読み込み、`assert discovery_pairs.isdisjoint(confirmation_pairs)` を追加。 |

---

## 2. 重要な設計上の判断（Design Decisions）

### 2.1 V3 RQ2 / Confirmatory の「いつ（When）」の整合化（Items 9, 10）
- **課題**: `run_rq2_spatiotemporal_maps.py` は事前登録アンカー `pre_V`, `pre_A` だけでなく、全マップ探索によるピーク（empirical peak）を見つける探索的側面を持っていたが、Confirmatory 側（`run_confirmatory_replication.py`）では `"pre_V"`, `"pre_A"` がハードコードされ、アーティファクトの動的受渡が断絶していた。
- **解決策**:
  - RQ2 出力 `v3_rq2_causal_sites.json` に `a_priori_test_stage_v/a`（事前アンカー）と `empirical_peak_stage_v/a`（データドリブン探索ピーク）を両方記録。
  - RQ3 の Mediator 解析を経て生成される最終検証アーティファクト `frozen_confirmatory_sites.json` に `temporal_stage_v`, `temporal_stage_a` を明示的に保存。
  - Confirmatory スクリプトは `frozen_confirmatory_sites.json` から動的に読み取り、その stage に基づいて時間的コントラストを算出するプロトコルに統一。

### 2.2 データリークおよび不正フォールバックの完全根絶（Items 2, 7, 11, 21）
- **課題**: サンプル数が少ない場合やエッジケースにおいて、同一の `pair_id` が train と test/eval の両方に割り当てられる、あるいは `test = train` で上書きする危険なフォールバックが存在していた。
- **解決策**:
  - Phase B, V2 RQ4, V3 RQ3, V1 Phase C / E6 のすべてにおいて、フォールバックを完全撤廃。
  - 各分割後に必ず `assert train_pairs.isdisjoint(test_pairs)` を実行し、数学的に互いに素（disjoint）であることを保証。

### 2.3 再現性基盤の強化（Items 13, 14, 15）
- **`run_id` 体系**: AGENTS.md 5.3 に準拠し、`YYYYMMDDTHHMMSSZ_<git-short-sha>_<config-short-hash>` を自動発行する `generate_run_id()` を `affective_empathy_eval.manifests` に実装。
- **Hugging Face revision**: `configs/models.yaml` に `base_revision: "main"`, `instruct_revision: "main"` を追加し、モデル登録レイヤーで追跡。
- **Prompt Hash**: `compute_prompt_hash()` によりテンプレート文字列のハッシュを manifest に記録。

---

## 3. テストと検証結果

### 3.1 ユニットテスト・回帰テストスイート
プロジェクト仮想環境（`.venv`）を用いてテストを実行：

```bash
.venv/bin/python -m pytest -q
```
**結果**:
```text
.............................................................................. [ 82%]
.................                                                             [100%]
95 passed, 1 deselected, 5 warnings in 9.06s
```
- 全 95 件のテストが 100% PASS。
- 新規追加した `tests/test_likelihood.py::test_prepare_joint_sequence_boundary_and_bpe_merge` を含むすべての検証が成功。

### 3.2 パイプライン dry-run 動作検証
全 primary スクリプトについて `--dry-run` を実行し、設定読み込み、アーティファクト入出力、disjoint 検証が正常に機能することを確認：

1. **V1 Phase A**: `v1/primary/run_phase_a.py --family qwen --dry-run` $\to$ **PASS**
2. **V1 Phase B (Reader)**: `v1/primary/run_phase_b.py --family qwen --task-type reader --dry-run` $\to$ **PASS**
3. **V1 Phase B (Self)**: `v1/primary/run_phase_b.py --family qwen --task-type self --dry-run` $\to$ **PASS**
4. **V1 Phase C**: `v1/primary/run_phase_c.py --family qwen --dry-run` $\to$ **PASS**
5. **V1 E6 Specialization**: `v1/primary/phase_c/run_e6_specialization.py --family qwen --dry-run` $\to$ **PASS**
6. **V2 RQ1/RQ2**: `v2/primary/run_rq1_rq2_cross_decoding.py --family qwen --dry-run` $\to$ **PASS**
7. **V2 RQ4**: `v2/primary/run_rq4_recovery_patching.py --family qwen --dry-run` $\to$ **PASS**
8. **V3 RQ1**: `v3/primary/run_rq1_state_induction.py --family qwen --dry-run` $\to$ **PASS**
9. **V3 RQ2**: `v3/primary/run_rq2_spatiotemporal_maps.py --family qwen --dry-run` $\to$ **PASS**
10. **V3 RQ3**: `v3/primary/run_rq3_path_mediation.py --family qwen --dry-run` $\to$ **PASS**
11. **V3 Confirmatory**: `v3/primary/run_confirmatory_replication.py --family llama --dry-run` $\to$ **PASS**

以上の修正により、監査指摘のあった全21項目が完全に是正され、厳密な科学的再現性と整合性が確保されました。
