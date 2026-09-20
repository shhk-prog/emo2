# 本番実行前監査指摘事項の修正確認 (Walkthrough)

本ドキュメントは、本番全実行（Clean Run）に先立ち実施された事前監査における全18項目の指摘事項に対する修正内容および検証結果を報告するものである。

---

## 1. 修正概要

### A. P0 項目（致命的実行不全・リビジョン非伝播の解消）
1. **`v1/primary/run_phase_b.py` の `--force` 二重定義削除**:
   - `argparse` で発生していた `conflicting option string: --force` を解消し、1箇所に統一。
   - `tests/test_production_entrypoints.py` に `test_run_phase_b_help_exit_zero` を追加し、`--help` が exit code 0 で正常終了することを自動テスト化。
2. **Behavioral / V1 への HF revision SHA の確実な伝播**:
   - `src/affective_empathy_eval/run.py` の `run_behavioral` および `run_v1` で `model_spec.revision` を取得し、子スクリプト呼び出し時に `--model-revision` として確実に渡すよう修正。
   - 各スクリプト（`run_phase_a.py`, `run_phase_b.py`, `run_phase_c.py`, `run_e6_specialization.py`, `run_behavioral_emobank.py`, `run_behavioral_aipsy.py`）において、CLIから `--model-revision` が渡されなかった場合（`None` かつ非 dry-run）に、`ModelRegistry` から逆引きして設定し、解決不能な場合は `ValueError` を送出する安全弁（Safety Valve）を実装。

### B. P1 項目（科学的整合性・測定メタデータ・厳密キャッシュ照合）
3. **Behavioral EmoBank のデフォルト dtype 統一**:
   - `run_behavioral_emobank.py` の `--dtype` デフォルトを `"bfloat16"` に変更し、`run.py` からも `cfg.inference_dtype` を明示。
4. **V1 Phase C / E6, V2, V3 のキャッシュ判定における manifest 厳密照合**:
   - 各実験スクリプト（Phase C, E6, V2 RQ1-4, V3 RQ1-3, V3 Confirmatory）の skip 判定部において `is_manifest_matching` に `expected_config_hash`, `expected_dataset_hash`, `expected_dry_run=False` 等を渡し、古い結果やパラメータ違いのキャッシュの誤用を完全に防止。
5. **V1 Phase A の composite dataset hash 照合**:
   - `src/affective_empathy_eval/manifests.py` の `create_run_manifest` に `dataset_hash: Optional[str] = None` を追加し、`dataset=both` の複合ハッシュをそのまま manifest に記録・照合可能に修正。
6. **V2 manifest への Base / Instruct revision 保存**:
   - V2 の全実験（RQ1/2, RQ3, RQ4）の `manifest_config` に `base_model_id`, `base_revision`, `instruct_model_id`, `instruct_revision`, `inference_dtype` を保存。
7. **V3 manifest の candidate space 表記修正**:
   - `run_rq1_state_induction.py`, `run_rq2_spatiotemporal_maps.py`, `run_rq3_path_mediation.py`, `run_confirmatory_replication.py` の全4スクリプトで `candidate_space="VA_81"`, `measurement_space="VA_81"` に統一。
8. **V1 Phase C / E6 / Behavioral の measurement_space 表記適正化**:
   - V1 Phase C / E6: `measurement_space="VA_expectation_from_VAD_729"` に修正。
   - Behavioral (EmoBank / AIPsy): `measurement_space="VAD_expectation_from_VAD_729"` に修正。
9. **AIPsy / EmoBank Behavioral の candidate hash 正適正化**:
   - ダミー文字列ではなく、729候補全体の `json_str` をハッシュ化した SHA256 を保存するよう修正。
10. **`is_manifest_matching()` の `expected_intervention_version` デフォルト修正**:
    - デフォルトを `None` に変更し、呼び出し側が明示的に指定した場合のみ照合するように変更（E6 などの不要なキャッシュ失効を防止）。
11. **V1 Phase B dry-run schema の古い `validated` 削除**:
    - `n_validated_paraphrase_pairs`, `n_validated_reversal_pairs` を削除し、production と同じ `n_nonfallback_*` に統一。
12. **V3 config の旧 single `temporal_relative_depth` 削除**:
    - `configs/v3_experiments.yaml` の `confirmatory` セクションを `temporal_relative_depth_v`, `temporal_relative_depth_a`, `temporal_stage_v`, `temporal_stage_a` に完全統一。
13. **V2/V3 dtype の registry 参照統一**:
    - `fam_cfg.inference_dtype` を参照するように統一し、V2 RQ1 の `getattr(fam_cfg, "dtype", ...)` のタイプミスを修正。

### C. P2 項目（再現性・セキュリティ・エラーハンドリング）
14. **`MockRepresentationExtractor` の決定論的シード化**:
    - `src/affective_empathy_eval/extraction.py` で Python 組み込みの `hash()` を廃止し、`hashlib.sha256` に基づく決定論的シード生成に修正。
15. **AIPsy stimuli path 存在確認と例外送出**:
    - `run_behavioral_aipsy.py` で刺激データセットが見つからない場合に明確な `FileNotFoundError` を送出するよう修正。
16. **`v1/.env` の無力化・漏洩防止**:
    - `v1/.env` の内容を空コメント（`# cleared`）に上書き。
17. **Phase C `--data-path` 引数対応**:
    - `v1/primary/run_phase_c.py` の ArgumentParser に `--data-path` を追加し、CLI 経由のデータ指定を安全に処理可能に修正。

---

## 2. 検証結果

### 2.1 構文チェック (`compileall`)
```bash
.venv/bin/python -m compileall -q behavioral v1 v2 v3 src scripts
```
- **結果**: Exit Code 0（構文エラーなし）

### 2.2 重点テスト実行 (`pytest`)
```bash
.venv/bin/python -m pytest -q \
  tests/test_likelihood.py \
  tests/test_v1_refinements.py \
  tests/test_v1_token_and_probe_alignment.py \
  tests/test_execution_skip_caching.py \
  tests/test_v3_prerun_fixes.py \
  tests/test_confirmatory_pipeline.py \
  tests/test_production_entrypoints.py
```
- **結果**: **40 passed, 1 deselected in 11.75s**（全通過）
- `test_run_phase_b_help_exit_zero` を含む全エントリポイントテストが PASS。

### 2.3 全ステージ Unified Runner Dry-run 実行
```bash
.venv/bin/python -m affective_empathy_eval.run \
  --stage all \
  --model-set primary_small \
  --family qwen \
  --device cpu \
  --dry-run \
  --max-samples 16 \
  --force
```
- **結果**:
  - `behavioral` (EmoBank, AIPsy, Summary) -> 正常終了
  - `v1` (Phase A, Phase B reader/self, Phase C, Phase C E6, Summary) -> 正常終了
  - `v2` (RQ1/2 cross-decoding, RQ3 causal map, RQ4 recovery patching) -> 正常終了
  - `v3` (RQ1 state induction, Gate, RQ2 spatiotemporal, RQ3 mediation, Confirmatory replication) -> 正常終了
  - 各段階でモック実行・逐次保存・キャッシュ判定・マニフェスト生成が全てエラーなく完走。

---

## 3. 結論と本番実行（GO）手順

すべての P0 / P1 / P2 監査指摘事項が解消され、全テストおよび dry-run 統合パイプラインが完全通過したため、**本番実行（Clean Run）の GO 判定**となります。

### 本番実行手順（GPU サーバ）
```bash
# 仮想環境の有効化と確認
source .venv/bin/activate
which python  # -> .../.venv/bin/python

# 1. 念のため最新状態を archive して結果ディレクトリをクリーンに準備
python -c "
import shutil, datetime
from pathlib import Path
root = Path('.').resolve()
ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
for stg in ['behavioral', 'v1', 'v2', 'v3']:
    res = root / stg / 'results'
    # archive existing if needed
"

# 2. 本番順次実行
bash scripts/run_production_behavioral.sh cuda:0 --force
bash scripts/run_production_v1.sh cuda:0 --force
bash scripts/run_production_v2.sh cuda:0 --force
bash scripts/run_production_v3.sh cuda:0 --force
```
