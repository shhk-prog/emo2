# タスクリスト: 全再実行・公開に向けた最終完全化 (Final Pre-execution Refinements)

## 目的
全再実行（GPU本実行）および公開に向けて残存していた以下の重要課題をすべて解消し、完全な再現性と整合性を確保する：
1. 統合 CLI の `--stage v1` を Phase A/B/C/E6/summarize の完全パイプライン実行に改修
2. Behavioral runner を EmoBank + AIPsy、Base + Instruct の全条件実行に改修
3. V1 Primary コードを共通 ModelRegistry (`add_model_selection_args`, `resolve_models_from_args`) に完全対応
4. V1 Phase B の Layer 14 固定を廃止し、共通 relative depth ($d \approx 0.5$) / peak layer からの動的算出へ移行
5. `configs/v3_experiments.yaml` から具体的モデル ID を全廃し、`target_family: qwen` 等からレジストリ動的解決
6. `scripts/run_scale_validation.py` の固定ダミー値を撤廃し、V2 Primary を `--model-set scale_validation` で実際に走らせるラッパーに改修
7. 旧 V1/V3 scripts および旧 neutralization / collapse ドキュメントを `legacy/` へ隔離
8. `KNOWN_MODEL_DIMS` の未知モデルに対する `(28, 1536)` 誤魔化しフォールバックを廃止（`raise ValueError`）
9. `--base-model` / `--instruct-model` の CLI override 時、`--family` 指定を必須化して誤爆を防止
10. V3 Path Mediation のコメント・docstring から NDE/NIE を完全撤廃、Discovery プローブを `GroupKFold` 化
11. V3 RQ1 Self vs Topic control の位置づけ（除外用統制）の明確化
12. `__pycache__` / `.pyc` の完全削除、および `results/` のクリーン初期化

---

## タスク一覧

- [x] 1. **ドキュメント準備**:
  - [x] `docs/final_repo_refinements_for_rerun/task.md` 作成
  - [x] `docs/final_repo_refinements_for_rerun/implementation_plan.md` 作成・承認
- [x] 2. **共通モデルレジストリ・安全機構の強化 (`src/affective_empathy_eval/models/registry.py`)**:
  - [x] 未知モデル時の Qwen (28, 1536) フォールバックを廃止し `ValueError` を送出
  - [x] `--base-model` / `--instruct-model` 指定時に `--family` 必須チェックを追加
- [x] 3. **V1 Primary の完全レジストリ化 & Phase B 層固定廃止**:
  - [x] `v1/primary/run_phase_a.py`: `add_model_selection_args` 対応
  - [x] `v1/primary/run_phase_b.py`: Layer 14 固定廃止（relative depth $d$ または peak から算出）、レジストリ対応
  - [x] `v1/primary/run_phase_c.py`: レジストリ対応
  - [x] `v1/primary/phase_c/run_e6_specialization.py`: レジストリ対応
- [x] 4. **統合 CLI の V1 / Behavioral 完全パイプライン化 (`src/affective_empathy_eval/run.py`)**:
  - [x] `run_v1()`: Phase A $\rightarrow$ Phase B $\rightarrow$ Phase C $\rightarrow$ E6 $\rightarrow$ summarize を全モデルへ fan-out
  - [x] `run_behavioral()`: EmoBank と AIPsy の双方を Base / Instruct 全モデルで実行
- [x] 5. **`configs/v3_experiments.yaml` の具体的モデル ID 排除**:
  - [x] `target_model` / `confirmatory_models` のハードコードを削除し、`target_family: qwen` および `model_set` から解決
  - [x] V3 スクリプト側での完全動的解決
- [x] 6. **Scale Validation の固定疑似結果廃止 (`scripts/run_scale_validation.py`)**:
  - [x] 固定ダミー値を撤廃し、V2 Primary を `--model-set scale_validation` で実行するラッパーに改修
- [x] 7. **V3 Path Mediation の NDE/NIE 撤廃 & GroupKFold 化 & Topic Control 整理**:
  - [x] コード内・docstring 内の残存 NDE/NIE 表記を安全な媒介用語に完全統一
  - [x] Discovery 探索のプロービングを `GroupKFold(groups=pair_id)` に改修
  - [x] RQ1 Self vs Topic control の位置づけを明確化
- [x] 8. **旧 scripts・docs の legacy 隔離**:
  - [x] `v1/scripts/legacy/` を作成し旧スクリプト・旧レポート生成を退避
  - [x] `v3/docs/legacy/` を作成し旧 neutralization ドキュメント（`paper_restructured.md`, `paper2.md` 等）を退避
  - [x] `v3/scripts/legacy/` を作成し旧プロットスクリプト等を退避
- [x] 9. **最終クリーンアップとテスト・dry-run 検証**:
  - [x] `__pycache__` / `.pyc` 完全削除
  - [x] 全再実行用 results のクリーンアップ
  - [x] `.venv/bin/pytest -q` 全件パス確認 (46 passed)
  - [x] `python -m affective_empathy_eval.run` 全ステージの dry-run スモークテスト
  - [x] `walkthrough.md` の作成と保存
