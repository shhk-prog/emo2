# 実装計画: E6 Task-Selectivity Site 選定・Fallback 撤去・表記統一・最終整理

全再実行（GPU本番実行）の直前段階として、ユーザーからご指摘いただいた実質3点および付随する整理項目を実施します。

---

## ユーザー確認が必要な事項 (User Review Required)

> [!IMPORTANT]
> **E6 サイト選定の No-Go 挙動について**:
> タスク選択性コントラスト $S_R(l) = C_R(l) - C_S(l)$, $S_S(l) = C_S(l) - C_R(l)$ において、両タスクの最大層が同一（または最大選択性 $\le 0$）となった場合、人工的な第2ピークへの置換を行わず、科学的誠実性に基づき「distinct task-selective sites were not identified」として **negative result / No-Go**（2x2 ablation は行わず、状態を記録して正常終了）とします。

---

## 提案される変更 (Proposed Changes)

### 1. V1 Phase C E6 サイト選定と Fallback 撤去
#### [MODIFY] [run_e6_specialization.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/phase_c/run_e6_specialization.py)
- `select_sites_from_e3()`:
  - $S_R(l) = C_R(l) - C_S(l)$, $S_S(l) = C_S(l) - C_R(l)$ を算出。
  - $L_R = \mathrm{argmax}_l S_R(l)$, $L_S = \mathrm{argmax}_l S_S(l)$ を選定。
  - $L_R == L_S$ または $\max S_R \le 0$ または $\max S_S \le 0$ の場合、`no_distinct_sites_identified` を返却。
  - `e3_csv_path` が存在しない場合は `FileNotFoundError` を raise（heuristic fallback を完全撤去）。
- `main()`:
  - `no_distinct_sites_identified` の場合は negative result として `e6_lmm_results.json` および manifest に記録し、エラー終了ではなく「特定不能」という科学的結果として終了。
  - `selectivity_info`（各タスクの最大選択性値、選択性プロファイル）を記録。

#### [MODIFY] [run_phase_c.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_c.py)
- dry-run mock の `e3_records` で、layer 0 を Reader-selective ($C_R > C_S$)、layer 14 を Self-selective ($C_S > C_R$) とし、モックテスト時にも自然に distinct sites が選出されるよう整合。

---

### 2. Double Dissociation 表記の完全統一
#### [MODIFY] [v1/README.md](file:///mnt/nas/home/hiromi/src/emo2/v1/README.md)
- 見出し `### 6.3 E6 Double Dissociation` $\rightarrow$ `### 6.3 E6 Task-Specific Causal Specialization` へ改定。

#### [MODIFY] [run.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/run.py)
- コメント・ログの `Double Dissociation` $\rightarrow$ `Task-Specific Causal Specialization` へ改定。

#### [MODIFY] [run_production_v1.sh](file:///mnt/nas/home/hiromi/scripts/run_production_v1.sh)
- スクリプト内コメントの統一。

#### [MODIFY] [summarize_phase_c.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/phase_c/summarize_phase_c.py)
- docstring およびレポート見出しを "Causal Specialization / Partial Dissociation" に統一。

---

### 3. V3 RQ2 の Discovery 明記
#### [MODIFY] [run_rq2_spatiotemporal_maps.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq2_spatiotemporal_maps.py)
- 出力 JSON および manifest のメタデータに `"analysis_role": "discovery"` を明記。

---

### 4. V2 Legacy Scripts / Config の隔離
#### [MOVE]
- `v2/scripts/run_lambda_dose_response.py` $\rightarrow$ `v2/scripts/legacy/run_lambda_dose_response.py`
- `v2/scripts/run_module_probing.py` $\rightarrow$ `v2/scripts/legacy/run_module_probing.py`
- `v2/scripts/run_all_models.sh` $\rightarrow$ `v2/scripts/legacy/run_all_models.sh`
- `v2/scripts/REPRODUCIBILITY.md` $\rightarrow$ `v2/scripts/legacy/REPRODUCIBILITY.md`
- `v2/configs/prompt_hashes.json` $\rightarrow$ `v2/configs/legacy/prompt_hashes.json`

---

## 検証計画 (Verification Plan)

### 自動テスト
- `pytest -q` で全単体テストが通過することを確認。
- `python v1/primary/phase_c/run_e6_specialization.py --help` の構文および引数チェック。
- `bash scripts/run_production_all.sh cpu --dry-run` を実行し、全4ステージがエラーなく高速完走することを確認。
- `results/` 配下が `.gitkeep` のみのクリーン状態を維持していることを確認。
